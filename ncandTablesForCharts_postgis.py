import xarray as xr
import geopandas as gpd
from sqlalchemy import create_engine, text, inspect
from geoalchemy2 import Geometry
import os
from tqdm import tqdm
import pandas as pd
from shapely.geometry import Point, LineString
import pypsa
import numpy as np

def clean_database(engine):
    print("Starting database cleanup...")
    inspector = inspect(engine)
    
    protected_tables = ['spatial_ref_sys', 'geography_columns', 'geometry_columns', 'raster_columns', 'raster_overviews']
    
    with engine.connect() as connection:
        tables = inspector.get_table_names(schema='public')
        
        for table in tables:
            if table not in protected_tables:
                print(f"Deleting table: {table}")
                connection.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        
        connection.commit()
    
    print("Database cleaned, maintaining essential PostGIS tables.")


def create_postgis_table(engine, table_name, columns):
    if table_name.lower() in ["all", "user", "group"]:
        table_name = f"table_{table_name}"
    
    print(f"Creating table {table_name} in the database...")
    column_defs = []
    for col, dtype in columns.items():
        if col == 'country_code':
            continue 
        col = col.replace(':', '_').replace('.', '_').replace(' ', '_')
        if dtype == 'geometry':
            column_defs.append(f"{col} GEOMETRY")
        elif isinstance(dtype, np.dtype):
            if np.issubdtype(dtype, np.floating):
                column_defs.append(f"{col} FLOAT")
            elif np.issubdtype(dtype, np.integer):
                column_defs.append(f"{col} INTEGER")
            elif np.issubdtype(dtype, np.bool_):
                column_defs.append(f"{col} BOOLEAN")
            else:
                column_defs.append(f"{col} TEXT")
        elif str(dtype).startswith('float'):
            column_defs.append(f"{col} FLOAT")
        elif str(dtype).startswith('int'):
            column_defs.append(f"{col} INTEGER")
        elif str(dtype) == 'bool':
            column_defs.append(f"{col} BOOLEAN")
        else:
            column_defs.append(f"{col} TEXT")
    
    columns_sql = ", ".join(column_defs)
    sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY, 
        country_code CHAR(2),
        {columns_sql}
    )
    """
    
    with engine.connect() as conn:
        conn.execute(text(sql))
    print(f"Table {table_name} created successfully.")

def get_total_demand(n):
    demand = n.loads_t.p_set.multiply(n.snapshot_weightings.objective, axis=0).sum().sum() / 1e6
    return demand.round(4)

def get_installed_capacities(n):
    gen_capacities = n.generators.groupby("carrier").p_nom.sum()
    storage_capacities = n.storage_units.groupby("carrier").p_nom.sum()
    capacities = (pd.concat([gen_capacities, storage_capacities], axis=0) / 1e3).round(4)
    if "load" in n.generators.carrier.unique():
        capacities.drop("load", inplace=True)
    return capacities

def get_generation_mix(n):
    gen_generation = n.generators_t.p.multiply(n.snapshot_weightings.objective, axis=0).groupby(n.generators.carrier, axis=1).sum().sum()
    storage_generation = n.storage_units_t.p.multiply(n.snapshot_weightings.objective, axis=0).groupby(n.storage_units.carrier, axis=1).sum().sum()
    generation_mix = pd.concat([gen_generation, storage_generation], axis=0) / 1e6
    return generation_mix.round(4)

def get_total_generation_by_ac_bus(n):
    ac_carrier = ["AC"]
    ac_buses = n.buses.query("carrier in @ac_carrier").index
    
    gen_generation_by_bus = n.generators_t.p.multiply(n.snapshot_weightings.objective, axis=0).groupby([n.generators.bus, n.generators.carrier], axis=1).sum().sum()
    gen_generation_by_ac_bus = gen_generation_by_bus.loc[gen_generation_by_bus.index.get_level_values(0).isin(ac_buses)]
    gen_generation_by_ac_bus.loc[gen_generation_by_ac_bus.index.get_level_values(1) == 'load'] *= 1e-3

    store_generation_by_bus = n.storage_units_t.p.multiply(n.snapshot_weightings.objective, axis=0).groupby([n.storage_units.bus, n.storage_units.carrier], axis=1).sum().sum()
    store_generation_by_ac_bus = store_generation_by_bus.loc[store_generation_by_bus.index.get_level_values(0).isin(ac_buses)]

    gen_generation_by_ac_bus_reset = gen_generation_by_ac_bus.reset_index()
    store_generation_by_ac_bus_reset = store_generation_by_ac_bus.reset_index()

    gen_generation_by_ac_bus_reset.columns = ['bus', 'carrier', 'generation']
    store_generation_by_ac_bus_reset.columns = ['bus', 'carrier', 'generation']

    total_generation_by_ac_bus = pd.concat([gen_generation_by_ac_bus_reset, store_generation_by_ac_bus_reset])
    return total_generation_by_ac_bus

def load_data_to_postgis(n, engine, country_code):
    tables = ['buses', 'carriers', 'generators', 'lines', 'loads', 'storage_units', 'stores']
    
    for table_name in tables:
        print(f"Loading data from {table_name} for country {country_code} into the database...")
        
        df = getattr(n, table_name)
        
        index_name = df.index.name if df.index.name else table_name.capitalize()
        df = df.reset_index()
        df = df.rename(columns={'index': index_name})
        
        columns = [index_name] + [col for col in df.columns if col != index_name]
        df = df[columns]
        
        dtypes = df.dtypes.to_dict()
        
        if table_name == 'buses':
            ac_carrier = ["AC"]
            ac_buses = n.buses.query("carrier in @ac_carrier").index
            df = df[df[index_name].isin(ac_buses)]
        
        df['country_code'] = country_code
        
        if table_name in ['buses', 'lines']:
            if table_name == 'buses':
                df['geometry'] = df.apply(lambda row: Point(row['x'], row['y']) if 'x' in row and 'y' in row else None, axis=1)
            elif table_name == 'lines':
                print("Processing 'lines' table")
                print(f"Columns in df: {df.columns}")
                print(f"First rows of df:\n{df.head()}")
                
                buses = n.buses[['x', 'y']]
                df = df.merge(buses.add_suffix('_0'), left_on='bus0', right_index=True)
                df = df.merge(buses.add_suffix('_1'), left_on='bus1', right_index=True)
                
                df['geometry'] = df.apply(lambda row: 
                    LineString([(row['x_0'], row['y_0']), (row['x_1'], row['y_1'])])
                    if all(x in row and pd.notnull(row[x]) for x in ['x_0', 'y_0', 'x_1', 'y_1']) 
                    else None, axis=1)
                
                print(f"Geometries created: {df['geometry'].notna().sum()}")
        
        df.columns = [col.replace(':', '_').replace('.', '_').replace(' ', '_') for col in df.columns]
        
        if 'geometry' in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        else:
            gdf = df
        
        columns = {**dtypes, 'country_code': 'object', 'geometry': 'geometry'}
        create_postgis_table(engine, table_name, columns)
        
        print(f"Inserting data into table {table_name}...")
        try:
            if 'geometry' in gdf.columns:
                gdf_with_geometry = gdf[gdf['geometry'].notnull()]
                gdf_without_geometry = gdf[gdf['geometry'].isnull()]
                if not gdf_with_geometry.empty:
                    gdf_with_geometry.to_postgis(table_name, engine, if_exists='append', index=False)
                if not gdf_without_geometry.empty:
                    gdf_without_geometry.to_sql(table_name, engine, if_exists='append', index=False)
            else:
                gdf.to_sql(table_name, engine, if_exists='append', index=False)
            
            print(f"Data inserted into {table_name}:")
            print(gdf.head())
            print(f"Data shape: {gdf.shape}")
        except Exception as e:
            print(f"Error inserting data into table {table_name}: {e}")
            print("First 5 rows of data:")
            print(gdf.head())
            print("\nData type information:")
            print(gdf.dtypes)
        print(f"Data from {table_name} for country {country_code} inserted successfully.")

    print(f"Loading additional data for country {country_code} into the database...")

    # Total demand
    total_demand = get_total_demand(n)
    demand_df = pd.DataFrame({'country_code': [country_code], 'total_demand_twh': [total_demand]})
    demand_df.to_sql('total_demand', engine, if_exists='append', index=False)

    # Installed capacities
    installed_capacities = get_installed_capacities(n)
    capacities_df = installed_capacities.reset_index()
    capacities_df.columns = ['carrier', 'capacity_gw']
    capacities_df['country_code'] = country_code
    capacities_df.to_sql('installed_capacities', engine, if_exists='append', index=False)

    # Generation mix
    generation_mix = get_generation_mix(n)
    mix_df = generation_mix.reset_index()
    mix_df.columns = ['carrier', 'generation_twh']
    mix_df['country_code'] = country_code
    mix_df.to_sql('generation_mix', engine, if_exists='append', index=False)

    # Total generation by AC bus
    total_generation_by_ac_bus = get_total_generation_by_ac_bus(n)
    total_generation_by_ac_bus['country_code'] = country_code
    total_generation_by_ac_bus.to_sql('total_generation_by_ac_bus', engine, if_exists='append', index=False)

    print(f"Additional data for country {country_code} inserted successfully.")

def process_netcdf(file_path, engine):
    print(f"Processing NetCDF file: {file_path}")
    country_code = os.path.basename(file_path).split('_')[0]
    n = pypsa.Network(file_path)
    load_data_to_postgis(n, engine, country_code)

def main():
    db_params = {
        'dbname': 'pypsa_earth_db',
        'user': 'postgres',
        'password': 'oetpostgres',
        'host': '34.68.214.20',
        'port': '5432'
    }
    engine = create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}")

    print("Starting database cleanup...")
    clean_database(engine)

    netcdf_files = [
        r"D:\Work\OET\Dashboard\Yerbol code\NetCDF\AU_2021.nc",
        r"D:\Work\OET\Dashboard\Yerbol code\NetCDF\BR_2021.nc",
        r"D:\Work\OET\Dashboard\Yerbol code\NetCDF\CO_2021.nc",
        r"D:\Work\OET\Dashboard\Yerbol code\NetCDF\DE_2021.nc",
        r"D:\Work\OET\Dashboard\Yerbol code\NetCDF\IN_2021.nc",
        r"D:\Work\OET\Dashboard\Yerbol code\NetCDF\IT_2021.nc",
        r"D:\Work\OET\Dashboard\Yerbol code\NetCDF\MX_2021.nc",
        r"D:\Work\OET\Dashboard\Yerbol code\NetCDF\NG_2021.nc",
        r"D:\Work\OET\Dashboard\Yerbol code\NetCDF\US_2021.nc",
        r"D:\Work\OET\Dashboard\Yerbol code\NetCDF\ZA_2021.nc"
    ]

    for netcdf_file in tqdm(netcdf_files, desc="Processing NetCDF files"):
        process_netcdf(netcdf_file, engine)

if __name__ == "__main__":
    main()
