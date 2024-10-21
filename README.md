# JI_GIS_Data_processing

# PyPSA Earth Data Processing and PostGIS Integration

This repository contains Python scripts for processing PyPSA-Earth network data from NetCDF files and integrating it into a PostGIS database. The scripts facilitate the extraction, transformation, and loading of power system data for multiple countries, making it suitable for further analysis and visualization.

## Key Features:

- Process NetCDF files containing PyPSA-Earth network data
- Clean and prepare the existing PostGIS database
- Create and populate PostGIS tables with network components (buses, lines, generators, etc.)
- Calculate and store aggregated data for charts and analysis:
  - Total demand
  - Installed capacities
  - Generation mix
  - Total generation by AC bus
- Handle geospatial data for network components
- Support for multiple countries

## Files:

1. `NetCDFtoPostGIS.py`: Main script for processing NetCDF files and loading data into PostGIS
2. `ncandTablesForCharts_postgis.py`: Extended script with additional data processing for chart generation

These scripts are designed to work with PyPSA-Earth data and require a PostgreSQL database with PostGIS extension.
