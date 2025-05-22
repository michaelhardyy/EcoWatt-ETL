import os, uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from dotenv import load_dotenv 
from utils.datsetup import *
import numpy as np

load_dotenv()

account_storage = os.environ.get('ACCOUNT_STORAGE')

azureDB = AzureDB()
azureDB.access_container("csv-files")
azureDB.list_blobs()

def extract_data(azureDB):
    """Extract data from CSV files"""
    print("Starting data extraction...")

    # Extract EV data
    ev_df = azureDB.access_blob_csv('Ev_Population.csv', delimiter=';')
    print(f"Extracted {len(ev_df)} EV records")
    
    # Extract electricity consumption data
    electricity_df = azureDB.access_blob_csv('Electricity_Consumption.csv', delimiter=';')
    print(f"Extracted {len(electricity_df)} electricity consumption records")
    
    # Extract pollution data
    pollution_df = azureDB.access_blob_csv(
    'Pollution_Index.csv',
    delimiter=',',        
    header=2,             
    parse_dates=['Date'],
    dayfirst=True
    )
    pollution_df.columns = pollution_df.columns.str.strip()
    print(f"Extracted {len(pollution_df)} pollution records")
    
    return ev_df, electricity_df, pollution_df

def transform_ev_data(ev_df):
    """Transform EV data"""
    print("Transforming EV data...")
    
    # Clean column names
    ev_df.columns = [col.strip().rstrip(';') for col in ev_df.columns]
    
    # Extract only BEV (Battery Electric Vehicles) and PHEV (Plug-in Hybrid Electric Vehicles)
    ev_df = ev_df[ev_df['FUEL_TYPE'].isin(['BEV', 'PHEV'])]
    
    # Create a vehicle type category
    ev_df['VEHICLE_CATEGORY'] = ev_df['VEHICLE_TYPE'].str.strip()
    
    # Extract year from model
    ev_df['MODEL_YEAR'] = ev_df['MODEL'].str.extract(r'(\d{4})').astype('float')
    
    # Clean price data
    ev_df['PRICE'] = ev_df['LISTED_PRICE'].str.replace('*', '').str.strip()
    ev_df['PRICE'] = pd.to_numeric(ev_df['PRICE'], errors='coerce')
    
    # Process range data
    ev_df['RANGE_KM'] = pd.to_numeric(ev_df['RANGE'], errors='coerce')
    # Clean suburb data
    ev_df['SUBURB'] = ev_df['SUBURB'].str.strip()
    
    # Group by suburb and count EVs
    suburb_ev_counts = ev_df.groupby(['SUBURB', 'FUEL_TYPE']).size().reset_index(name='COUNT')
    
    # Calculate summary metrics
    ev_summary = pd.DataFrame({
        'TOTAL_EVs': ev_df.groupby('SUBURB').size(),
        'BEV_COUNT': ev_df[ev_df['FUEL_TYPE'] == 'BEV'].groupby('SUBURB').size(),
        'PHEV_COUNT': ev_df[ev_df['FUEL_TYPE'] == 'PHEV'].groupby('SUBURB').size(),
        'AVG_RANGE_KM': ev_df.groupby('SUBURB')['RANGE_KM'].mean(),
        'AVG_PRICE': ev_df.groupby('SUBURB')['PRICE'].mean()
    }).reset_index()
    
    # Fill NaN values
    ev_summary = ev_summary.fillna(0)

    return ev_summary

def transform_electricity_data(electricity_df):
    """Transform electricity consumption data"""
    print("Transforming electricity consumption data...")
    
    # Clean column names
    electricity_df.columns = [col.strip() for col in electricity_df.columns]
    
    # Extract data for 2022-2023
    electricity_subset = electricity_df[['Name', 'F2021_22', 'F2022_23']]
    
    # Rename columns for clarity
    electricity_subset = electricity_subset.rename(columns={
        'Name': 'SUBURB',
        'F2021_22': 'CONSUMPTION_2022',
        'F2022_23': 'CONSUMPTION_2023'
    })
    
    # Clean suburb names
    electricity_subset['SUBURB'] = electricity_subset['SUBURB'].str.split('+').str[0].str.strip()
    
    # Calculate year-over-year change
    electricity_subset['CONSUMPTION_CHANGE_PCT'] = ((electricity_subset['CONSUMPTION_2023'] - 
                                                    electricity_subset['CONSUMPTION_2022']) / 
                                                   electricity_subset['CONSUMPTION_2022'] * 100)
    
    return electricity_subset

def transform_pollution_data(pollution_df):
    """Transform pollution data"""
    print("Transforming pollution data...")
    
    # Extract relevant columns for NO2 pollution
    pollution_cols = [col for col in pollution_df.columns if 'NO2 annual average' in col]
    pollution_cols.append('Date')
    
    pollution_subset = pollution_df[pollution_cols]
    
    # Reshape data from wide to long format
    pollution_long = pd.melt(
        pollution_subset,
        id_vars=['Date'],
        value_vars=[col for col in pollution_cols if col != 'Date'],
        var_name='LOCATION',
        value_name='NO2_LEVEL'
    )
    
    # Extract suburb name from location
    pollution_long['SUBURB'] = pollution_long['LOCATION'].str.extract(r'(.*) NO2 annual average')
    pollution_long['SUBURB'] = pollution_long['SUBURB'].str.title()
    
    # Map pollution measurement locations to suburbs
    suburb_mapping = {
        'Alexandria': 'Alexandria',
        'Rozelle': 'Rozelle',
        'Earlwood': 'Earlwood',
        'Cook And Phillip': 'Sydney',
        'Randwick': 'Randwick',
        'Macquarie Park': 'Macquarie Park',
        'Parramatta North': 'Parramatta'
    }
    
    # Filter for mapped suburbs only and rename
    pollution_long = pollution_long[pollution_long['SUBURB'].isin(suburb_mapping.keys())]
    pollution_long['SUBURB'] = pollution_long['SUBURB'].map(suburb_mapping)
    
    # Convert date to year
    pollution_long['YEAR'] = pd.to_datetime(pollution_long['Date']).dt.year
    
    # Keep only 2022 and 2023 data
    pollution_long = pollution_long[pollution_long['YEAR'].isin([2022, 2023])]
    
    # Pivot to get pollution by suburb and year
    pollution_pivot = pollution_long.pivot_table(
        index='SUBURB',
        columns='YEAR',
        values='NO2_LEVEL',
        aggfunc='mean'
    ).reset_index()
    
    pollution_pivot.columns = ['SUBURB', 'NO2_2022', 'NO2_2023']
    
    # Calculate year-over-year change
    pollution_pivot['NO2_CHANGE'] = pollution_pivot['NO2_2023'] - pollution_pivot['NO2_2022']
    pollution_pivot['NO2_CHANGE_PCT'] = ((pollution_pivot['NO2_2023'] - pollution_pivot['NO2_2022']) / 
                                        pollution_pivot['NO2_2022'] * 100)
    
    return pollution_pivot

def merge_datasets(ev_summary, electricity_subset, pollution_pivot):
    """Merge all transformed datasets"""
    print("Merging datasets...")
    
    # First merge EV and electricity data
    merged_df = pd.merge(ev_summary, electricity_subset, on='SUBURB', how='outer')
    
    # Then merge with pollution data
    final_df = pd.merge(merged_df, pollution_pivot, on='SUBURB', how='outer')
    
    # Fill NaN values
    final_df = final_df.fillna({
        'TOTAL_EVs': 0,
        'BEV_COUNT': 0,
        'PHEV_COUNT': 0,
        'AVG_RANGE_KM': 0,
        'AVG_PRICE': 0,
        'CONSUMPTION_2022': 0,
        'CONSUMPTION_2023': 0,
        'CONSUMPTION_CHANGE_PCT': 0,
        'NO2_2022': 0,
        'NO2_2023': 0,
        'NO2_CHANGE': 0,
        'NO2_CHANGE_PCT': 0
    })
    
    # Calculate additional metrics
    final_df['EV_PER_ENERGY_UNIT'] = final_df['TOTAL_EVs'] / (final_df['CONSUMPTION_2023'] / 1000000)
    final_df['NO2_PER_EV'] = final_df['NO2_2023'] / final_df['TOTAL_EVs'].replace(0, 1)
    final_df['EV_ADOPTION_SCORE'] = final_df['TOTAL_EVs'] * (1 - final_df['NO2_CHANGE_PCT'] / 100)
    
    return final_df

def create_dimension_tables(final_df, ev_df):
    """Create dimension tables for the star schema"""
    print("Creating dimension tables...")
    
    # Time dimension
    time_dim = pd.DataFrame({
        'id_time': [2022, 2023],
        'YEAR': [2022, 2023],
        'IS_CURRENT_YEAR': [False, True]
    })
    time_dim.to_csv('extracted/time_dim.csv')
    
    # Suburb dimension
    suburb_dim = pd.DataFrame({
        'id_suburb': range(1, len(final_df) + 1),
        'SUBURB_NAME': final_df['SUBURB'],
    })
    suburb_dim.to_csv('extracted/suburb_dim.csv')
    
    # Vehicle type dimension
    vehicle_type_dim = pd.DataFrame({
        'id_vehicle_type': range(1, len(ev_df['VEHICLE_TYPE'].unique()) + 1),
        'VEHICLE_TYPE': sorted(ev_df['VEHICLE_TYPE'].unique())
    })
    vehicle_type_dim.to_csv('extracted/vehicle_dim.csv')
    
    # Fuel type dimension
    fuel_type_dim = pd.DataFrame({
        'id_fuel_type': [1, 2],
        'FUEL_TYPE': ['BEV', 'PHEV'],
        'FUEL_DESCRIPTION': ['Battery Electric Vehicle', 'Plug-in Hybrid Electric Vehicle']
    })
    fuel_type_dim.to_csv('extracted/fuel_dim.csv')
    
    return time_dim, suburb_dim, vehicle_type_dim, fuel_type_dim

def create_fact_tables(final_df, suburb_dim):
    """Create fact tables for the star schema"""
    print("Creating fact tables...")
    
    # Join suburb dimension to get keys
    final_df_with_keys = pd.merge(
        final_df,
        suburb_dim,
        left_on='SUBURB',
        right_on='SUBURB_NAME',
        how='left'
    )
    
    # EV impact fact table
    ev_impact_fact = pd.DataFrame({
        'fact_ev_impact_id': range(1, len(final_df_with_keys)+1),
        'id_suburb': final_df_with_keys['id_suburb'],
        'YEAR': 2023,
        'TOTAL_EVS': final_df_with_keys['TOTAL_EVs'],
        'BEV_COUNT': final_df_with_keys['BEV_COUNT'],
        'PHEV_COUNT': final_df_with_keys['PHEV_COUNT'],
        'AVG_RANGE_KM': final_df_with_keys['AVG_RANGE_KM'],
        'AVG_PRICE': final_df_with_keys['AVG_PRICE'],
        'EV_ADOPTION_SCORE': final_df_with_keys['EV_ADOPTION_SCORE']
    })

    ev_impact_fact = ev_impact_fact.replace([float('inf'), float('-inf')], 0)
    ev_impact_fact = ev_impact_fact.fillna(0)
    
    # Energy vs Pollution fact table
    energy_pollution_fact = pd.DataFrame({
        'fact_energy_pollution_id': range(1, len(final_df_with_keys)+1),
        'id_suburb': final_df_with_keys['id_suburb'],
        'YEAR': 2023,
        'ENERGY_CONSUMPTION': final_df_with_keys['CONSUMPTION_2023'],
        'ENERGY_CHANGE_PCT': final_df_with_keys['CONSUMPTION_CHANGE_PCT'],
        'NO2_LEVEL': final_df_with_keys['NO2_2023'],
        'NO2_CHANGE': final_df_with_keys['NO2_CHANGE'],
        'NO2_CHANGE_PCT': final_df_with_keys['NO2_CHANGE_PCT'],
        'EV_PER_ENERGY_UNIT': final_df_with_keys['EV_PER_ENERGY_UNIT'],
        'NO2_PER_EV': final_df_with_keys['NO2_PER_EV']
    })
    
    energy_pollution_fact = energy_pollution_fact.replace([float('inf'), float('-inf')], 0)
    energy_pollution_fact = energy_pollution_fact.fillna(0)

    # Create historical rows for 2022
    energy_pollution_fact_2022 = pd.DataFrame({
        'fact_energy_pollution_id': range(
            len(final_df_with_keys)+1,   # start
            len(final_df_with_keys)*2+1  # stop
        ),
        'id_suburb': final_df_with_keys['id_suburb'],
        'YEAR': 2022,
        'ENERGY_CONSUMPTION': final_df_with_keys['CONSUMPTION_2022'],
        'ENERGY_CHANGE_PCT': 0,  # No previous year for comparison
        'NO2_LEVEL': final_df_with_keys['NO2_2022'],
        'NO2_CHANGE': 0,  # No previous year for comparison
        'NO2_CHANGE_PCT': 0,  # No previous year for comparison
        'EV_PER_ENERGY_UNIT': final_df_with_keys['TOTAL_EVs'] / (final_df_with_keys['CONSUMPTION_2022'] / 1000000),
        'NO2_PER_EV': final_df_with_keys['NO2_2022'] / final_df_with_keys['TOTAL_EVs'].replace(0, 1)
    })

    # Safely calculate EV_PER_ENERGY_UNIT
    def safe_ev_per_energy(row):
        try:
            if row['CONSUMPTION_2022'] <= 0:
                return 0
            result = row['TOTAL_EVs'] / (row['CONSUMPTION_2022'] / 1000000)
            # Check for infinity or too large values
            if result > 1e15 or pd.isna(result) or np.isinf(result):
                return 0
            return result
        except Exception:
            return 0
    
    # Safely calculate NO2_PER_EV
    def safe_no2_per_ev(row):
        try:
            if row['TOTAL_EVs'] <= 0:
                return 0
            result = row['NO2_2022'] / row['TOTAL_EVs']
            # Check for infinity or too large values
            if result > 1e15 or pd.isna(result) or np.isinf(result):
                return 0
            return result
        except Exception:
            return 0
    
    # Apply safe calculations
    energy_pollution_fact_2022['EV_PER_ENERGY_UNIT'] = final_df_with_keys.apply(safe_ev_per_energy, axis=1)
    energy_pollution_fact_2022['NO2_PER_EV'] = final_df_with_keys.apply(safe_no2_per_ev, axis=1)
    
    # Final cleanup
    energy_pollution_fact_2022 = energy_pollution_fact_2022.replace([float('inf'), float('-inf')], 0)
    energy_pollution_fact_2022 = energy_pollution_fact_2022.fillna(0)
    
    # Round all float columns to 6 decimal places to avoid precision issues
    for df in [energy_pollution_fact, energy_pollution_fact_2022, ev_impact_fact]:
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = df[col].round(6)
    
    # Combine 2022 and 2023 data
    energy_pollution_fact = pd.concat([energy_pollution_fact, energy_pollution_fact_2022])
    
    ev_impact_fact.to_csv('ev_fact.csv')
    energy_pollution_fact.to_csv('energy_fact.csv')

    
    return ev_impact_fact, energy_pollution_fact

def load_to_azure(azureDB, time_dim, suburb_dim, vehicle_type_dim, fuel_type_dim, ev_impact_fact, energy_pollution_fact):
    """Load dimension and fact tables to Azure"""
    print("\n=== LOADING DATA TO AZURE ===")
    
    # Load dimension tables
    print("\nLoading dimension tables to Azure SQL database...")
    azureDB.upload_dataframe_sqldatabase("dim_time", time_dim)
    azureDB.upload_dataframe_sqldatabase("dim_suburb", suburb_dim)
    azureDB.upload_dataframe_sqldatabase("dim_vehicle_type", vehicle_type_dim)
    azureDB.upload_dataframe_sqldatabase("dim_fuel_type", fuel_type_dim)
    
    # Load fact tables
    print("\nLoading fact tables to Azure SQL database...")
    azureDB.upload_dataframe_sqldatabase("fact_ev_impact", ev_impact_fact)
    azureDB.upload_dataframe_sqldatabase("fact_energy_pollution", energy_pollution_fact)
    
    # Save to blob storage as well for backup/archival
    print("\nSaving data to Azure Blob Storage for archival...")


def main():
    print("Extracting Data")
    ev_df, electricity_df, pollution_df = extract_data(azureDB)
    
    print("Transforming EV Data")
    ev_summary = transform_ev_data(ev_df)
    print("\nSample of transformed EV data:")
    print(ev_summary.head())

    print("Transforming Electricty Data")
    electricity_subset = transform_electricity_data(electricity_df)
    print("\nSample of transformed Electricity data:")
    print(electricity_subset.head())

    print("Transforming Pollution Data")
    pollution_pivot= transform_pollution_data(pollution_df)
    print("\nSample of transformed Pollution data:")
    print(pollution_pivot.head())
    
    # Merge datasets
    final_df = merge_datasets(ev_summary, electricity_subset, pollution_pivot)
    # Quick shape & head
    print("Final merged shape:", final_df.shape)
    print(final_df.head(), "\n")

    # Unique-SUBURB counts
    print("Unique suburbs:",
          "EV:", ev_summary['SUBURB'].nunique(),
          "Elec:", electricity_subset['SUBURB'].nunique(),
          "Poll:", pollution_pivot['SUBURB'].nunique(),
          "Final:", final_df['SUBURB'].nunique()
    )

    # Dimension Tables
    time_dim, suburb_dim, vehicle_type_dim, fuel_type_dim = \
        create_dimension_tables(final_df, ev_df)

    print("Time DT")
    print("Shape:", time_dim.shape)
    print(time_dim, "\n")

    print("Suburb DT")
    print("Shape:", suburb_dim.shape)
    print(suburb_dim.head(), "\n")

    print("Vehicle-Type DT")
    print("Shape:", vehicle_type_dim.shape)
    print(vehicle_type_dim, "\n")

    print("Fuel-Type DT")
    print("Shape:", fuel_type_dim.shape)
    print(fuel_type_dim, "\n")

    # Fact tables
    ev_impact_fact, energy_pollution_fact = create_fact_tables(final_df, suburb_dim)

    print("EV Impact Fact")
    print("Shape:", ev_impact_fact.shape)
    print(ev_impact_fact.head(), "\n")

    print("Energy vs Pollution Fact")
    print("Shape:", energy_pollution_fact.shape)
    print(energy_pollution_fact.head(), "\n")

    # … after create_dimension_tables and create_fact_tables …
    load_to_azure(azureDB, time_dim, suburb_dim, vehicle_type_dim, fuel_type_dim, 
                 ev_impact_fact, energy_pollution_fact)

if __name__ == "__main__":
    main()