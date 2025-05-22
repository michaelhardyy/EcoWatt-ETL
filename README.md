# EcoWatt-ETL
EcoWatt ETL Implementation with Python and Azure Cloud, extracts raw CSV files from Azure Blob Storage, transforms them with Python / Pandas, and loads the cleansed, model‑driven tables into Azure SQL Database. A Streamlit dashboard then lets analysts drill into yearly KPIs, interactive charts and geospatial insights.

The stack was chosen so I could practise end‑to‑end cloud data engineering while keeping costs low and everything in Microsoft’s ecosystem.


# To run the application locally, follow these steps:

Create .env file in the project directory and add the required environment variables

Create a virtual environment and install the required packages: pip install -r requirements.txt

python main.py  # creates tables & loads data

streamlit run app.py

# Pre-requisite 

Python 3.10+

Azure CLI (for default credential login)

An Azure account with Blob Storage and SQL Database already provisioned.

# Required Environment variables:

ACCOUNT_STORAGE= ""

USERNAME_AZURE= ""

PASSWORD= ""

SERVER= "" 

DATABASE= ""

AZURE_STORAGE_CONNECTION_STRING=""

AZURE_SQL_CONNECTIONSTRING  = ""

