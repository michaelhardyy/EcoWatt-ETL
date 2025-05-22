import os, pyodbc
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import io
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
import json

load_dotenv()

# Retrieve Azure SQL connection string and storage details from environment variables
username = os.environ.get('USERNAME_AZURE')
password = os.environ.get('PASSWORD')
server = os.environ.get('SERVER')
database = os.environ.get('DATABASE')
account_storage = os.environ.get('ACCOUNT_STORAGE')
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
connection_string = os.getenv('AZURE_SQL_CONNECTIONSTRING')


# Using pyodbc with connection string
# engine = create_engine(f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server')
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={connection_string}")


class AzureDB():
    def __init__(self, local_path = "./data", account_storage = account_storage):
        self.local_path = local_path
        self.account_url = f"https://{account_storage}.blob.core.windows.net"
        self.default_credential = DefaultAzureCredential()
        self.blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        # self.blob_service_client = BlobServiceClient(self.account_url, credential=self.default_credential)
        
    def access_container(self, container_name): 
        # Use this function to create/access a new container
        try:
            # Creating container if not exist
            self.container_client = self.blob_service_client.create_container(container_name)
            print(f"Creating container {container_name} since not exist in database")
            self.container_name = container_name
    
        except Exception as ex:
            print(f"Acessing container {container_name}")
            # Access the container
            self.container_client = self.blob_service_client.get_container_client(container=container_name)
            self.container_name = container_name
            
    def delete_container(self):
        # Delete a container
        print("Deleting blob container...")
        self.container_client.delete_container()
        print("Done")
        
    def upload_blob(self, blob_name, blob_data = None):
        # Create a file in the local data directory to upload as blob to Azure
        local_file_name = blob_name
        upload_file_path = os.path.join(self.local_path, local_file_name)
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=local_file_name)
        print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

        if blob_data is not None:
            blob_client.create_blob_from_text(container_name=self.container_name, blob_name=blob_name, text=blob_data)
        else:
            # Upload the created file
            with open(file=upload_file_path, mode="rb") as data:
                blob_client.upload_blob(data)
                
    def list_blobs(self):
        print("\nListing blobs...")
        # List the blobs in the container
        blob_list = self.container_client.list_blobs()
        for blob in blob_list:
            print("\t" + blob.name)  
            
    def download_blob(self, blob_name):
        # Download the blob to local storage
        download_file_path = os.path.join(self.local_path, blob_name)
        print("\nDownloading blob to \n\t" + download_file_path)
        with open(file=download_file_path, mode="wb") as download_file:
                download_file.write(self.container_client.download_blob(blob_name).readall())
                
    def delete_blob(self, container_name: str, blob_name: str):
        # Deleting a blob
        print("\nDeleting blob " + blob_name)
        blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.delete_blob()
        
    def access_blob_csv(self, blob_name: str, **read_csv_kwargs) -> pd.DataFrame:
        """
        Download a CSV blob and return a DataFrame.
        Any kwargs you pass here (delimiter, skiprows, dtype, etc.) are forwarded to pd.read_csv.
        """
        print(f"Accessing blob {blob_name}")
        content = self.container_client.download_blob(blob_name).readall().decode('utf-8')
        return pd.read_csv(io.StringIO(content), **read_csv_kwargs)
    
    def upload_dataframe_sqldatabase(self, blob_name, blob_data):
        print("\nUploading to Azure SQL server as table:\n\t" + blob_name)
        blob_data.to_sql(blob_name, engine, if_exists='replace', index=False)
        primary = blob_name.replace('dim', 'id')
        if 'fact' in blob_name.lower():
            with engine.connect() as con:
                trans = con.begin()
                con.execute(text(f'ALTER TABLE [dbo].[{blob_name}] alter column {blob_name}_id bigint NOT NULL'))
                con.execute(text(f'ALTER TABLE [dbo].[{blob_name}] ADD CONSTRAINT [PK_{blob_name}] PRIMARY KEY CLUSTERED ([{blob_name}_id] ASC);'))
                trans.commit() 
        else:        
            with engine.connect() as con:
                trans = con.begin()
                con.execute(text(f'ALTER TABLE [dbo].[{blob_name}] alter column {primary} bigint NOT NULL'))
                con.execute(text(f'ALTER TABLE [dbo].[{blob_name}] ADD CONSTRAINT [PK_{blob_name}] PRIMARY KEY CLUSTERED ([{primary}] ASC);'))
                trans.commit() 
                
    def append_dataframe_sqldatabase(self, blob_name, blob_data):
        print("\nAppending to table:\n\t" + blob_name)
        blob_data.to_sql(blob_name, engine, if_exists='append', index=False)
    
    def delete_sqldatabase(self, table_name):
        with engine.connect() as con:
            trans = con.begin()
            con.execute(text(f"DROP TABLE [dbo].[{table_name}]"))
            trans.commit()
            
    def get_sql_table(self, query):        
        # Create connection and fetch data using Pandas        
        df = pd.read_sql_query(query, engine)
        # Convert DataFrame to the specified JSON format
        result = df.to_dict(orient='records')
        return result