from google.cloud import bigquery
import pandas as pd

# Initialize the BigQuery client
client = bigquery.Client()

# Define your query
query = """
    SELECT *
    FROM `ncbi-2024-amr-codeathon.ESKAPE_AMR_only.AMR`
    """

# Execute the query and fetch the results into a pandas DataFrame
query_job = client.query(query)  # Make an API request.
df = query_job.result().to_dataframe()  # Fetch the results.

# Save the DataFrame to a CSV file on the VM
df.to_csv('/shared_venv/data_from_bigquery.csv', index=False)

print("Data saved successfully to 'data_from_bigquery.csv'")
