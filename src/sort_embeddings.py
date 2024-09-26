import polars as pl

alldata = "/shared_venv/data_from_bigquery.csv"
values = "temp/all_embeddings.txt"

data = pl.read_csv(alldata)

# Filter and sort by values in all_embeddings.txt
data = data.filter(pl.col('protein_acc').is_in(values)).sort(by=lambda x: x['protein_acc'].apply(lambda y: values.index(y)))

data.write_csv("filtered_data.csv")