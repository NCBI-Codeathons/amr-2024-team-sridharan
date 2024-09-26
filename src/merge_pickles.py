import os,pickle, argparse
import numpy as np
import polars as pl

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Get arguments')
parser.add_argument('-ep', '--embeddingspath', type=str, default="temp/embeddings", help='folder where the embedding pickles are')
parser.add_argument('-d', '--datafile', type=str, default="/shared_venv/data_from_bigquery.csv", help='file with downloaded database in csv format')
args = parser.parse_args()

pickledir = args.embeddingspath
alldata = args.datafile
output_path = alldata.rsplit('/',1)[0]

files = os.listdir(pickledir)

data = []
for file in files:
    print(f"reading {file}")
    with open(os.path.join(pickledir,file),'rb') as handle:
        data += pickle.load(handle)



embeddings = [i[0][1] for i in data]
accessions = [i[0][0][:-6] for i in data]

embeddings = np.array(embeddings)

dir_above_pickledir = pickledir.rsplit('/',1)[0]

with open(os.path.join(dir_above_pickledir,"all_embeddings.pickle"),'wb') as handle:
    pickle.dump(embeddings,handle)

with open(os.path.join(dir_above_pickledir,"all_accessions.txt"),'w') as handle:
    handle.write("\n".join(accessions))

values = accessions

data = pl.read_csv(alldata)
# Filter and sort by values in all_embeddings.txt
data = data.filter(pl.col('protein_acc').is_in(values))
data = data.with_columns(pl.Series([values.index(x) for x in data['protein_acc']]).alias('sort_index'))
data = data.sort(by='sort_index')

data.write_csv("/shared_venv/filtered_data.csv")