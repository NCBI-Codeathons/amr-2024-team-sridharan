import os,pickle
import numpy as np

pickledir = "temp/embeddings"

files = os.listdir(pickledir)

data = []
for file in files:
    print(f"reading {file}")
    with open(os.path.join(pickledir,file),'rb') as handle:
        data += pickle.load(handle)



embeddings = [i[0][0][1] for i in data]
accessions = [i[0][0][0] for i in data]

embeddings = np.array(embeddings)

dir_above_pickledir = '/'.join(pickledir.split('/')[:-1])

with open(os.path.join(dir_above_pickledir,"all_embeddings.pickle"),'wb') as handle:
    pickle.dump(embeddings,handle)

with open(os.path.join(dir_above_pickledir,"all_accessions.txt"),'w') as handle:
    handle.write("\n".join(accessions))