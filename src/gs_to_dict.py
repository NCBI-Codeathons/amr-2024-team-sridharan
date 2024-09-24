import gzip
from google.cloud import storage
from io import BytesIO, StringIO
from Bio import SeqIO

def parse_fasta_from_gcs(gcs_url,output_path='.'):
    # Extract bucket name and file path from the gs:// URL
    if gcs_url.startswith("gs://"):
        bucket_name, *file_path = gcs_url[5:].split("/")
        file_path = "/".join(file_path)
    else:
        raise ValueError("Invalid GCS URL. It should start with 'gs://'")

    # Initialize Google Cloud Storage client
    client = storage.Client()

    # Get the bucket and the blob (file) from GCS
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Download the file content as bytes
    file_data = blob.download_as_bytes()

    # save downloaded file
    with open(output_path,'wb') as file:
        file.write(file_data)

    # # Create a BytesIO buffer for the gzip file and extract the content
    # with gzip.GzipFile(fileobj=BytesIO(file_data)) as gz:
    #     fasta_data = gz.read().decode('utf-8')  # Read as a string (text mode)

    # # Parse the FASTA content
    # fasta_dict = {}
    
    # # Use StringIO to wrap the string in a text stream and parse it with SeqIO
    # fasta_io = StringIO(fasta_data)  # Convert string to a StringIO object for text mode parsing
    # for record in SeqIO.parse(fasta_io, "fasta"):
    #     fasta_dict['header'] = record.description
    #     fasta_dict['sequence'] = str(record.seq)

    # return fasta_dict

if __name__=="__main__":
    # Example usage
    gcs_url = "gs://ncbi-pathogen-assemblies/Enterococcus_faecium/1361/000/DAIGVV010000255.1.fna.gz"
    fasta_dict = parse_fasta_from_gcs(gcs_url)
    print(fasta_dict)
