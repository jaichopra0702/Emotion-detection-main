import requests

url = "https://www.dropbox.com/s/nilt43hyl1dx82k/dataset.zip?dl=1"
local_filename = "dataset.zip"

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192): 
            f.write(chunk)

print(f"Downloaded file saved as {local_filename}")
