# Get dummy fMRI and genomics data

import requests

host = 'https://hunimal.org/Hackathon/data'
files = ['fmri-FC.pkl', 'fmri-FC-slim.pkl', 'T2Dcounts.pkl', 'GSE202295_gene_counts.txt']
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

for file in files:
    r = requests.get(f'{host}/{file}', headers=headers)
    print(file)
    print(r.status_code)
    if r.status_code != 200:
        print('Something went wrong')
    else:
        with open(f'data/{file}', 'wb') as f:
            f.write(r.content)
        print('Done')
