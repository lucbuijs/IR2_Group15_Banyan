# download the pre-training corpora for the different languages
from datasets import load_dataset

langs = ['af', 'am', 'ar', 'en', 'es', 'ha', 'hi', 'id', 'mr', 'te']

for lang in langs:
    print(f'Downloading dataset for language: {lang}')
    ds = load_dataset("m0pper/Small-Multilingual-Corpora", lang)
    lines = []
    for i in range(len(ds['train'])):
        lines.append(ds['train'][i]['text'])

    with open(f'../data/{lang}_train.txt', 'w') as f:
        for line in lines:
            f.write(line + '\n')

    lines = []
    for i in range(len(ds['dev'])):
        lines.append(ds['dev'][i]['text'])

    with open(f'../data/{lang}_dev.txt', 'w') as f:
        for line in lines:
            f.write(line + '\n')
   