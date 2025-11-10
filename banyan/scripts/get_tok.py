# downloads all the necessary tokenizers from the BPEMB Package
from bpemb import BPEmb

langs = ['af', 'am', 'ar', 'en', 'es', 'ha', 'hi', 'id', 'mr', 'te']

for lang in langs:
    bpemb = BPEmb(lang=lang, vs=25000, dim=100)
    print('Downloaded BPEMB Tokeniser for language:', lang)

