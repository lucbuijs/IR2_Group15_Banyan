# evaluate retrieval performance on Arguana and Quora 
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from typing import List, Dict
from models import Banyan
from beir.retrieval.evaluation import EvaluateRetrieval
from bpemb import BPEmb
from tqdm import trange
import torch
import sys 


class StrAE_DE:
    def __init__(self, model, **kwargs):
        self.model = model 
        self.tokenizer = BPEmb(lang='en', vs=25000, dim=100)
        self.model.eval()
        self.model.to('cuda')
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                encoded = [torch.tensor(self.tokenizer.encode_ids(x)) for x in queries[start_idx:start_idx+batch_size]]
                encoded = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=25000).to(self.model.device)
                model_out, _ = self.model(encoded, encoded)
                query_embeddings.append(model_out)
        
        return torch.cat(query_embeddings, dim=0).cpu().numpy()

    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        corpus_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(corpus), batch_size):
                encoded = [torch.tensor(self.tokenizer.encode_ids(x['text']) + self.tokenizer.encode_ids(x['title'])) for x in corpus[start_idx:start_idx+batch_size]]
                encoded = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=25000).to(self.model.device)
                model_out, _ = self.model(encoded, encoded)
                corpus_embeddings.append(model_out)
        
        return torch.cat(corpus_embeddings, dim=0).cpu().numpy()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# these are the hyperparameters used in the paper (change as desired for your trained model)
model  = Banyan(25001, 256, 128, 0.1, device).to(device)

# load from checkpoint
path = sys.argv[1]
print('Loading model from:', path)
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model'])
model = DRES(StrAE_DE(model), batch_size=1024)
retriever = EvaluateRetrieval(model, score_function="cos_sim")


data_path = "data/arguana"
print('Evaluating on Arguana dataset...')
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

results = retriever.retrieve(corpus, queries)
print(retriever.evaluate(qrels, results, retriever.k_values))
print('\n')


data_path = "data/quora"
print('Evaluating on Quora dataset...')
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

results = retriever.retrieve(corpus, queries)
print(retriever.evaluate(qrels, results, retriever.k_values))

data_path = "data/nfcorpus"
print('Evaluating on NFCorpus dataset...')
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

results = retriever.retrieve(corpus, queries)
print(retriever.evaluate(qrels, results, retriever.k_values))

data_path = "data/scifact"
print('Evaluating on Scifact dataset...')
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

results = retriever.retrieve(corpus, queries)
print(retriever.evaluate(qrels, results, retriever.k_values))