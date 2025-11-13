# retrieval_sent2vec.py
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import trange
import numpy as np
from typing import List, Dict
import sent2vec
import sys
import os
import re

def preprocess_text(text: str) -> str:
    """
    Preprocess text the same way sent2vec training data was preprocessed.
    This is CRITICAL for getting good results.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

class Sent2VecRetriever:
    """
    BEIR-compatible wrapper for a Sent2Vec .bin model.
    Implements encode_queries() and encode_corpus() for DenseRetrievalExactSearch.
    """
    def __init__(self, model_path: str):
        self.model = sent2vec.Sent2vecModel()
        print(f"Loading Sent2Vec model from {model_path}...")
        self.model.load_model(model_path)
        print("Sent2Vec model loaded.")
        #print(f"Model vocabulary size: {self.model.get_vocab_size()}")
        print(f"Model embedding dimension: {self.model.get_emb_size()}")
    
    def encode_queries(self, queries: List[str], batch_size: int = 128, **kwargs) -> np.ndarray:
        query_embeddings = []
        for start_idx in trange(0, len(queries), batch_size, desc="Encoding queries"):
            batch = queries[start_idx:start_idx + batch_size]
            # CRITICAL: Preprocess the text before encoding
            batch_preprocessed = [preprocess_text(q) for q in batch]
            emb = self.model.embed_sentences(batch_preprocessed)
            query_embeddings.append(emb)
        return np.vstack(query_embeddings)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 128, **kwargs) -> np.ndarray:
        # Concatenate title and text
        texts = [x.get('title', '') + " " + x.get('text', '') for x in corpus]
        
        corpus_embeddings = []
        for start_idx in trange(0, len(texts), batch_size, desc="Encoding corpus"):
            batch = texts[start_idx:start_idx + batch_size]
            # CRITICAL: Preprocess the text before encoding
            batch_preprocessed = [preprocess_text(t) for t in batch]
            emb = self.model.embed_sentences(batch_preprocessed)
            corpus_embeddings.append(emb)
        return np.vstack(corpus_embeddings)

def evaluate_dataset(dense_model, data_path: str, dataset_name: str):
    print(f"\n=== Evaluating {dataset_name} dataset ===")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    # Perform retrieval
    results_dense = dense_model.search(corpus, queries, top_k=10, score_function="cos_sim")
    
    # Use EvaluateRetrieval to compute metrics
    retriever = EvaluateRetrieval(dense_model, score_function="cos_sim")
    metrics = retriever.evaluate(qrels, results_dense, retriever.k_values)
    
    print(f"\nDense Retrieval ({dataset_name})")
    print(metrics)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python retrieval_sent2vec.py <sent2vec_checkpoint.bin>")
        print("\nMake sure you're using the .bin file, NOT the .vec file!")
        sys.exit(1)
    
    sent2vec_path = sys.argv[1]
    
    # Verify it's a .bin file
    if not sent2vec_path.endswith('.bin'):
        print(f"WARNING: You provided '{sent2vec_path}'")
        print("Make sure you're using the .bin file (the full model), not the .vec file!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Wrap Sent2Vec model
    s2v_retriever = Sent2VecRetriever(sent2vec_path)
    
    # BEIR dense search wrapper
    dense_model = DRES(s2v_retriever, batch_size=1024)
    retriever = EvaluateRetrieval(dense_model, score_function="cos_sim")
    
    # Set data folder to banyan/data relative to this repo
    banyan_data_path = os.path.expanduser("~/Repo_Combined/banyan/data")
    
    # Evaluate on Arguana
    evaluate_dataset(dense_model, os.path.join(banyan_data_path, "arguana"), "Arguana")
    
    # Evaluate on Quora
    evaluate_dataset(dense_model, os.path.join(banyan_data_path, "quora"), "Quora")