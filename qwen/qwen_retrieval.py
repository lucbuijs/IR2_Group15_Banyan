# retrieval_qwen.py
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import trange
import numpy as np
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModel
import sys
import os

class QwenRetriever:
    """
    BEIR-compatible wrapper for Qwen3-Embedding-8B model.
    Implements encode_queries() and encode_corpus() for DenseRetrievalExactSearch.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B", device: str = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading Qwen3 model from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map=self.device
        )
        self.model.eval()
        print(f"Qwen3 model loaded on {self.device}")
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts using mean pooling."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return embeddings.cpu().numpy()
    
    def encode_queries(self, queries: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode queries in batches."""
        query_embeddings = []
        for start_idx in trange(0, len(queries), batch_size, desc="Encoding queries"):
            batch = queries[start_idx:start_idx + batch_size]
            emb = self._encode_batch(batch)
            query_embeddings.append(emb)
        return np.vstack(query_embeddings)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode corpus documents in batches."""
        # Concatenate title and text
        texts = [
            (doc.get('title', '') + ' ' + doc.get('text', '')).strip()
            for doc in corpus
        ]
        
        corpus_embeddings = []
        for start_idx in trange(0, len(texts), batch_size, desc="Encoding corpus"):
            batch = texts[start_idx:start_idx + batch_size]
            emb = self._encode_batch(batch)
            corpus_embeddings.append(emb)
        return np.vstack(corpus_embeddings)

def evaluate_dataset(dense_model, data_path: str, dataset_name: str):
    """Evaluate on a single dataset."""
    print(f"\n=== Evaluating {dataset_name} dataset ===")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    # Perform retrieval
    results_dense = dense_model.search(corpus, queries, top_k=10, score_function="cos_sim")
    
    # Use EvaluateRetrieval to compute metrics
    retriever = EvaluateRetrieval(dense_model, score_function="cos_sim")
    metrics = retriever.evaluate(qrels, results_dense, retriever.k_values)
    
    print(f"\nDense Retrieval ({dataset_name})")
    print(metrics)
    
    return metrics

if __name__ == "__main__":
    # Check for custom model path
    model_name = "Qwen/Qwen3-Embedding-8B"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    print(f"Using model: {model_name}")
    
    # Initialize Qwen retriever
    qwen_retriever = QwenRetriever(model_name)
    
    # BEIR dense search wrapper - use smaller batch size for large model
    dense_model = DRES(qwen_retriever, batch_size=32)
    
    # Set data folder to banyan/data
    banyan_data_path = os.path.expanduser("~/Repo_Combined/banyan/data")

    # Evaluate on Arguana
    arguana_metrics = evaluate_dataset(
        dense_model,
        os.path.join(banyan_data_path, "arguana"),
        "Arguana"
    )

    # Evaluate on Quora
    quora_metrics = evaluate_dataset(
        dense_model,
        os.path.join(banyan_data_path, "quora"),
        "Quora"
    )


    # Evaluate on NFCorpus
    nfcorpus_metrics = evaluate_dataset(
        dense_model,
        os.path.join(banyan_data_path, "nfcorpus"),
        "NFCorpus"
    )

    # Evaluate on SciFact
    scifact_metrics = evaluate_dataset(
        dense_model,
        os.path.join(banyan_data_path, "scifact"),
        "SciFact"
    )
    
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Arguana NDCG@10: {arguana_metrics[0]['NDCG@10']:.4f}")
    print(f"Quora NDCG@10: {quora_metrics[0]['NDCG@10']:.4f}")
    print(f"NFCorpus NDCG@10: {nfcorpus_metrics[0]['NDCG@10']:.4f}")
    print(f"SciFact NDCG@10: {scifact_metrics[0]['NDCG@10']:.4f}")