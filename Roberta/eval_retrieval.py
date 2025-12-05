import argparse
import torch
from transformers import AutoTokenizer, RobertaModel
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
import os
import logging
import sys

# Add current directory to path to import model_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_utils import encode_texts

class BanyanRobertaModel:
    def __init__(self, model_path, device="cuda", batch_size=64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.encoder = RobertaModel.from_pretrained(model_path).to(device)
        self.encoder.eval()
        self.device = device
        self.batch_size = batch_size
        
    def encode_queries(self, queries, batch_size=16, **kwargs):
        return encode_texts(queries, self.tokenizer, self.encoder, self.device, batch_size=batch_size).numpy()
    
    def encode_corpus(self, corpus, batch_size=8, **kwargs):
        # Extract text from corpus dictionaries
        texts = [doc.get("title", "") + " " + doc.get("text", "") for doc in corpus]
        return encode_texts(texts, self.tokenizer, self.encoder, self.device, batch_size=batch_size).numpy()

def evaluate_dataset(dataset_name, model, data_path="beir_data"):
    print(f"Evaluating on {dataset_name}...")
    
    # Download/Load dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = os.path.join(data_path, dataset_name)
    data_path_dataset = util.download_and_unzip(url, out_dir)
    
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path_dataset).load(split="test")
    
    # BEIR Dense Retrieval
    retriever = EvaluateRetrieval(DenseRetrievalExactSearch(model, batch_size=128), score_function="cos_sim")
    results = retriever.retrieve(corpus, queries)
    
    ndcg, _map, recall, _ = retriever.evaluate(qrels, results, retriever.k_values)
    
    print(f"Results for {dataset_name}:")
    print(f"NDCG@1: {ndcg['NDCG@1']}")
    print(f"NDCG@10: {ndcg['NDCG@10']}")
    print(f"Recall@1: {recall['Recall@1']}")
    print(f"Recall@10: {recall['Recall@10']}")
    
    return {
        "NDCG@1": ndcg['NDCG@1'],
        "NDCG@10": ndcg['NDCG@10'],
        "Recall@1": recall['Recall@1'],
        "Recall@10": recall['Recall@10']
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_path", type=str, default="beir_data", help="Path to store BEIR data")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BanyanRobertaModel(args.model_path, device=device)
    
    results = {}
    for dataset in ["quora", "arguana"]:
        results[dataset] = evaluate_dataset(dataset, model, args.data_path)
        
    print("\nFinal Retrieval Results:")
    print(results)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    main()
