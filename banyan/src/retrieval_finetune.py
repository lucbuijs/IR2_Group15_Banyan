#!/usr/bin/env python3
"""
Compare vanilla Banyan vs fine-tuned Banyan (with projection) on retrieval tasks.

This script evaluates both:
1. Vanilla Banyan (using raw root embeddings via STS mode)
2. Fine-tuned Banyan (using projection head + normalization)

Usage:
    python compare_retrieval.py \
        --vanilla checkpoints/original_banyan.pt \
        --finetuned checkpoints/banyan_retrieval_frozen.pt \
        --datasets data/arguana,data/quora,data/nfcorpus,data/scifact
"""

import os
import sys
import argparse
from typing import List, Dict
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from bpemb import BPEmb

from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from models import Banyan


class BanyanRetriever(nn.Module):
    """Wrapper for Banyan with optional projection head"""
    def __init__(self, banyan_model, projection_dim=None):
        super().__init__()
        self.banyan = banyan_model
        self.embedding_size = banyan_model.E
        
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.embedding_size, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim)
            )
        else:
            self.projection = None


class VanillaBanyanEncoder:
    """Original Banyan encoder using STS mode (no projection, no normalization)"""
    def __init__(self, model, **kwargs):
        if hasattr(model, 'banyan'):
            self.banyan = model.banyan
        else:
            self.banyan = model
        
        self.tokenizer = BPEmb(lang='en', vs=25000, dim=100)
        self.banyan.eval()
        self.banyan.to('cuda')
        print('[Vanilla] Using STS mode (raw root embeddings)')
    
    def _get_embeddings(self, encoded):
        # Use STS mode: model(encoded, encoded)
        emb, _ = self.banyan(encoded, encoded)
        return emb
    
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size, desc='[Vanilla] Queries'):
                encoded = []
                for q in queries[start_idx:start_idx+batch_size]:
                    ids = self.tokenizer.encode_ids(q)
                    if ids and isinstance(ids[0], list):
                        ids = [item for sublist in ids for item in sublist]
                    encoded.append(torch.tensor(ids))
                
                encoded = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=25000)
                encoded = encoded.to(self.banyan.device)
                
                embs = self._get_embeddings(encoded)
                query_embeddings.append(embs.cpu())
        
        return torch.cat(query_embeddings, dim=0).numpy()
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        corpus_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(corpus), batch_size, desc='[Vanilla] Corpus'):
                encoded = []
                for doc in corpus[start_idx:start_idx+batch_size]:
                    text_ids = self.tokenizer.encode_ids(doc.get('text', ''))
                    title_ids = self.tokenizer.encode_ids(doc.get('title', ''))
                    
                    if text_ids and isinstance(text_ids[0], list):
                        text_ids = [item for sublist in text_ids for item in sublist]
                    if title_ids and isinstance(title_ids[0], list):
                        title_ids = [item for sublist in title_ids for item in sublist]
                    
                    combined = text_ids + title_ids
                    encoded.append(torch.tensor(combined))
                
                encoded = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=25000)
                encoded = encoded.to(self.banyan.device)
                
                embs = self._get_embeddings(encoded)
                corpus_embeddings.append(embs.cpu())
        
        return torch.cat(corpus_embeddings, dim=0).numpy()


class FinetunedBanyanEncoder:
    """Fine-tuned Banyan encoder using projection head + normalization"""
    def __init__(self, model, **kwargs):
        # model should be BanyanRetriever with projection
        self.model = model
        if hasattr(model, 'banyan'):
            self.banyan = model.banyan
        else:
            self.banyan = model
        
        self.projection = getattr(model, 'projection', None)
        self.tokenizer = BPEmb(lang='en', vs=25000, dim=100)
        
        self.model.eval()
        self.model.to('cuda')
        
        if self.projection is not None:
            print('[Fine-tuned] Using projection head + normalization')
        else:
            print('[Fine-tuned] WARNING: No projection head found!')
    
    def _get_embeddings(self, encoded):
        # Get root embeddings
        root_embs = self.banyan.compose(encoded, roots=True)
        
        # Apply projection if available
        if self.projection is not None:
            root_embs = self.projection(root_embs)
        
        # Normalize (as done in contrastive training)
        root_embs = F.normalize(root_embs, p=2, dim=1)
        return root_embs
    
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size, desc='[Fine-tuned] Queries'):
                encoded = []
                for q in queries[start_idx:start_idx+batch_size]:
                    ids = self.tokenizer.encode_ids(q)
                    if ids and isinstance(ids[0], list):
                        ids = [item for sublist in ids for item in sublist]
                    encoded.append(torch.tensor(ids))
                
                encoded = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=25000)
                encoded = encoded.to(self.banyan.device)
                
                embs = self._get_embeddings(encoded)
                query_embeddings.append(embs.cpu())
        
        return torch.cat(query_embeddings, dim=0).numpy()
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        corpus_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(corpus), batch_size, desc='[Fine-tuned] Corpus'):
                encoded = []
                for doc in corpus[start_idx:start_idx+batch_size]:
                    text_ids = self.tokenizer.encode_ids(doc.get('text', ''))
                    title_ids = self.tokenizer.encode_ids(doc.get('title', ''))
                    
                    if text_ids and isinstance(text_ids[0], list):
                        text_ids = [item for sublist in text_ids for item in sublist]
                    if title_ids and isinstance(title_ids[0], list):
                        title_ids = [item for sublist in title_ids for item in sublist]
                    
                    combined = text_ids + title_ids
                    encoded.append(torch.tensor(combined))
                
                encoded = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=25000)
                encoded = encoded.to(self.banyan.device)
                
                embs = self._get_embeddings(encoded)
                corpus_embeddings.append(embs.cpu())
        
        return torch.cat(corpus_embeddings, dim=0).numpy()


def load_model(checkpoint_path, device):
    """Load a Banyan model (handles both vanilla and fine-tuned)"""
    print(f'Loading: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    
    # Check for projection head
    has_projection = any('projection' in k for k in state_dict.keys())
    has_banyan_prefix = any(k.startswith('banyan.') for k in state_dict.keys())
    
    # Determine projection dim
    projection_dim = None
    if has_projection:
        for k in state_dict.keys():
            if 'projection' in k and k.endswith('.weight'):
                projection_dim = state_dict[k].shape[0]
                break
        print(f'  Detected projection head (dim={projection_dim})')
    else:
        print(f'  No projection head (vanilla Banyan)')
    
    # Load base Banyan
    banyan = Banyan(25001, 256, 128, 0.1, device).to(device)
    
    # If has projection, load into BanyanRetriever wrapper
    if has_projection:
        model = BanyanRetriever(banyan, projection_dim=projection_dim).to(device)
        
        # Remap keys if needed
        if has_banyan_prefix:
            remapped = {}
            for k, v in state_dict.items():
                # Keep 'banyan.' and 'projection.' prefixes for BanyanRetriever
                remapped[k] = v
            state_dict = remapped
        
        model.load_state_dict(state_dict, strict=False)
    else:
        # Vanilla Banyan - remove 'banyan.' prefix if exists
        if has_banyan_prefix:
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith('banyan.'):
                    remapped[k.replace('banyan.', '')] = v
                else:
                    remapped[k] = v
            state_dict = remapped
        
        banyan.load_state_dict(state_dict, strict=False)
        model = banyan
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Total params: {total_params:,}')
    
    return model, has_projection


def evaluate_on_dataset(encoder, dataset_path, dataset_name):
    """Evaluate on a single BEIR dataset"""
    print(f'\n{"="*60}')
    print(f'Dataset: {dataset_name}')
    print(f'{"="*60}')
    
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split='test')
    print(f'{len(corpus)} docs, {len(queries)} queries')
    
    dres = DRES(encoder, batch_size=1024)
    retriever = EvaluateRetrieval(dres, score_function='cos_sim')
    
    results = retriever.retrieve(corpus, queries)
    scores = retriever.evaluate(qrels, results, retriever.k_values)
    
    # Extract NDCG@10
    if isinstance(scores, tuple):
        ndcg10 = scores[0].get('NDCG@10', 0.0)
    else:
        ndcg10 = scores.get('NDCG', {}).get('NDCG@10', 0.0)
    
    print(f'NDCG@10: {ndcg10:.4f}')
    print(scores)
    
    return ndcg10, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vanilla', type=str, required=True,
                       help='Path to vanilla pretrained Banyan')
    parser.add_argument('--finetuned', type=str, required=True,
                       help='Path to fine-tuned Banyan (with projection)')
    parser.add_argument('--datasets', type=str,
                       default='data/arguana,data/quora,data/nfcorpus,data/scifact',
                       help='Comma-separated dataset paths')
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')
    
    # Load both models
    print('='*60)
    print('LOADING MODELS')
    print('='*60)
    vanilla_model, _ = load_model(args.vanilla, device)
    finetuned_model, has_proj = load_model(args.finetuned, device)
    
    if not has_proj:
        print('\nWARNING: Fine-tuned model has no projection head!')
        print('This comparison may not be meaningful.\n')
    
    # Parse datasets
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    
    # Results storage
    vanilla_results = {}
    finetuned_results = {}
    
    # Evaluate each dataset
    for ds_path in datasets:
        if not os.path.exists(ds_path):
            print(f'Skipping {ds_path} (not found)')
            continue
        
        ds_name = os.path.basename(ds_path)
        
        # Vanilla evaluation
        print('\n' + '='*60)
        print(f'VANILLA BANYAN - {ds_name}')
        print('='*60)
        vanilla_encoder = VanillaBanyanEncoder(vanilla_model)
        v_ndcg10, v_scores = evaluate_on_dataset(vanilla_encoder, ds_path, ds_name)
        vanilla_results[ds_name] = v_ndcg10
        
        # Fine-tuned evaluation
        print('\n' + '='*60)
        print(f'FINE-TUNED BANYAN - {ds_name}')
        print('='*60)
        finetuned_encoder = FinetunedBanyanEncoder(finetuned_model)
        f_ndcg10, f_scores = evaluate_on_dataset(finetuned_encoder, ds_path, ds_name)
        finetuned_results[ds_name] = f_ndcg10
    
    # Final comparison
    print('\n' + '='*60)
    print('FINAL COMPARISON')
    print('='*60)
    print(f'{"Dataset":<15} {"Vanilla":<10} {"Fine-tuned":<10} {"Δ":<10}')
    print('-'*60)
    
    for ds_name in vanilla_results.keys():
        v = vanilla_results[ds_name]
        f = finetuned_results[ds_name]
        delta = f - v
        symbol = '✓' if delta > 0 else '✗'
        print(f'{ds_name:<15} {v:<10.4f} {f:<10.4f} {delta:+.4f} {symbol}')
    
    # Average
    v_avg = np.mean(list(vanilla_results.values()))
    f_avg = np.mean(list(finetuned_results.values()))
    delta_avg = f_avg - v_avg
    symbol = '✓' if delta_avg > 0 else '✗'
    
    print('-'*60)
    print(f'{"Average":<15} {v_avg:<10.4f} {f_avg:<10.4f} {delta_avg:+.4f} {symbol}')
    
    if delta_avg > 0:
        print(f'\n✓ Fine-tuning improved performance by {delta_avg:.4f} NDCG@10')
    else:
        print(f'\n✗ Fine-tuning hurt performance by {abs(delta_avg):.4f} NDCG@10')


if __name__ == '__main__':
    main()