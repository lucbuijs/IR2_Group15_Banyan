#!/usr/bin/env python3
"""
Prepare MSMARCO data for Banyan retrieval training
Combines queries.jsonl, corpus.jsonl, and qrels/*.tsv into training format
"""

import json
from collections import defaultdict
from tqdm import tqdm
import argparse
import random


def load_corpus(corpus_path):
    """Load corpus from JSONL"""
    print(f'Loading corpus from {corpus_path}...')
    corpus = {}
    with open(corpus_path, 'r') as f:
        for line in tqdm(f):
            doc = json.loads(line)
            # Handle different corpus formats
            if '_id' in doc:
                corpus[doc['_id']] = doc.get('text', doc.get('contents', ''))
            elif 'docid' in doc:
                corpus[doc['docid']] = doc.get('text', doc.get('contents', ''))
            else:
                # Fallback: assume first field is ID, second is text
                fields = list(doc.values())
                if len(fields) >= 2:
                    corpus[str(fields[0])] = str(fields[1])
    print(f'Loaded {len(corpus)} documents')
    return corpus


def load_queries(queries_path):
    """Load queries from JSONL"""
    print(f'Loading queries from {queries_path}...')
    queries = {}
    with open(queries_path, 'r') as f:
        for line in f:
            query = json.loads(line)
            # Handle different query formats
            if '_id' in query:
                queries[query['_id']] = query.get('text', query.get('query', ''))
            elif 'qid' in query:
                queries[query['qid']] = query.get('text', query.get('query', ''))
            else:
                # Fallback
                fields = list(query.values())
                if len(fields) >= 2:
                    queries[str(fields[0])] = str(fields[1])
    print(f'Loaded {len(queries)} queries')
    return queries


def load_qrels(qrels_path):
    """Load relevance judgments from TSV"""
    print(f'Loading qrels from {qrels_path}...')
    qrels = defaultdict(list)
    with open(qrels_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid = parts[0]
                docid = parts[2] if len(parts) == 4 else parts[1]
                qrels[qid].append(docid)
    print(f'Loaded {len(qrels)} query-document pairs')
    return qrels


def create_training_examples(queries, corpus, qrels, max_examples=None):
    """Create training examples from queries, corpus, and qrels"""
    print('Creating training examples...')
    examples = []
    
    skipped_queries = 0
    skipped_docs = 0
    
    for qid, docids in tqdm(qrels.items()):
        # Check if query exists
        if qid not in queries:
            skipped_queries += 1
            continue
        
        query_text = queries[qid]
        
        # Create example for each relevant document
        for docid in docids:
            if docid not in corpus:
                skipped_docs += 1
                continue
            
            passage_text = corpus[docid]
            
            # Basic filtering
            if len(query_text.strip()) == 0 or len(passage_text.strip()) == 0:
                continue
            
            examples.append({
                'qid': qid,
                'docid': docid,
                'query': query_text,
                'positive_passage': passage_text
            })
            
            # Limit examples if specified
            if max_examples and len(examples) >= max_examples:
                break
        
        if max_examples and len(examples) >= max_examples:
            break
    
    print(f'Created {len(examples)} training examples')
    if skipped_queries > 0:
        print(f'Skipped {skipped_queries} queries (not found in queries file)')
    if skipped_docs > 0:
        print(f'Skipped {skipped_docs} documents (not found in corpus)')
    
    return examples


def split_data(examples, train_ratio=0.95, seed=42):
    """Split data into train and validation sets"""
    random.seed(seed)
    random.shuffle(examples)
    
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    print(f'Train: {len(train_examples)} examples')
    print(f'Val: {len(val_examples)} examples')
    
    return train_examples, val_examples


def write_jsonl(examples, output_path):
    """Write examples to JSONL file"""
    print(f'Writing to {output_path}...')
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    print(f'Wrote {len(examples)} examples')


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MSMARCO data for Banyan retrieval training'
    )
    parser.add_argument('--corpus', type=str, required=True,
                       help='Path to corpus.jsonl')
    parser.add_argument('--queries', type=str, required=True,
                       help='Path to queries.jsonl')
    parser.add_argument('--qrels', type=str, required=True,
                       help='Path to qrels TSV file (e.g., qrels/train.tsv)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for prepared data')
    parser.add_argument('--train_ratio', type=float, default=0.95,
                       help='Ratio of data to use for training (default: 0.95)')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Maximum number of examples to create (for testing)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for train/val split')
    
    args = parser.parse_args()
    
    # Load data
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    
    # Create examples
    examples = create_training_examples(
        queries, corpus, qrels, 
        max_examples=args.max_examples
    )
    
    if len(examples) == 0:
        print('ERROR: No examples created. Check your data format.')
        return
    
    # Split data
    train_examples, val_examples = split_data(
        examples, 
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    # Write outputs
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.output_dir, 'train.jsonl')
    val_path = os.path.join(args.output_dir, 'val.jsonl')
    
    write_jsonl(train_examples, train_path)
    write_jsonl(val_examples, val_path)
    
    print('\n=== Data Preparation Complete ===')
    print(f'Training data: {train_path}')
    print(f'Validation data: {val_path}')
    print(f'\nYou can now run training with:')
    print(f'python train_retrieval.py \\')
    print(f'    --pretrained_path <your_pretrained_model.pt> \\')
    print(f'    --train_path {train_path} \\')
    print(f'    --dev_path {val_path} \\')
    print(f'    --save_path banyan_retrieval.pt')


if __name__ == '__main__':
    main()