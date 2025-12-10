# Data loading utilities for MSMARCO with Banyan's BPEmb tokenization
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
from bpemb import BPEmb
from tqdm import tqdm


class MSMARCORetrievalDataset(Dataset):
    """
    Dataset for MSMARCO passage ranking with BPEmb tokenization
    Compatible with Banyan's tokenization scheme
    """
    def __init__(self, data_path, bpe, max_query_len=32, max_passage_len=200):
        self.bpe = bpe
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        self.data = []
        
        print(f'Loading data from {data_path}...')
        with open(data_path, 'r') as f:
            for line in tqdm(f):
                item = json.loads(line)
                self.data.append(item)
        
        print(f'Loaded {len(self.data)} query-passage pairs')
    
    def sent_to_bpe(self, sent, max_len):
        """Convert sentence to BPE IDs, matching Banyan's preprocessing"""
        encoded = self.bpe.encode_ids(sent)
        # Flatten the list of lists if needed
        if isinstance(encoded[0], list):
            encoded = [item for sublist in encoded for item in sublist]
        
        # Truncate to max length
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize query and passage using BPEmb
        query_tokens = self.sent_to_bpe(item['query'], self.max_query_len)
        passage_tokens = self.sent_to_bpe(item['positive_passage'], self.max_passage_len)
        
        # Filter out very short sequences (matching Banyan's original filtering)
        if query_tokens.size(0) < 3:
            # Return a default/skip item, or handle in collate_fn
            query_tokens = torch.tensor([0, 0, 0], dtype=torch.long)
        
        if passage_tokens.size(0) < 3:
            passage_tokens = torch.tensor([0, 0, 0], dtype=torch.long)
        
        return {
            'query': query_tokens,
            'passage': passage_tokens
        }


def collate_fn_retrieval(batch):
    """
    Collate function for retrieval dataloader
    Pads sequences to match Banyan's approach (padding_value=25000)
    """
    queries = [item['query'] for item in batch]
    passages = [item['passage'] for item in batch]
    
    # Pad sequences (25000 is the padding token in Banyan)
    queries_padded = pad_sequence(queries, batch_first=True, padding_value=25000)
    passages_padded = pad_sequence(passages, batch_first=True, padding_value=25000)
    
    return {
        'query': queries_padded,
        'passage': passages_padded
    }


def create_retrieval_dataloader(data_path, batch_size, shuffle=True, 
                               max_query_len=32, max_passage_len=200,
                               lang='en', num_workers=4):
    """
    Create DataLoader for retrieval training using BPEmb tokenization
    
    Args:
        data_path: Path to prepared JSONL data
        batch_size: Batch size
        shuffle: Whether to shuffle
        max_query_len: Maximum query length (default: 32)
        max_passage_len: Maximum passage length (default: 200)
        lang: Language code (default: 'en')
        num_workers: Number of worker processes
    
    Returns:
        DataLoader
    """
    # Initialize BPEmb with same parameters as Banyan pretraining
    print(f'Initializing BPEmb for language: {lang}')
    bpe = BPEmb(lang=lang, vs=25000, dim=100)
    
    dataset = MSMARCORetrievalDataset(
        data_path,
        bpe,
        max_query_len=max_query_len,
        max_passage_len=max_passage_len
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_retrieval,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# For backward compatibility with the training script
def create_dataloader_retrieval(*args, **kwargs):
    """Alias for create_retrieval_dataloader"""
    return create_retrieval_dataloader(*args, **kwargs)


if __name__ == '__main__':
    # Test the dataloader
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to prepared JSONL file')
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    print('Testing dataloader...')
    dataloader = create_retrieval_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print(f'\nDataloader created with {len(dataloader)} batches')
    print('\nFetching first batch...')
    batch = next(iter(dataloader))
    
    print(f"Query shape: {batch['query'].shape}")
    print(f"Passage shape: {batch['passage'].shape}")
    print(f"\nSample query tokens: {batch['query'][0][:20]}")
    print(f"Sample passage tokens: {batch['passage'][0][:20]}")
    print('\nDataloader test successful!')