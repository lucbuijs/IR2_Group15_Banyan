import os
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_data(output_dir="data"):
    print("Downloading WikiText-103...")
    # Using wikitext-103-raw-v1 as it's the standard raw version
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    # We only need the train split for pre-training as per instructions
    train_data = dataset["train"]
    
    print(f"Loaded {len(train_data)} training examples.")
    
    # Save to disk for easier loading later if needed, or we can just use the cache
    # For this implementation, we'll rely on the HF cache but ensure the tokenizer is ready
    
    print("Loading Tokenizer...")
    # Reusing roberta-base tokenizer as planned
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Save tokenizer to local directory to ensure reproducibility/offline usage
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    return dataset, tokenizer

if __name__ == "__main__":
    prepare_data()
