import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, RobertaModel
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import sys
import os
from sklearn.metrics import f1_score, accuracy_score

# Add current directory to path to import model_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_utils import encode_texts

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def prepare_glue_data(dataset_name, tokenizer, encoder, device, split="train"):
    dataset = load_dataset("glue", dataset_name, split=split)
    
    embeddings = []
    labels = []
    
    print(f"Encoding {dataset_name} {split} set...")
    
    # Process in chunks to avoid OOM
    chunk_size = 1000
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i+chunk_size]
        
        if dataset_name == "sst2":
            texts = chunk["sentence"]
            embs = encode_texts(texts, tokenizer, encoder, device)
        elif dataset_name == "mrpc":
            # Concatenate strategy as planned
            embs1 = encode_texts(chunk["sentence1"], tokenizer, encoder, device)
            embs2 = encode_texts(chunk["sentence2"], tokenizer, encoder, device)
            # [emb1; emb2] -> 512 dim
            embs = torch.cat([embs1, embs2], dim=1)
            
        embeddings.append(embs)
        labels.extend(chunk["label"])
        
    return torch.cat(embeddings, dim=0), torch.tensor(labels)

def train_and_evaluate(dataset_name, model_path, device="cuda", epochs=10, lr=1e-3):
    print(f"--- Processing {dataset_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder = RobertaModel.from_pretrained(model_path).to(device)
    encoder.eval()
    
    # Prepare Data
    X_train, y_train = prepare_glue_data(dataset_name, tokenizer, encoder, device, split="train")
    # Use validation set for evaluation as per standard practice when test labels aren't available
    X_val, y_val = prepare_glue_data(dataset_name, tokenizer, encoder, device, split="validation")
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    input_dim = X_train.shape[1]
    classifier = MLPClassifier(input_dim).to(device)
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training MLP for {epochs} epochs...")
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = classifier(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    # Evaluation
    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(X_val.to(device))
        val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
        y_val_np = y_val.numpy()
        
        acc = accuracy_score(y_val_np, val_preds)
        f1 = f1_score(y_val_np, val_preds) if dataset_name == "mrpc" else 0.0
        
    print(f"{dataset_name} Results - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return {"accuracy": acc, "f1": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}
    results["sst2"] = train_and_evaluate("sst2", args.model_path, device)
    results["mrpc"] = train_and_evaluate("mrpc", args.model_path, device, epochs=15) # MRPC might need more epochs
    
    print("\nFinal Classification Results:")
    print(results)

if __name__ == "__main__":
    main()
