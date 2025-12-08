import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn.functional as F


class MSMARCODataset(Dataset):
    """Dataset for MS MARCO query-passage pairs"""
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize query
        query_encoded = self.tokenizer(
            item['query'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize positive passage
        passage_encoded = self.tokenizer(
            item['positive_passage'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_encoded['input_ids'].squeeze(0),
            'query_attention_mask': query_encoded['attention_mask'].squeeze(0),
            'passage_input_ids': passage_encoded['input_ids'].squeeze(0),
            'passage_attention_mask': passage_encoded['attention_mask'].squeeze(0),
        }


class BiEncoder(nn.Module):
    """Bi-encoder model for query and passage encoding"""
    def __init__(self, model_name_or_path, pooling='mean'):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name_or_path)
        self.pooling = pooling
    
    def mean_pool(self, last_hidden_state, attention_mask):
        """Mean pooling over sequence length"""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.pooling == 'mean':
            embeddings = self.mean_pool(outputs.last_hidden_state, attention_mask)
        elif self.pooling == 'cls':
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # L2 normalize for contrastive learning
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss with in-batch negatives"""
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.ce_criterion = nn.CrossEntropyLoss()
    
    def forward(self, query_embs, passage_embs):
        """
        Args:
            query_embs: (batch_size, embed_dim) normalized query embeddings
            passage_embs: (batch_size, embed_dim) normalized passage embeddings
        Returns:
            loss: scalar contrastive loss
        """
        batch_size = query_embs.shape[0]
        
        # Compute similarity matrix: (batch_size, batch_size)
        # Each query should match its corresponding passage (diagonal)
        sim_matrix = torch.matmul(query_embs, passage_embs.T) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=query_embs.device)
        
        # InfoNCE loss
        loss = self.ce_criterion(sim_matrix, labels)
        
        # Calculate accuracy for monitoring
        predictions = sim_matrix.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        
        return loss, accuracy


def get_args():
    parser = argparse.ArgumentParser(description="Finetune RoBERTa for retrieval with contrastive learning")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to pretrained RoBERTa model")
    parser.add_argument("--train_data", type=str, default="data/msmarco/train.jsonl",
                        help="Path to training JSONL file")
    parser.add_argument("--val_data", type=str, default="data/msmarco/val.jsonl",
                        help="Path to validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="checkpoints/roberta-retrieval",
                        help="Directory to save finetuned model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio of total steps")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Temperature for contrastive loss")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"],
                        help="Pooling strategy")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    return parser.parse_args()


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            passage_input_ids = batch['passage_input_ids'].to(device)
            passage_attention_mask = batch['passage_attention_mask'].to(device)
            
            # Encode queries and passages
            query_embs = model(query_input_ids, query_attention_mask)
            passage_embs = model(passage_input_ids, passage_attention_mask)
            
            # Compute loss
            loss, accuracy = criterion(query_embs, passage_embs)
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def main():
    args = get_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MSMARCODataset(args.train_data, tokenizer, args.max_length)
    val_dataset = MSMARCODataset(args.val_data, tokenizer, args.max_length)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    model = BiEncoder(args.model_path, pooling=args.pooling)
    model.to(device)
    
    # Initialize loss
    criterion = ContrastiveLoss(temperature=args.temperature)
    
    # Setup optimizer and scheduler
    num_training_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    print(f"Total training steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}")
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            passage_input_ids = batch['passage_input_ids'].to(device)
            passage_attention_mask = batch['passage_attention_mask'].to(device)
            
            # Forward pass
            query_embs = model(query_input_ids, query_attention_mask)
            passage_embs = model(passage_input_ids, passage_attention_mask)
            
            # Compute loss
            loss, accuracy = criterion(query_embs, passage_embs)
            loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            epoch_accuracy += accuracy.item()
            
            # Update weights
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{epoch_loss / (step + 1):.4f}",
                    'acc': f"{epoch_accuracy / (step + 1):.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Evaluation
                if global_step % args.eval_steps == 0:
                    print("\nRunning evaluation...")
                    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
                    print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"New best model! Saving to {args.output_dir}/best_model")
                        model.encoder.save_pretrained(f"{args.output_dir}/best_model")
                        tokenizer.save_pretrained(f"{args.output_dir}/best_model")
                    
                    model.train()
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_dir = f"{args.output_dir}/checkpoint-{global_step}"
                    print(f"\nSaving checkpoint to {checkpoint_dir}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.encoder.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
        
        # End of epoch evaluation
        print("\nEnd of epoch evaluation...")
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model! Saving to {args.output_dir}/best_model")
            model.encoder.save_pretrained(f"{args.output_dir}/best_model")
            tokenizer.save_pretrained(f"{args.output_dir}/best_model")
    
    # Save final model
    print(f"\nSaving final model to {args.output_dir}/final_model")
    model.encoder.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()