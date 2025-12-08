# Fine-tuning script for Banyan with contrastive loss for dense retrieval
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Banyan
from utils import *
import argparse
from tqdm import tqdm


class BanyanRetriever(nn.Module):
    """Wrapper around Banyan for retrieval tasks"""
    def __init__(self, banyan_model, projection_dim=None):
        super(BanyanRetriever, self).__init__()
        self.banyan = banyan_model
        self.embedding_size = banyan_model.E
        
        # Optional projection head for contrastive learning
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.embedding_size, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim)
            )
        else:
            self.projection = None
    
    def encode(self, seqs):
        """Encode sequences to root embeddings"""
        # Get root embeddings from Banyan's compose function
        root_embs = self.banyan.compose(seqs, roots=True)
        
        # Apply projection if available
        if self.projection is not None:
            root_embs = self.projection(root_embs)
        
        # Normalize for cosine similarity
        root_embs = F.normalize(root_embs, p=2, dim=1)
        return root_embs
    
    def forward(self, queries, passages):
        """Forward pass for contrastive learning"""
        query_embs = self.encode(queries)
        passage_embs = self.encode(passages)
        return query_embs, passage_embs


class RetrievalLossHandler:
    def __init__(self, device, temp=0.05, combine_with_reconstruction=True, 
                 reconstruction_weight=0.1):
        self.device = device
        self.temp = temp
        self.combine_with_reconstruction = combine_with_reconstruction
        self.reconstruction_weight = reconstruction_weight
        self.ce_criterion = nn.CrossEntropyLoss()
        
    def contrastive_loss(self, query_embs, passage_embs):
        """
        Compute InfoNCE contrastive loss with in-batch negatives
        
        Args:
            query_embs: (batch_size, embed_dim) normalized query embeddings
            passage_embs: (batch_size, embed_dim) normalized passage embeddings
        """
        batch_size = query_embs.shape[0]
        
        # Compute similarity matrix: (batch_size, batch_size)
        # Each query should match with its corresponding passage (diagonal)
        sim_matrix = torch.matmul(query_embs, passage_embs.T) / self.temp
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=self.device)
        
        # Compute cross-entropy loss (InfoNCE)
        loss = self.ce_criterion(sim_matrix, labels)
        return loss
    
    def reconstruction_loss(self, model, tokens):
        """Original Banyan reconstruction loss"""
        tokens = tokens.to(self.device)
        logits, labels = model.banyan(tokens)
        loss = self.ce_criterion(logits, labels)
        return loss
    
    def combined_loss(self, model, queries, passages, query_tokens=None):
        """
        Combined loss: contrastive + reconstruction
        
        Args:
            model: BanyanRetriever model
            queries: query token sequences (batch_size, seq_len)
            passages: passage token sequences (batch_size, seq_len)
            query_tokens: optional, for reconstruction loss on queries
        """
        # Contrastive loss
        query_embs, passage_embs = model(queries, passages)
        contrastive = self.contrastive_loss(query_embs, passage_embs)
        
        # Reconstruction loss (optional)
        if self.combine_with_reconstruction and query_tokens is not None:
            recon = self.reconstruction_loss(model, query_tokens)
            total_loss = contrastive + self.reconstruction_weight * recon
            return total_loss, contrastive.item(), recon.item()
        
        return contrastive, contrastive.item(), 0.0


def train_retrieval(dataloader, model, loss_handler, optimizer, epoch):
    """Training loop for retrieval fine-tuning"""
    model.train()
    total_loss = 0
    total_contrastive = 0
    total_reconstruction = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        queries, passages = batch['query'], batch['passage']
        queries = queries.to(loss_handler.device)
        passages = passages.to(loss_handler.device)
        
        optimizer.zero_grad()
        
        # Compute loss
        if loss_handler.combine_with_reconstruction:
            # Use passages for reconstruction (queries are typically shorter)
            loss, cont_loss, recon_loss = loss_handler.combined_loss(
                model, queries, passages, query_tokens=passages
            )
        else:
            loss, cont_loss, recon_loss = loss_handler.combined_loss(
                model, queries, passages
            )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_contrastive += cont_loss
        total_reconstruction += recon_loss
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cont': f'{cont_loss:.4f}',
            'recon': f'{recon_loss:.4f}'
        })
    
    n = len(dataloader)
    print(f'Epoch {epoch} - Loss: {total_loss/n:.4f}, '
          f'Contrastive: {total_contrastive/n:.4f}, '
          f'Reconstruction: {total_reconstruction/n:.4f}')
    
    return total_loss / n


@torch.no_grad()
def validate_retrieval(dataloader, model, loss_handler):
    """Validation loop"""
    model.eval()
    total_loss = 0
    total_contrastive = 0
    
    for batch in tqdm(dataloader, desc='Validation'):
        queries, passages = batch['query'], batch['passage']
        queries = queries.to(loss_handler.device)
        passages = passages.to(loss_handler.device)
        
        loss, cont_loss, _ = loss_handler.combined_loss(
            model, queries, passages
        )
        
        total_loss += loss.item()
        total_contrastive += cont_loss
    
    n = len(dataloader)
    print(f'Validation - Loss: {total_loss/n:.4f}, '
          f'Contrastive: {total_contrastive/n:.4f}')
    
    return total_loss / n


def main(args, device):
    # Load pretrained Banyan model
    print('Loading pretrained Banyan model...')
    pretrained = torch.load(args.pretrained_path, map_location=device)
    banyan = Banyan(25001, args.e_dim, args.channels, args.r, device).to(device)
    banyan.load_state_dict(pretrained['model'])
    
    # Create retrieval model
    model = BanyanRetriever(
        banyan, 
        projection_dim=args.projection_dim if args.use_projection else None
    ).to(device)
    
    # Freeze options
    if args.freeze_embeddings:
        print('Freezing embedding layer...')
        model.banyan.embedding.requires_grad_(False)
    
    if args.freeze_composition:
        print('Freezing composition/decomposition functions...')
        model.banyan.comp_fn.requires_grad_(False)
        model.banyan.decomp_fn.requires_grad_(False)
    
    # Create dataloaders using BPEmb tokenization
    from msmarco_dataloader import create_retrieval_dataloader
    
    train_dataloader = create_retrieval_dataloader(
        args.train_path, 
        args.batch_size, 
        shuffle=True,
        max_query_len=args.max_query_len,
        max_passage_len=args.max_passage_len,
        lang='en',
        num_workers=args.num_workers
    )
    dev_dataloader = create_retrieval_dataloader(
        args.dev_path, 
        args.batch_size, 
        shuffle=False,
        max_query_len=args.max_query_len,
        max_passage_len=args.max_passage_len,
        lang='en',
        num_workers=args.num_workers
    )
    
    # Loss and optimizer
    loss_handler = RetrievalLossHandler(
        device,
        temp=args.temperature,
        combine_with_reconstruction=args.combine_reconstruction,
        reconstruction_weight=args.recon_weight
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs
    )
    
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_retrieval(train_dataloader, model, loss_handler, 
                                     optimizer, epoch)
        val_loss = validate_retrieval(dev_dataloader, model, loss_handler)
        
        scheduler.step()
        
        # Save best model
        if val_loss < best_loss and args.save_path:
            print(f'Model improved! Saving to {args.save_path}')
            state_dict = {
                'model': model.state_dict(),
                'banyan_only': model.banyan.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': val_loss,
                'args': args
            }
            torch.save(state_dict, args.save_path)
            best_loss = val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune Banyan for Retrieval')
    
    # Model args
    parser.add_argument('--pretrained_path', type=str, required=True,
                       help='Path to pretrained Banyan model')
    parser.add_argument('--e_dim', type=int, default=256,
                       help='Embedding dimensionality')
    parser.add_argument('--channels', type=int, default=128,
                       help='Number of channels')
    parser.add_argument('--r', type=float, default=0.1,
                       help='Embedding init range')
    
    # Retrieval-specific args
    parser.add_argument('--use_projection', action='store_true',
                       help='Use projection head for contrastive learning')
    parser.add_argument('--projection_dim', type=int, default=128,
                       help='Projection head dimension')
    parser.add_argument('--temperature', type=float, default=0.05,
                       help='Temperature for contrastive loss')
    
    # Training strategy
    parser.add_argument('--combine_reconstruction', action='store_true',
                       help='Combine contrastive with reconstruction loss')
    parser.add_argument('--recon_weight', type=float, default=0.1,
                       help='Weight for reconstruction loss')
    parser.add_argument('--freeze_embeddings', action='store_true',
                       help='Freeze embedding layer')
    parser.add_argument('--freeze_composition', action='store_true',
                       help='Freeze composition/decomposition functions')
    
    # Data args
    parser.add_argument('--train_path', type=str, required=True,
                       help='Path to training data (JSONL)')
    parser.add_argument('--dev_path', type=str, required=True,
                       help='Path to validation data (JSONL)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--max_query_len', type=int, default=32,
                       help='Maximum query length')
    parser.add_argument('--max_passage_len', type=int, default=200,
                       help='Maximum passage length')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    # Optimization args
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    
    # Misc
    parser.add_argument('--save_path', type=str, required=True,
                       help='Path to save fine-tuned model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    main(args, device)