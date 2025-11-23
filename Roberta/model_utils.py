import torch
import torch.nn.functional as F

def mean_pool(last_hidden_state, attention_mask):
    """
    Mean pooling strategy as described in the paper.
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def encode_texts(texts, tokenizer, encoder, device="cuda", batch_size=32, max_length=256):
    """
    Encodes a list of texts into embeddings using the frozen encoder and mean pooling.
    """
    encoder.eval()
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = encoder(**encoded)
            emb = mean_pool(outputs.last_hidden_state, encoded.attention_mask)
            # L2 normalize for retrieval tasks (standard practice)
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu())
            
    return torch.cat(all_embeddings, dim=0)
