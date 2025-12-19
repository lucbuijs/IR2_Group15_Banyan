# Banyan with Learned Merge Scoring (MLP instead of cosine similarity)
import torch
import torch.nn as nn
import dgl
import numpy as np
from model_utils import create_index, reduce_frontier, get_complete
from funcs import Compose, Decompose


class MergeScorer(nn.Module):
    """
    Learnable merge scoring function.
    
    Replaces fixed cosine similarity with an MLP that scores merge candidates.
    Input features: [left, right, left*right, |left-right|]
    Output: scalar merge score
    """
    
    def __init__(self, embedding_size, hidden_size=128):
        super(MergeScorer, self).__init__()
        # Input: concat of [left, right, left*right, |left-right|] = 4 * E
        self.mlp = nn.Sequential(
            nn.Linear(4 * embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, left, right):
        """
        Score merge candidates.
        
        Args:
            left: (batch, seq_len-1, E) left embeddings of adjacent pairs
            right: (batch, seq_len-1, E) right embeddings of adjacent pairs
            
        Returns:
            scores: (batch, seq_len-1) merge scores
        """
        # Compute features
        product = left * right
        diff = torch.abs(left - right)
        
        # Concatenate: [left, right, left*right, |left-right|]
        features = torch.cat([left, right, product, diff], dim=-1)
        
        # Score through MLP
        scores = self.mlp(features).squeeze(-1)  # (batch, seq_len-1)
        
        return scores



class BanyanMLP(nn.Module):
    """
    Banyan model with learned merge scoring.
    
    Uses a trainable MLP to score merge candidates instead of fixed cosine similarity.
    This allows the model to learn task-specific merge preferences.
    """
    
    def __init__(self, vocab_size, embedding_size, channels, r, device, hidden_size=128):
        super(BanyanMLP, self).__init__()
        self.E = embedding_size
        self.c = channels
        self.e = int(self.E / self.c)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=vocab_size - 1)
        if r != 0.0:
            nn.init.uniform_(self.embedding.weight, -r, r)
        self.embedding.weight.data[vocab_size - 1] = -np.inf
        self.comp_fn = Compose(self.E, self.c)
        self.decomp_fn = Decompose(self.E, self.c)
        self.vocab_size = vocab_size
        self.device = device
        self.dropout = nn.Dropout(p=0.1, inplace=True)
        self.out = nn.Linear(self.E, self.vocab_size - 1)
        
        # Learned merge scorer (replaces cosine similarity)
        self.merge_scorer = MergeScorer(embedding_size, hidden_size)
        
    def get_merge_scores(self, nodes, index):
        """
        Compute learned merge scores for adjacent pairs.
        
        Replaces the cosine similarity in get_sims().
        """
        # Get embeddings for all positions
        sims = torch.full((index.shape[0], index.shape[1], nodes.shape[1]), -np.inf, device=nodes.device)
        sims[index != -1] = nodes[index[index != -1]]
        
        # Get left and right embeddings for adjacent pairs
        left = sims[:, :-1, :]   # (batch, seq_len-1, E)
        right = sims[:, 1:, :]   # (batch, seq_len-1, E)
        
        # Create padding mask
        padding_mask = (sims == -np.inf).all(dim=2)[:, 1:]  # (batch, seq_len-1)
        
        # Replace -inf with zeros for MLP input (will be masked later)
        left_clean = left.clone()
        right_clean = right.clone()
        left_clean[left == -np.inf] = 0
        right_clean[right == -np.inf] = 0
        
        # Score through MLP
        scores = self.merge_scorer(left_clean, right_clean)  # (batch, seq_len-1)
        
        # Mask padded positions
        scores = scores.masked_fill_(padding_mask, -np.inf)
        
        # Get the best merge position
        max_sim = torch.argmax(scores, dim=1)
        retrieval = torch.cat((max_sim.unsqueeze(0), (max_sim + 1).unsqueeze(0)), dim=0).T.reshape(-1)
        
        return max_sim.long(), retrieval.long()

    def compose_words(self, word_sequence):
        """Compose word embeddings into a single representation (for lexical eval)."""
        word_sequence = self.embedding(word_sequence)
        while word_sequence.shape[0] != 1:
            # Use learned scorer for single sequence
            left = word_sequence[:-1].unsqueeze(0)  # (1, n-1, E)
            right = word_sequence[1:].unsqueeze(0)  # (1, n-1, E)
            scores = self.merge_scorer(left, right).squeeze(0)  # (n-1,)
            
            max_indices = torch.argmax(scores, dim=0)
            retrieval = torch.cat((max_indices.unsqueeze(0), (max_indices + 1).unsqueeze(0)), dim=0).T.reshape(-1)
            batch_selected = word_sequence[retrieval.long()].view(2, self.E)
            parent = self.comp_fn(batch_selected.view(2, self.c, self.e), words=True)
            word_sequence[max_indices.long() + 1] = parent
            batch_remaining_mask = torch.ones(word_sequence.shape).bool()
            batch_remaining_mask[max_indices.long()] = False
            word_sequence = word_sequence[batch_remaining_mask].view(-1, self.E)
        return word_sequence.squeeze()

    def update_graph(self, graph, retrieval, index):
        """Update graph with new nodes (same as original Banyan)."""
        range_tensor = torch.arange(index.shape[0], device=index.device, dtype=torch.long).repeat_interleave(2)
        src = index[range_tensor, retrieval].view(-1, 2)
        ex_src, ex_dst = graph.edges() if graph.num_edges() > 0 else (None, None)
        
        if ex_src is not None:
            mask = ~torch.eq(src.unsqueeze(1), ex_src.view(-1, 2)).all(dim=2).any(dim=1)
            src = src[mask]
            src = torch.unique(src, dim=0)
            dst = torch.max(ex_dst) + 1 + torch.arange(src.shape[0], device=src.device)
        else:
            src = torch.unique(src, dim=0)
            dst = torch.max(index) + 1 + torch.arange(src.shape[0], device=src.device)
        
        graph.add_nodes(dst.shape[0], {'comp': self.comp_fn(graph.ndata['comp'][src].view(-1, 2, self.c, self.e))})
        graph.add_edges(src.flatten(), dst.repeat_interleave(2).flatten())
        
        src = index[range_tensor, retrieval].view(-1, 2)
        ex_src, ex_dst = graph.edges()
        locs = torch.where(src.unsqueeze(1) == ex_src.view(-1, 2), 1, 0).all(dim=-1).nonzero()[:, 1]
        update = ex_dst.view(-1, 2)[locs]
        index[range_tensor, retrieval] = update.view(-1)
        
        return graph, index

    def compose(self, seqs, roots=False):
        """Compose sequences into tree representations using learned merge scoring."""
        range_tensor = torch.tensor(range(seqs.shape[0]), dtype=torch.long, device=self.device)
        index, tokens, leaf_inds = create_index(seqs)
        g = dgl.graph(([], []), device=self.device)
        g.add_nodes(tokens.shape[0])
        g.ndata['comp'] = self.dropout(self.embedding(tokens))

        while index.shape[1] != 1:
            # Use learned merge scoring instead of cosine similarity
            max_sim, retrieval = self.get_merge_scores(g.ndata['comp'], index)
            completion_mask = get_complete(index)
            g, index[completion_mask] = self.update_graph(g, retrieval[completion_mask.repeat_interleave(2)], index[completion_mask])
            index = reduce_frontier(index, completion_mask, range_tensor, max_sim)
        
        if roots:
            return g.ndata['comp'][index.flatten()]

        rg = g.reverse(copy_ndata=True)
        rt = [t.to(self.device) for t in dgl.topological_nodes_generator(rg)]
        rg.ndata['feat'] = rg.ndata['comp'].view(-1, self.c, self.e)
        rg.edata['pos'] = torch.tensor([[1, 0], [0, 1]], device=self.device).repeat(rg.num_edges() // 2, 1)
        return rg, rt, tokens, leaf_inds

    def forward(self, seqs, seqs2=None, words=False):
        """Forward pass - same interface as original Banyan."""
        if words:
            return self.compose_words(seqs)
        if seqs2 is not None:
            r1 = self.compose(seqs, roots=True)
            r2 = self.compose(seqs2, roots=True)
            return r1, r2

        rg, rt, tokens, leaf_inds = self.compose(seqs)
        rg.prop_nodes(rt[1:], message_func=self.decomp_fn.message_func, reduce_func=self.decomp_fn.reduce_func)
        return self.out(rg.ndata['feat'][leaf_inds].view(-1, self.E)), tokens
