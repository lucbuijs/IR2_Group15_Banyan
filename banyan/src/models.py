# main func defining the Banyan Self-Structuring AutoEncoder
import torch
import torch.nn as nn
import dgl
import numpy as np
from model_utils import create_index, get_sims, reduce_frontier, get_complete
from funcs import Compose, Decompose


class Banyan(nn.Module):
    def __init__(self, vocab_size, embedding_size, channels, r, device):
        super(Banyan, self).__init__()
        self.E = embedding_size
        self.c = channels
        self.e = int(self.E/self.c)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=vocab_size - 1)
        if r != 0.0:
            nn.init.uniform_(self.embedding.weight, -r, r)
        self.embedding.weight.data[vocab_size - 1] = -np.inf
        self.comp_fn = Compose(self.E, self.c)
        self.decomp_fn = Decompose(self.E, self.c)
        self.vocab_size = vocab_size
        self.device = device
        self.dropout = nn.Dropout(p=0.1, inplace=True)
        # projection layer back to vocab size
        self.out = nn.Linear(self.E, self.vocab_size-1)
        
    def compose_words(self, word_sequence):
        word_sequence = self.embedding(word_sequence)
        while word_sequence.shape[0] != 1:
            # find that subwords that should be merged first
            cosines = torch.nn.functional.cosine_similarity(word_sequence[:-1], word_sequence[1:], dim=1)
            # get the indices of the subwords that should be merged
            max_indices = torch.argmax(cosines, dim=0)
            retrieval = torch.cat((max_indices.unsqueeze(0), (max_indices + 1).unsqueeze(0)), dim=0).T.reshape(-1)
            # compose accordingly
            batch_selected = word_sequence[retrieval.long()].view(2, self.E)
            parent = self.comp_fn(batch_selected.view(2, self.c, self.e), words=True)
            # substitute the second of the composed words with the parent embedding
            word_sequence[max_indices.long() + 1] = parent
            # mask out the first of the composed words
            batch_remaining_mask = torch.ones(word_sequence.shape).bool()
            batch_remaining_mask[max_indices.long()] = False
            # update the queue
            word_sequence = word_sequence[batch_remaining_mask].view(-1, self.E)
        return word_sequence.squeeze()


    def update_graph(self, graph, retrieval, index):
        # range used for indexing
        range_tensor = torch.arange(index.shape[0], device=index.device, dtype=torch.long).repeat_interleave(2)
        # get src
        src = index[range_tensor, retrieval].view(-1,2)
        # get existing edges in the graph
        ex_src, ex_dst = graph.edges() if graph.num_edges() > 0 else (None, None)
        # get the new src and dst 
        if ex_src is not None:
            # check whether edges already exist in graph
            mask = ~torch.eq(src.unsqueeze(1), ex_src.view(-1, 2)).all(dim=2).any(dim=1)
            # filter src according to mask
            src = src[mask]
            # make sure the new edges are unique
            src = torch.unique(src, dim=0) 
            # set indices for the new nodes
            dst = torch.max(ex_dst) + 1 + torch.arange(src.shape[0], device=src.device)
        else:
            src = torch.unique(src, dim=0)
            dst = torch.max(index) + 1 + torch.arange(src.shape[0], device=src.device)
        # update the graph
        # 1. add the new nodes and their representations 
        graph.add_nodes(dst.shape[0], {'comp': self.comp_fn(graph.ndata['comp'][src].view(-1, 2, self.c, self.e))})
        # 2. add the new edges
        graph.add_edges(src.flatten(), dst.repeat_interleave(2).flatten())
        # update index tensor 
        # 1. recreate the original src tensor
        src = index[range_tensor, retrieval].view(-1,2)
        # 2. find which edges contain src
        ex_src, ex_dst = graph.edges()
        locs = torch.where(src.unsqueeze(1) == ex_src.view(-1, 2), 1, 0).all(dim=-1).nonzero()[:, 1]
        # 3. get the corresponding values from dst and fill index accordingly 
        update = ex_dst.view(-1, 2)[locs] 
        index[range_tensor, retrieval] = update.view(-1)
        # return the updated graph and index tensor
        return graph, index


    def compose(self, seqs, roots=False):
        # we need the range for several operations and no point casting it to cuda each time
        range_tensor = torch.tensor(range(seqs.shape[0]), dtype=torch.long, device=self.device)
        # get the node indexes for the embeddings and the tokens they map to 
        index, tokens, leaf_inds = create_index(seqs) # index shape (b_size, seq_len) tokens shape set(tokens in seqs)
        # create the graph 
        g = dgl.graph(([], []), device=self.device) # empty graph
        g.add_nodes(tokens.shape[0]) # add the nodes
        g.ndata['comp'] = self.dropout(self.embedding(tokens)) # add the embeddings for the nodes

        # reduce the frontiers till we get to the root 
        while index.shape[1] != 1:
            # get the merge indices
            max_sim, retrieval = get_sims(g.ndata['comp'].detach(), index)
            # get the completion mask
            completion_mask = get_complete(index)
            # update the graph 
            g, index[completion_mask] = self.update_graph(g, retrieval[completion_mask.repeat_interleave(2)], index[completion_mask])
            # reduce the frontiers
            index = reduce_frontier(index, completion_mask, range_tensor, max_sim)
        
        if roots:
            return g.ndata['comp'][index.flatten()]

        # return the graph and the traversal order
        rg = g.reverse(copy_ndata=True) # reverse the graph: root -> leaves
        rt = [t.to(self.device) for t in dgl.topological_nodes_generator(rg)]
        # init features for the decoder 
        rg.ndata['feat'] = rg.ndata['comp'].view(-1, self.c, self.e)
        # binary indicator for whether an edge leads to left or right child
        rg.edata['pos'] = torch.tensor([[1,0], [0,1]], device=self.device).repeat(rg.num_edges()//2, 1)
        return rg, rt, tokens, leaf_inds

    def forward(self, seqs, seqs2=None, words=False):
        # for lex eval
        if words:
            return self.compose_words(seqs)
        # for STS 
        if seqs2 is not None: 
            r1 = self.compose(seqs, roots=True)
            r2 = self.compose(seqs2, roots=True)
            return r1, r2

        # get the graph and traversal order
        rg, rt, tokens, leaf_inds = self.compose(seqs)
        # propagate the embeddings
        rg.prop_nodes(rt[1:], message_func=self.decomp_fn.message_func, reduce_func=self.decomp_fn.reduce_func)
        # project leaf nodes to vocab and return
        return self.out(rg.ndata['feat'][leaf_inds].view(-1, self.E)), tokens

    
            




    



