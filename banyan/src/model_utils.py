# helper functions for mapping sequences to indices and reducing frontiers
import torch
import numpy as np
import torch.nn.functional as f



def create_index(seqs):
    # Creates the index tensor for updating the adjacency matrix -> maps each token to a unique index matching its node
    unique_values = torch.unique(seqs)
    lookup = torch.zeros(unique_values.max() + 1, dtype=torch.long, device=seqs.device)
    lookup[unique_values] = torch.arange(len(unique_values), device=seqs.device)
    # set the final value to -1 to represent padding
    lookup[-1] = -1
    index = lookup[seqs]
    leaf_inds = torch.unique(index)[1:] # padding is set to -1 so 1:
    unique_values = unique_values[:-1] # remove the padding value, :-1 because pad = 25k for tokens
    return index, unique_values, leaf_inds


def get_complete(frontiers):
    return ~torch.all(frontiers[:, 1:] == -1, dim=1)

@torch.no_grad()
def get_sims(nodes, index):
    # empty to tensor to perform the similarity check
    sims = torch.full((index.shape[0], index.shape[1], nodes.shape[1]), -np.inf, device=nodes.device)
    # fill sims with the actual node embedding values
    sims[index != -1] = nodes[index[index != -1]]
    # take similarity between adjacent nodes in each frontier 
    cosines = f.cosine_similarity(sims[:, :-1, :], sims[:, 1:, :], dim=2)
    # mask the padded values
    cosines = cosines.masked_fill_((sims == -np.inf).all(dim=2)[:, 1:], -np.inf)
    # get the most similar pairs in each frontier
    max_sim = torch.argmax(cosines, dim=1)
    # additionally get the retrieval tensor (max_sim, max_sim + 1)
    retrieval = torch.cat((max_sim.unsqueeze(0), (max_sim + 1).unsqueeze(0)), dim=0).T.reshape(-1)
    return max_sim.long(), retrieval.long()

@torch.no_grad()
def reduce_frontier(index, completion_mask, range_tensor, max_indices):
    # create a mask to perform frontier reduction
    batch_remaining_mask = torch.ones_like(index, dtype=torch.bool)
    # remove left child of composed sequences
    batch_remaining_mask[range_tensor[completion_mask], max_indices[completion_mask]] = False
    # for completed sequences remove padding element so shapes fit
    if torch.where(completion_mask == 0)[0].numel() != 0:
        batch_remaining_mask[torch.where(completion_mask == 0, True, False), -1] = False
    # reduce the index tensor
    index = index[batch_remaining_mask].view(index.shape[0], -1)
    return index