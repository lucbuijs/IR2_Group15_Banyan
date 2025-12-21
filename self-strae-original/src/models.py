import torch
import torch.nn as nn
import dgl
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import itertools
import torch.nn.functional as f

def tensor_delete(tensor, indices):
    mask = torch.ones(tensor.shape, dtype=bool)
    mask[indices] = False
    return tensor[mask].view(-1, tensor.shape[1])



class SelfStrAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, channel_size, r, device):
        super(SelfStrAE, self).__init__()
        self.V = vocab_size
        self.E = embedding_size
        self.c = channel_size
        self.e = int(self.E / self.c)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=vocab_size - 1)
        if r != 0.0:
            nn.init.uniform_(self.embedding.weight, -r, r)

        self.embedding.weight.data[vocab_size - 1] = -np.inf
        self.comp_fn = Compose(self.E, self.c)
        self.decomp_fn = Decompose(self.E, self.c)
        self.vocab_size = vocab_size
        self.device = device
        self.dropout = nn.Dropout(p=0.2)
        # projection layer back to vocab size
        self.out = nn.Linear(self.E, self.vocab_size-1)


    def compose_words(self, word_sequence):
        while word_sequence.shape[0] != 1:
            # find that subwords that should be merged first
            cosines = f.cosine_similarity(word_sequence[:-1],
                                                            word_sequence[1:], dim=1)

            # get the indices of the subwords that should be merged
            max_indices = torch.argmax(cosines, dim=0)
            retrieval = torch.cat((max_indices.unsqueeze(0),
                                   (max_indices + 1).unsqueeze(0)), dim=0).T.reshape(-1)

            # compose accordingly
            batch_selected = word_sequence[retrieval.long()]
            parent = self.comp_fn(batch_selected.view(2, self.c, self.e), words=True)

            # substitute the second of the composed words with the parent embedding
            word_sequence[max_indices.long() + 1] = parent

            # mask out the first of the composed words
            batch_remaining_mask = torch.ones(word_sequence.shape).bool()
            batch_remaining_mask[max_indices.long()] = False

            # update the queue
            word_sequence = word_sequence[batch_remaining_mask].view(-1, self.E)

        return word_sequence.squeeze()

    def compose(self, padded, roots=False):
        b_size = padded.shape[0]

        # initialise our storage tensors
        # keep a tally of what the index of a given vector in the graph should be
        # used for filling in the adjacency matrix
        index_tracker = torch.arange(padded.shape[1], device=self.device).repeat(b_size).view(b_size, -1)
        # add the padded elements
        index_tracker[padded.isinf()[:, :, 0]] = -1

        # we know the total number of nodes in all graphs will be equal to the (number of leaf nodes * 2) -1
        # therefore our adjacency matrix and embedding matrices will have to correspond to this size
        adjacency_matrix = torch.zeros(b_size, (2 * padded.shape[1]) - 1, (2 * padded.shape[1]) - 1, device=self.device)

        # we already know the embeddings for the leaf nodes so we can fill first from the input sequences
        embedding_matrix = torch.cat(
            (padded, torch.full((b_size, padded.shape[1] - 1, self.E), -np.inf, device=self.device)), dim=1) # adds a stack of -inf vectors to the end of each sequence
                                                                                                             # shape -1 of seq len and will be filled with the non-leaf node embeddings


        # we need the range for several operations and no point casting it to cuda each time
        range_tensor = torch.tensor(range(b_size), dtype=torch.long, device=self.device)
        while padded.shape[1] != 1:
            #  calculate similarity to neighbour
            cosines = f.cosine_similarity(padded[:, :-1, :], padded[:, 1:, :], dim=2)
            # mask all the padded values
            cosines = cosines.masked_fill_((padded == -np.inf).all(dim=2)[:, 1:], -np.inf)
            # find the max
            max_indices = torch.argmax(cosines, dim=1)

            # select the max from the input
            # retrieval tensor contains the index of the max similarity
            # for a given sequence followed by the index subsequent to it
            retrieval = torch.cat((max_indices.unsqueeze(0), (max_indices + 1).unsqueeze(0)), dim=0).T.reshape(-1)
            # check which members of the batch only have one none negative inf value
            # this means that the root node has been reached for that sequence in the batch
            # and they should no longer be based to the composition function
            completion_mask = ~torch.eq(torch.sum(~torch.all(torch.isinf(padded) & (padded < 0), dim=2), dim=1), 1)
            # the first argument just specifies which element of the batch to take from
            batch_selected = padded[range_tensor[completion_mask].repeat_interleave(2),
                                    retrieval.long()[completion_mask.repeat_interleave(2)]]
            # this tells us which rows of the adjacency matrix needs to be updated i.e. the source
            src = index_tracker[range_tensor[completion_mask].repeat_interleave(2),
                                retrieval.long()[completion_mask.repeat_interleave(2)]]

            # which tell us which column we should be filling in the adjacency matrix i.e. the dst
            dst = torch.max(index_tracker, dim=1)[0][completion_mask] + 1
            # update adjacency matrix order is flipped because we actually want to store the reverse graph
            adjacency_matrix[range_tensor[completion_mask].repeat_interleave(2), dst.repeat_interleave(2), src] = 1

            # replace the rows corresponding to the second child being composed with
            # the embedding corresponding to the parent
            parent = self.comp_fn(batch_selected.view(-1, 2, self.c, self.e))
            padded[range_tensor[completion_mask], (max_indices.long() + 1)[completion_mask]] = parent

            # update embedding matrices to contain the new parent
            embedding_matrix[range_tensor[completion_mask], dst] = parent

            # update the index tracker to reflect the composition
            index_tracker[range_tensor[completion_mask], (max_indices.long() + 1)[completion_mask]] = dst

            # create a mask to filter out the first children because their rows have
            # no longer been updated
            batch_remaining_mask = torch.ones(padded.shape, device=self.device).bool()
            batch_remaining_mask[range_tensor[completion_mask], max_indices.long()[completion_mask]] = False

            # remove a padding row from completed sequences because less padded longer
            # sequences are still being composed
            if torch.where(completion_mask == 0)[0].numel() != 0:
                # print('great success!')
                batch_remaining_mask[torch.where(completion_mask == 0, True, False), -1, :] = False

            # filter out the relevant row from the updated sequences
            #padded = torch.masked_select(padded, batch_remaining_mask).view(b_size, -1, self.E)
            padded = padded[batch_remaining_mask].view(b_size, -1, self.E)
            index_tracker = index_tracker[batch_remaining_mask[:, :, 1]].view(b_size, -1)

        # the padded input is already reduced to the root representation for each element so just return that for STS
        if roots:
            return padded.squeeze()

        return embedding_matrix, adjacency_matrix

    def create_graph(self, adj):
        n_z = torch.nonzero(adj)
        graph = dgl.graph((n_z[:, 0], n_z[:, 1])).to(self.device)
        return graph

    def forward(self, sequences, seqs2=None, words=False, ceco=False):
        if words:
            ws = self.embedding(sequences)
            return self.compose_words(ws)
        
        if seqs2 is not None:
            padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.vocab_size - 1).to(self.device)
            padded_sequences = self.embedding(padded_sequences)
            roots_1 = self.compose(padded_sequences, roots=True)

            padded_sequences_2 = pad_sequence(seqs2, batch_first=True, padding_value=self.vocab_size - 1).to(self.device)
            padded_sequences_2 = self.embedding(padded_sequences_2)
            roots_2 = self.compose(padded_sequences_2, roots=True)

            return roots_1, roots_2

        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.vocab_size - 1).to(self.device)
        # the amount of times they need to be repeated
        node_nums = [2*len(x)-2 for x in sequences]
        # the number to be added to each of the non-zero elements
        node_index = [0]+list(np.cumsum(np.add(node_nums, 1))[:-1])
        node_index = torch.tensor(list(itertools.chain.from_iterable([[node_index[x]]*node_nums[x] for x in range(len(node_nums))]))).to(self.device)

        padded_sequences = self.embedding(padded_sequences)


        # create a mask so we don't apply dropout to the padded values
        mask = ~torch.isinf(padded_sequences).any(dim=-1)
        padded_sequences[mask] = self.dropout(padded_sequences[mask])

        embeddings, adj_matrices = self.compose(padded_sequences)
        adj_matrices = (torch.nonzero(adj_matrices)[:, 1:] + node_index[:, None])
        rg = dgl.graph((adj_matrices[:, 0], adj_matrices[:, 1])).to(self.device)
        embeddings = embeddings[~torch.isinf(embeddings)].view(-1, self.E)
        rg.ndata['feat'] = embeddings.view(-1, self.c, self.e)
        rt = [t.to(self.device) for t in dgl.topological_nodes_generator(rg)]
        rg.prop_nodes(rt[1:], message_func=self.decomp_fn.message_func, reduce_func=self.decomp_fn.reduce_func)



        if ceco:
            g = rg.reverse()
            tr = [t.to(self.device) for t in dgl.topological_nodes_generator(g)]
            return self.out(g.ndata['feat'].index_select(0, tr[0]).view(-1, self.E)), tensor_delete(embeddings, tr[0]), tensor_delete(rg.ndata['feat'].view(-1, self.E), tr[0])
        

        return embeddings, rg.ndata['feat'].view(-1, self.E)

class Compose(nn.Module):
    def __init__(self, embedding_size, channel_size):
        super(Compose, self).__init__()
        self.E = embedding_size
        self.c = channel_size
        self.e = int(self.E / self.c)
        self.compose = nn.Linear(2 * self.e, self.e)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, in_feats, words=False):
        if words:
            return self.compose(torch.cat((in_feats[0], in_feats[1]), dim=-1)).view(-1, self.E)
        N, children, _, _ = in_feats.shape
        assert children == 2, "Expected to have only 2 children"
        t_in = in_feats.transpose(0, 1)
        return self.dropout(self.compose(torch.cat((t_in[0], t_in[1]), dim=-1))).view(-1, self.E)


class Decompose(nn.Module):
    def __init__(self, embedding_size, channel_size):
        super(Decompose, self).__init__()
        self.E = embedding_size
        self.c = channel_size
        self.e = int(self.E / self.c)
        self.decompose = nn.Linear(self.e, 2 * self.e)
        self.dropout = nn.Dropout(p=0.1)

    def message_func(self, edges):
        N, _, _ = edges.src['feat'].shape
        _, c = edges.src['_ID'].unique_consecutive(return_counts=True, dim=0)
        assert c.eq(2).all().item(), 'Repeated pairs assumption broken!'
        t_out = torch.split(self.decompose(edges.src['feat'][::2]), self.e, dim=-1)
        return {'feat': self.dropout(torch.stack(t_out, dim=1).view(N, self.c, self.e))}

    def reduce_func(self, nodes):
        return {'feat': nodes.mailbox['feat'].squeeze(1)}





    
            




    



