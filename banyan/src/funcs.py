# Diagonal Composition and Decomposition functions
import torch
import torch.nn as nn
import torch.nn.functional as f


class Compose(nn.Module):
    def __init__(self, embedding_size, channel_size):
        super(Compose, self).__init__()
        self.E = embedding_size
        self.c = channel_size
        self.e = int(self.E / self.c)
        self.comp_l = nn.Parameter(torch.rand(self.e).uniform_(0.6, 0.9), requires_grad=True)
        self.comp_r = nn.Parameter(torch.rand(self.e).uniform_(0.6, 0.9), requires_grad=True)
        self.cb = nn.Parameter(torch.zeros(self.e), requires_grad=True)
        self.dropout = nn.Dropout(p=0.1, inplace=True)

    def forward(self, in_feats, words=False):
        if words:
            l_c = in_feats[0] * f.sigmoid(self.comp_l)
            r_c = in_feats[1] * f.sigmoid(self.comp_r)
            return (l_c + r_c + self.cb).view(-1, self.E)

        N, children, _, _ = in_feats.shape
        assert children == 2, "Expected to have only 2 children"
        t_in = in_feats.transpose(0, 1)
        l_c = t_in[0] * f.sigmoid(self.comp_l)
        r_c = t_in[1] * f.sigmoid(self.comp_r)
        return self.dropout(((l_c + r_c) + self.cb).view(-1, self.E))


class Decompose(nn.Module):
    def __init__(self, embedding_size, channel_size):
        super(Decompose, self).__init__()
        self.E = embedding_size
        self.c = channel_size
        self.e = int(self.E / self.c)
        self.decomp_l = nn.Parameter(torch.rand(self.e).uniform_(0.6, 0.9), requires_grad=True)
        self.decomp_r = nn.Parameter(torch.rand(self.e).uniform_(0.6, 0.9), requires_grad=True)
        self.lb = nn.Parameter(torch.zeros(self.e), requires_grad=True)
        self.rb = nn.Parameter(torch.zeros(self.e), requires_grad=True)
        self.dropout = nn.Dropout(p=0.1, inplace=True)


    def message_func(self, edges):
        N, _, _ = edges.src['feat'].shape
        lc = edges.src['feat'] * f.sigmoid(self.decomp_l) + self.lb
        rc = edges.src['feat'] * f.sigmoid(self.decomp_r) + self.rb
        t_out = torch.stack([lc, rc], dim=1).view(-1, self.c, self.e)
        t_out = self.dropout(t_out[edges.data['pos'].flatten().bool()])
        return {'feat': t_out}  

    def reduce_func(self, nodes):
        return {'feat': nodes.mailbox['feat'].mean(dim=1)}









