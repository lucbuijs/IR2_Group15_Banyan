# dataloading, seed setting and general helper functions
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from bpemb import BPEmb
from tqdm import tqdm 
import torch
from torch.nn.utils.rnn import pad_sequence


def sent_to_bpe(sent, bpe):
    encoded = bpe.encode_ids(sent)
    return torch.tensor([torch.tensor(x) for x in encoded])


def process_dataset(data_path, lang):
    dataset = []
    bpemb_en = BPEmb(lang=lang, vs=25000, dim=100)
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines[:-1]): # skip the last line which is empty
            # parse the line
            line = line.strip('\n')
            emb = sent_to_bpe(line, bpemb_en)
            if 3 < emb.size(dim=0) <= 200:
                dataset.append(emb)
    return dataset

class StrAEDataset(Dataset):
    def __init__(self, data):
        self.sequences = data
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.sequences[index]
    
    def __len__(self):
        return self.n_samples



def collate_fn(data):
    return data

def process_sts_dataset(data_path, lang):
    df = pd.read_csv(data_path)
    bpemb_en = BPEmb(lang=lang, vs=25000, dim=100)
    sents1 = [sent_to_bpe(x.strip('\n'), bpemb_en) for x in df['sent1']]
    sents2 = [sent_to_bpe(x.strip('\n'), bpemb_en) for x in df['sent2']]
    scores = [torch.tensor(x) for x in df['score']]
    dataset = [(sents1[x], sents2[x], scores[x]) for x in range(len(sents1))]
    return dataset


class STSDataset(Dataset):
    def __init__(self, data):
        self.sequences = data
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.sequences[index]

    def __len__(self):
        return self.n_samples


def collate_fn_sts(data):
    sents_1 = [x[0] for x in data]
    sents_2 = [x[1] for x in data]
    scores = torch.stack([x[2] for x in data], dim=0)
    return sents_1, sents_2, scores



def create_sts_dataloader(data_path, batch_size, shuffle=False, lang='en'):
    data = process_sts_dataset(data_path, lang)
    dataset = STSDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_sts, shuffle=shuffle)
    return dataloader


def create_dataloader(data_path, batch_size, shuffle=False, lang='en'):
    data = process_dataset(data_path, lang)
    dataset = StrAEDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    return dataloader


def set_seed(seed=None):
    rseed = seed if seed else torch.initial_seed()
    rseed = rseed & ((1 << 63) - 1)  # protect against uint64 vs int64 issues
    torch.manual_seed(rseed)
    return rseed
