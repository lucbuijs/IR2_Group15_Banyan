# evaluate the Banyan model on SST and MRPC datasets
import torch.optim as optim
from utils import *
from models import Banyan
import numpy as np
from tqdm import tqdm 
import torch.nn as nn
import sys

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def f1_score(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    
    tp = ((y_pred_tag == 1) & (y_test == 1)).sum().float()
    fp = ((y_pred_tag == 1) & (y_test == 0)).sum().float()
    fn = ((y_pred_tag == 0) & (y_test == 1)).sum().float()
    precision = tp / (tp + fp + 1e-7) 
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return torch.round(f1 * 100)



def process_sst_dataset(data_path):
    df = pd.read_csv(data_path, sep='\t')
    bpemb_en = BPEmb(lang='en', vs=25000, dim=100)
    sents1 = [sent_to_bpe(x.strip('\n'), bpemb_en) for x in df['sentence']]
    scores = [torch.tensor(x) for x in df['label']]
    dataset = [(sents1[x], scores[x]) for x in range(len(sents1))]
    return dataset


class SSTDataset(Dataset):
    def __init__(self, data):
        self.sequences = data
        self.n_samples = len(data)

    def __getitem__(self, index):
        return self.sequences[index]

    def __len__(self):
        return self.n_samples


def collate_fn_sst(data):
    sents_1 = pad_sequence([x[0] for x in data], batch_first=True, padding_value=25000)
    scores = torch.stack([x[1] for x in data], dim=0)
    return sents_1, scores


def create_sst_dataloader(data_path, batch_size, shuffle=False):
    data = process_sst_dataset(data_path)
    dataset = SSTDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_sst, shuffle=shuffle)
    return dataloader

@torch.no_grad()
def embed_sentence(model, path):
    model.eval()
    embeddings = []
    labels = []
    dataloader = create_sst_dataloader(path, 128)
    for inputs in tqdm(dataloader):
        tokens_1 = inputs[0]
        out = model(tokens_1.to(model.device), seqs2=tokens_1.to(model.device))
        embeddings.append(out[0])
        labels.append(inputs[-1])

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

@torch.no_grad()
def embed_sentence_pairs(model, path):
    model.eval()
    embeddings = []
    labels = []
    dataloader = create_sts_dataloader(path, 128)
    for inputs in tqdm(dataloader):
        tokens_1 = inputs[0]
        tokens_2 = inputs[1]
        out = model(tokens_1.to(model.device), seqs2=tokens_2.to(model.device))
        embeddings.append(torch.cat((out[0], out[1]), dim=1))
        labels.append(inputs[-1])

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

class S1ClassDataset(Dataset):
    def __init__(self, model, path):
        data = embed_sentence(model, path)
        self.x = data[0]
        self.y = data[1]
        self.n_samples = data[0].shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
class SPClassDataset(Dataset):
    def __init__(self, model, path):
        data = embed_sentence_pairs(model, path)
        self.x = data[0]
        self.y = data[1]
        self.n_samples = data[0].shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

class SPClassModel(nn.Module):
    def __init__(self, embedding_size , output_size, fancy=False):
        super(SPClassModel, self).__init__()
        if fancy: 
            self.layer_out = nn.Sequential(nn.Linear(2 * embedding_size, 512),
                                           nn.GELU(),
                                           nn.Linear(512, output_size))
        else:
            self.layer_out = nn.Linear(2 * embedding_size, output_size)

    def forward(self, x):
        return self.layer_out(x)


def singleclass_eval(model, train_path, dev_path, fancy=False):
    train_dataset = SPClassDataset(model, train_path)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    dev_dataset = SPClassDataset(model, dev_path)
    dev_dataloader = DataLoader(dev_dataset, batch_size=5000)

    results = []
    for seed in tqdm(range(5)):
        classifier = SPClassModel(256, 1, fancy=fancy).to('cuda')
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
        classifier.train()
        acc = []
        dev_acc = []

        best_acc = 0.0
        best_f1 = 0.0

        for epoch in range(40):
            epoch_loss = 0
            epoch_acc = 0

            for x, labels in train_dataloader:
                optimizer.zero_grad()
                labels = labels.to(model.device)

                logits = torch.squeeze(classifier(x.float()))
                loss = criterion(logits, labels.squeeze().float())

                epoch_loss += loss.item()
                epoch_acc += binary_acc(logits, labels.squeeze()).item()


                loss.backward()
                optimizer.step()

            acc.append(epoch_acc / len(train_dataloader))
            with torch.no_grad():
                for x, labels in dev_dataloader:
                    labels = labels.to(model.device)
                    logits = torch.squeeze(classifier(x.float()))

                    loss = criterion(logits, labels.squeeze().float())


                    if binary_acc(logits, labels.squeeze()).item() > best_acc:
                        best_acc = binary_acc(logits, labels.squeeze()).item()

                    if f1_score(logits, labels.squeeze()).item() > best_f1:
                        best_f1 = f1_score(logits, labels.squeeze()).item()

        results.append((best_acc, best_f1))

    return np.mean([x[0] for x in results]), np.mean([x[1] for x in results])


def sst_eval(model, train_path, dev_path, fancy=False):
    train_dataset = S1ClassDataset(model, train_path)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    dev_dataset = S1ClassDataset(model, dev_path)
    dev_dataloader = DataLoader(dev_dataset, batch_size=5000)

    results = []
    for seed in tqdm(range(5)):
        classifier = SPClassModel(128, 1, fancy=fancy).to('cuda')
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
        classifier.train()
        acc = []
        dev_acc = []

        best_acc = 0.0
        best_f1 = 0.0

        for epoch in range(40):
            epoch_loss = 0
            epoch_acc = 0

            for x, labels in train_dataloader:
                optimizer.zero_grad()
                labels = labels.to(model.device)

                logits = torch.squeeze(classifier(x.float()))
                loss = criterion(logits, labels.squeeze().float())

                epoch_loss += loss.item()
                epoch_acc += binary_acc(logits, labels.squeeze()).item()


                loss.backward()
                optimizer.step()

            acc.append(epoch_acc / len(train_dataloader))
            with torch.no_grad():
                for x, labels in dev_dataloader:
                    labels = labels.to(model.device)
                    logits = torch.squeeze(classifier(x.float()))

                    loss = criterion(logits, labels.squeeze().float())


                    if binary_acc(logits, labels.squeeze()).item() > best_acc:
                        best_acc = binary_acc(logits, labels.squeeze()).item()

                    if f1_score(logits, labels.squeeze()).item() > best_f1:
                        best_f1 = f1_score(logits, labels.squeeze()).item()

        results.append((best_acc, best_f1))

    return np.mean([x[0] for x in results]), np.mean([x[1] for x in results])

path = sys.argv[1]
print(f'Loading model from {path}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# these are the hyperparameters used in the paper (change as desired for your trained model)
model  = Banyan(25001, 256, 128, 0.1, device).to(device)

# load from checkpoint
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)


out = singleclass_eval(model, 'data/mrpc_train.csv', 'data/mrpc_test.csv', fancy=True)
print(f'MRPC Accuracy: {out[0]} F1: {out[1]}')
out = sst_eval(model, 'data/sst_train.tsv', 'data/sst_dev.tsv', fancy=True)
print(f'SST Accuracy: {out[0]} F1: {out[1]}')
