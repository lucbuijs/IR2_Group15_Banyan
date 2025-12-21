# main training script for Banyan model
import torch.nn as nn
from utils import *
from models import SelfStrAE
import argparse
from tqdm import tqdm 
from eval import IntrinsicEvaluator
import torch.nn.functional as f

def nxn_cos_sim(A, B, dim=1):
    a_norm = f.normalize(A, p=2, dim=dim)
    b_norm = f.normalize(B, p=2, dim=dim)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class LossHandler:
    def __init__(self, loss_type, device):
        self.criterion = nn.CrossEntropyLoss()
        self.loss_type = loss_type
        self.device = device
        self.tau = 0.2

    def apply_contrastive(self, model, tokens):
        out = model(tokens)
        prediction_matrix = nxn_cos_sim(out[0], out[1]) * torch.exp(torch.tensor(self.tau).to(self.device))
        n = out[0].shape[0]
        labels = torch.tensor(range(n), dtype=torch.long).to(self.device)
        loss = self.criterion(prediction_matrix, labels)
        loss += self.criterion(prediction_matrix.T, labels)
        loss = loss / 2
        return loss

    def apply_ceco(self, model, tokens):
        out = model(tokens, ceco=True)
        prediction_matrix = nxn_cos_sim(out[1], out[2]) * torch.exp(torch.tensor([self.tau]).to(self.device))
        n = out[1].shape[0]
        labels = torch.tensor(range(n), dtype=torch.long).to(self.device)
        loss = self.criterion(prediction_matrix, labels)
        loss += self.criterion(prediction_matrix.T, labels)
        loss = loss / 2
        loss += self.criterion(out[0], torch.cat(tokens, dim=0).to(self.device))
        loss = loss / 2
        return loss

    def loss(self, model, tokens):
        if self.loss_type == 'CON':
            return self.apply_contrastive(model, tokens)
        elif self.loss_type == 'CECO':
            return self.apply_ceco(model, tokens)


def train(dataloader, model, obj, optimizer):
    model.train()
    epoch_loss = 0
    for tokens in tqdm(dataloader):
        optimizer.zero_grad()
        loss = obj.loss(model, tokens)
        epoch_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    print('Train Loss: {}'.format(epoch_loss / len(dataloader)), flush=True)
    return epoch_loss / len(dataloader)

@torch.no_grad()
def validate(dataloader, model, obj):
    model.eval()
    validation_loss = 0
    for tokens in tqdm(dataloader):
        loss = obj.loss(model, tokens)
        validation_loss += loss.item()
    print('Val Loss: {}'.format(validation_loss / len(dataloader)), flush=True)
    return validation_loss / len(dataloader)

def main(args, device):
    train_dataloader = create_dataloader(args.train_path, args.batch_size, shuffle=True, lang=args.lang)
    dev_dataloader = create_dataloader(args.dev_path, args.batch_size, shuffle=False, lang=args.lang)
    model = SelfStrAE(25001, args.e_dim, args.channels, args.r, device).to(device)
    objective = LossHandler(args.loss_type, device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    evaluator = IntrinsicEvaluator(device, args.lang)
    best_loss = 10000000

    for epoch in range(args.epochs):
        print(epoch, flush=True)
        train_loss = train(train_dataloader, model, objective, optimizer)
        val_loss = validate(dev_dataloader, model, objective)

        if args.lang == 'en':
            print('Lexical Evaluation', flush=True)
            sl_score, ws_score, wr_score = evaluator.evaluate_word_level(model)
            lex_score = (sl_score + ws_score + wr_score) / 3
            print('Average Lex Score: {}'.format(lex_score), flush=True)
            print('\n')

            print('STS Evaluation', flush=True) 
            sts12_score, sts13_score, sts14_score, sts15_score, sts16_score, stsb_score, sick_score, sem_score = evaluator.evaluate_sts(model, device)
            sts_score = (sts12_score + sts13_score + sts14_score + sts15_score + sts16_score + stsb_score + sick_score + sem_score) / 8
            print('Average STS Score: {}'.format(sts_score), flush=True)
            print('\n')
        
        else:
            print(f'{args.lang.upper()} STR Evaluation', flush=True)
            str_score = evaluator.evaluate_lang(model)
        
        if args.save_path:
            if val_loss < best_loss:
                print('Model Improved! Saving Progress...')
                state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'seed': args.seed, 'epoch': epoch, 'loss': val_loss}
                torch.save(state_dict, args.save_path)
                best_loss = val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Training Script')
    parser.add_argument('--train_path', help='specify the path to the training set', type=str,
                        default=None)
    parser.add_argument('--dev_path', help='specify path to validation set', type=str, default=None)
    parser.add_argument('--r', help='specify range for uniform embedding init, 0.0 means guassian init instead',
                        type=float, default=0.1)
    parser.add_argument('--e_dim', help='embedding dimensionality', type=int, default=256)
    parser.add_argument('--epochs', help='specify the number of epochs for which to train the model', type=int,
                        default=15)
    parser.add_argument('--batch_size', help='specify the batch size for training and dev', type=int, default=512)
    parser.add_argument('--lr', help='set the learning rate', type=float, default=1e-4)
    parser.add_argument('--save_path', help='specify the path to save the trained model', type=str)
    parser.add_argument('--load_path', help='specify the path to load a pre-trained model', type=str)
    parser.add_argument('--seed', help='set the random seed for the model', type=int)
    parser.add_argument('--channels', help='specify the number of channels for the embeddings', type=int, default=16)
    parser.add_argument('--lang', help='specify the language of the model', type=str, default='en')
    parser.add_argument('--loss_type', help='specify the type of loss to use', type=str, default='CON')
    
    args = parser.parse_args()
    args.seed = set_seed(args.seed)
    print(args.seed)
    assert args.e_dim % args.channels == 0, 'Embedding dimensionality must be divisible by the number of channels'
    assert args.lang in ['af', 'am', 'ar', 'en', 'es', 'ha', 'hi', 'id', 'mr', 'te'], 'Language must be one of the supported languages: af, am, ar, en, es, ha, hi, id, mr, te'

    if not args.train_path:
        args.train_path = f'../data/{args.lang}_train.txt'
        print(f'Train Set: {args.train_path}', flush=True)
    if not args.dev_path:
        args.dev_path = f'../data/{args.lang}_dev.txt'
        print(f'Dev Set: {args.dev_path}', flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    main(args, device)
