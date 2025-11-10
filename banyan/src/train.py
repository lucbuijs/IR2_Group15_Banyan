# main training script for Banyan model
import torch.nn as nn
from utils import *
from models import Banyan
import argparse
from tqdm import tqdm 
from eval import IntrinsicEvaluator


class LossHandler:
    def __init__(self, device):
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def apply_ce(self, model, tokens):
        tokens = tokens.to(self.device)
        logits, labels = model(tokens)
        loss = self.criterion(logits, labels)
        return loss

    def loss(self, model, tokens):
        return self.apply_ce(model, tokens)

def train(dataloader, model, obj, optimizer):
    model.train()
    epoch_loss = 0
    for tokens in tqdm(dataloader):
        optimizer.zero_grad()
        loss = obj.loss(model, tokens)
        epoch_loss += loss.item()
        loss.backward()
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
    model = Banyan(25001, args.e_dim, args.channels, args.r, device).to(device)
    objective = LossHandler(device)
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
    parser.add_argument('--lr', help='set the learning rate', type=float, default=1e-3)
    parser.add_argument('--save_path', help='specify the path to save the trained model', type=str)
    parser.add_argument('--seed', help='set the random seed for the model', type=int)
    parser.add_argument('--channels', help='specify the number of channels for the embeddings', type=int, default=128)
    parser.add_argument('--lang', help='specify the language of the model', type=str, default='en')
    
    args = parser.parse_args()
    args.seed = set_seed(args.seed)
    print(args.seed)
    assert args.e_dim % args.channels == 0, 'Embedding dimensionality must be divisible by the number of channels'
    assert args.lang in ['af', 'am', 'ar', 'en', 'es', 'ha', 'hi', 'id', 'mr', 'te'], 'Language must be one of the supported languages: af, am, ar, en, es, ha, hi, id, mr, te'

    if not args.train_path:
        args.train_path = f'data/{args.lang}_train.txt'
        print(f'Train Set: {args.train_path}', flush=True)
    if not args.dev_path:
        args.dev_path = f'data/{args.lang}_dev.txt'
        print(f'Dev Set: {args.dev_path}', flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    main(args, device)
