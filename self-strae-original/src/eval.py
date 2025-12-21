# run intrinisic evaluation, lexical + sts (where available)
from utils import *
from models import *
from scipy import stats
import numpy as np


class IntrinsicEvaluator:
    def __init__(self, device, lang):
        self.device = device
        self.lang = lang
        self.cos_sents = nn.CosineSimilarity(dim=1)

        if self.lang == 'en':
            # lexical 
            self.sl_data = self.load_word_level('../data/simlex.tsv')
            self.ws_data = self.load_word_level('../data/wordsim_similarity_goldstandard.txt')
            self.wr_data = self.load_word_level('../data/wordsim_relatedness_goldstandard.txt')
            self.cos_words = nn.CosineSimilarity(dim=0)
            # sentence
            self.sts12_dataloader = create_sts_dataloader('../data/sts12_test.csv', 128)
            self.sts13_dataloader = create_sts_dataloader('../data/sts13_test.csv', 128)
            self.sts14_dataloader = create_sts_dataloader('../data/sts14_test.csv', 128)
            self.sts15_dataloader = create_sts_dataloader('../data/sts15_test.csv', 128)
            self.sts16_dataloader = create_sts_dataloader('../data/sts16_test.csv', 128)
            self.stsb_dataloader = create_sts_dataloader('../data/stsb_test.csv', 128)
            self.sick_dataloader = create_sts_dataloader('../data/sick_test.csv', 128)
            self.sem_dataloader = create_sts_dataloader('../data/semrel_test.csv', 128)
        
        # afrikaans
        elif self.lang == 'af':
            self.lang_dataloader = create_sts_dataloader('../data/af_test.csv', 128, lang='af')
        # indonesian
        elif self.lang == 'id':
            self.lang_dataloader = create_sts_dataloader('../data/id_test.csv', 128, lang='id')
        # telugu
        elif self.lang == 'te':
            self.lang_dataloader = create_sts_dataloader('../data/te_test.csv', 128, lang='te')
        # arabic
        elif self.lang == 'ar':
            self.lang_dataloader = create_sts_dataloader('../data/ar_test.csv', 128, lang='ar')
        # hindi
        elif self.lang == 'hi':
            self.lang_dataloader = create_sts_dataloader('../data/hi_test.csv', 128, lang='hi')
        # marathi
        elif self.lang == 'mr':
            self.lang_dataloader = create_sts_dataloader('../data/mr_test.csv', 128, lang='mr')
        # hausa
        elif self.lang == 'ha':
            self.lang_dataloader = create_sts_dataloader('../data/ha_test.csv', 128, lang='ha')
        # amharic
        elif self.lang == 'am':
            self.lang_dataloader = create_sts_dataloader('../data/am_test.csv', 128, lang='am')
        # spanish
        elif self.lang == 'es':
            self.lang_dataloader = create_sts_dataloader('../data/es_test.csv', 128, lang='es')
        
    def load_word_level(self, path):
        dataset = []
        bpemb_en = BPEmb(lang='en', vs=25000, dim=100)
        with open(path, 'r') as f:
            for line in f.readlines():
                dataset.append((bpemb_en.encode_ids(line.split()[0].lower()), bpemb_en.encode_ids(line.split()[1].lower()), line.split()[2]))
        return dataset

    @torch.no_grad()
    def embed_word(self, word, model, embeddings):
        model.eval()
        if len(word) > 1:
            embed = model(torch.tensor(word, dtype=torch.long).to(self.device), words=True)
        else:
            embed = embeddings[word[0]]
        return embed.cpu()

    def evaluate_word_level(self, model):
        embeddings = model.embedding.weight.detach().cpu()
        sl_predictions = [self.cos_words(self.embed_word(x[0], model, embeddings), self.embed_word(x[1], model, embeddings)).item()
                          for x in self.sl_data]
        sl_score = stats.spearmanr(np.array(sl_predictions), np.array([x[2] for x in self.sl_data]))
        print('SimLex Score: {}'.format(sl_score.correlation.round(3)), flush=True)

        ws_predictions = [self.cos_words(self.embed_word(x[0], model, embeddings), self.embed_word(x[1], model,
                                                                                                   embeddings)).item()
                          for x in self.ws_data]
        ws_score = stats.spearmanr(np.array(ws_predictions), np.array([x[2] for x in self.ws_data]))
        print('Wordsim S Score: {}'.format(ws_score.correlation.round(3)), flush=True)

        wr_predictions = [self.cos_words(self.embed_word(x[0], model, embeddings), self.embed_word(x[1], model,
                                                                                                   embeddings)).item()
                          for x in self.wr_data]
        wr_score = stats.spearmanr(np.array(wr_predictions), np.array([x[2] for x in self.wr_data]))
        print('Wordsim R Score: {}'.format(wr_score.correlation.round(3)), flush=True)

        return sl_score.correlation, ws_score.correlation, wr_score.correlation

    @torch.no_grad()
    def embed_sts(self, model, dataloader):
        model.eval()
        predicted_sims = []
        all_scores = []
        for tokens_1, tokens_2, scores in dataloader:
            out = model(tokens_1, seqs2=tokens_2)
            predicted_sims.append(self.cos_sents(out[0], out[1]))
            all_scores.append(scores)
        return predicted_sims, all_scores

    def evaluate_sts(self, model, device):
        predicted_sims, all_scores = self.embed_sts(model, self.sts12_dataloader)
        sts12_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                                    np.array(torch.cat(all_scores, dim=0).cpu()))
        print('STS-12: {}'.format(sts12_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.sts13_dataloader)
        sts13_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                                    np.array(torch.cat(all_scores, dim=0).cpu()))
        print('STS-13: {}'.format(sts13_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.sts14_dataloader)
        sts14_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                                    np.array(torch.cat(all_scores, dim=0).cpu()))
        print('STS-14: {}'.format(sts14_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.sts15_dataloader)
        sts15_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                                    np.array(torch.cat(all_scores, dim=0).cpu()))
        print('STS-15: {}'.format(sts15_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.sts16_dataloader)
        sts16_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                                    np.array(torch.cat(all_scores, dim=0).cpu()))
        print('STS-16: {}'.format(sts16_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.stsb_dataloader)
        stsb_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                                    np.array(torch.cat(all_scores, dim=0).cpu()))
        print('STS-B: {}'.format(stsb_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.sick_dataloader)
        sick_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                        np.array(torch.cat(all_scores, dim=0).cpu()))
        print('SICK-R: {}'.format(sick_score.correlation.round(3)), flush=True)

        predicted_sims, all_scores = self.embed_sts(model, self.sem_dataloader)
        sem_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                        np.array(torch.cat(all_scores, dim=0).cpu()))
        print('SemRel: {}'.format(sem_score.correlation.round(3)), flush=True)

        return sts12_score.correlation, sts13_score.correlation, sts14_score.correlation, sts15_score.correlation, sts16_score.correlation, stsb_score.correlation, sick_score.correlation, sem_score.correlation


    def evaluate_lang(self, model):
        predicted_sims, all_scores = self.embed_sts(model, self.lang_dataloader)
        str_score = stats.spearmanr(np.array(torch.cat(predicted_sims, dim=0).cpu()),
                        np.array(torch.cat(all_scores, dim=0).cpu()))
        print('SCORE: {}'.format(str_score.correlation.round(3)), flush=True)
        return str_score.correlation