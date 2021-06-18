import json
import os

import itertools
from collections import Counter
from collections import defaultdict

from nltk.util import ngrams

import difflib
import re
from unidecode import unidecode

import numpy as np

import torch
import torch.nn.functional as F

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset

import soundfile as sf

import networkx as nx

TOPN = 500  # number of candidate labels to consider as hypotheses space
CTC_depth = 1  # number of top predictions by ASR model considered


def resize(a, size):
    new = np.array(a)
    new.resize(size)
    return new


def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return pos_a, pos_b, size


def embed_bvr(strings, bigram_vocabulary):
    bv = np.zeros((1, len(bigram_vocabulary)))
    for word in strings:
#     for word in words.split():
        # split into bigrams
        bigrams = list(ngrams(word, 2))
        # optimize tokenizer TODO
        for i, b in enumerate(bigrams):
            if b in bigram_vocabulary:
                # record bigram
                for c in range(bv.shape[0]):
                    if bv[c][bigram_vocabulary[b]] == 0: 
                        bv[c][bigram_vocabulary[b]] = i + 1
    #                     print(b, bigram_vocabulary[b], i + 1)
                        break
                    elif c == bv.shape[0] - 1:  # add dimension to record duplicate bigram
                        bvn = np.zeros(len(bigram_vocabulary))
                        bvn[bigram_vocabulary[b]] = i + 1
                        bv = np.vstack([bv, bvn])
    return bv


def embed_ctc(strings, bigram_vocabulary, predictions_logits, PAD = '<pad>'):
    bv = np.zeros((1, len(bigram_vocabulary)))
    b_scores = {}
    # make all combinations
    # iterate over columns
    os = np.ones(len(strings))
    for c in range(len(strings[0])-1):
        # iterate over rows
        for i in range(len(strings)):
            c1 = strings[i][c]
#             print(i, c, c1)
            c1_score = predictions_logits[i][c]
            added = False
            if c1 != PAD and c1_score > 0:
                for j in range(len(strings)):
                    # iterate over columns if pad symbol
                    for s in range(len(strings[0])-1-c):
                        c2 = strings[j][c+s+1]
                        c2_score = predictions_logits[j][c+s+1]
                        if c2 != PAD and c2 != c1 and c2_score > 0:
                            if (c1, c2) in bigram_vocabulary:
                                b_id = bigram_vocabulary[(c1, c2)]
#                                 if (c1, c2) c1_score + c2_score > 
                                b_scores[(c1, c2)] = c1_score + c2_score
                                
                                # record bigram
                                for w in range(bv.shape[0]):
#                                     print(bv[w][b_id])
                                    if bv[w][b_id] != 0 and bv[w][b_id] == os[i]:
                                        break # do not record duplicates
                                    if bv[w][b_id] != 0 and bv[w][b_id] == os[i]-1:
                                        break # do not record duplicates
                                    elif bv[w][b_id] == 0: 
#                                         print((c1, c2), os[i])
                                        bv[w][b_id] = os[i]
                                        added = True
#                                         os[i] += 1
                                        break
                                    elif w == bv.shape[0] - 1:  # add dimension to record duplicate bigram
#                                         print((c1, c2), os[i], 'added')
                                        bvn = np.zeros(len(bigram_vocabulary))
                                        bvn[b_id] = os[i]
#                                         os[i] += 1
                                        bv = np.vstack([bv, bvn])
                                        added = True
                                        break
                                break
            # advance counter to next column
            if added:
                os[i] += 1
#     print(b_scores)
    return bv, b_scores


class Recognizer():

    def __init__(self, sampling_rate=16_000, topn=TOPN, lexicon_path='../data/'):
        # load the ASR model
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.model.eval()
        self.model.to('cuda')
        
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.tokenizer = self.processor.tokenizer
        
        # load lexicon
        self.load_lexicon(lexicon_path)
        
        # params
        self.sampling_rate = sampling_rate
        self.topn = topn
        
        # prepare bigram dictionary for bvr
        self.prepare_bigrams()
        
    def prepare_bigrams(self):
        all_chars = list(self.tokenizer.encoder.keys())[5:]
        all_chars = ''.join(all_chars)
        all_bigrams = list(set(itertools.permutations(all_chars, 2)))  # unique bigrams

        self.bigram_vocabulary = {}
        for i, b in enumerate(all_bigrams):
            self.bigram_vocabulary[b] = i
        assert len(self.bigram_vocabulary) == len(all_bigrams)
    
    def load_lexicon(self, lexicon_path):
        with open(lexicon_path+'entities_bigrams_index.json', 'r') as fin:
            self.entity_bigrams = json.load(fin)
            
        with open(lexicon_path+'entities_labels2ids.json', 'r') as fin:
            self.entities_labels2ids = json.load(fin)
#         print(len(entities_labels2ids), 'entity labels in total')

#         print(len(entity_bigrams), 'entity bigrams')
        # old: relations_bigrams_index new:properties_bigrams_index
        with open(lexicon_path+'properties_bigrams_index.json', 'r') as fin:
            self.property_bigrams = json.load(fin)
        
        # old: relations_labels2ids new:properties_bigrams_index
        with open(lexicon_path+'properties_labels2ids.json', 'r') as fin:
            self.relations_labels2ids = json.load(fin)
#         print(len(relations_labels2ids), 'relation labels in total') 

    def predict(self, path):
        speech, samplerate = sf.read(path)
        self.input_values = self.processor(speech, sampling_rate=self.sampling_rate,
                                      return_tensors="pt", padding="longest").input_values
        
        self.input_values = self.input_values.to('cuda')
        
        with torch.no_grad():
            logits = self.model(self.input_values).logits
        return logits
        
    def greedy_decode(self, path):
        # best path translation
        self.logits = self.predict(path)
        transcription = self.tokenizer.batch_decode(torch.argmax(self.logits, dim=-1))[0].lower()
        print(transcription)
        return transcription
                    
    def score_labels(self, path, match_entities=True, match_predicates=True):
        # greedy decode first
        transcription = self.greedy_decode(path)
        
        # 1) produce bigrams from the bp translation O(max len transcript)
        bottom_bigrams = []
        for word in transcription.split():
            for i in range(len(word)-1):
                bigram = word[i] + word[i+1]
                if bigram not in bottom_bigrams and bigram[0] != bigram[1]:
                    bottom_bigrams.append(bigram)
        
        # 2) match to the bigrams from the greedy decoded transcription
        e_labels, p_labels = Counter(), Counter()
        for b in bottom_bigrams:
            if b in self.entity_bigrams:
                matched_e_labels = self.entity_bigrams[b]
                for e_label in matched_e_labels:
                    # how many bigrams did the label match normalised by the length of the label
                    e_labels[e_label] += 1 / len(e_label) #/ len(entities2bigrams[e_label])
            if b in self.property_bigrams:
                matched_p_labels = self.property_bigrams[b]
                for p_label in matched_p_labels:
                    # how many bigrams did the label match normalised by the length of the label
                    p_labels[p_label] += 1 / len(p_label) #/ len(entities2bigrams[e_label])
    
        return e_labels, p_labels
    
    def top_decode(self, path, ctc_depth=CTC_depth):
        # best path translation
        speech, samplerate = sf.read(path)
        self.input_values = self.processor(speech, sampling_rate=self.sampling_rate,
                                      return_tensors="pt", padding="longest").input_values
        
        self.input_values = self.input_values.to('cuda')
        
        # run ASR
        with torch.no_grad():
            self.logits = self.model(self.input_values).logits

        # topk
        ctc_table = torch.topk(self.logits, k=ctc_depth, dim=-1)
        # print(ctc_table.indices)
        
        predicted_ids = ctc_table.indices[0]
        predictions = np.transpose(np.array(predicted_ids.cpu()))
        strings = self.tokenizer.batch_decode(predictions)
        
        logits = ctc_table.values[0]
        predictions_logits = np.transpose(logits.cpu().detach().numpy())
        
        # embed speech as bvr
        self.speech, self.b_scores = embed_ctc(strings, self.bigram_vocabulary, predictions_logits)
                
        # top hypothesis from greedy decoding
        transcription = strings[0].lower()
        print(transcription)
        return transcription
    
    def score_labels_new(self, path, match_entities=True, match_predicates=True):
        # decode topk
        transcription = self.top_decode(path)
        
        # 1) produce bigrams from the bp translation O(max len transcript)
        bottom_bigrams = []
        for word in transcription.split():
            for i in range(len(word)-1):
                bigram = word[i] + word[i+1]
                if bigram not in bottom_bigrams and bigram[0] != bigram[1]:
                    bottom_bigrams.append(bigram)
        
        # 2) match to the bigrams from the greedy decoded transcription
        e_labels, p_labels = Counter(), Counter()
        for b in bottom_bigrams:
            if b in self.entity_bigrams:
                matched_e_labels = self.entity_bigrams[b]
                for e_label in matched_e_labels:
                    # how many bigrams did the label match normalised by the length of the label
                    e_labels[e_label] += 1 / len(e_label) #/ len(entities2bigrams[e_label])
            if b in self.property_bigrams:
                matched_p_labels = self.property_bigrams[b]
                for p_label in matched_p_labels:
                    # how many bigrams did the label match normalised by the length of the label
                    p_labels[p_label] += 1 / len(p_label) #/ len(entities2bigrams[e_label])
    
        return e_labels, p_labels
    
    def rescore_new(self, e_hypotheses, topk=100):
#         print(e_hypotheses)
        lexicon = e_hypotheses#[]
#         for label in e_hypotheses:
#             lexicon.append(label.split(' ')[0])  # todo fix considering only the first word here
        
        # encode strings as bigrams
#         bvs = []
#         for word in lexicon:
#             bv = embed_bvr([word], self.bigram_vocabulary)
#             bvs.append(bv)
            
        bvs = []
        wmap = defaultdict(list)
        for i, words in enumerate(lexicon):
            for word in words.split(' '):
                if len(word) > 1:
                    bv = embed_bvr([word], self.bigram_vocabulary)
                    wmap[i].append(len(bvs))
                    bvs.append(bv)

        # calculate differences
        seq_scores = []
        for i in range(len(bvs)):
            sample = bvs[i][0]
            word_length = np.count_nonzero(sample) # minimum number of overlaps to get the score 1
            if not word_length:
                seq_scores.append(0)
                continue
            mask = self.speech * sample # non-zero overlap
            set_overlap = np.count_nonzero(mask[0]) # number of overlapping bigrams as a set

            set_score = set_overlap / word_length
            if set_overlap:

                diff = sample - self.speech
                # apply mask
                m = np.ma.masked_where(mask==0, diff)

                # find clusters
                u, c = np.unique(m, return_counts=True)
                dup = u[c > 1]
                seq_overlap = max(c[:-1]) # do not count -- symbols 

                seq_score = seq_overlap / word_length + seq_overlap
                if seq_overlap / word_length < 0.5: # min overlap threshold
                    seq_score = 0

            else:
                seq_score = 0
        #     print(seq_score)
            seq_scores.append(seq_score)
        seq_scores = np.array(seq_scores)
        
        scores = np.zeros(len(lexicon))
        for i, windices in wmap.items():
            _score = sum(seq_scores[windices]) / len(windices) + sum(seq_scores[windices]) #+ len(windices)
            word = lexicon[i]
            bigrams = list(ngrams(word, 2))
            score = 0
            for b in bigrams:
                if b in self.b_scores:
                    score += self.b_scores[b] #* scores[i]
            scores[i] = score * _score

        rank = (-scores).argsort()

        return rank, scores
    
    
    def rescore(self, e_hypotheses):
        with self.processor.as_target_processor():
            labels_batch = self.processor(e_hypotheses, return_tensors="pt", padding=True)
             # replace padding with -100 to ignore loss correctly
            labels = labels_batch.input_ids.masked_fill(labels_batch.attention_mask.ne(1), -100)
    #                 print(labels)

        labels = labels.to('cuda')

        # compute loss

        # retrieve loss input_lengths from attention_mask
        attention_mask = torch.ones_like(self.input_values, dtype=torch.long)
        attention_mask = torch.cat(labels.shape[0]*[attention_mask])
        input_lengths = self.model.module._get_feat_extract_output_lengths(attention_mask.sum(-1))

        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)

        logits = torch.cat(labels.shape[0]*[self.logits])
        logits[logits < 0] = 0  # drop negative logits

        logits = logits.transpose(0, 1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                logits,
                flattened_targets,
                input_lengths,
                target_lengths,
                blank=self.model.module.config.pad_token_id,
                reduction="none",
                zero_infinity=True
            )
        rank = torch.argsort(-loss, dim=-1, descending=True)
        
        return rank, loss
