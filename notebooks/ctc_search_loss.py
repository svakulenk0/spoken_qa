import json
import os

import itertools
from collections import Counter

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


def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return pos_a, pos_b, size


class Recognizer():

    def __init__(self, sampling_rate=16_000, topn=TOPN, lexicon_path='../data/'):
        # load the ASR model
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.tokenizer = self.processor.tokenizer
        
        # load lexicon
        self.load_lexicon(lexicon_path)
        
        # params
        self.sampling_rate = sampling_rate
        self.topn = topn
    
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
#         self.model.to('cuda')
        
#         self.model.eval()
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
        
        return rank
