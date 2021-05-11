import os
import itertools

import difflib
import re
from unidecode import unidecode

import numpy as np
import torch

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset

import soundfile as sf

import networkx as nx


CTC_DEPTH = 5  # size of the ctc matrix considered for search
NPATHS = 10 # number of longest paths on the bigram graph


def get_overlap(s1, s2):
        s = difflib.SequenceMatcher(None, s1, s2)
        pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
        return s1[pos_a:pos_a+size]


class Recognizer():

    def __init__(self, sampling_rate=16_000, ctc_depth=CTC_DEPTH, n_paths=NPATHS):
        # load the ASR model
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.tokenizer = self.processor.tokenizer
        
        # params
        self.sampling_rate = sampling_rate
        self.ctc_depth = ctc_depth
        self.n_paths = n_paths

    def connect(self, predictions, t, k, n):
        '''
        Generate adjacencies
        '''
        edges = []
        for j in range(predictions.shape[0]):  # offset
            if predictions[j][k] != 0:
                edges.append([n*predictions.shape[1]+t, j*predictions.shape[1]+k])
            else:
                # skip to next if exists
                if k < predictions.shape[1]-1:
                    edges.extend(self.connect(predictions, t, k+1, n))
        return edges

    def predict(self, path):
        speech, samplerate = sf.read(path)

        input_values = self.processor(speech, sampling_rate=self.sampling_rate, return_tensors="pt", padding="longest").input_values
        self.logits = self.model(input_values).logits

        # find where s_tokens appear in the table
        ctc_table = torch.topk(self.logits, k=self.ctc_depth, dim=-1)
        predicted_ids = ctc_table.indices[0]

        predictions = np.transpose(np.array(predicted_ids))
        self.indices = predictions.flatten()

        self.edges = []
        for t in range(predictions.shape[1]-1):  # columns
            for n in range(predictions.shape[0]):  # rows
                if predictions[n][t] != 0:
                    self.edges.extend(self.connect(predictions, t, t+1, n))

    def match(self, query_str):
        query = self.tokenizer(query_str)['input_ids']
        query = [query[i:i + 2] for i in range(0, len(query)-1, 1)]
        
        # filter bigrams
        bigrams = []
        for e in self.edges:
            bigram = [self.indices[e[0]], self.indices[e[1]]]
            if bigram in query:
                bigrams.append(e)

        # build graph
        DG = nx.DiGraph()
        DG.add_edges_from(bigrams)
        
        # find all paths
    #     all_paths = []
    #     for (x, y) in itertools.combinations(DG.nodes, 2):
    #         for path in nx.all_simple_paths(DG, x, y):
    #             all_paths.append(path)
    #     # sort all paths
    #     all_paths.sort(key=len, reverse=True)
        
        all_paths = [nx.dag_longest_path(DG)]

        # lookup maximum overlap between strings
        for path in all_paths[:self.n_paths]:
            word = ''.join(self.tokenizer.convert_ids_to_tokens([self.indices[i] for i in path]))
            # print(word)
            overlap = get_overlap(query_str, word)
            print(overlap)
            return len(overlap) / len(query_str)
        return 0

    def search(self, query_str, chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'):
        '''
        Return the score for query_str wrt ctc matrix
        '''
        query_str = re.sub(chars_to_ignore_regex, '', query_str).lower()
        query_str = unidecode(query_str)
        query_str = ''.join([j for i, j in enumerate(query_str) if j != query_str[i-1]])  # remove repeated letters
        print(query_str)

        q_words = [w for w in query_str.split() if len(w) > 1]
        # print(q_words)

        matches = 0
        for word in q_words:
            query_str = word.upper()
            matches += self.match(query_str)

        return matches/len(q_words)
