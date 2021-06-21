import pathlib, os
import json

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models

model_path = '/ivi/ilps/personal/svakule/spoken_qa/models/'

model_names = [["msmarco-distilbert-base-v3", 'cos_sim'],
               ["msmarco-distilbert-base-tas-b", 'dot'],
               [model_path+"msmarco-distilbert-base-v3-cos_sim-WD18_entities_original", 'cos_sim'],
               [model_path+"msmarco-distilbert-base-tas-b-dot-WD18_entities_original", 'dot'],
               [model_path+"msmarco-distilbert-base-tas-b-cos_sim-WD18_entities_original", 'cos_sim'],
               [model_path+"msmarco-distilbert-base-tas-b-dot-WD18_relations_original", 'dot'],
               [model_path+"msmarco-distilbert-base-tas-b-cos_sim-WD18_relations_original", 'cos_sim']]

similarities = ['cos_sim', 'dot']

data_path = '/ivi/ilps/personal/svakule/spoken_qa/datasets'
dataset_name = 'WD18'
splits = ['train', 'valid']

queries_versions = ['original', 'wav2vec2-base-960h']
corpus_versions = ['entities', 'relations']

dataset_versions = {'entities': ['WD18_entities_original', 'WD18_entities_wav2vec2-base-960h'],
                    'relations': ['WD18_relations_original', 'WD18_relations_wav2vec2-base-960h']}


def load_data(dataset_version, split='train', data_path=data_path):
    dataset_name, corpus_version, queries_version = dataset_version.split('_')
    print('Loaded:', dataset_name, corpus_version, queries_version, split)
    
    data_path = os.path.join(data_path, dataset_name)
    corpus_path = os.path.join(data_path, 'corpus/%s.jsonl'%corpus_version)
    query_path = os.path.join(data_path, 'queries/%s.jsonl'%queries_version)  # original.jsonl wav2vec2-base-960h.jsonl
    qrels_path = os.path.join(data_path, 'qrels/%s_%s.tsv'%(split, corpus_version))  # valid_relations.tsv train_entities.tsv
    
    return GenericDataLoader(corpus_file=corpus_path, query_file=query_path, qrels_file=qrels_path).load_custom()


def load_queries(queries_version='original', dataset_name=dataset_name, data_path=data_path):
    data_path = os.path.join(data_path, dataset_name)
    query_path = os.path.join(data_path, 'queries/%s.jsonl'%queries_version)  # original.jsonl wav2vec2-base-960h.jsonl
    queries = {}
    with open(query_path) as fin:
        for line in fin:
            line = json.loads(line)
            queries[line.get("_id")] = line.get("text")
    return queries


def evaluate_model(model_name, corpus_version, similarities=similarities):
    # iterate over models
#     for model_name, similarity in model_names[:]:
        # load model
    model = DRES(models.SentenceBERT(model_name))

    # wiggle similarity function at inference time
    for similarity in similarities:
        print(model_name, similarity)
        retriever = EvaluateRetrieval(model, score_function=similarity)

        # iterate over all dataset versions
        metrics = [] 
        for dataset_version in dataset_versions[corpus_version]:
            corpus, queries, qrels = load_data(dataset_version, 'valid', data_path)

            results = retriever.retrieve(corpus, queries)
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            metrics.extend([recall['Recall@1'], recall['Recall@3'],
                            recall['Recall@5'], recall['Recall@10'], recall['Recall@100']])

        #     break

        print('%.3f\t'*5*len(dataset_versions) % tuple(metrics))
