import pathlib, os

from beir.datasets.data_loader import GenericDataLoader


def load_data(split, dataset_version, data_path):
    dataset_name, corpus_version, queries_version = dataset_version.split('_')
    print(dataset_name, corpus_version, queries_version, split)
    
    data_path = os.path.join(data_path, dataset_name)
    corpus_path = os.path.join(data_path, 'corpus/%s.jsonl'%corpus_version)
    query_path = os.path.join(data_path, 'queries/%s.jsonl'%queries_version)  # original.jsonl wav2vec2-base-960h.jsonl
    qrels_path = os.path.join(data_path, 'qrels/%s_%s.tsv'%(split, corpus_version))  # valid_relations.tsv train_entities.tsv
    
    return GenericDataLoader(corpus_file=corpus_path, query_file=query_path, qrels_file=qrels_path).load_custom()
