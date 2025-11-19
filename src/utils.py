import os
import json
import torch
import hdbscan

import numpy as np
import pandas as pd

from InstructorEmbedding import INSTRUCTOR
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import Birch, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score, adjusted_mutual_info_score, normalized_mutual_info_score

from src.eval_metrics import evaluate, singleton_accuracy, precision_recall_f1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_embedding(embedding):
    tokenizer = None
    if embedding == 'sbert':
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
    elif str(embedding).startswith('all'):
        embedding_model = SentenceTransformer(embedding)
    elif str(embedding).startswith('one') or str(embedding).startswith('two') or str(embedding).startswith('Multi'):
        embedding_model = SentenceTransformer(f'swardiantara/{embedding}')
    elif embedding == 'drone-sbert':
        model_path = 'swardiantara/drone-sbert'
        embedding_model = SentenceTransformer(model_path)
    elif embedding == 'simcse':
        model_path = 'princeton-nlp/sup-simcse-roberta-large'
        embedding_model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif str(embedding).startswith('instructor'):
        embedding_model = INSTRUCTOR(f'hkunlp/{embedding}')
    
    return embedding_model, tokenizer


def get_features(dataset, embedding, device=device, normalize_embeddings=False):
    corpus = dataset['message'].to_list()
    if embedding == 'sbert':
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        corpus_embeddings = embedding_model.encode(corpus)
    elif str(embedding).startswith('all'):
        embedding_model = SentenceTransformer(embedding)
        corpus_embeddings = embedding_model.encode(corpus, normalize_embeddings=normalize_embeddings)
    elif str(embedding).startswith('one') or str(embedding).startswith('two') or str(embedding).startswith('Multi'):
        embedding_model = SentenceTransformer(f'swardiantara/{embedding}')
        corpus_embeddings = embedding_model.encode(corpus, normalize_embeddings=normalize_embeddings)
    elif embedding == 'drone-sbert':
        model_path = 'swardiantara/drone-sbert'
        embedding_model = SentenceTransformer(model_path)
        corpus_embeddings = embedding_model.encode(corpus)
    elif embedding == 'simcse':
        model_path = 'princeton-nlp/sup-simcse-roberta-large'
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        inputs = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            corpus_embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    elif str(embedding).startswith('instructor'):
        embedding_model = INSTRUCTOR(f'hkunlp/{embedding}')
        log_dict = []
        for ind in dataset.index:
            log_dict.append(['Represent the Log message for clustering: ', dataset['message'][ind]])
        corpus_embeddings = embedding_model.encode(sentences=log_dict, device=device, normalize_embeddings=normalize_embeddings)
    
    return corpus_embeddings


def get_pred_df(clustering, dataset, sample_order):
    corpus = dataset['message'].to_list()
    pseudo_label = []
    log_message = []
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(clustering):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus[sentence_id])
    
    for i, cluster in clustered_sentences.items():
        for element in cluster:
            pseudo_label.append(i)
            log_message.append(element)

    cluster_label = pd.DataFrame({
        'message': log_message,
        'cluster_id': pseudo_label
    }).sort_values(by='message', ascending=sample_order).reset_index(drop=True)

    return cluster_label


def round_score(score_dict: dict, decimal=3):
    for key, value in score_dict.items():
        score_dict[key] = str(round(value, decimal))

    return score_dict


def evaluation_score(true_df: pd.DataFrame, pred_df: pd.DataFrame, sample_order):
    true_df = true_df.sort_values(by='message', ascending=sample_order).reset_index(drop=True)
    pred_df = pred_df.sort_values(by='message', ascending=sample_order).reset_index(drop=True)
    labels_pred = pred_df['cluster_id']
    labels_true = true_df['cluster_id']
    ami_score = adjusted_mutual_info_score(labels_true, labels_pred)
    fga, group_accuracy = evaluate(true_df, pred_df)
    singleton_acc, true_singleton_indices, pred_singleton_indices = singleton_accuracy(true_df, pred_df)
    _, _, singleton_f1 = precision_recall_f1(true_singleton_indices, pred_singleton_indices)
    nmi_score = normalized_mutual_info_score(labels_true, labels_pred)
    ari_score = adjusted_rand_score(labels_true, labels_pred)
    hgi_score = homogeneity_score(labels_true, labels_pred)
    cpi_score = completeness_score(labels_true, labels_pred)
    vmi_score = v_measure_score(labels_true, labels_pred)
    fmi_score = fowlkes_mallows_score(labels_true, labels_pred)
    num_cluster = len(pred_df['cluster_id'].unique())

    score_dict =  {
        'f1_group_accuracy': fga,
        'group_accuracy': group_accuracy,
        'singleton_f1': singleton_f1,
        'num_cluster': num_cluster,
        'ami_score': ami_score,
        'nmi_score': nmi_score,
        'ari_score': ari_score,
        'hgi_score': hgi_score,
        'cpi_score': cpi_score,
        'vmi_score': vmi_score,
        'fmi_score': fmi_score,
    }

    return round_score(score_dict)


def compute_distance_matrix(corpus_embeddings, is_norm=True):
    if is_norm:
        corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    distance_matrix = pairwise_distances(corpus_embeddings, corpus_embeddings, metric='cosine')
    return distance_matrix


def save_results(arguments_dict: dict, cluster_label_df: pd.DataFrame, output_dir: str):
    with open(os.path.join(output_dir, 'scenario_arguments.json'), 'w') as json_file:
        json.dump(arguments_dict, json_file, indent=4)
    if len(cluster_label_df) <= 2000:
        file_path = os.path.join(output_dir, 'prediction.xlsx')
        cluster_label_df.to_excel(file_path, index=False)


def get_model(args):
    if args.model == 'birch':
        return Birch(threshold=args.threshold, n_clusters=None)
    elif args.model == 'agglomerative':
        return AgglomerativeClustering(n_clusters=None,
                                    metric='precomputed',
                                    linkage=args.linkage,
                                    distance_threshold=args.threshold)
    elif args.model == 'dbscan':
        return DBSCAN(eps=args.threshold, min_samples=1, metric='cosine')
    elif args.model == 'hdbscan':
        return hdbscan.HDBSCAN(min_cluster_size=2,
                                    metric='precomputed',
                                    cluster_selection_epsilon=args.threshold)
    elif args.model == 'optics':
        return OPTICS(max_eps=args.threshold, min_samples=2, metric='cosine')
    else:
        raise(f"The clustering model '{args.model}' is not supported")
