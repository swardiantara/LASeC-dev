import os
import torch
import json
import datetime
import random
import argparse
import numpy as np
import pandas as pd
from src.utils import get_features, get_pred_df, evaluation_score, get_model, save_results, compute_distance_matrix


parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='grid-search',
                    help="Folder to store the experimental results. Default: experiments")
parser.add_argument('--dataset_type', type=str, default='sample',
                    help="Dataset to run, between sample or full. Default: sample")
parser.add_argument('--dataset', choices=['Apache', 'Drone', 'DroneOvs', 'Android', 'BGL', 'Hadoop', 'HDFS', 'HealthApp', 'HPC', 'Linux', 'Mac', 'OpenSSH', 'OpenStack', 'Proxifier', 'Spark', 'Thunderbird', 'Windows', 'Zookeeper', 'MultiSource', 'MultiUnique'], default='Drone',
                    help="Dataset to test. Default: drone")
parser.add_argument('--model', choices=['agglomerative', 'birch', 'dbscan', 'hdbscan', 'optics'], default='agglomerative', help="Embedding model to extract the log's feature. Default: sbert")
parser.add_argument('--linkage', choices=['ward', 'single', 'average', 'complete'], default='average', help="Linkage criteria for Agglomerative clustering. Default: `average`")
parser.add_argument('--embedding', default='sbert',
                    help="Embedding model to extract the log's feature. Default: sbert")
parser.add_argument('--threshold', type=float,
                    help="Distance threshold for intra-cluster criteria [0.01,0.2]. Default: 0.07")
parser.add_argument('--sample_order', type=str, default='asc',
                    help="How to sort the sample based on the message alphabetically. Default: `asc (A-Z)`")
parser.add_argument('--sample_size', type=int, default=2000,
                    help="Sample size for efficiency test. Default: 2000")
parser.add_argument('--held_out', action='store_true',
                    help="Whether to run the abstraction on the held-out dataset.")
parser.add_argument('--overwrite', action='store_true',
                    help="Whether to overwrite existing results.")
parser.add_argument('--normalize_embedding', action='store_true',
                    help="Wether to normalize the embedding vectors")
parser.add_argument('--seed', type=int, default=42,
                    help="Random seed for reproducibility. Default: `42`")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# swardiantara/MultiSource-full-cdk0-m0.5-e5-b128-L6
# only when using fine-tuned embedding model
def get_heldout_sample(model_path: str, source_name: str, source_sample: pd.DataFrame) -> pd.DataFrame:
    initial_model = 'all-MiniLM-L6-v2' if model_path.split('-')[-1] == 'L6' else 'all-MiniLM-L12-v2'
    dataset = model_path.split("-cdk")[0]  # MultiSource-full
    sampling_strategy = 'random' if model_path.split('-')[1][1] == 'r' else 'distance'
    num_sample = model_path.split('-')[2][2:]
    held_out_dir = os.path.join('dataset_heldout', source_name, f"{sampling_strategy}-{num_sample}")
    os.makedirs(held_out_dir, exist_ok=True)
    held_out_file = os.path.join(held_out_dir, f'heldout_{sampling_strategy}_{num_sample}.xlsx')

    if os.path.exists(held_out_file):
        print(f"Held out dataset is found in {held_out_file}")
        test_sample = pd.read_excel(held_out_file)
    else:
        # embeddings/MultiSource-full/all-MiniLM-L6-v2/random-k1/random_k1_selected_sample.xlsx
        training_file_path = os.path.join('embeddings', dataset, initial_model, f"{sampling_strategy}-{num_sample}", f'{sampling_strategy}_{num_sample}_selected_sample.xlsx')
        if not os.path.exists(training_file_path) or num_sample == 0:
            print('The training samples file path is not found or k=0!')
            return source_sample
        
        train_sample = pd.read_excel(training_file_path).reset_index(drop=True)
        train_source = train_sample[train_sample['Source'] == source_name] if not source_name.startswith('Multi') else train_sample
        test_sample = source_sample.copy()
        for content in train_source['Content'].to_list():
            index_to_remove = test_sample[test_sample['message'] == content].index
            test_sample.drop(index_to_remove, inplace=True)
        test_sample.to_excel(held_out_file, index=False)
    return test_sample

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    args = parser.parse_args()
    set_seed(args.seed)
    
    train_started_at = datetime.datetime.now()
    if args.dataset_type == 'sample':
        if args.dataset == 'Drone':
            dataset = pd.read_csv(os.path.join('dataset_corrected', 'Drone_584.log_structured.csv')).sort_values(by='Content', ascending=args.sample_order).reset_index(drop=True)
            dataset.rename(columns = {'Content': 'message', 'EventId': 'cluster_id'}, inplace = True)
            labels_true = dataset['cluster_id'].to_list()
        elif args.dataset == 'DroneOvs':
            dataset = pd.read_csv(os.path.join('dataset_corrected', 'DroneOvs_815.log_structured.csv')).sort_values(by='Content', ascending=args.sample_order).reset_index(drop=True)
            dataset.rename(columns = {'Content': 'message', 'EventId': 'cluster_id'}, inplace = True)
            labels_true = dataset['cluster_id'].to_list()
        elif str(args.dataset).startswith('Multi'):
            dataset = pd.read_csv(os.path.join('dataset_corrected', f'{args.dataset}_2k.log_structured.csv')).sort_values(by='Content', ascending=args.sample_order).reset_index(drop=True)
            dataset.rename(columns = {'Content': 'message'}, inplace = True)
            dataset['cluster_id'] = dataset['Source'] + "-" + dataset['EventId']
        elif args.sample_size == 2000:
            dataset = pd.read_csv(os.path.join('dataset_corrected', f'{args.dataset}_2k.log_structured.csv')).sort_values(by='Content', ascending=args.sample_order).reset_index(drop=True)
            dataset.rename(columns = {'Content': 'message', 'EventId': 'cluster_id'}, inplace = True)
            labels_true = dataset['cluster_id'].to_list()
        elif args.sample_size > 2000:
            dataset = pd.read_csv(os.path.join('dataset_efficiency', f'{args.dataset}_{args.sample_size}.log_structured.csv')).sort_values(by='Content', ascending=args.sample_order).reset_index(drop=True)
            dataset.rename(columns = {'Content': 'message', 'EventId': 'cluster_id'}, inplace = True)
    elif args.dataset_type == 'full' and args.sample_size != 2000: # efficiency test first time
        out_dir = os.path.join('dataset_efficiency', f'{args.dataset}-{str(args.sample_size)}')
        os.makedirs(out_dir, exist_ok=True)
        dataset = pd.read_csv(os.path.join(out_dir, f'{args.dataset}_full.log_structured.csv')).sort_values(by='Content', ascending=args.sample_order).reset_index(drop=True).sample(args.sample_size, random_state=args.seed)
        dataset.to_csv(os.path.join(out_dir, f'{args.dataset}_{str(args.sample_size)}.log_structured.csv'), index=False)
        with open(os.path.join(out_dir, f'{args.dataset}_{str(args.sample_size)}.log'), 'w') as f:
            f.write('\n'.join(dataset['Content'].to_list()))
        dataset.drop_duplicates('EventId')[['EventId', 'EventTemplate']].to_csv(os.path.join(out_dir, f'{args.dataset}_{str(args.sample_size)}.log_templates.csv'), index=False)
        dataset.rename(columns = {'Content': 'message', 'EventId': 'cluster_id'}, inplace = True)

    
    if args.held_out:
        dataset = get_heldout_sample(args.embedding, args.dataset, dataset).sort_values(by='message', ascending=args.sample_order).reset_index(drop=True)
    
    if args.held_out:
        dataset_scenario = f"HO-{args.dataset}-{str(args.sample_size)}" if args.sample_size > 2000 else f"HO-{args.dataset}"
    else:
        dataset_scenario = f"{args.dataset}-{str(args.sample_size)}" if args.sample_size > 2000 else args.dataset

    model_scenario = args.model
    if args.model == 'agglomerative':
        model_scenario = f"{args.model}-{args.linkage}"
    
    workdir = os.path.join('experiments', args.output_dir, dataset_scenario, model_scenario, args.embedding, str(args.threshold))

    print(f"[{args.model}] - Current scenario: {workdir}")
    if not args.overwrite and os.path.exists(workdir):
        out_file = os.path.join(workdir, 'scenario_arguments.json')
        if os.path.exists(out_file):
            with open(out_file, 'r') as file:
                load_file = json.load(file)
                if 'eval_score' in load_file:
                    print(f"[{args.model}] - Scenario has been executed.")
                    return 0
    else:
        os.makedirs(workdir, exist_ok=True)

    corpus_embeddings = get_features(dataset, args.embedding, device, args.normalize_embedding)
    clustering_model = get_model(args)

    if args.model == 'agglomerative':
        distance_matrix = compute_distance_matrix(corpus_embeddings)
        clustering_model.fit(distance_matrix)
    elif args.model == 'hdbscan':
        distance_matrix = compute_distance_matrix(corpus_embeddings)
        clustering_model.fit(distance_matrix.astype(np.float64))
    else:
        clustering_model.fit(corpus_embeddings)
    train_end_at = datetime.datetime.now()

    arguments_dict = vars(args)
    duration = train_end_at - train_started_at
    arguments_dict['train_started_at'] = str(train_started_at)
    arguments_dict['train_end_at'] = str(train_end_at)
    arguments_dict['duration'] = str(round(duration.total_seconds() * 1000, 5)) + ' miliseconds'
    pred_df = pd.DataFrame()

    eval_start_at = datetime.datetime.now()
    pred_df = get_pred_df(clustering_model.labels_, dataset, args.sample_order)
    eval_score = evaluation_score(dataset, pred_df, args.sample_order)
    eval_end_at = datetime.datetime.now()
    eval_duration = eval_end_at - eval_start_at
    arguments_dict['eval_start_at'] = str(eval_start_at)
    arguments_dict['eval_end_at'] = str(eval_end_at)
    arguments_dict['eval_duration'] = str(round(eval_duration.total_seconds() * 1000, 5)) + ' miliseconds'
    arguments_dict['eval_score'] = eval_score
    arguments_dict['sample_size'] = len(dataset)
            
    save_results(arguments_dict, pred_df, workdir)
    
    return 0


if __name__ == "__main__":
    main()
