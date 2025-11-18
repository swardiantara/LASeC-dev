import time
import os
import json
import argparse

import pandas as pd
from datetime import datetime
from src.utils_embedding import (
    split_by_template,
    set_seed,
)
from src.embedding import LogEmbedding

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='embeddings',
                    help="Folder to store the fine-tuned embedding models. Default: `embeddings`")
parser.add_argument('--initial_model_path', default='all-MiniLM-L6-v2',
                    help="Initial model_name_or_path. Default: `all-MiniLM-L6-v2`")
parser.add_argument('--sampling_strategy', choices=['random', 'distance'], default='random', 
                    help="Sampling strategy to select `k` positive samples. Default: `random`")
parser.add_argument('--dataset', default='MultiSource',
                    help="Dataset to use for fine-tuning. Default=`Drone`")
parser.add_argument('--k', type=int, default=10,
                    help="Number of samples to be paired with their template to construct positive pairs. Default: `10`")
parser.add_argument('--template_portion', type=str, choices=['full', 'partial'], default='full',
                    help="The portion of unique templates to use for fine-tuning. Default: `full` (use all templates)")
parser.add_argument('--margin', type=float, default=0.5,
                    help="Margin for the first stage's negative pairs. Default: `0.5`")
parser.add_argument('--batch_size', type=int, default=128,
                    help="Batch size for the first stage. Default: `128`")
parser.add_argument('--epoch', type=int, default=5,
                    help="Training iterations for the first stage. Default: `5`")
parser.add_argument('--seed', type=int, default=42,
                    help="Random seed for reproducibility. Default: `42`")
parser.add_argument('--push_embedding', action='store_true', help="Whether to push the fine-tuned embeddings to Huggingface.")


def main():
    args = parser.parse_args()
    set_seed(args.seed)
    
    if args.dataset == 'Drone':
        filename = "Drone_584.log_structured.csv"
    else:
        filename = f"{args.dataset}_2k.log_structured.csv"
    
    dataset = pd.read_csv(os.path.join('dataset', filename))
    dataset.drop_duplicates(subset=['Content', 'EventTemplate'], inplace=True)

    # Initialize and run fine-tuning
    args_dict = vars(args)
    k = args_dict['k']
    sampling_strategy = args_dict['sampling_strategy']
    initial_model = args_dict['initial_model_path']
    output_path = os.path.join(args.output_dir, f'{args.dataset}-{args.template_portion}', initial_model, f"{sampling_strategy}-k{k}")
    precomputed_dir = os.path.join(args_dict['output_dir'], f'{args.dataset}-{args.template_portion}', initial_model)
    args_dict['precomputed_dir'] = precomputed_dir
    
    os.makedirs(output_path, exist_ok=True)
    
    # check for the dataset portion
    if args.template_portion != 'full':
        train_path = os.path.join(args_dict['output_dir'], f'{args.dataset}-{args.template_portion}', 'train_dataset.csv')
        test_path = os.path.join(args_dict['output_dir'], f'{args.dataset}-{args.template_portion}', 'test_dataset.csv')
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"Loading pre-split dataset from {train_path} and {test_path}")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            # Verify no template leakage
            train_templates = set(train_data['EventId'].unique())
            test_templates = set(test_data['EventId'].unique())
            overlap = train_templates.intersection(test_templates)
            
            print(f"\n=== Verification ===")
            print(f"Template overlap between train and test: {len(overlap)}")
            if len(overlap) == 0:
                print("No template leakage - split is valid!")
                dataset = train_data  # use only training data for fine-tuning
            else:
                print(f"!WARNING: {len(overlap)} templates appear in both sets!")
        else:
            # Split by templates (80% train, 20% test)
            train_data, test_data = split_by_template(
                dataset, 
                test_size=0.2, 
                random_state=42,
                output_train=train_path,
                output_test=test_path,
            )
            
            # Verify no template leakage
            train_templates = set(train_data['EventId'].unique())
            test_templates = set(test_data['EventId'].unique())
            overlap = train_templates.intersection(test_templates)
            
            print(f"\n=== Verification ===")
            print(f"Template overlap between train and test: {len(overlap)}")
            if len(overlap) == 0:
                print("No template leakage - split is valid!")
            else:
                print(f"!WARNING: {len(overlap)} templates appear in both sets!")
            dataset = train_data  # use only training data for fine-tuning

    # return 0
    start_time = time.time()
    log_embedding = LogEmbedding(args_dict, dataset, output_path)
    log_embedding.fine_tuning()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    args_dict['start_time'] = str(datetime.fromtimestamp(start_time))
    args_dict['end_time'] = str(datetime.fromtimestamp(end_time))
    args_dict['total_time_seconds'] = elapsed_time
    args_dict['total_time_hms'] = f"{hours}h {minutes}m {seconds:.2f}s"

    with open(os.path.join(output_path, 'execution_args.json'), "w") as json_file:
        json.dump(args_dict, json_file, indent=4)

if __name__ == '__main__':
    main()