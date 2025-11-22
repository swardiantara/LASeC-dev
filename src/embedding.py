import torch
import time
import math
import os
import gc
import shutil
import json
import logging
import pickle

import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer, losses, InputExample

class LogEmbedding:
    def __init__(self,
                 args_dict: dict,
                 dataset: pd.DataFrame,
                 output_path: str = './embeddings'):
        """
        Initialize LogEmbedding learning
        :param args_dict: Dictionary of arguments
        :param datasets: List of log DataFrames
        :param output_path: Directory to save results
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.args = args_dict
        self.output_path = output_path
        self.sampling_strategy = args_dict['sampling_strategy']
        # Initialize model and datasets
        self.model_path = args_dict['initial_model_path']
        self.model = SentenceTransformer(args_dict['initial_model_path'])
        self.dataset = dataset
        self.sampling_scenario = f"{self.args['sampling_strategy']}_k{self.args['k']}"

    def _reload_model(self):
        self.model = SentenceTransformer(self.model_path)
        
    def precompute_embeddings(self, column, embedding_path):
        """Compute embeddings for all log messages"""
        if os.path.exists(embedding_path):
            # embedding has been computed, load from file
            self.logger.info(f"{column}s' embedding for {self.model_path} is found!")
            with open(embedding_path, 'rb') as f:
                self.dataset[f'{column}_embedding'] = pickle.load(f)
            self.logger.info(f"Loaded from {embedding_path}.")
        else:
            # embedding has not computed, compute now
            self.logger.info(f"{column}s' embedding for {self.model_path} has not computed. Computing...")
            embedding_vectors = list(
                self.model.encode(self.dataset[column].tolist(), show_progress_bar=False, normalize_embeddings=True)
            )
            self.dataset[f'{column}_embedding'] = embedding_vectors
            # save after finish computing
            self.logger.info(f"Saving the computed {column}s' embedding for {self.model_path} to {embedding_path}")
            with open(embedding_path, 'wb') as f:
                pickle.dump(embedding_vectors, f)
            
            # Explicitly clear CUDA cache to prevent OOM
            torch.cuda.empty_cache()
            gc.collect()
        self.model.to('cpu')
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        self._reload_model()

    def construct_pairs(self):
        """
        Construct pairs for stage 1 learning
        
        Returns:
        - positive_pairs: Pairs of templates with their messages
        - negative_pairs: Pairs of templates from different EventIDs
        """

        # check if there's pre-computed embedding for the same initial_model
        message_embedding_path = os.path.join(self.args['precomputed_dir'], f"Content_{self.model_path}.pkl")
        self.precompute_embeddings('Content', message_embedding_path)
        template_embedding_path = os.path.join(self.args['precomputed_dir'], f"EventTemplate_{self.model_path}.pkl")
        self.precompute_embeddings('EventTemplate', template_embedding_path)
        
        negative_pairs_folder = os.path.join(self.args['output_dir'], f'{self.args["dataset"]}-{self.args["template_portion"]}')
        # check if there's pre-constructed sample pairs based on the same strategy and k
        selected_sample_path = os.path.join(self.output_path, f"{self.sampling_scenario}_selected_sample.xlsx")
        positive_pairs_path = os.path.join(self.output_path, f'{self.sampling_scenario}_positive_pairs.json')
        
        # print(f"Debug columns: {self.dataset.columns.tolist()}")
        # group by EventTemplate
        grouped = self.dataset.groupby(by='EventTemplate')
        
        positive_pairs = []
        if os.path.exists(positive_pairs_path):
            self.logger.info(f"Positive pairs for sampling scenario={self.sampling_scenario} is found: {positive_pairs_path}")
            with open(positive_pairs_path, 'r') as f:
                positive_pairs = json.load(f)
        elif self.args['k'] > 0:
            self.logger.info(f"Positive pairs for sampling scenario={self.sampling_scenario} is not found. Constructing...")
            # Positive pairs: Templates with their messages
            selected_sample = pd.DataFrame()
            for template, group in grouped:
                # Get template embedding
                template_embedding = self.model.encode([template])[0]
                per_template = group.drop_duplicates(subset=['Content'])
                per_template['distance'] = float(0)
               
                # Compute the distance of unique samples to their template
                for idx, sample in per_template.iterrows():
                    # Compute cosine distance
                    cosine_dist = 1 - cosine_similarity(
                        [template_embedding], 
                        [sample['Content_embedding']]
                    )[0][0]
                    per_template.at[idx, 'distance'] = cosine_dist

                # sample k messages for this template
                if self.sampling_strategy == 'random':
                    sample_messages = per_template.sample(min(self.args['k'], len(per_template))) # should be reproducible after setting the global seed
                elif self.sampling_strategy == 'distance':
                    # sort by distance in descending order and take top-k
                    sample_messages = per_template.sort_values(
                        by='distance', 
                        ascending=False
                    ).head(min(self.args['k'], len(per_template)))

                for idx, row in sample_messages.iterrows():
                    positive_pairs.append({
                        'template': template,
                        'message': row['Content'],
                        'cosine_distance': float(row['distance'])
                    })
                selected_sample = pd.concat([selected_sample, sample_messages], ignore_index=True)

            # save to file after finish sampling
            print(f"Debug columns before saving: {selected_sample.columns.tolist()}")
            selected_sample = selected_sample.drop(['Content_embedding', 'EventTemplate_embedding'], axis='columns')
            selected_sample.to_excel(selected_sample_path)
            self._save_log_pairs(positive_pairs, positive_pairs_path)
            self.logger.info(f"Finished constructing positive pairs for sampling scenario={self.sampling_scenario}")
            self.logger.info(f"Saved to {positive_pairs_path}")
            
        negative_pairs_path = os.path.join(negative_pairs_folder, f'negative_pairs.json')
        if os.path.exists(negative_pairs_path):
            self.logger.info(f"Negative pairs for sampling scenario={self.sampling_scenario} is found: {negative_pairs_path}")
            with open(negative_pairs_path, 'r') as f:
                negative_pairs = json.load(f)
        else:
            self.logger.info(f"Negative pairs for sampling scenario={self.sampling_scenario} is not found.")
            negative_pairs = []
            # Negative pairs: Templates from different EventIDs
            template_df = self.dataset.drop_duplicates(subset='EventTemplate')
            total_combinations = math.comb(len(template_df), 2)
            pbar = tqdm(total=total_combinations, desc="Constructing negative pairs...")
            for i in range(0, len(template_df)):
                for j in range(i+1, len(template_df)):
                    # Compute template embedding distances
                    template_dist = 1 - cosine_similarity(
                        [template_df.iloc[i]['EventTemplate_embedding']], 
                        [template_df.iloc[j]['EventTemplate_embedding']]
                    )[0][0]
                    
                    negative_pairs.append({
                        'template1': template_df.iloc[i]['EventTemplate'],
                        'template2': template_df.iloc[j]['EventTemplate'],
                        'template_distance': float(template_dist)
                    })
                    pbar.update(1)
            pbar.close()
            self._save_log_pairs(negative_pairs, negative_pairs_path)
            self.logger.info(f"Finished constructing negative pairs for sampling scenario={self.sampling_scenario}")
            self.logger.info(f"Saved to {negative_pairs_path}")
        
        return positive_pairs, negative_pairs, positive_pairs_path, negative_pairs_path, selected_sample_path
    

    def _save_log_pairs(self,
                    sample_pairs: list,
                    filename: str):
        """
        Save pair information to JSON file

        :param sample_pairs: List of sample pairs to store
        :param filename: filename to store the log pairs
        """
        with open(filename, 'w') as f:
            json.dump(sample_pairs, f, indent=2)
    
    def fine_tuning(self):
        """
        Contrastive fine-tuning process
        
        """
        start_time = time.time()
        # construct pairs
        positive_pairs, negative_pairs, positive_pairs_path, negative_pairs_path, selected_sample_path = self.construct_pairs()
        
        # prepare training samples
        stage_one_examples = []
        
        # positive pairs (template-message pairs)
        for pair in positive_pairs:
            stage_one_examples.append(
                InputExample(
                    texts=[pair['template'], pair['message']], 
                    label=1
                )
            )
        
        # negative pairs (different template pairs)
        for pair in negative_pairs:
            stage_one_examples.append(
                InputExample(
                    texts=[pair['template1'], pair['template2']], 
                    label=0
                )
            )
        
        # training setup
        train_dataloader = DataLoader(
            stage_one_examples, 
            shuffle=True, 
            batch_size=self.args['batch_size']
        )
        
        # binary contrastive loss for pair-based learning
        loss_fc = losses.ContrastiveLoss(
            model=self.model,
            margin=self.args['margin']
        )

        self.logger.info("Starting fine-tuning...")
        out_model_dir = f"m{self.args['margin']}-e{self.args['epoch']}-b{self.args['batch_size']}"
        output_path=os.path.join(self.output_path, f'{out_model_dir}')
        os.makedirs(output_path, exist_ok=True)
        self.model.fit(
            train_objectives=[(train_dataloader, loss_fc)],
            epochs=self.args['epoch'],
            output_path=output_path
        )


        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        self.args['one_start_time'] = str(datetime.fromtimestamp(start_time))
        self.args['one_end_time'] = str(datetime.fromtimestamp(end_time))
        self.args['one_total_time_seconds'] = elapsed_time
        self.args['one_total_time_hms'] = f"{hours}h {minutes}m {seconds:.2f}s"

        with open(os.path.join(output_path, 'running_time.json'), "w") as json_file:
            json.dump(self.args, json_file, indent=4)

        if self.args['push_embedding']:
            # push model to Huggingface
            api = HfApi(token=os.getenv("HF_TOKEN"))
            repo_name = f'{self.args['dataset']}-{self.args['template_portion']}-c{self.args['sampling_strategy'][0]}k{self.args['k']}-{out_model_dir}-{self.model_path.split('-')[-2]}'
            api.create_repo(repo_id=f'swardiantara/{repo_name}', exist_ok=True, repo_type="model")

            # copy selected samples and sample pairs to the folder for documentation
            files_to_move = [selected_sample_path, positive_pairs_path, negative_pairs_path] if self.args['k'] > 0 else [negative_pairs_path]
            # copy each file to the model directory
            for file_name in files_to_move:
                shutil.copy(file_name, os.path.join(output_path, file_name.split('/')[-1]))
            
            api.upload_folder(
                folder_path=output_path,
                repo_id=f"swardiantara/{repo_name}",
                repo_type="model",
            )

        self.logger.info("Fine-Tuning Complete")

