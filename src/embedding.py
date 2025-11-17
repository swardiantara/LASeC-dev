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
from huggingface_hub import HfApi, hf_hub_download, repo_exists
from sentence_transformers import SentenceTransformer, losses, InputExample

from src.utils_embedding import CustomContrastiveLoss


class GeometricConstrainedLogEmbedding:
    def __init__(self,
                 args_dict: dict,
                 dataset: pd.DataFrame,
                 sampling_strategy: str = 'random',
                 stage: str = 'both',
                 initial_model: str = 'all-MiniLM-L6-v2',
                 output_dir: str = './log_embedding_results'):
        """
        Initialize multi-stage log embedding learning
        
        :param datasets: List of log DataFrames
        :param sampling_strategy: Sampling strategy to select `k`-positive samples
        :param stage: Stage of the fine-tuning. `both`, `one`, or `two`.
        :param initial_model: Base SBERT model
        :param output_dir: Directory to save results
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.sampling_strategy = sampling_strategy
        self.stage = stage
        # Initialize model and datasets
        self.args = args_dict
        self.model_path = initial_model
        self.model = SentenceTransformer(initial_model)
        self.dataset = dataset
        self.sampling_scenario = None

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

    def construct_stage_one_pairs(self, k=10):
        """
        Construct pairs for stage 1 learning
        
        Returns:
        - positive_pairs: Pairs of templates with their messages
        - negative_pairs: Pairs of templates from different EventIDs
        - sample_sample_pairs: Pairs of samples from the same EventIDs
        """

        # check if there's pre-computed embedding for the same initial_model
        message_embedding_path = os.path.join(self.output_dir, f"Content_{self.model_path}.pkl")
        self.precompute_embeddings('Content', message_embedding_path)
        template_embedding_path = os.path.join(self.output_dir, f"EventTemplate_{self.model_path}.pkl")
        self.precompute_embeddings('EventTemplate', template_embedding_path)
        
        negative_pairs_folder = os.path.join(self.args['output_dir'], f'{self.args["dataset"]}-{self.args["template_portion"]}')
        # check if there's pre-constructed sample pairs based on the same strategy and k
        selected_sample_path = os.path.join(self.output_dir, f"selected_{self.sampling_scenario}.xlsx")
        positive_pairs_path = os.path.join(self.output_dir, f'one_{self.sampling_scenario}_positive_pairs.json')
        
        # Group by EventTemplate
        grouped = self.dataset.groupby(by='EventTemplate')
        
        if os.path.exists(positive_pairs_path):
            self.logger.info(f"Positive pairs for sampling scenario={self.sampling_scenario} is found: {positive_pairs_path}")
            with open(positive_pairs_path, 'r') as f:
                positive_pairs = json.load(f)
        elif k > 0:
            self.logger.info(f"Positive pairs for sampling scenario={self.sampling_scenario} is not found. Constructing...")
            # Positive pairs: Templates with their messages
            positive_pairs = []
            sample_sample_pairs = []
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
                    sample_messages = per_template.sample(min(k, len(per_template))) # should be reproducible after setting the global seed
                elif self.sampling_strategy == 'distance':
                    # sort by distance in descending order and take top-k
                    sample_messages = per_template.sort_values(
                        by='distance', 
                        ascending=False
                    ).head(min(k, len(per_template)))

                for idx, row in sample_messages.iterrows():
                    positive_pairs.append({
                        'template': template,
                        'message': row['Content'],
                        'cosine_distance': float(row['distance'])
                    })
                    for _, inner_row in sample_messages.iloc[idx+1:].iterrows():
                        sample_sample_pairs.append({
                            'sample1': row['Content'],
                            'sample2': inner_row['Content'],
                        })
                selected_sample = pd.concat([selected_sample, sample_messages], ignore_index=True)

            # save to file after finish sampling
            selected_sample = selected_sample.drop(['Content_embedding', 'EventTemplate_embedding'], axis='columns')
            self._log_selected_sample(selected_sample, selected_sample_path)
            self._log_pairs(positive_pairs, positive_pairs_path)
            self.logger.info(f"Finished constructing positive pairs for sampling scenario={self.sampling_scenario}")
            self.logger.info(f"Saved to {positive_pairs_path}")
            
        negative_pairs_path = os.path.join(negative_pairs_folder, f'one_negative_pairs.json')
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
            self._log_pairs(negative_pairs, negative_pairs_path)
            self.logger.info(f"Finished constructing negative pairs for sampling scenario={self.sampling_scenario}")
            self.logger.info(f"Saved to {negative_pairs_path}")
        
        return positive_pairs, negative_pairs, positive_pairs_path, negative_pairs_path, selected_sample_path, sample_sample_pairs
    
    def construct_stage_two_pairs(self, stage_one_positive_pairs):
        """
        Construct pairs for stage 2 geometric learning
        
        :param stage_one_positive_pairs: Positive pairs from stage 1
        """
        # check if the stage two pairs has been constructed for this sampling strategy
        stage_two_pairs_path = os.path.join(self.output_dir, 'stage_two_pairs.json')
        if os.path.exists(stage_two_pairs_path):
            self.logger.info(f"Stage-two pairs for sampling scenario={self.sampling_scenario} is found: {stage_two_pairs_path}")
            with open(stage_two_pairs_path, 'r') as f:
                stage_two_pairs = json.load(f)
        else:
            self.logger.info(f"Stage-two pairs for sampling scenario={self.sampling_scenario} is not found. Constructing...")
            # Organize data by EventTemplate
            grouped_by_template = {}
            for pair in stage_one_positive_pairs:
                template = pair['template']
                message = pair['message']
                
                if template not in grouped_by_template:
                    grouped_by_template[template] = []
                grouped_by_template[template].append(message)
            
            # Construct stage two pairs
            stage_two_pairs = []
            
            for template, messages in grouped_by_template.items():
                # Positive pairs: Template with its messages
                template_embedding = self.model.encode([template])[0]
                
                # Positive pairs: Template with each message
                for message in messages:
                    message_embedding = self.model.encode([message])[0]
                    
                    pos_cosine_dist = 1 - cosine_similarity(
                        [template_embedding],
                        [message_embedding]
                    )[0][0]
                    
                    stage_two_pairs.append({
                        'type': 'positive',
                        'template': template,
                        'message': message,
                        'cosine_distance': float(pos_cosine_dist)
                    })
                
                # Negative pairs: Messages within the same template
                for i, msg1 in enumerate(messages):
                    for msg2 in messages[i+1:]:
                        msg1_embedding = self.model.encode([msg1], show_progress_bar=False)[0]
                        msg2_embedding = self.model.encode([msg2], show_progress_bar=False)[0]
                        
                        neg_cosine_dist = 1 - cosine_similarity(
                            [msg1_embedding],
                            [msg2_embedding]
                        )[0][0]
                        
                        stage_two_pairs.append({
                            'type': 'negative',
                            'template': template,
                            'message1': msg1,
                            'message2': msg2,
                            'cosine_distance': float(neg_cosine_dist)
                        })
        
            # Log stage two pairs
            self._log_pairs(stage_two_pairs, stage_two_pairs_path)
            self.logger.info(f"Finished constructing stage-two pairs for sampling scenario={self.sampling_scenario}")
            self.logger.info(f"Saved to {stage_two_pairs_path}")
        
        return stage_two_pairs, stage_two_pairs_path
    
    def _log_selected_sample(self,
                             selected_sample: pd.DataFrame,
                             file_name: str):
        """
        Log selected sample based on the strategy
        
        :param file_name: The path of the file
        """
        selected_sample.to_excel(file_name)
    
    def _load_log_pairs(self,
                    stage: str = 'one',
                    sampling_scenario: str = 'random_10',
                    sample_side: str = 'positive'):
        """
        Load the log pairs information to JSON files
        
        :param stage: Current learning stage
        :param sampling_scenario: Scenario of sampling: `random-10`, `distance-10`
        :param sample_side: Positive or negative
        """
        with open(os.path.join(self.output_dir, f'{stage}_{sampling_scenario}_{sample_side}_pairs.json'), 'r') as f:
            return json.load(f)

    def _log_pairs(self,
                    sample_pairs: list,
                    filename: str):
        """
        Log pair information to JSON file

        :param sample_pairs: List of sample pairs to store
        :param filename: filename to store the log pairs
        """
        with open(filename, 'w') as f:
            json.dump(sample_pairs, f, indent=2)
    
    def multi_stage_fine_tuning(self,
                                k: int = 10,
                                stage_one_margin: float = 0.5,
                                stage_two_margin: float = 0.05,
                                stage_one_epochs: int = 5,
                                stage_two_epochs: int = 2,
                                stage_one_batch: int = 128,
                                stage_two_batch: int = 128):
        """
        Multi-stage fine-tuning process
        
        :param k: Number of positive samples to pair with log template
        :param stage_one_margin: Margin for template separation
        :param stage_two_margin: Margin for geometric constraint
        :param stage_one_epochs: Stage one training epochs
        :param stage_two_epochs: Stage two training epochs
        :param stage_one_batch: Stage one training batch size
        :param stage_two_batch: Stage two training batch size
        """
        sampling_scenario = f"{self.sampling_strategy}_{k}"
        self.sampling_scenario = sampling_scenario
        if self.stage in ['both', 'one']:
            start_time = time.time()
            # Stage 1: Template Semantic Separation
            positive_pairs, negative_pairs, positive_pairs_path, negative_pairs_path, selected_sample_path, sample_sample_pairs = self.construct_stage_one_pairs(k)
            
            # Prepare training examples for stage 1
            stage_one_examples = []
            
            # Positive pairs (template-message pairs)
            for pair in positive_pairs:
                stage_one_examples.append(
                    InputExample(
                        texts=[pair['template'], pair['message']], 
                        label=0.0
                    )
                )
            
            # Negative pairs (different template pairs)
            for pair in negative_pairs:
                stage_one_examples.append(
                    InputExample(
                        texts=[pair['template1'], pair['template2']], 
                        label=stage_one_margin
                    )
                )

            # Negative pairs (different sample pairs)
            for pair in sample_sample_pairs:
                stage_one_examples.append(
                    InputExample(
                        texts=[pair['sample1'], pair['sample2']], 
                        label=stage_two_margin
                    )
                )
            
            # Stage 1 Training
            train_dataloader = DataLoader(
                stage_one_examples, 
                shuffle=True, 
                batch_size=stage_one_batch
            )
            
            stage_one_loss = CustomContrastiveLoss(
                model=self.model,
                margin=stage_one_margin
            )

            self.logger.info("Starting Stage 1 Fine-Tuning")
            out_model_dir = f"m{stage_one_margin}-e{stage_one_epochs}-b{stage_one_batch}"
            output_path=os.path.join(self.output_dir, f'stage-one-{out_model_dir}')
            os.makedirs(output_path, exist_ok=True)
            self.model.fit(
                train_objectives=[(train_dataloader, stage_one_loss)],
                epochs=stage_one_epochs,
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
                repo_name = f'{self.args['dataset']}-one-c{self.sampling_strategy[0]}k{k}-{out_model_dir}-{self.model_path.split('-')[2]}'
                api.create_repo(repo_id=f'swardiantara/{repo_name}', exist_ok=True, repo_type="model")

                # copy selected samples and sample pairs to the folder for documentation
                files_to_move = [selected_sample_path, positive_pairs_path, negative_pairs_path]
                # copy each file to the model directory
                for file_name in files_to_move:
                    shutil.copy(file_name, os.path.join(output_path, file_name.split('/')[-1]))
                
                api.upload_folder(
                    folder_path=output_path,
                    repo_id=f"swardiantara/{repo_name}",
                    repo_type="model",
                )

        if self.stage in ['two', 'both']:
            if self.stage == 'two':
                # check if the first stage is on Huggingface
                api = HfApi(token=os.getenv("HF_TOKEN"))
                model_scenario = f"m{stage_one_margin}-e{stage_one_epochs}-b{stage_one_batch}"
                out_model_dir = os.path.join(self.output_dir, f'stage-one-{model_scenario}')
                repo_name = f'one-c{self.sampling_strategy[0]}k{k}-{model_scenario}-{self.model_path.split('-')[2]}'
                if api.repo_exists(f'swardiantara/{repo_name}'):
                    self.model = SentenceTransformer(f'swardiantara/{repo_name}')
                    local_path = hf_hub_download(repo_id=f'swardiantara/{repo_name}', filename=f'one_{sampling_scenario}_positive_pairs.json', repo_type="model")
                    with open(local_path, 'r') as f:
                        positive_pairs = json.load(f) # load from pre-constructed
                    selected_sample_path = hf_hub_download(repo_id=f'swardiantara/{repo_name}', filename=f'selected_{sampling_scenario}.xlsx', repo_type="model")
                # or in the local directory
                elif os.path.exists(out_model_dir):
                    self.model = SentenceTransformer(out_model_dir) # load from the pre-trained on the first stage
                    positive_pairs_path = os.path.join(self.output_dir, f'one_{sampling_scenario}_positive_pairs.json') # load from pre-constructed
                    selected_sample_path = os.path.join(self.output_dir, f'selected_{sampling_scenario}.xlsx') # load from pre-constructed
                    with open(positive_pairs_path, 'r') as f:
                        positive_pairs = json.load(f)
                else:
                    raise FileNotFoundError('The first stage model is not found')
            
            # stage 2: Geometric Constraint Learning
            start_time = time.time()
            stage_two_pairs, stage_two_pairs_path = self.construct_stage_two_pairs(positive_pairs)
            
            stage_two_examples = []
            
            # positive pairs (template-message)
            for pair in stage_two_pairs:
                if pair['type'] == 'positive':
                    stage_two_examples.append(
                        InputExample(
                            texts=[pair['template'], pair['message']], 
                            label=0.0
                        )
                    )
            
            # negative pairs (messages within same template)
            for pair in stage_two_pairs:
                if pair['type'] == 'negative':
                    stage_two_examples.append(
                        InputExample(
                            texts=[pair['message1'], pair['message2']], 
                            label=stage_two_margin
                        )
                    )
            
            # stage 2 Training
            train_dataloader = DataLoader(
                stage_two_examples, 
                shuffle=True, 
                batch_size=stage_two_batch
            )
            
            stage_two_loss = CustomContrastiveLoss(
                model=self.model, 
                margin=stage_two_margin
            )
            
            self.logger.info("Starting Stage 2 Fine-Tuning")
            out_model_dir = f"m{stage_two_margin}-e{stage_two_epochs}-b{stage_two_batch}"
            output_path=os.path.join(self.output_dir, f'stage-two-{out_model_dir}')
            os.makedirs(output_path, exist_ok=True)
            self.model.fit(
                train_objectives=[(train_dataloader, stage_two_loss)],
                epochs=stage_two_epochs,
                output_path=output_path
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = elapsed_time % 60
            self.args['two_start_time'] = str(datetime.fromtimestamp(start_time))
            self.args['two_end_time'] = str(datetime.fromtimestamp(end_time))
            self.args['two_total_time_seconds'] = elapsed_time
            self.args['two_total_time_hms'] = f"{hours}h {minutes}m {seconds:.2f}s"
            
            with open(os.path.join(output_path, 'running_time.json'), "w") as json_file:
                json.dump(self.args, json_file, indent=4)

            if self.args['push_embedding']:
                # push model to Huggingface
                api = HfApi(token=os.getenv("HF_TOKEN"))
                repo_name = f'two-c{self.sampling_strategy[0]}k{k}-{out_model_dir}-{self.model_path.split('-')[2]}'
                api.create_repo(repo_id=f'swardiantara/{repo_name}', exist_ok=True, repo_type="model")

                # move selected samples and sample pairs to the folder for documentation
                files_to_move = [selected_sample_path, stage_two_pairs_path]
                for file_name in files_to_move:
                    shutil.copy(file_name, os.path.join(output_path, file_name.split('/')[-1]))
                
                api.upload_folder(
                    folder_path=output_path,
                    repo_id=f"swardiantara/{repo_name}",
                    repo_type="model",
                )

        self.logger.info("Multi-stage Fine-Tuning Complete")

