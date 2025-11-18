import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Iterable, Dict
from torch import Tensor
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


# taken from sentence-transformers
class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

    Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two embeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.

    Example::

        from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
        from torch.utils.data import DataLoader

        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.ContrastiveLoss(model=model)

        model.fit([(train_dataloader, train_loss)], show_progress_bar=True)

    """

    def __init__(
        self,
        model: SentenceTransformer,
        distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
        margin: float = 0.5,
        size_average: bool = True,
    ):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.model = model
        self.size_average = size_average

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(SiameseDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "SiameseDistanceMetric.{}".format(name)
                break

        return {"distance_metric": distance_metric_name, "margin": self.margin, "size_average": self.size_average}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        assert len(reps) == 2
        rep_anchor, rep_other = reps
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (
            labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2)
        )
        return losses.mean() if self.size_average else losses.sum()


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across numpy, random, torch, and cuda
    
    :param seed: Random seed value (default: 42)
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def split_by_template(merged_dataset, test_size=0.2, random_state=42, 
                      output_train='train_dataset.csv', 
                      output_test='test_dataset.csv'):
    """
    Split merged log dataset by unique EventTemplates into train and test sets.
    
    Args:
        merged_dataset (pd.DataFrame): Merged dataset with columns 
                                       [Source, Content, EventId, EventTemplate]
        test_size (float): Proportion of templates for test set (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        output_train (str): Output filename for training set
        output_test (str): Output filename for test set
        
    Returns:
        tuple: (train_df, test_df) - Training and test dataframes
    """
    
    print("=== Splitting Dataset by Unique Templates ===\n")
    
    # Get unique templates (EventIds)
    unique_templates = merged_dataset['EventTemplate'].unique()
    print(f"Total unique templates (EventIds): {len(unique_templates)}")
    
    # Split templates into train and test
    train_templates, test_templates = train_test_split(
        unique_templates, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Train templates: {len(train_templates)} ({len(train_templates)/len(unique_templates)*100:.1f}%)")
    print(f"Test templates: {len(test_templates)} ({len(test_templates)/len(unique_templates)*100:.1f}%)")
    
    # Split logs based on template membership
    train_df = merged_dataset[merged_dataset['EventId'].isin(train_templates)].copy()
    test_df = merged_dataset[merged_dataset['EventId'].isin(test_templates)].copy()
    
    # Reset indices
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    print(f"\n=== Split Statistics ===")
    print(f"Training logs: {len(train_df)} ({len(train_df)/len(merged_dataset)*100:.1f}%)")
    print(f"Test logs: {len(test_df)} ({len(test_df)/len(merged_dataset)*100:.1f}%)")
    
    # Show template statistics
    print(f"\n=== Training Set - Template Statistics ===")
    train_template_counts = train_df['EventId'].value_counts()
    print(f"Min logs per template: {train_template_counts.min()}")
    print(f"Max logs per template: {train_template_counts.max()}")
    print(f"Mean logs per template: {train_template_counts.mean():.2f}")
    print(f"Median logs per template: {train_template_counts.median():.1f}")
    
    print(f"\n=== Test Set - Template Statistics ===")
    test_template_counts = test_df['EventId'].value_counts()
    print(f"Min logs per template: {test_template_counts.min()}")
    print(f"Max logs per template: {test_template_counts.max()}")
    print(f"Mean logs per template: {test_template_counts.mean():.2f}")
    print(f"Median logs per template: {test_template_counts.median():.1f}")
    
    # Save to CSV
    if output_train:
        train_df.to_csv(output_train, index=False)
        print(f"\nTraining set saved to: {output_train}")
    
    if output_test:
        test_df.to_csv(output_test, index=False)
        print(f"Test set saved to: {output_test}")
    
    return train_df, test_df
