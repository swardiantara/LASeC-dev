"""
Description : This file implements the function to evaluation accuracy of log parsing
Author      : LogPAI team
License     : MIT
"""

import sys
import pandas as pd
from collections import defaultdict
try:
    from scipy.misc import comb
except ImportError as e:
    from scipy.special import comb


def evaluate(df_groundtruth, df_parsedlog, verbose=0):
    """ Evaluation function to org_benchmark log parsing accuracy
    
    Arguments
    ---------
        df_groundtruth : pandas.DataFrame
            A pandas Dataframe of the ground truth
        df_parsedlog : pandas.DataFrame
            A pandas Dataframe of the parsing results
        verbose : int
            wether to log the scores. default=0

    Returns
    -------
        f_measure : float
        accuracy : float
    """ 
    # df_groundtruth = pd.read_excel(groundtruth)
    # df_parsedlog = pd.read_excel(parsedresult)
    
    # Remove invalid groundtruth event Ids
    null_logids = df_groundtruth[~df_groundtruth['cluster_id'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]
    
    (precision, recall, f_measure, accuracy) = get_accuracy(df_groundtruth['cluster_id'], df_parsedlog['cluster_id'])
    
    if verbose > 0:
        print('Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Grouping_Accuracy (GA): %.4f'%(precision, recall, f_measure, accuracy))
    
    return f_measure, accuracy

def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    """ Compute accuracy metrics between log parsing results and ground truth
    
    Arguments
    ---------
        series_groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        series_parsedlog : pandas.Series
            A sequence of parsed event Ids
        debug : bool, default False
            print error log messages when set to True

    Returns
    -------
        precision : float
        recall : float
        f_measure : float
        accuracy : float
    """
    # print(f'series_groundtruth: {series_groundtruth}')
    # print(f'series_parsedlog: {series_parsedlog}')
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    # print(f'series_groundtruth_valuecounts: \n{series_groundtruth_valuecounts}')
    real_pairs = 0
    # print(f'real_pairs: {real_pairs}')
    for count in series_groundtruth_valuecounts:
        if count > 1:
            # print(f'comb(count, 2): {comb(count, 2)}')
            real_pairs += comb(count, 2)
    # print(f'real_pairs: {real_pairs}')
    
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    # print(f'series_parsedlog_valuecounts: \n{series_parsedlog_valuecounts}')
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += comb(count, 2)
    # print(f'parsed_pairs: {parsed_pairs}')

    accurate_pairs = 0
    accurate_events = 0 # determine how many lines are correctly parsed

    for parsed_eventId in series_parsedlog_valuecounts.index:
        # print(f'parsed_eventId: {parsed_eventId}')
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        # print(f'logIds: {logIds}, len: {len(logIds)}')
        # print(f'series_groundtruth[logIds]: {series_groundtruth[logIds]}')
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        # print(f'series_groundtruth_logId_valuecounts: \n{series_groundtruth_logId_valuecounts}')
        
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        # print(f'error_eventIds: {error_eventIds}')
        
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs if parsed_pairs > 0 else 0
    recall = float(accurate_pairs) / real_pairs if real_pairs > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy


def singleton_accuracy(df_groundtruth, df_parsedlog):
    """ Compute accuracy metrics between log parsing results and ground truth
        for singleton clusters only
    Arguments
    ---------
        df_groundtruth : pandas.DataFrame
            A pandas Dataframe of the ground truth
        df_parsedlog : pandas.DataFrame
            A pandas Dataframe of the parsing results

    Returns
    -------
        accuracy : float
        true_singleton_indices: sorted set
        pred_singleton_indices: sorted set
    """
    null_logids = df_groundtruth[~df_groundtruth['cluster_id'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]

    series_groundtruth, series_parsedlog = df_groundtruth['cluster_id'], df_parsedlog['cluster_id']
    
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    series_parsed_valuecounts = series_parsedlog.value_counts()
    # print(f'series_groundtruth_valuecounts: {series_groundtruth_valuecounts.columns}')
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    singleton_groundtruth = series_groundtruth_valuecounts[series_groundtruth_valuecounts == 1]
    if len(singleton_groundtruth) < 1:
        return 0, 0, 0
    
    singleton_parsed = series_parsed_valuecounts[series_parsed_valuecounts == 1]
    
    # print(f'num of singleton: {len(singleton_groundtruth)}')
    # print(f'singleton_groundtruth: {singleton_groundtruth}')
    accurate_events = 0
    for golden_clusterId in singleton_groundtruth.index:
        samplesIdx = series_groundtruth[series_groundtruth == golden_clusterId].index
        # print(f'samplesIdx: {samplesIdx}')
        # print(f'golden_clusterId: {golden_clusterId}, size: {series_groundtruth_valuecounts[golden_clusterId]}')
        # print(f'sampleIdx: {samplesIdx}')
        for sampleIdx in samplesIdx:
            current_clusterId = series_parsedlog[sampleIdx]
            if series_parsedlog_valuecounts[current_clusterId] == 1:
                # print(f'pred_clusterId: {current_clusterId}, size: {series_parsedlog_valuecounts[current_clusterId]}')
                accurate_events += 1
            # else:
                # print(f'pred_clusterId: {current_clusterId}, size: {series_parsedlog_valuecounts[current_clusterId]}')
                # if current_clusterId in preds_dict:
            #     preds_dict[current_clusterId].append(sampleIdx)
            # else:
            #     preds_dict[current_clusterId] = [sampleIdx]
            # if i == 0:
            #     continue
            # else:
            #     if current_clusterId ==
    accuracy = float(accurate_events) / len(singleton_groundtruth)

    singleton_groundtruth_indices = [series_groundtruth[series_groundtruth == golden_clusterId].index[0] for golden_clusterId in singleton_groundtruth.index]
    singleton_parsed_indices = [series_parsedlog[series_parsedlog == parsed_clusterId].index[0] for parsed_clusterId in singleton_parsed.index]
    singleton_groundtruth_indices = sorted(singleton_groundtruth_indices)
    singleton_parsed_indices = sorted(singleton_parsed_indices)
    # print(f'singleton_groundtruth_indices: {singleton_groundtruth_indices}')
    # print(f'singleton_parsed_indices: {singleton_parsed_indices}')
    return accuracy, singleton_groundtruth_indices, singleton_parsed_indices


def precision_recall_f1(true_singleton_indices, pred_singleton_indices):
  """Computes precision, recall, and F1-score for singleton cluster prediction.

  Args:
    true_singleton_indices: List of indices of true singleton events.
    pred_singleton_indices: List of indices of predicted singleton events.

  Returns:
    precision: float.
    recall: float.
    f1: float.
  """

  if true_singleton_indices == 0:
      return 0, 0, 0

  # Convert indices to sets for efficient intersection and union operations
  true_singletons = set(true_singleton_indices)
  pred_singletons = set(pred_singleton_indices)

  # Calculate true positives (correctly predicted singletons)
  true_positives = len(true_singletons.intersection(pred_singletons))

  # Calculate precision, recall, and f1-score
  precision = true_positives / len(pred_singletons) if len(pred_singletons) > 0 else 0
  recall = true_positives / len(true_singletons)
  f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

  return precision, recall, f1


