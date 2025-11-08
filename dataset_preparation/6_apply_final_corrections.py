# --- 0. SETUP ---
# !pip install pandas numpy scikit-learn networkx
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from difflib import SequenceMatcher
import networkx as nx
import sys
import os

# --- Configuration ---
ANNOTATION_DIR = 'annotations'
MASTER_DATA = 'initial_data.pkl'

# --- Input Files (Must be 100% COMPLETED by you) ---
MERGE_ADJ_IN = os.path.join(ANNOTATION_DIR, 'merge_review.adjudicated.xlsx')
SPLIT_ADJ_IN = os.path.join(ANNOTATION_DIR, 'split_review.adjudicated.xlsx')

# --- Output Files ---
CONFLICT_LOG_DIR = 'conflict_logs'
MERGE_CONFLICT_OUT = os.path.join(CONFLICT_LOG_DIR, 'merge_conflicts.xlsx')
SPLIT_CLEANUP_OUT = os.path.join(CONFLICT_LOG_DIR, 'split_cleanup.xlsx')
FINAL_DATASET_OUT = 'gold_standard_dataset.csv'

STRICT_THRESHOLD = 0.15 # Use the same threshold for re-clustering

# --- Helper Functions ---
def get_string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def calculate_purity(messages):
    """Calculates the syntactic purity of a list of messages."""
    if not messages or len(messages) < 2:
        return 1.0
    
    # Use the first message as the prototype for this check
    prototype = messages[0]
    purity_scores = [get_string_similarity(prototype, msg) for msg in messages[1:]]
    return np.mean(purity_scores)

def clean_verdict(verdict, default='keep'):
    """Standardizes the final verdict. Assumes any blank is 'keep'."""
    if pd.isna(verdict):
        return default
    return str(verdict).strip().lower()

# --- 1. Load All Data ---
print("--- Starting Final Correction Pipeline ---")

# Create output dirs
if not os.path.exists(CONFLICT_LOG_DIR):
    os.makedirs(CONFLICT_LOG_DIR)

try:
    # This .pkl file has all 3000+ rows with metadata
    df = pd.read_pickle(MASTER_DATA)
    merge_verdicts = pd.read_excel(MERGE_ADJ_IN)
    split_verdicts = pd.read_excel(SPLIT_ADJ_IN)
except FileNotFoundError as e:
    print(f"Error: Could not find a required file. {e}")
    print("Please make sure 'initial_data.pkl' and the two '.adjudicated.xlsx' files exist.")
    sys.exit()

# Check that the user has completed the files
if merge_verdicts['verdict'].isna().any() or split_verdicts['verdict'].isna().any():
    print("="*60)
    print("!! WARNING !!")
    print("Blank 'verdict' rows were found in your adjudicated files.")
    print("The script will treat all blanks as 'keep'.")
    print("If this was not your intention, press Ctrl+C to stop.")
    print("="*60)
    try:
        input("Press Enter to continue...")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit()

# Start our new labels as a copy. This Series aligns with the full df.
final_labels = df['initial_label'].copy()
new_label_counter = final_labels.max() + 1
merge_conflicts_list = []
split_cleanup_list = []

# --- 2. Apply MERGE Verdicts (with Validation) ---
print("Step 2: Applying 'MERGE' verdicts...")

# Clean all verdicts first
merge_verdicts['verdict'] = merge_verdicts['verdict'].apply(clean_verdict)
merge_decisions = merge_verdicts[merge_verdicts['verdict'] == 'merge']

# Build a graph to find all "transitive" merge groups
G = nx.Graph()
for _, row in merge_decisions.iterrows():
    G.add_edge(row['first_id'], row['second_id'])

merge_groups = list(nx.connected_components(G))
print(f"Found {len(merge_groups)} 'super-clusters' to merge from 'merge' verdicts.")

for group in merge_groups:
    group_labels = list(group)
    
    # Get all messages (from all 3000+ rows) belonging to this new super-cluster
    messages_to_merge = df[df['initial_label'].isin(group_labels)]['message'].unique().tolist()
    
    # VALIDATION: Check the purity of this new super-cluster
    purity = calculate_purity(messages_to_merge)
    
    if purity > 0.8: # You can set this threshold
        # --- PURE: This merge is auto-approved ---
        new_label = new_label_counter
        final_labels[df['initial_label'].isin(group_labels)] = new_label
        new_label_counter += 1
    else:
        # --- IMPURE: Flag for manual review ---
        print(f"  -> Conflict: Merge group {group_labels} is 'impure' (Score: {purity:.2f}). Flagging.")
        merge_conflicts_list.append({
            'group': str(group_labels),
            'purity': purity,
            'messages': ' | '.join(messages_to_merge)
        })

# --- 3. Apply SPLIT Verdicts (with Re-Clustering) ---
print("Step 3: Applying 'SPLIT' verdicts...")

# Clean all verdicts
split_verdicts['verdict'] = split_verdicts['verdict'].apply(clean_verdict)
split_decisions = split_verdicts[split_verdicts['verdict'] == 'split']

# This list contains the *unique* message strings to split
split_messages = split_decisions['member_message'].unique().tolist()

if split_messages:
    # Find all rows in the *full* DataFrame that match these messages
    split_indices = df[df['message'].isin(split_messages)].index
    
    # "Orphan" all matching rows
    final_labels.loc[split_indices] = -99 # Mark for re-clustering

    # 3a. Re-Cluster the "Orphaned" Bucket
    split_bucket_df = df.loc[final_labels == -99]
    print(f"  -> Re-clustering {len(split_bucket_df)} total 'SPLIT' messages...")
    bucket_embeddings = np.array(split_bucket_df['embedding'].tolist())
    
    split_cluster_model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=STRICT_THRESHOLD,
        metric='cosine', linkage='average'
    )
    new_split_labels = split_cluster_model.fit_predict(bucket_embeddings)
    
    # 3b. Validate and assign these new clusters
    split_bucket_df['new_label'] = new_split_labels
    unique_new_labels = split_bucket_df['new_label'].unique()
    
    for new_label in unique_new_labels:
        new_cluster_messages = split_bucket_df[split_bucket_df['new_label'] == new_label]['message'].unique().tolist()
        purity = calculate_purity(new_cluster_messages)
        indices_to_update = split_bucket_df[split_bucket_df['new_label'] == new_label].index
        
        if purity > 0.8:
            # --- PURE: This new cluster is auto-approved ---
            final_labels.loc[indices_to_update] = new_label_counter
            new_label_counter += 1
        else:
            # --- IMPURE: Flag for cleanup, but assign as singletons ---
            print(f"  -> Cleanup: New split cluster is 'impure' (Score: {purity:.2f}). Flagging.")
            split_cleanup_list.append({
                'purity': purity,
                'messages': ' | '.join(new_cluster_messages)
            })
            # Assign as singletons (one new ID per message)
            for idx in indices_to_update:
                final_labels.loc[idx] = new_label_counter
                new_label_counter += 1
else:
    print("  -> No messages were marked for 'SPLIT'.")

# --- 4. Final Output ---
print("Step 4: Saving final files...")
df['final_label'] = final_labels

# Save the "conflict" files for your final manual check
if merge_conflicts_list:
    pd.DataFrame(merge_conflicts_list).to_excel(MERGE_CONFLICT_OUT, index=False)
    print(f"  -> WARNING: Found {len(merge_conflicts_list)} merge conflicts. Review '{MERGE_CONFLICT_OUT}'.")
else:
    print(f"  -> No merge conflicts found. ({MERGE_CONFLICT_OUT})")

if split_cleanup_list:
    pd.DataFrame(split_cleanup_list).to_excel(SPLIT_CLEANUP_OUT, index=False)
    print(f"  -> WARNING: Found {len(split_cleanup_list)} impure split clusters. Review '{SPLIT_CLEANUP_OUT}'.")
else:
    print(f"  -> No split cleanup needed. ({SPLIT_CLEANUP_OUT})")

# Save the final gold-standard dataset with all original metadata
final_dataset = df[['set', 'source_file', 'message', 'final_label']]
# Clean up any -99 labels that might have slipped through (shouldn't happen)
final_dataset = final_dataset[final_dataset['final_label'] != -99]

final_dataset.to_csv(FINAL_DATASET_OUT, index=False)

print("\n--- Script 6 Complete ---")
print(f"Your final dataset is in '{FINAL_DATASET_OUT}'")