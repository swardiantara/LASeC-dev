# --- 0. SETUP ---
# !pip install pandas numpy sentence-transformers scikit-learn open-pyxl
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from difflib import SequenceMatcher
import sys

OUTPUT_DIR = 'review_sheets_raw'

# --- Helper Function for Purity ---
def get_string_similarity(a, b):
    """Helper function to get string similarity ratio."""
    return SequenceMatcher(None, a, b).ratio()

# --- 1. Load Data ---
print("--- Starting Pipeline ---")

try:
    # --- MODIFIED: Load from merged.xlsx ---
    df_raw = pd.read_excel('merged_data.xlsx')
    if not all(col in df_raw.columns for col in ['set', 'source_file', 'message']):
        print("Error: 'merged.xlsx' must have columns 'set', 'source_file', and 'message'.")
        sys.exit()
except FileNotFoundError:
    print("Error: 'merged.xlsx' not found.")
    sys.exit()

print(f"Loaded {len(df_raw)} total log messages from 'merged.xlsx'.")

# --- 2. Get Unique Messages for Clustering ---
print("Extracting unique messages for embedding...")
unique_messages = df_raw['message'].unique()
print(f"Found {len(unique_messages)} unique messages.")

print("Generating embeddings for unique messages...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
unique_embeddings = embed_model.encode(unique_messages)

# --- 3. Initial Strict Clustering (on Unique Messages) ---
print("Running initial strict clustering (threshold=0.15)...")
STRICT_THRESHOLD = 0.15
cluster_model = AgglomerativeClustering(
    n_clusters=None, 
    distance_threshold=STRICT_THRESHOLD,
    metric='cosine', linkage='average'
)
# unique_labels will correspond to each message in unique_messages
unique_labels = cluster_model.fit_predict(unique_embeddings)
num_clusters = len(np.unique(unique_labels))
print(f"Found {num_clusters} initial clusters from unique messages.")

# --- 4. Map Labels & Embeddings back to Full DataFrame ---
print("Mapping labels and embeddings back to all 3016 rows...")

# Create mapping dictionaries
message_to_label_map = dict(zip(unique_messages, unique_labels))
message_to_embedding_map = dict(zip(unique_messages, list(unique_embeddings)))

# --- NEW: Map unique labels/embeddings back to all rows ---
df_raw['initial_label'] = df_raw['message'].map(message_to_label_map)
df_raw['embedding'] = df_raw['message'].map(message_to_embedding_map)

# export the df_raw with the initial labels for analysis
initial_df = df_raw[['set', 'source_file', 'message', 'initial_label']]
initial_df.to_excel(os.path.join(OUTPUT_DIR, 'initial_label_data.xlsx'), index=False)

# Save this full, enriched DataFrame for the next script
df_raw.to_pickle('initial_data.pkl')
print("Saved 'initial_data.pkl' with full set/source_file/message/label info.")

# --- 5. Prepare Cluster Info (from Unique Data) ---
print("Calculating centroids and prototypes...")
cluster_info = {}
for i, label in enumerate(unique_labels):
    if label not in cluster_info:
        # We only need to store info based on the unique messages
        cluster_info[label] = {'messages': [], 'embeddings': []}
    cluster_info[label]['messages'].append(unique_messages[i])
    cluster_info[label]['embeddings'].append(unique_embeddings[i])

# Calculate centroid, prototype, and size for each cluster
for label, data in cluster_info.items():
    cluster_embeddings_arr = np.array(data['embeddings'])
    data['centroid'] = np.mean(cluster_embeddings_arr, axis=0)
    data['size'] = len(data['messages']) # Note: this is size in *unique messages*
    sims = cosine_similarity(cluster_embeddings_arr, data['centroid'].reshape(1, -1))
    data['prototype'] = data['messages'][np.argmax(sims)]

# --- 6. Generate MERGE Review Sheet (Task 1) ---
print("Generating 'merge_review_sheet.xlsx'...")
labels_list = list(cluster_info.keys())
centroids_list = [cluster_info[label]['centroid'] for label in labels_list]
centroid_distances = cosine_distances(centroids_list)

merge_candidates = []
MERGE_GATE_THRESHOLD = 0.30 

for i in range(len(labels_list)):
    for j in range(i + 1, len(labels_list)):
        distance = centroid_distances[i, j]
        if distance < MERGE_GATE_THRESHOLD:
            label_1, label_2 = labels_list[i], labels_list[j]
            merge_candidates.append({
                'first_id': label_1, 'first_prototype': cluster_info[label_1]['prototype'],
                'distance': distance,
                'second_prototype': cluster_info[label_2]['prototype'], 'second_id': label_2,
                'verdict': '' # For annotator
            })

merge_review_df = pd.DataFrame(merge_candidates).sort_values(by='distance')
review_path = os.path.join(OUTPUT_DIR, 'merge_review_sheet.xlsx')
merge_review_df.to_excel(review_path, index=False)
print(f"Saved {review_path} with {len(merge_review_df)} candidates.")

# --- 7. Generate SPLIT Review Sheet (Task 2) ---
print("Generating 'split_review_members_sheet.xlsx'...")

# 7a. Find impure clusters
split_candidates_impure = []
for label, data in cluster_info.items():
    if data['size'] <= 1: continue
    
    prototype_msg = data['prototype']
    purity_scores = [get_string_similarity(prototype_msg, msg) for msg in data['messages'] if msg != prototype_msg]
    avg_purity = np.mean(purity_scores) if purity_scores else 1.0
        
    if avg_purity < 1.0: 
        split_candidates_impure.append({'cluster_id': label, 'purity_score': avg_purity})

# 7b. "Explode" impure clusters into the review sheet
split_review_rows = []
impure_cluster_ids = [c['cluster_id'] for c in sorted(split_candidates_impure, key=lambda x: x['purity_score'])]

for label in impure_cluster_ids:
    data = cluster_info[label]
    prototype_msg = data['prototype']
    # Iterate over unique messages in that cluster
    for msg in data['messages']: 
        if msg == prototype_msg: continue
        sim_to_prototype = get_string_similarity(prototype_msg, msg)
        split_review_rows.append({
            'cluster_id': label,
            'prototype_message': prototype_msg,
            'member_message': msg,
            'similarity_to_prototype': sim_to_prototype,
            'verdict': '' 
        })

split_members_df = pd.DataFrame(split_review_rows)
split_path = os.path.join(OUTPUT_DIR, 'split_review_members_sheet.xlsx')
split_members_df.to_excel(split_path, index=False)
print(f"Saved split_path with {len(split_members_df)} members to review.")
print("--- Script 1 Complete ---")