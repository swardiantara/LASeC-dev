import pandas as pd
import numpy as np
import os
import sys

# --- Configuration ---
ANNOTATION_DIR = 'annotations'
MASTER_DIR = 'review_sheets_raw' # Assuming master files are in the root

# --- File Names ---
MERGE_MASTER_FILE = os.path.join(MASTER_DIR, 'merge_review_sheet.xlsx')
SPLIT_MASTER_FILE = os.path.join(MASTER_DIR, 'split_review_members_sheet.xlsx')

MERGE_A_IN = os.path.join(ANNOTATION_DIR, 'merge_review.annotator_A.xlsx')
MERGE_B_IN = os.path.join(ANNOTATION_DIR, 'merge_review.annotator_B.xlsx')
SPLIT_A_IN = os.path.join(ANNOTATION_DIR, 'split_review.annotator_A.xlsx')
SPLIT_B_IN = os.path.join(ANNOTATION_DIR, 'split_review.annotator_B.xlsx')

MERGE_ADJ_IN = os.path.join(ANNOTATION_DIR, 'merge_disagreements_for_adjudication.xlsx')
SPLIT_ADJ_IN = os.path.join(ANNOTATION_DIR, 'split_disagreements_for_adjudication.xlsx')

MERGE_ADJ_OUT = os.path.join(ANNOTATION_DIR, 'merge_review.adjudicated.xlsx')
SPLIT_ADJ_OUT = os.path.join(ANNOTATION_DIR, 'split_review.adjudicated.xlsx')

def clean_verdict(verdict, default='keep'):
    """Standardizes verdict for comparison."""
    if pd.isna(verdict):
        return default
    return str(verdict).strip().lower()

print("--- Starting Final Adjudication File Prep ---")

# --- 1. Process MERGE Task ---
print(f"\nProcessing MERGE task...")
try:
    df_merge_master = pd.read_excel(MERGE_MASTER_FILE)
    df_merge_A = pd.read_excel(MERGE_A_IN)
    df_merge_B = pd.read_excel(MERGE_B_IN)
    df_merge_resolved = pd.read_excel(MERGE_ADJ_IN)
except FileNotFoundError:
    print("Error: Make sure all merge files are in the correct locations.")
    sys.exit()

# Check if you filled out the adjudication file
if df_merge_resolved['adj_verdict'].isna().any():
    print("WARNING: You have un-filled 'adj_verdict' rows in the merge disagreement file.")

# --- 1a. Get AGREEMENTS
df_merge_A['verdict_A'] = df_merge_A['verdict'].apply(clean_verdict)
df_merge_B['verdict_B'] = df_merge_B['verdict'].apply(clean_verdict)
df_merge_A['key'] = df_merge_A['first_id'].astype(str) + '_' + df_merge_A['second_id'].astype(str)
df_merge_B['key'] = df_merge_B['first_id'].astype(str) + '_' + df_merge_B['second_id'].astype(str)

df_compare_merge = pd.merge(df_merge_A[['key', 'verdict_A']], df_merge_B[['key', 'verdict_B']], on='key')
merge_agreements = df_compare_merge[df_compare_merge['verdict_A'] == df_compare_merge['verdict_B']]
agreed_merge_map = pd.Series(merge_agreements.verdict_A.values, index=merge_agreements.key).to_dict()

# --- 1b. Get RESOLVED DISAGREEMENTS
df_merge_resolved['key'] = df_merge_resolved['first_id'].astype(str) + '_' + df_merge_resolved['second_id'].astype(str)
resolved_merge_map = pd.Series(df_merge_resolved.adj_verdict.values, index=df_merge_resolved.key).to_dict()

# --- 1c. Combine maps
final_merge_map = {**agreed_merge_map, **resolved_merge_map}

# --- 1d. Create final file
df_merge_adjudicated = df_merge_master.copy()
df_merge_adjudicated['key'] = df_merge_adjudicated['first_id'].astype(str) + '_' + df_merge_adjudicated['second_id'].astype(str)
df_merge_adjudicated['verdict'] = df_merge_adjudicated['key'].map(final_merge_map)
df_merge_adjudicated = df_merge_adjudicated.drop(columns=['key'])

df_merge_adjudicated.to_excel(MERGE_ADJ_OUT, index=False)
print(f"  -> Created '{MERGE_ADJ_OUT}' for you to complete.")

# --- 2. Process SPLIT Task ---
print(f"\nProcessing SPLIT task...")
try:
    df_split_master = pd.read_excel(SPLIT_MASTER_FILE)
    df_split_A = pd.read_excel(SPLIT_A_IN)
    df_split_B = pd.read_excel(SPLIT_B_IN)
    df_split_resolved = pd.read_excel(SPLIT_ADJ_IN)
except FileNotFoundError:
    print("Error: Make sure all split files are in the correct locations.")
    sys.exit()

if df_split_resolved['adj_verdict'].isna().any():
    print("WARNING: You have un-filled 'adj_verdict' rows in the split disagreement file.")

# --- 2a. Get AGREEMENTS
df_split_A['verdict_A'] = df_split_A['verdict'].apply(clean_verdict)
df_split_B['verdict_B'] = df_split_B['verdict'].apply(clean_verdict)
df_split_A['key'] = df_split_A['cluster_id'].astype(str) + '_' + df_split_A['member_message']
df_split_B['key'] = df_split_B['cluster_id'].astype(str) + '_' + df_split_B['member_message']

df_compare_split = pd.merge(df_split_A[['key', 'verdict_A']], df_split_B[['key', 'verdict_B']], on='key')
split_agreements = df_compare_split[df_compare_split['verdict_A'] == df_compare_split['verdict_B']]
agreed_split_map = pd.Series(split_agreements.verdict_A.values, index=split_agreements.key).to_dict()

# --- 2b. Get RESOLVED DISAGREEMENTS
df_split_resolved['key'] = df_split_resolved['cluster_id'].astype(str) + '_' + df_split_resolved['member_message']
resolved_split_map = pd.Series(df_split_resolved.adj_verdict.values, index=df_split_resolved.key).to_dict()

# --- 2c. Combine maps
final_split_map = {**agreed_split_map, **resolved_split_map}

# --- 2d. Create final file
df_split_adjudicated = df_split_master.copy()
df_split_adjudicated['key'] = df_split_adjudicated['cluster_id'].astype(str) + '_' + df_split_adjudicated['member_message']
df_split_adjudicated['verdict'] = df_split_adjudicated['key'].map(final_split_map)
df_split_adjudicated = df_split_adjudicated.drop(columns=['key'])

df_split_adjudicated.to_excel(SPLIT_ADJ_OUT, index=False)
print(f"  -> Created '{SPLIT_ADJ_OUT}' for you to complete.")

print("\n--- Final File Prep Complete ---")
print("Your next step: Open the '...adjudicated.xlsx' files and fill in the remaining blank 'verdict' rows.")