import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import cohen_kappa_score

# --- Configuration ---
ANNOTATION_DIR = 'annotations'

# --- Input File Names ---
MERGE_A_IN = os.path.join(ANNOTATION_DIR, 'merge_review.annotator_A.xlsx')
MERGE_B_IN = os.path.join(ANNOTATION_DIR, 'merge_review.annotator_B.xlsx')
SPLIT_A_IN = os.path.join(ANNOTATION_DIR, 'split_review.annotator_A.xlsx')
SPLIT_B_IN = os.path.join(ANNOTATION_DIR, 'split_review.annotator_B.xlsx')

# --- Output File Names ---
MERGE_ADJ_TODO = os.path.join(ANNOTATION_DIR, 'merge_disagreements_for_adjudication.xlsx')
SPLIT_ADJ_TODO = os.path.join(ANNOTATION_DIR, 'split_disagreements_for_adjudication.xlsx')
IAA_REPORT_FILE = os.path.join(ANNOTATION_DIR, 'iaa_report.txt')


def clean_verdict(verdict, default='keep'):
    """Standardizes verdict for comparison. Assumes blank is 'keep'."""
    if pd.isna(verdict):
        return default
    return str(verdict).strip().lower()

print("--- Starting Disagreement Finder & IAA Calculation ---")

# We will build this report string
iaa_report_content = "--- Inter-Annotator Agreement (IAA) Report ---\n\n"

# --- 1. Process MERGE Task ---
print(f"\nProcessing MERGE task...")
try:
    df_merge_A = pd.read_excel(MERGE_A_IN)
    df_merge_B = pd.read_excel(MERGE_B_IN)
except FileNotFoundError:
    print("Error: Make sure annotator files are in the 'annotations/' folder.")
    sys.exit()

# Clean verdicts
df_merge_A['verdict_A'] = df_merge_A['verdict'].apply(clean_verdict, default='keep')
df_merge_B['verdict_B'] = df_merge_B['verdict'].apply(clean_verdict, default='keep')

# Create a unique key for merging
df_merge_A['key'] = df_merge_A['first_id'].astype(str) + '_' + df_merge_A['second_id'].astype(str)
df_merge_B['key'] = df_merge_B['first_id'].astype(str) + '_' + df_merge_B['second_id'].astype(str)

# Merge the two annotators' decisions
# This merge correctly isolates just the 30% sample they both did.
df_compare_merge = pd.merge(
    df_merge_A,
    df_merge_B[['key', 'verdict_B']],
    on='key',
    how='inner' # Use 'inner' to only get rows they BOTH annotated
)

# --- 1a. Calculate MERGE IAA ---
merge_labels_A = df_compare_merge['verdict_A']
merge_labels_B = df_compare_merge['verdict_B']

merge_kappa = cohen_kappa_score(merge_labels_A, merge_labels_B)
merge_agreement_pct = (merge_labels_A == merge_labels_B).mean() * 100
merge_total_reviewed = len(df_compare_merge)

iaa_report_content += "MERGE TASK\n"
iaa_report_content += "----------\n"
iaa_report_content += f"- Total pairs reviewed by both annotators: {merge_total_reviewed}\n"
iaa_report_content += f"- Simple Agreement: {merge_agreement_pct:.2f}%\n"
iaa_report_content += f"- Cohen's Kappa: {merge_kappa:.4f}\n\n"

print(f"  -> Merge Task IAA: {merge_agreement_pct:.2f}% Agreement, {merge_kappa:.4f} Kappa")

# --- 1b. Find MERGE Disagreements ---
merge_disagreements = df_compare_merge[
    df_compare_merge['verdict_A'] != df_compare_merge['verdict_B']
].copy()

# Prepare the new adjudication file
merge_disagreements['adj_verdict'] = '' # Add the blank column for you
merge_disagreements = merge_disagreements.drop(columns=['verdict']) 

merge_disagreements.to_excel(MERGE_ADJ_TODO, index=False)
print(f"  -> Found {len(merge_disagreements)} merge disagreements.")
print(f"  -> Saved your 'to-do' list to: {MERGE_ADJ_TODO}")

# --- 2. Process SPLIT Task ---
print(f"\nProcessing SPLIT task disagreements...")
try:
    df_split_A = pd.read_excel(SPLIT_A_IN)
    df_split_B = pd.read_excel(SPLIT_B_IN)
except FileNotFoundError:
    print("Error: Make sure annotator files are in the 'annotations/' folder.")
    sys.exit()

# Clean verdicts
df_split_A['verdict_A'] = df_split_A['verdict'].apply(clean_verdict, default='keep')
df_split_B['verdict_B'] = df_split_B['verdict'].apply(clean_verdict, default='keep')

# Create a unique key
df_split_A['key'] = df_split_A['cluster_id'].astype(str) + '_' + df_split_A['member_message']
df_split_B['key'] = df_split_B['cluster_id'].astype(str) + '_' + df_split_B['member_message']

# Merge decisions
df_compare_split = pd.merge(
    df_split_A,
    df_split_B[['key', 'verdict_B']],
    on='key',
    how='inner' # Use 'inner' to only get rows they BOTH annotated
)

# --- 2a. Calculate SPLIT IAA ---
split_labels_A = df_compare_split['verdict_A']
split_labels_B = df_compare_split['verdict_B']

split_kappa = cohen_kappa_score(split_labels_A, split_labels_B)
split_agreement_pct = (split_labels_A == split_labels_B).mean() * 100
split_total_reviewed = len(df_compare_split)

iaa_report_content += "SPLIT TASK\n"
iaa_report_content += "----------\n"
iaa_report_content += f"- Total rows reviewed by both annotators: {split_total_reviewed}\n"
iaa_report_content += f"- Simple Agreement: {split_agreement_pct:.2f}%\n"
iaa_report_content += f"- Cohen's Kappa: {split_kappa:.4f}\n\n"

print(f"  -> Split Task IAA: {split_agreement_pct:.2f}% Agreement, {split_kappa:.4f} Kappa")

# --- 2b. Find SPLIT Disagreements ---
split_disagreements = df_compare_split[
    df_compare_split['verdict_A'] != df_compare_split['verdict_B']
].copy()

# Prepare the new adjudication file
split_disagreements['adj_verdict'] = '' # Add the blank column for you
split_disagreements = split_disagreements.drop(columns=['verdict']) 

split_disagreements.to_excel(SPLIT_ADJ_TODO, index=False)
print(f"  -> Found {len(split_disagreements)} split disagreements.")
print(f"  -> Saved your 'to-do' list to: {SPLIT_ADJ_TODO}")

# --- 3. Save Final IAA Report ---
iaa_report_content += "--- Interpretation of Cohen's Kappa (k) ---\n"
iaa_report_content += "< 0.00: Poor\n"
iaa_report_content += "0.00 - 0.20: Slight\n"
iaa_report_content += "0.21 - 0.40: Fair\n"
iaa_report_content += "0.41 - 0.60: Moderate\n"
iaa_report_content += "0.61 - 0.80: Substantial\n"
iaa_report_content += "0.81 - 1.00: Almost Perfect\n"

with open(IAA_REPORT_FILE, 'w') as f:
    f.write(iaa_report_content)

print(f"\nSuccessfully saved full IAA report to: {IAA_REPORT_FILE}")
print("--- Script 4 Complete ---")