import pandas as pd
import os
import sys

# --- Configuration ---
SAMPLE_FRACTION = 0.3
RANDOM_SEED = 42 # Ensures the same sample is generated every time
OUTPUT_DIR = 'review_sheets_raw'

# --- File Names ---
MERGE_MASTER_FILE = os.path.join(OUTPUT_DIR, 'merge_review_sheet.xlsx')
SPLIT_MASTER_FILE = os.path.join(OUTPUT_DIR, 'split_review_members_sheet.xlsx')

MERGE_A_OUT = os.path.join(OUTPUT_DIR, 'merge_review.annotator_A.xlsx')
MERGE_B_OUT = os.path.join(OUTPUT_DIR, 'merge_review.annotator_B.xlsx')
SPLIT_A_OUT = os.path.join(OUTPUT_DIR, 'split_review.annotator_A.xlsx')
SPLIT_B_OUT = os.path.join(OUTPUT_DIR, 'split_review.annotator_B.xlsx')

print("--- Starting Sample Generation ---")

# --- 1. Create Directories ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

# --- 2. Process MERGE File ---
print(f"\nProcessing {MERGE_MASTER_FILE}...")
try:
    df_merge_master = pd.read_excel(MERGE_MASTER_FILE)
except FileNotFoundError:
    print(f"Error: Master file not found at {MERGE_MASTER_FILE}")
    sys.exit()

# Take the random sample
df_merge_sample = df_merge_master.sample(
    frac=SAMPLE_FRACTION, 
    random_state=RANDOM_SEED
)
print(f"  -> Full sheet has {len(df_merge_master)} pairs.")
print(f"  -> Created 30% sample with {len(df_merge_sample)} pairs.")

# Save identical copies for both annotators
df_merge_sample.sort_values(by='distance').to_excel(MERGE_A_OUT, index=False)
df_merge_sample.sort_values(by='distance').to_excel(MERGE_B_OUT, index=False)
print(f"  -> Saved identical samples to {MERGE_A_OUT} and {MERGE_B_OUT}")


# --- 3. Process SPLIT File ---
print(f"\nProcessing {SPLIT_MASTER_FILE}...")
try:
    df_split_master = pd.read_excel(SPLIT_MASTER_FILE)
except FileNotFoundError:
    print(f"Error: Master file not found at {SPLIT_MASTER_FILE}")
    sys.exit()

# Take the random sample
df_split_sample = df_split_master.sample(
    frac=SAMPLE_FRACTION, 
    random_state=RANDOM_SEED
)
print(f"  -> Full sheet has {len(df_split_master)} rows.")
print(f"  -> Created 30% sample with {len(df_split_sample)} rows.")

# Save identical copies for both annotators
df_split_sample.sort_values(by='similarity_to_prototype').to_excel(SPLIT_A_OUT, index=False)
df_split_sample.sort_values(by='similarity_to_prototype').to_excel(SPLIT_B_OUT, index=False)
print(f"  -> Saved identical samples to {SPLIT_A_OUT} and {SPLIT_B_OUT}")

print("\n--- Sample Generation Complete ---")