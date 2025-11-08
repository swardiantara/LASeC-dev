# Drone Log Abstraction: Gold-Standard Dataset Pipeline

This repository contains the complete data preparation pipeline used to create a gold-standard, human-annotated dataset for drone flight log abstraction. The scripts and methodology are designed to be reproducible, measurable, and robust against common clustering failures.

## 1. Overview and Methodology

### 1.1. The Problem
Standard automated clustering of log messages (e.g., S-BERT embeddings + Agglomerative Clustering) fails in two critical, opposing ways:
1.  **Under-clustering (Logical Opposites):** Messages with similar topics but opposite meanings (e.g., `Your palm is too close` vs. `Your palm is too far`) are incorrectly grouped together due to high semantic similarity.
2.  **Over-clustering (Syntactic Differences):** Messages with identical semantic meaning but different syntax (e.g., `GCS link established` vs. `GCS connection OK`) are incorrectly split into separate clusters.

### 1.2. The Solution
This pipeline addresses these failures by implementing a rigorous, two-stage Human-in-the-Loop (HITL) validation process built upon an initial "strict" automated clustering.

* **Stage 1: Merge Review (Fixes Over-clustering):** Annotators review semantically-close cluster pairs to identify candidates for merging.
* **Stage 2: Split Review (Fixes Under-clustering):** Annotators review "syntactically impure" clusters to flag individual messages for splitting.

The pipeline includes automated validation checks (e.g., "Syntactic Purity") to verify human decisions and ensure the final dataset is coherent. The entire process is validated using Inter-Annotator Agreement (IAA) metrics.

---

## 2. 🚀 The Annotation Pipeline Workflow

The pipeline is executed via a series of Python scripts and manual annotation steps.

### Phase 1: Preparation (Script 1)
**Script:** `1_generate_review_sheets.py`
1.  Loads the raw dataset from `merged.xlsx`, preserving `set` and `source_file` metadata.
2.  Extracts all unique messages and generates `all-MiniLM-L6-v2` embeddings.
3.  Performs a "strict" `AgglomerativeClustering` (`distance_threshold=0.15`) on the unique messages to create a baseline of small, "pure" clusters.
4.  Generates `merge_review_sheet.xlsx` by identifying all cluster pairs with a centroid cosine distance `< 0.35`.
5.  Generates `split_review_members_sheet.xlsx` by calculating a "Syntactic Purity Score" for all clusters and "exploding" any cluster with a score `< 1.0` into its constituent messages.

### Phase 2: Sampling & IAA Calculation (Scripts 3 & 4)
1.  **Script `3_generate_samples.py`:** Extracts a 30% random sample (using a fixed `random_state=42` for reproducibility) from both master review sheets. It saves these as identical files for two independent annotators (Annotator A, Annotator B).
2.  **[Manual Step]:** Annotators A and B independently fill in the `verdict` column on their 30% sample files and return them to the `annotations/` folder.
3.  **Script `4_find_disagreements.py`:**
    * Compares the two completed sample files.
    * Calculates and saves the **Inter-Annotator Agreement (IAA)** (Simple Agreement and Cohen's Kappa) to `annotations/iaa_report.txt`.
    * Generates `..._for_adjudication.xlsx` files, which contain *only* the rows where the annotators disagreed, with a new `adj_verdict` column.

### Phase 3: Adjudication & Completion (Script 5 & Manual)
1.  **[Manual Step 1 - Adjudication]:** The Adjudicator reviews the `..._for_adjudication.xlsx` files and fills in the `adj_verdict` column to resolve all conflicts.
2.  **Script `5_prepare_adjudicator_file.py`:** Merges the *agreements* from Phase 2 and the *resolved disagreements* from this phase back into the master (100%) review sheets, creating the final `...adjudicated.xlsx` files.
3.  **[Manual Step 2 - Completion]:** The Adjudicator opens the two `...adjudicated.xlsx` files and completes the annotation by filling in the `verdict` for the remaining 70% of rows, following the established guidelines.

### Phase 4: Finalization & Validation (Script 6)
**Script:** `6_apply_final_corrections.py`
This script reads the 100% completed `...adjudicated.xlsx` files and builds the final dataset.
1.  **Applies Merges (with Validation):**
    * Resolves all `merge` verdicts, including transitive merges (A->B, B->C), using a graph-based connected components algorithm (NetworkX).
    * **Safety Net:** Before finalizing a merge, it runs a **Syntactic Purity Check**. If a merge would create an "impure" cluster, it is *rejected* and logged in `conflict_logs/merge_conflicts.xlsx`.
2.  **Applies Splits (with Validation):**
    * Collects all messages marked `[SPLIT]` into an "orphaned" bucket.
    * **Re-Clustering:** Re-clusters *only* this bucket using the same strict `AgglomerativeClustering` parameters. This allows split messages to form new, valid clusters (e.g., the "palm" examples).
    * **Safety Net:** Runs the **Purity Check** on these newly formed clusters. Any that are still "impure" are flagged in `conflict_logs/split_cleanup.xlsx` and are assigned as singletons (one cluster per message).
3.  **Generates Final CSV:** Maps all `final_label` IDs back to the original data, preserving the `set`, `source_file`, and `message` columns. The final product is saved as `🏆_gold_standard_dataset.csv`.

---

## 3. 📁 Repository Structure

```
└── 📁dataset_preparation
    └── 📁annotations
        ├── iaa_report.txt
        ├── merge_disagreements_for_adjudication.xlsx
        ├── merge_review.adjudicated.xlsx
        ├── merge_review.annotator_A.xlsx
        ├── merge_review.annotator_B.xlsx
        ├── split_disagreements_for_adjudication.xlsx
        ├── split_review.adjudicated.xlsx
        ├── split_review.annotator_A.xlsx
        ├── split_review.annotator_B.xlsx
    └── 📁conflict_logs
        ├── merge_conflicts.xlsx
        ├── split_cleanup.xlsx
    └── 📁review_sheets_raw
        ├── initial_label_data.xlsx
        ├── merge_review_sheet.xlsx
        ├── merge_review.annotator_A.xlsx
        ├── merge_review.annotator_B.xlsx
        ├── split_review_members_sheet.xlsx
        ├── split_review.annotator_A.xlsx
        ├── split_review.annotator_B.xlsx
    └── 📁test
        ├── flight_log_1.xlsx
        ├── flight_log_2.xlsx
        ├── flight_log_3.xlsx
        ├── flight_log_4.xlsx
        ├── flight_log_5.xlsx
        ├── flight_log_6.xlsx
        ├── flight_log_7.xlsx
    └── 📁train
        ├── airdata_logs.xlsx
        ├── vto_logs.xlsx
    ├── 0_preprocessing.ipynb
    ├── 1_generate_review_sheets.py
    ├── 2_generate_samples.py
    ├── 3_highlight_overlaps.py
    ├── 4_find_disagreements.py
    ├── 5_prepare_adjudicator_file.py
    ├── 6_apply_final_corrections.py
    ├── gold_standard_dataset.csv
    ├── initial_data.pkl
    ├── merged_data.xlsx
    ├── random_annotation.ipynb
    ├── readme.md
    └── readme2.md
```

--- 

## 4. How to Reproduce

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Generate Master Sheets:**
    ```bash
    python 1_generate_review_sheets.py
    ```
3.  **Generate Annotator Samples:**
    ```bash
    python 3_generate_samples.py
    ```
4.  **[MANUAL] Simulate Annotation:**
    * Copy the `annotator_A` and `annotator_B` files from `review_sheets_raw/` to `annotations/`.
    * Fill in the `verdict` columns in these copied files.
5.  **Calculate IAA & Find Disagreements:**
    ```bash
    python 4_find_disagreements.py
    ```
    * This generates `iaa_report.txt` and the `..._for_adjudication.xlsx` files.
6.  **[MANUAL] Simulate Adjudication:**
    * Open the `..._for_adjudication.xlsx` files.
    * Fill in the `adj_verdict` column.
7.  **Prepare Final Adjudicator Files:**
    ```bash
    python 5_prepare_adjudicator_file.py
    ```
    * This merges all decisions from the 30% sample into the final `...adjudicated.xlsx` files.
8.  **[MANUAL] Complete Annotation:**
    * Open `annotations/merge_review.adjudicated.xlsx` and `annotations/split_review.adjudicated.xlsx`.
    * Fill in the `verdict` for all remaining (70%) blank rows.
9.  **Generate Final Dataset:**
    ```bash
    python 6_apply_final_corrections.py
    ```
10. **Final Review:**
    * Check the `conflict_logs/` folder. If any files were generated, perform a final manual review and edit the `gold_standard_dataset.csv` as needed.
    * The `gold_standard_dataset.csv` file is now complete.