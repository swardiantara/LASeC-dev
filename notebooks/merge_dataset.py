import pandas as pd
import glob
import os
from pathlib import Path

def merge_log_datasets(dataset_folder='dataset'):
    """
    Merge multiple log datasets and re-assign EventIds based on unique EventTemplates.
    
    Args:
        dataset_folder (str): Path to the folder containing CSV files
        
    Returns:
        pd.DataFrame: Merged dataset with columns [Source, Content, EventId, EventTemplate]
    """
    
    # Find all CSV files matching the pattern
    csv_pattern = os.path.join(dataset_folder, '*_2k.log_structured.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern: {csv_pattern}")
    
    print(f"Found {len(csv_files)} dataset files")
    
    # List to store dataframes
    dataframes = []
    
    # Read each CSV file
    for csv_file in csv_files:
        # Extract source name from filename
        # e.g., "Android_2k.log_structured.csv" -> "Android"
        filename = Path(csv_file).name
        source_name = filename.replace('_2k.log_structured.csv', '')
        
        print(f"Reading {filename}...")
        
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        required_cols = ['Content', 'EventTemplate']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: {filename} missing columns: {missing_cols}. Skipping...")
            continue
        
        # Select and rename columns
        df_subset = df[['Content', 'EventTemplate']].copy()
        df_subset['Source'] = source_name
        
        dataframes.append(df_subset)
    
    if not dataframes:
        raise ValueError("No valid datasets were loaded. Check your CSV files.")
    
    # Merge all dataframes
    print("\nMerging datasets...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"Total logs: {len(merged_df)}")
    
    # Get unique EventTemplates and create EventId mapping
    unique_templates = merged_df['EventTemplate'].unique()
    print(f"Unique EventTemplates: {len(unique_templates)}")
    
    # Create EventId mapping: EventTemplate -> EventId (E1, E2, E3, ...)
    template_to_eventid = {
        template: f"E{idx+1}" 
        for idx, template in enumerate(sorted(unique_templates))
    }
    
    # Assign EventIds based on EventTemplate
    merged_df['EventId'] = merged_df['EventTemplate'].map(template_to_eventid)
    
    # Reorder columns
    merged_df = merged_df[['Source', 'Content', 'EventId', 'EventTemplate']]
    
    print("\nDataset merging complete!")
    print(f"EventId range: E1 to E{len(unique_templates)}")
    
    # Display summary statistics
    print("\n=== Summary ===")
    print(f"Total logs: {len(merged_df)}")
    print(f"Unique EventTemplates: {len(unique_templates)}")
    print(f"Sources: {merged_df['Source'].nunique()}")
    print("\nLogs per source:")
    print(merged_df['Source'].value_counts().sort_index())
    
    return merged_df


# Example usage
if __name__ == "__main__":
    # Merge datasets
    source_dir = 'dataset'
    merged_dataset = merge_log_datasets(source_dir)
    
    # Save merged dataset
    output_file = os.path.join(source_dir, 'MultiSource_2k.log_structured.csv')
    merged_dataset.to_csv(output_file, index=False)
    print(f"\nMerged dataset saved to: {output_file}")
    
    # Display sample
    print("\n=== Sample of merged dataset ===")
    print(merged_dataset.head(10))
    
    # Show EventId distribution
    print("\n=== EventId distribution (top 10) ===")
    print(merged_dataset['EventId'].value_counts().head(10))