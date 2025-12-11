"""
MID2.csv Data Extraction Script
Extracts drug composition and medical uses from MID2.csv for NLP training

Usage: python extract_MID2_data.py
"""

import pandas as pd
import json
from collections import Counter

def load_mid2_data(filepath='MID2.csv'):
    """
    Load MID2 data from Excel file, preserving newlines in USES column for multi-label extraction
    
    Args:
        filepath: Path to the Excel file (MID.csv or MID2.xlsx)
    
    Returns:
        DataFrame with all columns, USES column preserves newlines for label extraction
    """
    print(f"Loading {filepath}...")
    
    # Try reading as Excel first (since MID.csv is actually an Excel file)
    try:
        # Try .xls format
        df = pd.read_excel(filepath, engine='xlrd')
        print("✓ Successfully loaded as Excel (.xls) format")
    except Exception:
        try:
            # Try .xlsx format
            df = pd.read_excel(filepath, engine='openpyxl')
            print("✓ Successfully loaded as Excel (.xlsx) format")
        except Exception:
            # Fall back to CSV with error handling
            print("Excel read failed, trying CSV format...")
            df = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip', engine='python')
            print("✓ Successfully loaded as CSV format")
    
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)[:5]}...")
    
    # Clean text columns BUT preserve newlines in USES for multi-label extraction
    print("\nCleaning text columns...")
    text_columns = df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        if col == 'USES':
            # For USES column: preserve newlines but clean other whitespace
            df[col] = df[col].astype(str).str.replace('\r', '', regex=False)
            # Don't remove \n - we need it for splitting conditions
        else:
            # For other columns: remove all newlines and clean whitespace
            df[col] = df[col].astype(str).str.replace('\n', ' ', regex=False)
            df[col] = df[col].str.replace('\r', ' ', regex=False)
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            df[col] = df[col].str.strip()
        
        # Replace 'nan' strings back to actual NaN
        df[col] = df[col].replace('nan', '')
    
    print(f"✓ Processed {len(text_columns)} text columns (USES column preserves newlines)")
    
    return df


def clean_and_filter(df):
    """
    Clean data and filter records with both CONTAINS and USES
    """
    print("\n" + "="*60)
    print("Cleaning and filtering data...")
    print("="*60)
    
    initial_count = len(df)
    print(f"Initial records: {initial_count}")
    
    # Remove whitespace
    df['CONTAINS'] = df['CONTAINS'].str.strip()
    df['USES'] = df['USES'].str.strip()
    df['NAME'] = df['NAME'].str.strip()
    
    # Filter: must have both CONTAINS and USES
    df_clean = df[(df['CONTAINS'] != '') & (df['USES'] != '')].copy()
    print(f"Records with CONTAINS and USES: {len(df_clean)} ({len(df_clean)/initial_count*100:.1f}%)")
    
    # Remove duplicates based on NAME and CONTAINS
    df_clean = df_clean.drop_duplicates(subset=['NAME', 'CONTAINS'])
    print(f"After removing duplicates: {len(df_clean)}")
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean


def extract_condition_labels(df):
    """
    Extract individual medical conditions from USES column for multi-label classification
    Similar to data_preprocess.ipynb approach
    """
    print("\n" + "="*60)
    print("Extracting Medical Condition Labels")
    print("="*60)
    
    import re
    
    def extract_conditions(uses_text):
        """
        Extract disease/condition list from USES field
        Splits by newline to get individual conditions
        """
        if pd.isna(uses_text) or uses_text == '':
            return []
        
        # Split by newline (main delimiter in original data)
        conditions = uses_text.split('\n')
        
        # Clean each condition
        cleaned = []
        for cond in conditions:
            cond = cond.strip()
            
            # Remove numbering (e.g., "1. Cancer" -> "Cancer")
            cond = re.sub(r'^\d+\.\s*', '', cond)
            
            # Remove "Treatment of" prefix for consistency
            cond = re.sub(r'^Treatment of\s+', '', cond, flags=re.IGNORECASE)
            
            # Skip very short or empty entries
            if len(cond) > 2:
                cleaned.append(cond)
        
        return cleaned
    
    # Extract condition labels
    df['condition_labels'] = df['USES'].apply(extract_conditions)
    df['num_conditions'] = df['condition_labels'].apply(len)
    
    print(f"Total drugs: {len(df)}")
    print(f"Drugs with at least 1 condition: {(df['num_conditions'] > 0).sum()}")
    print(f"Average conditions per drug: {df['num_conditions'].mean():.2f}")
    print(f"Max conditions for a drug: {df['num_conditions'].max()}")
    
    return df


def analyze_data(df):
    """
    Perform exploratory data analysis on extracted conditions
    """
    print("\n" + "="*60)
    print("Data Analysis")
    print("="*60)
    
    # Basic stats
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check key columns
    print("\n--- CONTAINS (Composition) Analysis ---")
    print(f"Non-empty: {df['CONTAINS'].notna().sum()}")
    print("Sample values:")
    for i, val in enumerate(df['CONTAINS'].head(5)):
        if len(val) > 100:
            print(f"  {i+1}. {val[:100]}...")
        else:
            print(f"  {i+1}. {val}")
    
    print("\n--- USES (Medical Conditions) Analysis ---")
    print(f"Non-empty: {df['USES'].notna().sum()}")
    
    # Show distribution
    df_temp = df[df['num_conditions'] > 0]
    print(f"Drugs with single condition: {(df_temp['num_conditions'] == 1).sum()}")
    print(f"Drugs with multiple conditions: {(df_temp['num_conditions'] > 1).sum()}")
    
    # Count all unique medical conditions
    all_conditions = []
    for cond_list in df['condition_labels']:
        all_conditions.extend(cond_list)
    
    condition_counter = Counter(all_conditions)
    print(f"\nTotal unique medical conditions: {len(condition_counter)}")
    print("\nTop 30 most common conditions:")
    for condition, count in condition_counter.most_common(30):
        display_cond = condition[:60] + "..." if len(condition) > 60 else condition
        print(f"  {display_cond}: {count}")
    
    # Sample records
    print("\n--- Sample Drug Records ---")
    for i in range(min(5, len(df))):
        print(f"\n{i+1}. {df.iloc[i]['NAME']}")
        print(f"   CONTAINS: {df.iloc[i]['CONTAINS'][:80]}...")
        print(f"   CONDITIONS: {df.iloc[i]['condition_labels']}")
        if df.iloc[i]['HOW_WORKS'].strip():
            print(f"   HOW_WORKS: {df.iloc[i]['HOW_WORKS'][:80]}...")
    
    return df


def create_multilabel_dataset(df, min_freq=5):
    """
    Create multi-label classification dataset compatible with 1_data_preprocessing.ipynb
    Filters conditions by minimum frequency and prepares for MultiLabelBinarizer
    """
    from sklearn.preprocessing import MultiLabelBinarizer
    
    print("\n" + "="*60)
    print("Creating Multi-Label Dataset")
    print("="*60)
    
    # Count condition frequency
    all_conditions = []
    for cond_list in df['condition_labels']:
        all_conditions.extend(cond_list)
    
    condition_counts = Counter(all_conditions)
    print(f"\nTotal unique conditions: {len(condition_counts)}")
    
    # Filter by minimum frequency
    frequent_conditions = {cond for cond, count in condition_counts.items() 
                          if count >= min_freq}
    
    print(f"Conditions appearing >= {min_freq} times: {len(frequent_conditions)}")
    
    # Filter labels to only include frequent conditions
    df['filtered_labels'] = df['condition_labels'].apply(
        lambda labels: [l for l in labels if l in frequent_conditions]
    )
    
    # Remove samples without any valid labels
    df_filtered = df[df['filtered_labels'].apply(len) > 0].copy()
    print(f"\nDrugs after filtering: {len(df_filtered)} (from {len(df)})")
    
    # Multi-label binarization
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df_filtered['filtered_labels'])
    
    print(f"\nFinal multi-label dataset:")
    print(f"  Number of samples: {len(df_filtered)}")
    print(f"  Number of labels: {len(mlb.classes_)}")
    print(f"  Label matrix shape: {y.shape}")
    print(f"  Average labels per sample: {y.sum(axis=1).mean():.2f}")
    print(f"  Max labels per sample: {y.sum(axis=1).max()}")
    print(f"  Min labels per sample: {y.sum(axis=1).min()}")
    
    # Show label distribution
    label_counts = y.sum(axis=0)
    print(f"\nLabel frequency distribution:")
    print(f"  Most common label: {label_counts.max()} samples")
    print(f"  Least common label: {label_counts.min()} samples")
    print(f"  Median: {int(pd.Series(label_counts).median())} samples")
    
    return df_filtered, y, mlb


def save_cleaned_csv(df, output_file='MID_cleaned.csv'):
    """
    Save the cleaned Excel data as a proper CSV file without embedded newlines
    """
    print("\n" + "="*60)
    print("Saving cleaned CSV file (no embedded newlines)...")
    print("="*60)
    
    # Save to CSV with proper formatting
    df.to_csv(output_file, index=False, encoding='utf-8', lineterminator='\n')
    print(f"✓ Saved clean CSV to: {output_file}")
    print(f"  Total records: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    
    return output_file


def save_processed_data(df, y, mlb, output_base='MID2'):
    """
    Save processed data compatible with 1_data_preprocessing.ipynb notebook
    """
    import pickle
    import numpy as np
    
    print("\n" + "="*60)
    print("Saving processed data...")
    print("="*60)
    
    # 1. Save full processed CSV
    output_csv = f'{output_base}_processed.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✓ Saved full dataset: {output_csv}")
    
    # 2. Save simplified format for notebook (similar to medicine_processed.csv)
    df_simple = df[['NAME', 'CONTAINS', 'HOW_WORKS', 'filtered_labels']].copy()
    
    # Create input_text (composition + mechanism) similar to notebook format
    df_simple['input_text'] = (
        "Drug Composition: " + df['CONTAINS'].fillna('') + " " +
        "Mechanism of Action: " + df['HOW_WORKS'].fillna('')
    )
    
    output_simple = f'{output_base}_for_notebook.csv'
    df_simple.to_csv(output_simple, index=False, encoding='utf-8')
    print(f"✓ Saved notebook-ready format: {output_simple}")
    
    # 3. Save numpy arrays for ML (X and y)
    X_file = f'X_{output_base}.npy'
    y_file = f'y_{output_base}.npy'
    np.save(X_file, df_simple['input_text'].values)
    np.save(y_file, y)
    print(f"✓ Saved feature matrix: {X_file}")
    print(f"✓ Saved label matrix: {y_file}")
    
    # 4. Save MultiLabelBinarizer
    mlb_file = f'mlb_{output_base}.pkl'
    with open(mlb_file, 'wb') as f:
        pickle.dump(mlb, f)
    print(f"✓ Saved label encoder: {mlb_file}")
    
    # 5. Save statistics JSON
    all_conditions = []
    for cond_list in df['condition_labels']:
        all_conditions.extend(cond_list)
    
    stats = {
        'total_records': len(df),
        'total_unique_conditions': len(set(all_conditions)),
        'filtered_conditions': len(mlb.classes_),
        'avg_conditions_per_drug': float(df['num_conditions'].mean()),
        'max_conditions_per_drug': int(df['num_conditions'].max()),
        'label_matrix_shape': list(y.shape),
        'columns': list(df.columns),
        'top_20_conditions': Counter(all_conditions).most_common(20),
        'sample_records': [
            {
                'name': df.iloc[i]['NAME'],
                'contains': df.iloc[i]['CONTAINS'][:100],
                'conditions': df.iloc[i]['filtered_labels']
            }
            for i in range(min(5, len(df)))
        ]
    }
    
    stats_file = f'{output_base}_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved statistics: {stats_file}")
    
    return stats


def create_training_data_legacy(df, output_file='MID2_training_data_legacy.csv'):
    """
    Create simplified training data in legacy format (for backwards compatibility)
    """
    print("\n" + "="*60)
    print("Creating legacy training data format...")
    print("="*60)
    
    # Create simple input-output pairs
    training_data = pd.DataFrame({
        'drug_name': df['NAME'],
        'composition': df['CONTAINS'],
        'mechanism': df['HOW_WORKS'].fillna(''),
        'chemical_class': df['CHEMICAL_CLASS'].fillna(''),
        'medical_conditions': df['condition_labels'].apply(lambda x: ' | '.join(x)),
        'num_conditions': df['num_conditions']
    })
    
    # Create combined input text
    training_data['input_text'] = (
        training_data['composition'] + ' [SEP] ' + 
        training_data['mechanism'].str[:200]  # Limit mechanism length
    )
    
    training_data.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ Saved legacy training data: {output_file}")
    print(f"  Shape: {training_data.shape}")
    
    return training_data


def main():
    """
    Main execution function - prepares data for multi-label classification
    """
    print("="*60)
    print("MID Data Extraction Pipeline for Multi-Label Classification")
    print("="*60)
    
    # Step 1: Load data from Excel file (preserving newlines in USES)
    df = load_mid2_data('MID.xlsx')
    
    # Step 2: Clean and filter basic data quality
    df_clean = clean_and_filter(df)
    
    # Step 3: Extract condition labels from USES column (split by newlines)
    df_labeled = extract_condition_labels(df_clean)
    
    # Step 4: Analyze extracted conditions
    df_analyzed = analyze_data(df_labeled)
    
    # Step 5: Create multi-label dataset (filter by frequency)
    df_final, y_labels, mlb = create_multilabel_dataset(df_analyzed, min_freq=5)
    
    # Step 6: Save all outputs
    save_processed_data(df_final, y_labels, mlb, output_base='MID2')
    
    # Step 7: Create legacy format (optional)
    create_training_data_legacy(df_final)
    
    print("\n" + "="*60)
    print("✅ Extraction Complete!")
    print("="*60)
    print("\nGenerated files for multi-label classification:")
    print("  1. MID2_processed.csv - Full dataset with extracted labels")
    print("  2. MID2_for_notebook.csv - Simplified format for 1_data_preprocessing.ipynb")
    print("  3. X_MID2.npy - Feature matrix (drug compositions + mechanisms)")
    print("  4. y_MID2.npy - Multi-label matrix (binary encoded conditions)")
    print("  5. mlb_MID2.pkl - MultiLabelBinarizer (for label encoding/decoding)")
    print("  6. MID2_stats.json - Dataset statistics")
    print("  7. MID2_training_data_legacy.csv - Legacy format")
    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("1. Load data in 1_data_preprocessing.ipynb:")
    print("   df = pd.read_csv('MID2_for_notebook.csv')")
    print("   X = np.load('X_MID2.npy')")
    print("   y = np.load('y_MID2.npy')")
    print("   mlb = pickle.load(open('mlb_MID2.pkl', 'rb'))")
    print("\n2. Use biomedical NER pipeline on input_text column")
    print("\n3. Train multi-label classifier with extracted features")
    print("="*60)


if __name__ == "__main__":
    main()
