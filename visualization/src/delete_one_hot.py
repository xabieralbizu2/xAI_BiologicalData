import pandas as pd
import os

files = [
    "dataset_onehot.csv"
]

# Prefixes identified from previous context
one_hot_prefixes = ['Teeth_', 'Upper/Lower_', 'Side_', 'Sex_', 'Localización_']

created_files = []

for filename in files:
    try:
        df = pd.read_csv(filename)
        
        # Identify columns to drop
        cols_to_drop = []
        for col in df.columns:
            for prefix in one_hot_prefixes:
                if col.startswith(prefix):
                    cols_to_drop.append(col)
                    break
        
        # Drop columns
        df_clean = df.drop(columns=cols_to_drop)
        
        # Save new file
        new_filename = filename.replace('.csv', '_no_onehot.csv')
        df_clean.to_csv(new_filename, index=False)
        created_files.append(new_filename)
        
        print(f"Processed {filename}: Removed {len(cols_to_drop)} columns. Remaining: {len(df_clean.columns)}")
        print(f"Sample remaining columns: {df_clean.columns.tolist()[:5]}")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")