#!/usr/bin/env python3
"""
Script to format and validate field_analysis.csv for error location visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path

def standardize_model_name(x):
    """Standardize model names for consistency"""
    if pd.isna(x):
        return 'Unknown'
    x = str(x).lower()
    if "nuextract" in x:
        return "NuExtract"
    elif "gpt" in x:
        return "GPT-4.1"
    elif "llama" in x: 
        return "Llama3.1"
    elif "mistral" in x:
        return "Mistral"
    elif "gemma" in x:
        return "Gemma"
    elif "biomed" in x:
        return "BioMed GliNER"
    elif "gliner" in x:
        return "GliNER"
    return x

def is_valid_field(field_name):
    """Check if a field name is valid (represents an actual data field)"""
    valid_fields = [
        "date_collected", "date_received", "date_verified", "report_type",
        "testing_context", "ordering_clinic", "testing_laboratory", 
        "sequencing_scope", "sample_type", "analysis_type", "num_tested_genes",
        "refseq_mrna", "num_variants", "reference_genome", "gene_symbol",
        "variant_id", "chromosome", "hgvsg", "hgvsc", "hgvsp", "transcript_id",
        "exon", "zygosity", "interpretation", "mafac", "mafan", "mafaf",
        "mega_hgvs", "type", "total"
    ]
    
    if pd.isna(field_name):
        return False
    
    field_clean = str(field_name).strip().lower()
    # Remove prefixes like fp_, fn_, _fp, _fn
    field_clean = field_clean.replace("fp_", "").replace("fn_", "")
    field_clean = field_clean.replace("_fp", "").replace("_fn", "")
    field_clean = field_clean.strip()
    
    return field_clean in valid_fields

def format_field_analysis_data(input_file, output_file=None):
    """
    Format the field_analysis.csv file to have the correct structure
    
    Expected output format:
    - model: standardized model names
    - field: field names where errors occurred
    """
    
    print(f"Reading {input_file}...")
    
    try:
        # Read the file in chunks to handle large files
        chunk_size = 10000
        chunks = []
        
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            chunks.append(chunk)
            print(f"Processed chunk of {len(chunk)} rows")
        
        df = pd.concat(chunks, ignore_index=True)
        print(f"Total rows loaded: {len(df)}")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Original shape: {df.shape}")
    
    # Try to identify model and field columns
    model_col = None
    field_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        if 'model' in col_lower or 'llm' in col_lower:
            model_col = col
            print(f"Found model column: {col}")
        elif any(keyword in col_lower for keyword in ['field', 'error', 'missing', 'fn_', 'fp_']):
            if is_valid_field(col):
                field_cols.append(col)
    
    print(f"Detected model column: {model_col}")
    print(f"Detected field columns: {field_cols[:10]}...")  # Show first 10
    
    if not model_col:
        print("Could not find model column. Looking for columns containing model data...")
        # Try to infer from data content
        for col in df.columns:
            sample_values = df[col].dropna().astype(str).head(10).tolist()
            if any(any(model_name in val.lower() for model_name in 
                      ['gpt', 'llama', 'mistral', 'gemma', 'gliner', 'nuextract']) 
                   for val in sample_values):
                model_col = col
                print(f"Inferred model column from content: {col}")
                break
    
    if not model_col:
        print("ERROR: Could not identify model column")
        return None
    
    # Create formatted dataframe
    formatted_data = []
    
    print("Processing data...")
    
    for idx, row in df.iterrows():
        if int(idx) % 1000 == 0: # type: ignore
            print(f"Processing row {idx}...")
        
        model_name = standardize_model_name(row[model_col])
        
        # For each field column that has an error indicator
        for field_col in field_cols:
            # Check if this field has an error for this row
            field_value = row[field_col]
            
            # Determine if there's an error (you may need to adjust this logic)
            # Common patterns: non-null values, True values, positive numbers
            has_error = False
            
            if pd.notna(field_value): # type: ignore
                if isinstance(field_value, bool):
                    has_error = field_value
                elif isinstance(field_value, (int, float)):
                    has_error = field_value > 0
                elif isinstance(field_value, str):
                    has_error = field_value.lower() not in ['false', '0', 'none', 'null', '']
                else:
                    has_error = True
            
            if has_error: # type: ignore
                # Clean field name
                clean_field = field_col.lower().strip()
                clean_field = clean_field.replace("fp_", "").replace("fn_", "")
                clean_field = clean_field.replace("_fp", "").replace("_fn", "")
                clean_field = clean_field.strip()
                
                formatted_data.append({
                    'model': model_name,
                    'field': clean_field
                })
    
    if not formatted_data:
        print("No error data found. Please check the data format.")
        return None
    
    # Create final dataframe
    result_df = pd.DataFrame(formatted_data)
    
    print(f"\nFormatted data summary:")
    print(f"Total error records: {len(result_df)}")
    print(f"Unique models: {result_df['model'].nunique()}")
    print(f"Unique fields: {result_df['field'].nunique()}")
    print(f"\nModel distribution:")
    print(result_df['model'].value_counts())
    print(f"\nTop 10 error fields:")
    print(result_df['field'].value_counts().head(10))
    
    # Save formatted data
    if output_file is None:
        output_file = input_file.replace('.csv', '_formatted.csv')
    
    result_df.to_csv(output_file, index=False)
    print(f"\nFormatted data saved to: {output_file}")
    
    return result_df

def create_sample_field_data():
    """Create a sample field_analysis.csv file with the correct format"""
    
    models = ["GPT-4.1", "Llama3.1", "NuExtract", "Mistral", "Gemma", "GliNER", "BioMed GliNER"]
    fields = [
        "date_collected", "date_received", "gene_symbol", "hgvsg", "hgvsc", 
        "chromosome", "interpretation", "variant_id", "zygosity", "transcript_id"
    ]
    
    # Generate sample error data
    sample_data = []
    np.random.seed(42)  # For reproducible results
    
    for _ in range(1000):  # Generate 1000 error records
        model = np.random.choice(models)
        field = np.random.choice(fields, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
        sample_data.append({'model': model, 'field': field})
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('field_analysis_sample.csv', index=False)
    print("Sample field_analysis.csv created with correct format")
    return sample_df

if __name__ == "__main__":
    import sys
    
    input_file = "/Users/ayu/PDF_benchmarking/graphs/field_analysisfinal.csv"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    if not Path(input_file).exists():
        print(f"File {input_file} not found.")
        print("Creating a sample file to show the expected format...")
        create_sample_field_data()
    else:
        print(f"Formatting {input_file}...")
        formatted_df = format_field_analysis_data(input_file)
        
        if formatted_df is not None:
            print("Formatting completed successfully!")
        else:
            print("Formatting failed. Please check the input data format.")