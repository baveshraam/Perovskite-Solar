"""
02_feature_engineering.py
=========================
This script generates rich compositional features using matminer for the 
filtered perovskite dataset. It uses ElementProperty featurizer with the 
'magpie' preset to generate comprehensive elemental properties.

The script processes chemical formulas and generates features based on 
elemental properties like atomic mass, radius, electronegativity, etc.
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PEROVSKITE FEATURE ENGINEERING WITH MATMINER")
print("=" * 80)

# Import matminer libraries
print("Importing matminer libraries...")
MATMINER_AVAILABLE = True

try:
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    print("Matminer libraries imported successfully")
except ImportError as e:
    print(f"Matminer import failed: {e}")
    print("Please install matminer: pip install matminer")
    MATMINER_AVAILABLE = False

# Load the filtered dataset
print("\nLoading filtered perovskite dataset...")
try:
    df = pd.read_csv('data/perovskite_filtered.csv')
    print(f"   Loaded {len(df)} perovskite materials")
except FileNotFoundError:
    print("   Error: perovskite_filtered.csv not found!")
    print("   Please run 01_data_filtering.py first")
    exit(1)

# Check if matminer is available before proceeding
if not MATMINER_AVAILABLE:
    print("Cannot proceed without matminer. Exiting...")
    exit(1)

# Initialize the matminer featurizer with magpie preset
print(f"\nInitializing matminer ElementProperty featurizer...")
try:
    featurizer = ElementProperty.from_preset("magpie")
    print(f"   Featurizer will generate {len(featurizer.feature_labels())} features")
except Exception as e:
    print(f"   Error initializing featurizer: {e}")
    exit(1)

# Prepare chemical formulas for featurization
print(f"\nPreparing chemical formulas for featurization...")

def create_formula_column(df):
    """Create a formula column from composition data"""
    formulas = []
    
    for _, row in df.iterrows():
        try:
            # Create formula from composition
            if 'composition' in df.columns and pd.notna(row['composition']):
                formula = str(row['composition'])
            else:
                # Build from A, B, X site information if available
                A_site = str(row.get('A_site', ''))
                B_site = str(row.get('B_site', ''))
                X_site = str(row.get('X_site', ''))
                formula = f"{A_site}{B_site}{X_site}3"  # ABX3 structure
            
            formulas.append(formula)
        except Exception:
            formulas.append(None)
    
    return formulas

df['formula'] = create_formula_column(df)
print(f"   Created formula column for {len(df)} materials")
print(f"   Example: '{df['composition'].iloc[0]}' -> '{df['formula'].iloc[0]}'")

# Generate rich compositional features
print(f"\nGenerating rich compositional features...")
print("   This may take several minutes for large datasets...")

start_time = time.time()
feature_data = []
failed_materials = []

for idx, row in df.iterrows():
    try:
        formula = row['formula']
        if pd.isna(formula) or formula == 'nan':
            failed_materials.append((idx, formula, "Invalid formula"))
            continue
            
        # Create Composition object
        try:
            comp = Composition(formula)
        except Exception as e:
            failed_materials.append((idx, formula, f"Composition error: {str(e)[:30]}"))
            continue
        
        # Generate features
        try:
            features = featurizer.featurize(comp)
            
            # Create row with original data + new features
            new_row = row.to_dict()
            feature_labels = featurizer.feature_labels()
            
            for label, value in zip(feature_labels, features):
                new_row[label] = value
                
            feature_data.append(new_row)
            
        except Exception as e:
            failed_materials.append((idx, formula, f"Featurization error: {str(e)[:30]}"))
            continue
            
    except Exception as e:
        print(f"   Failed to process {formula}: {str(e)[:50]}...")
        failed_materials.append((idx, formula, f"General error: {str(e)[:30]}"))
        continue

# Create DataFrame with rich features
if feature_data:
    df_rich_features = pd.DataFrame(feature_data)
    print(f"   Successfully generated features for {len(df_rich_features)} materials")
else:
    print(f"   No materials were successfully processed!")
    exit(1)

# Report failed materials
if failed_materials:
    print(f"   {len(failed_materials)} materials failed featurization:")
    for idx, formula, error in failed_materials[:5]:  # Show first 5
        print(f"      {idx}: {formula} - {error}")
    if len(failed_materials) > 5:
        print(f"      ... and {len(failed_materials) - 5} more")

total_time = time.time() - start_time
print(f"   Total processing time: {total_time:.1f} seconds")

# Feature generation summary
print(f"\nFeature Generation Summary:")
print(f"   Original materials: {len(df):,}")
print(f"   Successfully processed: {len(df_rich_features):,}")
print(f"   Failed materials: {len(failed_materials):,}")
print(f"   Total features generated: {len(featurizer.feature_labels())}")

# Clean and organize final dataset
print(f"\nCleaning and organizing final dataset...")

# Check for NaN/inf values in numeric columns
numeric_columns = df_rich_features.select_dtypes(include=[np.number]).columns
total_issues = 0

for col in numeric_columns:
    issues = df_rich_features[col].isna().sum() + np.isinf(df_rich_features[col]).sum()
    total_issues += issues

if total_issues > 0:
    print(f"   Found {total_issues} NaN/inf values in numeric columns")
    print(f"   Replacing NaN/inf values with median values...")
    
    for col in numeric_columns:
        median_val = df_rich_features[col].median()
        df_rich_features[col] = df_rich_features[col].replace([np.inf, -np.inf], np.nan)
        df_rich_features[col] = df_rich_features[col].fillna(median_val)
else:
    print(f"   No NaN/inf values found in numeric columns")

# Save the enhanced dataset
print(f"\nSaving enhanced dataset...")
output_file = 'data/perovskite_features_rich.csv'
df_rich_features.to_csv(output_file, index=False)
print(f"   Saved to: {output_file}")
print(f"   Dataset shape: {df_rich_features.shape}")
print(f"   Ready for machine learning model training")

# Display sample of generated features
if len(df_rich_features) > 0:
    matminer_features = [col for col in df_rich_features.columns if col.startswith('MagpieData')]
    print(f"\nSample of matminer features generated:")
    for i, feature in enumerate(matminer_features[:10], 1):
        sample_val = df_rich_features[feature].iloc[0]
        print(f"   {i:2d}. {feature:<40} = {sample_val:.4f}")
    if len(matminer_features) > 10:
        print(f"   ... and {len(matminer_features) - 10} more matminer features")

print(f"\n" + "=" * 80)
print(f"COMPREHENSIVE FEATURE ENGINEERING COMPLETE!")
print(f"=" * 80)
print(f"Enhanced dataset with {len(matminer_features)} additional features")
print(f"Ready for advanced machine learning model training")