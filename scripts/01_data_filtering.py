"""
01_data_filtering.py
====================
This script filters the raw perovskite data to identify materials with ABX3 structure.

The ABX3 perovskite structure is characterized by:
- A: Large cation (typically alkali metals, alkaline earth metals, or organic cations)
- B: Smaller cation (typically transition metals or post-transition metals)  
- X: Anion (typically halides or oxygen)
- Stoichiometry: Exactly 1:1:3 ratio (one A, one B, three X atoms)

This filtering ensures we only work with true perovskite materials for solar cell applications.
"""

import pandas as pd
import re

print("Starting data filtering for ABX3 perovskite structures...")
print("=" * 60)

df_raw = pd.read_csv('data/Perovskite_data.csv')
print(f"Loaded {len(df_raw)} raw materials from database")

def is_perovskite_formula(formula):
    """
    Check if a chemical formula follows ABX3 structure.
    
    This function parses chemical formulas to identify true perovskite materials:
    1. Splits the formula into individual elements with their counts
    2. Uses regex to extract element names and stoichiometric coefficients
    3. Verifies that there are exactly 3 different elements
    4. Checks that the stoichiometric ratios are exactly [1, 1, 3]
    
    Args:
        formula (str): Chemical formula string (e.g., "Ac1 Mn1 O3")
        
    Returns:
        bool: True if the formula represents an ABX3 perovskite structure
        
    Examples:
        >>> is_perovskite_formula("Ac1 Mn1 O3")
        True
        >>> is_perovskite_formula("Ca1 Ti1 O3")  
        True
        >>> is_perovskite_formula("Ca2 Ti1 O4")  # Not 1:1:3 ratio
        False
    """
    # Split formula into individual element components
    elements = formula.strip().split()
    
    # Must have exactly 3 different elements for ABX3 structure
    if len(elements) != 3:
        return False
    
    element_counts = {}
    
    # Parse each element component using regex to extract element and count
    for element in elements:
        # Match pattern: Element name (letters) followed by count (digits)
        match = re.match(r'([A-Za-z]+)(\d+)', element)
        if not match:
            return False
        elem, count = match.groups()
        element_counts[elem] = int(count)
    
    # Extract the stoichiometric counts and sort them
    counts = list(element_counts.values())
    counts.sort()
    
    # Check if counts match the ABX3 pattern: [1, 1, 3]
    return counts == [1, 1, 3]

def extract_abx_sites(formula):
    """
    Extract A, B, and X site elements from a validated ABX3 chemical formula.
    
    In perovskite materials:
    - A-site: Large cation (count = 1)
    - B-site: Smaller cation (count = 1) 
    - X-site: Anion (count = 3)
    
    Args:
        formula (str): Validated ABX3 chemical formula string
        
    Returns:
        tuple: (A_site_element, B_site_element, X_site_element)
        
    Note:
        This function assumes the formula has already been validated 
        as ABX3 structure by is_perovskite_formula()
    """
    elements = formula.strip().split()
    element_counts = {}
    
    # Parse element counts from formula
    for element in elements:
        match = re.match(r'([A-Za-z]+)(\d+)', element)
        elem, count = match.groups()
        element_counts[elem] = int(count)
    
    # Initialize site assignments
    a_site = None
    b_site = None
    x_site = None
    
    # Assign elements to sites based on stoichiometric counts
    for elem, count in element_counts.items():
        if count == 3:
            x_site = elem  # X-site always has count of 3
        elif count == 1:
            # Two elements have count of 1 (A and B sites)
            if a_site is None:
                a_site = elem
            else:
                b_site = elem
    
    return a_site, b_site, x_site


# Apply ABX3 structure filter to identify true perovskite materials
print(f"\n🔍 Applying ABX3 structure filter...")
print(f"   Looking for materials with exactly 1:1:3 stoichiometry...")

perovskite_mask = df_raw['composition'].apply(is_perovskite_formula)
filtered_df = df_raw[perovskite_mask].copy()

print(f"   ✅ Found {len(filtered_df)} materials with ABX3 structure")
print(f"   📉 Filtered out {len(df_raw) - len(filtered_df)} non-perovskite materials")

# Extract A, B, and X site elements for each perovskite material
print(f"\n🧪 Extracting A, B, and X site elements...")
a_sites, b_sites, x_sites = zip(*filtered_df['composition'].apply(extract_abx_sites))
filtered_df['A_site'] = a_sites
filtered_df['B_site'] = b_sites
filtered_df['X_site'] = x_sites

print(f"   ✅ Successfully identified site assignments for all materials")


# Select and organize columns for the filtered dataset
print(f"\n📋 Organizing columns for filtered dataset...")

# Define columns to keep for downstream analysis
columns_to_keep = [
    'mp_id',                        # Materials Project identifier
    'composition',                  # Chemical formula with stoichiometry
    'a_edge (angstrom)',           # Unit cell parameters
    'b_edge (angstrom)',
    'c_edge (angstrom)',
    'alpha_ang (deg)',
    'beta_ang (deg)',
    'gamma_ang (deg)',
    'crystal_system',              # Crystal structure information
    'space_group',
    'formation_energy (eV/atom)',  # Thermodynamic properties
    'energy_above_hull (eV/atom)', # Stability metric (critical for synthesis)
    'band_gap (eV)',               # Electronic property (critical for solar cells)
    'A_site',                      # Site assignments for feature engineering
    'B_site',
    'X_site'
]

# Create final filtered dataset with selected columns
final_df = filtered_df[columns_to_keep]

# Save the filtered dataset for feature engineering
final_df.to_csv('data/perovskite_filtered.csv', index=False)

print(f"✅ Found {len(final_df)} perovskite materials")
print(f"✅ Saved filtered data to 'data/perovskite_filtered.csv'")
print(f"✅ Data filtering complete!")
print(f"\n📊 Summary:")
print(f"   Original materials: {len(df_raw):,}")
print(f"   ABX3 perovskites: {len(final_df):,}")
print(f"   Filtering efficiency: {len(final_df)/len(df_raw)*100:.1f}%")
