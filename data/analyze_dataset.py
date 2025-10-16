import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Perovskite_data.csv')

print('=' * 60)
print('PEROVSKITE MATERIALS DATASET ANALYSIS')
print('=' * 60)

print(f'\n📊 DATASET OVERVIEW:')
print(f'   • Total Materials: {df.shape[0]:,}')
print(f'   • Total Features: {df.shape[1]}')
print(f'   • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')

print(f'\n📋 COLUMN DETAILS:')
for i, col in enumerate(df.columns, 1):
    print(f'   {i:2d}. {col}')

print(f'\n🔍 DATA TYPES:')
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f'   • {dtype}: {count} columns')

print(f'\n❌ MISSING VALUES:')
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    for col, count in missing_cols.items():
        pct = (count / len(df)) * 100
        print(f'   • {col}: {count} ({pct:.1f}%)')
else:
    print('   • No missing values found!')

print(f'\n📈 KEY STATISTICS:')
print(f'   • Band Gap Range: {df["band_gap (eV)"].min():.3f} - {df["band_gap (eV)"].max():.3f} eV')
print(f'   • Stability Range: {df["energy_above_hull (eV/atom)"].min():.3f} - {df["energy_above_hull (eV/atom)"].max():.3f} eV/atom')
print(f'   • Stable Materials: {df["stable"].sum()} ({df["stable"].sum()/len(df)*100:.1f}%)')

print(f'\n🔬 CRYSTAL SYSTEMS:')
crystal_systems = df['crystal_system'].value_counts()
for system, count in crystal_systems.items():
    print(f'   • {system}: {count} materials')

print(f'\n⚛️  COMPOSITION ANALYSIS:')
compositions = df['composition'].value_counts()
print(f'   • Unique Compositions: {len(compositions)}')
print(f'   • Most Common: {compositions.index[0]} ({compositions.iloc[0]} materials)')

print(f'\n🎯 SOLAR CELL RELEVANCE:')
ideal_bandgap = df[(df['band_gap (eV)'] >= 1.1) & (df['band_gap (eV)'] <= 1.7)]
stable_materials = df[df['energy_above_hull (eV/atom)'] <= 0.1]
promising = df[(df['band_gap (eV)'] >= 1.1) & (df['band_gap (eV)'] <= 1.7) & (df['energy_above_hull (eV/atom)'] <= 0.1)]

print(f'   • Ideal Band Gap (1.1-1.7 eV): {len(ideal_bandgap)} materials ({len(ideal_bandgap)/len(df)*100:.1f}%)')
print(f'   • Stable Materials (≤0.1 eV/atom): {len(stable_materials)} materials ({len(stable_materials)/len(df)*100:.1f}%)')
print(f'   • Promising Candidates: {len(promising)} materials ({len(promising)/len(df)*100:.1f}%)')
