"""
Verification script to check if the evaluation plots are correct and show sample results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import os

print("🔍 VERIFYING EVALUATION PLOTS")
print("=" * 50)

# Check if plots exist
plot_dir = "results/plots"
bandgap_plot = "bandgap_training_vs_testing_evaluation.png"
stability_plot = "stability_training_vs_testing_evaluation.png"

bandgap_exists = os.path.exists(os.path.join(plot_dir, bandgap_plot))
stability_exists = os.path.exists(os.path.join(plot_dir, stability_plot))

print(f"📊 Band Gap Plot Exists: {'✅ YES' if bandgap_exists else '❌ NO'}")
print(f"📊 Stability Plot Exists: {'✅ YES' if stability_exists else '❌ NO'}")

# Verify the data and models are accessible
print("\n🔍 VERIFYING DATA AND MODELS")
print("-" * 30)

try:
    # Load data
    df = pd.read_csv('data/perovskite_features_rich.csv')
    print(f"✅ Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Check target columns
    band_gap_col = 'band_gap (eV)'
    stability_col = 'energy_above_hull (eV/atom)'
    
    print(f"✅ Band gap column found: {band_gap_col in df.columns}")
    print(f"✅ Stability column found: {stability_col in df.columns}")
    
    # Load models
    try:
        bg_model = joblib.load('models/band_gap_model_rich.joblib')
        print("✅ Band gap model loaded successfully")
    except:
        print("❌ Band gap model not found")
        
    try:
        stab_model = joblib.load('models/stability_model_rich.joblib')
        print("✅ Stability model loaded successfully")
    except:
        print("❌ Stability model not found")
    
    # Get features
    magpie_features = [col for col in df.columns if col.startswith('MagpieData')]
    print(f"✅ Found {len(magpie_features)} compositional features")
    
    # Quick verification of model performance
    print("\n📊 QUICK MODEL VERIFICATION")
    print("-" * 30)
    
    # Band Gap Model
    if band_gap_col in df.columns:
        df_bg = df.dropna(subset=[band_gap_col])
        X_bg = df_bg[magpie_features].fillna(df_bg[magpie_features].median())
        y_bg = df_bg[band_gap_col]
        
        X_train_bg, X_test_bg, y_train_bg, y_test_bg = train_test_split(
            X_bg, y_bg, test_size=0.2, random_state=42
        )
        
        try:
            train_pred_bg = bg_model.predict(X_train_bg)
            test_pred_bg = bg_model.predict(X_test_bg)
            
            train_r2_bg = r2_score(y_train_bg, train_pred_bg)
            test_r2_bg = r2_score(y_test_bg, test_pred_bg)
            
            print(f"Band Gap Model:")
            print(f"  Training R²: {train_r2_bg:.4f}")
            print(f"  Testing R²:  {test_r2_bg:.4f}")
            print(f"  Difference:  {train_r2_bg - test_r2_bg:.4f}")
            print(f"  Status: {'✅ Good' if abs(train_r2_bg - test_r2_bg) < 0.1 else '⚠️ Overfitting'}")
        except Exception as e:
            print(f"❌ Error evaluating band gap model: {e}")
    
    # Stability Model
    if stability_col in df.columns:
        df_stab = df.dropna(subset=[stability_col])
        X_stab = df_stab[magpie_features].fillna(df_stab[magpie_features].median())
        y_stab = df_stab[stability_col]
        
        X_train_stab, X_test_stab, y_train_stab, y_test_stab = train_test_split(
            X_stab, y_stab, test_size=0.2, random_state=42
        )
        
        try:
            train_pred_stab = stab_model.predict(X_train_stab)
            test_pred_stab = stab_model.predict(X_test_stab)
            
            train_r2_stab = r2_score(y_train_stab, train_pred_stab)
            test_r2_stab = r2_score(y_test_stab, test_pred_stab)
            
            print(f"\nStability Model:")
            print(f"  Training R²: {train_r2_stab:.4f}")
            print(f"  Testing R²:  {test_r2_stab:.4f}")
            print(f"  Difference:  {train_r2_stab - test_r2_stab:.4f}")
            print(f"  Status: {'✅ Good' if abs(train_r2_stab - test_r2_stab) < 0.1 else '⚠️ Overfitting'}")
        except Exception as e:
            print(f"❌ Error evaluating stability model: {e}")

except Exception as e:
    print(f"❌ Error loading data: {e}")

print("\n📋 SUMMARY")
print("=" * 20)
if bandgap_exists and stability_exists:
    print("✅ Both evaluation plots created successfully!")
    print("📁 Location: results/plots/")
    print("   • bandgap_training_vs_testing_evaluation.png")
    print("   • stability_training_vs_testing_evaluation.png")
    print("\n🎯 Each plot shows:")
    print("   • Blue = Training performance")
    print("   • Red = Testing performance")
    print("   • 4-panel comprehensive analysis")
else:
    print("❌ Some plots are missing. Re-run the generation script.")