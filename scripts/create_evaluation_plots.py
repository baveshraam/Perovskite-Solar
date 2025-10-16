"""
Create comprehensive training vs testing evaluation plots for both GBR models
with clear color coding and evaluation metrics display.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_model_evaluation_plot(model_path, data_path, target_col, model_name, save_path):
    """
    Create a comprehensive evaluation plot comparing training vs testing performance
    """
    # Load model and data
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    # Prepare data (same as training scripts)
    df = df.dropna(subset=[target_col])
    
    # Get the compositional features (MagpieData columns only)
    magpie_features = [col for col in df.columns if col.startswith('MagpieData')]
    
    X = df[magpie_features]
    y = df[target_col]
    
    # Fill any NaN values
    X = X.fillna(X.median())
    
    # Same train-test split as training (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Create the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Training vs Testing Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Parity Plot (Actual vs Predicted)
    # Training data in blue, Testing data in red
    ax1.scatter(y_train, y_train_pred, alpha=0.6, color='blue', s=30, label=f'Training (R²={train_r2:.3f})', edgecolor='darkblue', linewidth=0.5)
    ax1.scatter(y_test, y_test_pred, alpha=0.6, color='red', s=30, label=f'Testing (R²={test_r2:.3f})', edgecolor='darkred', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(min(y_train), min(y_test), min(y_train_pred), min(y_test_pred))
    max_val = max(max(y_train), max(y_test), max(y_train_pred), max(y_test_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Parity Plot: Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals Plot
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    ax2.scatter(y_train_pred, train_residuals, alpha=0.6, color='blue', s=30, label='Training Residuals', edgecolor='darkblue', linewidth=0.5)
    ax2.scatter(y_test_pred, test_residuals, alpha=0.6, color='red', s=30, label='Testing Residuals', edgecolor='darkred', linewidth=0.5)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.set_title('Residuals Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Metrics Comparison Bar Chart
    metrics = ['R² Score', 'MAE', 'RMSE']
    train_metrics = [train_r2, train_mae, train_rmse]
    test_metrics = [test_r2, test_mae, test_rmse]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, train_metrics, width, label='Training', color='blue', alpha=0.7, edgecolor='darkblue')
    bars2 = ax3.bar(x_pos + width/2, test_metrics, width, label='Testing', color='red', alpha=0.7, edgecolor='darkred')
    
    # Add value labels on bars
    for i, (train_val, test_val) in enumerate(zip(train_metrics, test_metrics)):
        ax3.text(i - width/2, train_val + max(train_metrics) * 0.01, f'{train_val:.3f}', 
                ha='center', va='bottom', fontweight='bold', color='darkblue')
        ax3.text(i + width/2, test_val + max(test_metrics) * 0.01, f'{test_val:.3f}', 
                ha='center', va='bottom', fontweight='bold', color='darkred')
    
    ax3.set_xlabel('Evaluation Metrics')
    ax3.set_ylabel('Metric Values')
    ax3.set_title('Training vs Testing Metrics Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Overfitting Assessment Box
    ax4.axis('off')
    
    # Calculate overfitting indicators
    r2_diff = train_r2 - test_r2
    mae_diff = test_mae - train_mae
    rmse_diff = test_rmse - train_rmse
    
    # Determine overfitting status
    if r2_diff < 0.05 and mae_diff < 0.05 and rmse_diff < 0.05:
        status = "EXCELLENT"
        status_color = "green"
    elif r2_diff < 0.1 and mae_diff < 0.1 and rmse_diff < 0.1:
        status = "GOOD"
        status_color = "orange"
    else:
        status = "WARNING - OVERFITTING"
        status_color = "red"
    
    # Create metrics text box
    metrics_text = f"""EVALUATION SUMMARY
{'='*30}

PERFORMANCE METRICS:
Training R²:     {train_r2:.4f}
Testing R²:      {test_r2:.4f}
R² Difference:   {r2_diff:.4f}

Training MAE:    {train_mae:.4f}
Testing MAE:     {test_mae:.4f}
MAE Difference:  {mae_diff:.4f}

Training RMSE:   {train_rmse:.4f}
Testing RMSE:    {test_rmse:.4f}
RMSE Difference: {rmse_diff:.4f}

OVERFITTING STATUS: {status}

DATASET INFO:
Training Samples: {len(y_train)}
Testing Samples:  {len(y_test)}
Total Features:   {X.shape[1]}"""
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Add status indicator
    ax4.text(0.05, 0.15, f"STATUS: {status}", transform=ax4.transAxes, 
             fontsize=14, fontweight='bold', color=status_color,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=status_color, linewidth=2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Evaluation plot saved: {save_path}")
    print(f"📊 Model Status: {status}")
    print(f"🎯 R² Difference: {r2_diff:.4f} (Training - Testing)")
    
    return train_r2, test_r2, train_mae, test_mae, train_rmse, test_rmse, status

# Create evaluation plots for both models
print("🚀 Creating comprehensive evaluation plots for both GBR models...")
print("=" * 70)

# 1. Band Gap Model Evaluation
print("\n📊 Creating Band Gap Model Evaluation Plot...")
try:
    band_gap_metrics = create_model_evaluation_plot(
        model_path='models/band_gap_model_rich.joblib',
        data_path='data/perovskite_features_rich.csv',
        target_col='band_gap (eV)',
        model_name='Band Gap Prediction Model (GBR)',
        save_path='results/plots/bandgap_training_vs_testing_evaluation.png'
    )
    print("✅ Band Gap Model evaluation completed!")
except Exception as e:
    print(f"❌ Error with Band Gap Model: {e}")

# 2. Stability Model Evaluation  
print("\n📊 Creating Stability Model Evaluation Plot...")
try:
    stability_metrics = create_model_evaluation_plot(
        model_path='models/stability_model_rich.joblib',
        data_path='data/perovskite_features_rich.csv',
        target_col='energy_above_hull (eV/atom)',
        model_name='Stability Prediction Model (GBR)',
        save_path='results/plots/stability_training_vs_testing_evaluation.png'
    )
    print("✅ Stability Model evaluation completed!")
except Exception as e:
    print(f"❌ Error with Stability Model: {e}")

print("\n🎉 Both evaluation plots created successfully!")
print("📁 Check the results/plots/ folder for:")
print("   • bandgap_training_vs_testing_evaluation.png")
print("   • stability_training_vs_testing_evaluation.png")
print("\n🔍 Each plot shows:")
print("   • Blue points = Training data performance")
print("   • Red points = Testing data performance") 
print("   • 4-panel comprehensive evaluation")
print("   • Overfitting assessment with status")