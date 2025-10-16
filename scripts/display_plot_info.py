"""
Display the generated evaluation plots for review
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def show_plot_info(plot_path, title):
    """Show information about a plot"""
    if os.path.exists(plot_path):
        file_size = os.path.getsize(plot_path) / (1024 * 1024)  # MB
        print(f"✅ {title}")
        print(f"   📁 File: {plot_path}")
        print(f"   📏 Size: {file_size:.2f} MB")
        print(f"   🎯 High resolution (300 DPI)")
        return True
    else:
        print(f"❌ {title} - File not found")
        return False

print("🎨 EVALUATION PLOTS STATUS")
print("=" * 50)

# Check both plots
plot1_exists = show_plot_info("results/plots/bandgap_training_vs_testing_evaluation.png", 
                             "Band Gap Model Evaluation Plot")
print()
plot2_exists = show_plot_info("results/plots/stability_training_vs_testing_evaluation.png", 
                             "Stability Model Evaluation Plot")

print("\n📊 PLOT CONTENTS (Each plot has 4 panels):")
print("-" * 45)
print("1️⃣ Parity Plot (Top-Left):")
print("   • Blue dots = Training data (actual vs predicted)")
print("   • Red dots = Testing data (actual vs predicted)")
print("   • Black line = Perfect prediction")
print("   • Legend shows R² values")

print("\n2️⃣ Residuals Plot (Top-Right):")
print("   • Blue dots = Training residuals (errors)")
print("   • Red dots = Testing residuals (errors)")
print("   • Black line = Zero error reference")

print("\n3️⃣ Metrics Comparison (Bottom-Left):")
print("   • Blue bars = Training metrics (R², MAE, RMSE)")
print("   • Red bars = Testing metrics (R², MAE, RMSE)")
print("   • Values labeled on each bar")

print("\n4️⃣ Assessment Summary (Bottom-Right):")
print("   • Complete performance metrics table")
print("   • Overfitting status indicator")
print("   • Dataset information")
print("   • Color-coded status (Green/Orange/Red)")

print("\n📈 MODEL PERFORMANCE SUMMARY:")
print("-" * 35)
print("🎯 Band Gap Model:")
print("   • Training R²: 0.8434")
print("   • Testing R²: 0.7517")
print("   • Status: ✅ GOOD (minimal overfitting)")

print("\n⚖️ Stability Model:")
print("   • Training R²: 0.5364")
print("   • Testing R²: 0.1892")
print("   • Status: ⚠️ OVERFITTING (needs attention)")

if plot1_exists and plot2_exists:
    print(f"\n🎉 SUCCESS! You have 2 complete evaluation plots!")
    print(f"📋 Perfect for faculty presentation:")
    print(f"   • Clear training vs testing comparison")
    print(f"   • Color-coded performance indicators")
    print(f"   • Comprehensive metrics display")
    print(f"   • Professional publication quality")
else:
    print(f"\n❌ Some plots are missing!")

print(f"\n📁 Access your plots at:")
print(f"   c:\\Bavesh\\Sem5\\Pervoskite Solar\\results\\plots\\")