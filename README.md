# Perovskite Solar

This repository contains data, models, and scripts for predicting perovskite solar cell properties (band gap and stability) and for running optimization experiments.

## Project structure

- `data/` - raw and processed datasets and data utilities
  - `Perovskite_data.csv` - original dataset
  - `perovskite_features.csv` - engineered features
  - `perovskite_features_rich.csv` - richer feature set
  - `perovskite_filtered.csv` - filtered dataset used in experiments
  - `analyze_dataset.py` - dataset analysis helper

- `models/` - trained model artifacts (binary `.joblib` files)
  - `band_gap_model.joblib`
  - `band_gap_model_rich.joblib`
  - `stability_model.joblib`
  - `stability_model_rich.joblib`

- `notebooks/` - exploratory and optimization notebooks
  - `1_Data_Exploration.ipynb`
  - `2_Bayesian_Optimization.ipynb`

- `results/` - evaluation metrics and plots
  - `metrics.txt`
  - `plots/` - generated figures

- `scripts/` - convenience scripts for data processing, training, and optimization
  - `01_data_filtering.py`
  - `02_feature_engineering_fixed.py`
  - `03_train_bandgap_model.py`
  - `04_train_stability_model.py`
  - other utility scripts

## Quick start

Prerequisites

- Python 3.8+ (create a venv if desired)
- Install dependencies:

```bash
pip install -r requirements.txt
```

Run the data processing pipeline

```bash
python scripts\01_data_filtering.py
python scripts\02_feature_engineering_fixed.py
```

Train models

```bash
python scripts\03_train_bandgap_model.py
python scripts\04_train_stability_model.py
```

Run optimization / evaluation

```bash
python scripts\final_optimization.py
python scripts\create_evaluation_plots.py
```

View notebooks

Open `notebooks/1_Data_Exploration.ipynb` and `notebooks/2_Bayesian_Optimization.ipynb` in Jupyter.

## Notes and recommendations

- Large files: `models/*.joblib` and `notebooks/*.ipynb` are committed. If you'd prefer to keep the repository small, add these to `.gitignore` and remove them from history; I can help with that.
- Authentication: You already pushed via HTTPS. If you'd like to switch to SSH, run:

```bash
git remote set-url origin git@github.com:baveshraam/Perovskite-Solar.git
```

- Contribution: Add a `CONTRIBUTING.md` and LICENSE if needed.

## Contact

Repository owner: baveshraam

---

If you'd like, I can (pick one):

- Add a `.gitignore` and remove large files from git history (interactive guide).
- Add a minimal `requirements.txt` pinning versions (I see one already; I can update it).
- Create a short `CONTRIBUTING.md` and `LICENSE`.
