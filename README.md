# SGW Infrastructure Risk Scoring & Decision System

![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green?style=flat-square&logo=xgboost)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?style=flat-square&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-blueviolet?style=flat-square)
![Cond-Forge](https://img.shields.io/badge/Conda--Forge-llvmlite_%7C_numba-black?style=flat-square&logo=anaconda)

> An AI-enabled decision support platform built for **Southeastern Grid & Water (SGW)** to predict infrastructure risk and streamline emergency response.

---

### ⚡ Quick Start & Setup

```bash
# 1. Clone the repository and navigate inside
git clone [https://github.com/mmesim/sgw-infrastructure-risk.git](https://github.com/mmesim/sgw-infrastructure-risk.git)
cd sgw-infrastructure-risk

# 2. Initialize environment
conda create -n sgw_env python=3.10 -y
conda activate sgw_env

# 3. Install core C/C++ dependencies via Conda
conda install -c conda-forge llvmlite numba -y

# 4. Install Python dependencies 
pip install -r requirements.txt
```


### 📂 Pipeline Execution: Step 1 (Data Ingestion & Preprocessing)

The pipeline is modularized under `src/` and executed via Jupyter notebooks:
* **Notebook:** `notebooks/01_data_ingestion.ipynb`
* **Library:** `src/data_prep.py`

**Operations Performed:**
1. **Data Loading:** Ingests raw utility fault data (`fault_data.csv`) with path and schema checks.
2. **Missing Value Imputation:** Fills numerical missing values with the column **median** and categorical features with the **mode**.
3. **Binary Target Mapping:** Maps asset health states (`'Normal'` $\rightarrow$ `0`, `'Faulty'`/`'Overheated'` $\rightarrow$ `1`) into a new binary target column (`is_critical`).
4. **Export:** Outputs the cleaned dataset to `data/processed_faults.csv` for Step 2 feature engineering.