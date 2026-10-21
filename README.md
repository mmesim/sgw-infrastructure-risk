# SGW Infrastructure Risk Scoring & Decision System

![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green?style=flat-square&logo=xgboost)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?style=flat-square&logo=scikit-learn)
![Cond-Forge](https://img.shields.io/badge/Conda--Forge-llvmlite_%7C_numba-black?style=flat-square&logo=anaconda)

> An AI-enabled decision support platform built for **Southeastern Grid & Water (SGW)** to predict infrastructure risk and streamline emergency response.

---

### 📊 Dataset Source & Full Column Preview
The raw input dataset used for this project is sourced from Kaggle:
🔗 **[Power System Faults Dataset](https://www.kaggle.com/datasets/ziya07/power-system-faults-dataset)**

**Full Dataset Schema (All 13 Columns):**
1. `Fault ID`: Unique identifier for each recorded asset event.
2. `Fault Type`: Categorical classification of the failure event (e.g., *Line Breakage, Transformer Failure, Overheating*).
3. `Fault Location (Latitude, Longitude)`: Geospatial coordinates of the impacted asset.
4. `Voltage (V)`: Electrical voltage telemetry reading.
5. `Current (A)`: Electrical current telemetry reading.
6. `Power Load (MW)`: Active power load carried by the asset.
7. `Temperature (°C)`: Operational thermal measurement.
8. `Wind Speed (km/h)`: Environmental weather stress indicator.
9. `Weather Condition`: Meteorological state (*Clear, Rainy, Snowy, Windstorm, Thunderstorm*).
10. `Maintenance Status`: Service tracking log (*Scheduled, Pending, Completed*).
11. `Component Health`: Asset operational status (*Normal, Faulty, Overheated*).
12. `Duration of Fault (hrs)`: Total active duration of the fault condition.
13. `Down time (hrs)`: Total asset downtime resulting from the incident.

---

### ⚡ Quick Start & Setup

```bash
# 1. Clone the repository and navigate inside
git clone https://github.com/mmesim/sgw-infrastructure-risk.git
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

---

### 📂 Pipeline Execution: Step 2 (Feature Engineering & Risk Input Matrix)

The pipeline is modularized under `src/` and executed via Jupyter notebooks:
- **Notebook:** `notebooks/02_feature_engineering.ipynb`
- **Library:** `src/features.py`

**Operations Performed:**
1. **Categorical Encoding:** One-hot encodes categorical stress factors (e.g., `Weather Condition`, `Maintenance Status`) using Pandas dummy variables for XGBoost compatibility.
2. **Interaction Feature Engineering:** Derives physical stress interaction proxies (e.g., electrical stress via `Voltage * Power Load`, environmental stress via `Wind Speed * Age`) to capture compounding infrastructure hazards.
3. **Feature Matrix Splitting:** Prepares the final training matrices by isolating target labels (`is_critical`) and stripping non-feature IDs and raw strings into clean $X$ and $y$ arrays for model training.


---

### 📂 Pipeline Execution: Step 3 (XGBoost Model Training & Risk Scoring)
The pipeline is modularized under `src/` and executed via Jupyter notebooks:
- **Notebook:** `notebooks/03_model_training.ipynb`
- **Library:** `src/model.py`

**Operations Performed:**
1. **Train-Test Splitting:** Splits the feature matrix into stratified training and testing subsets (`80/20` ratio).
2. **Post-Split Feature Scaling:** Normalizes numerical features using `StandardScaler` fitted **exclusively on the training subset** to prevent data leakage.
3. **XGBoost Classification & Evaluation:** Trains an `XGBoost` classifier (`XGBClassifier`) and evaluates performance using ROC-AUC scores, classification reports, and confusion matrix metrics.
4. **Dynamic Risk Scoring:** Converts model failure probabilities ($0.0 - 1.0$) into actionable continuous $0–100$ asset risk scores for prioritization.
---

### 📂 Pipeline Execution: Step 4 (Decision Framework & Operational Dashboard)
The decision support and geospatial visualization layer is executed via Jupyter notebooks:
- **Notebook:** `notebooks/04_decision_framework.ipynb`

**Operations Performed:**
1. **Risk Threshold Categorization:** Maps continuous $0\text{--}100\%$ risk scores into operational tiers (🟢 GREEN: 0–39%, 🟡 AMBER: 40–74%, 🔴 RED: 75–100%) with explicit action protocols.
2. **Model Explainability (SHAP):** Decomposes asset risk predictions into feature-level contributions to provide operators transparent reasoning.
3. **Control Room Dashboard:** Renders interactive summary metrics, high-risk counts, and score distribution histograms.
4. **Geospatial Risk Map:** Displays an interactive Folium map plotting assets with traffic-light color-coded markers and rich popup metadata.