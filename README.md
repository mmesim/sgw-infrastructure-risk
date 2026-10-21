# SGW Infrastructure Risk Scoring & Decision System

![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green?style=flat-square&logo=xgboost)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?style=flat-square&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-blueviolet?style=flat-square)
![Cond-Forge](https://img.shields.io/badge/Conda--Forge-llvmlite_%7C_numba-black?style=flat-square&logo=anaconda)

> An AI-enabled decision support platform built for **Southeastern Grid & Water (SGW)** to predict infrastructure risk and streamline emergency response[cite: 2].

---

### ⚡ Quick Start & Setup

```bash
# 1. Initialize environment
conda create -n sgw_env python=3.10 -y
conda activate sgw_env

# 2. Install core C/C++ dependencies via Conda
conda install -c conda-forge llvmlite numba -y

# 3. Install Python dependencies 
pip install -r requirements.txt