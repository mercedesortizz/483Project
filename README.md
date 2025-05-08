# 🧠 Negation-Aware Information Retrieval

## 👥 Authors
- **Mercedes Ortiz** 
- **Mikos Bazerkanian**
- **Rythm Sanghvi** 

This repository contains our work on developing and evaluating models for **negation-aware information retrieval (IR)**. We focused on handling complex negation patterns in text — an area where traditional IR systems often struggle. Our models were trained and evaluated using the [NevIR: Negation in Neural Information Retrieval](https://aclanthology.org/2024.eacl-long.139.pdf) dataset, which emphasizes challenging negation cases.

## 📊 Results Summary

We evaluated multiple models using **pairwise accuracy**:

| Model                           | Test Acc. |
|--------------------------------|-----------|
| XGBoost                        | 48.12%    |
| MonoT5-Base                    | 65.73%    |
| **MonoT5-Large (Full Precision)** | **71.11%** ✅ |

The best-performing model was **MonoT5-Large (Full Precision)**, achieving a **71.11%** pairwise accuracy on the test set — a notable improvement over both baseline and earlier models.

> Yes, it once ran for 15 hours and crashed. And yes, we did too.

## 📁 Project Structure

- 483Project
    - [README.md](README.md)
    - [monot5_base.py](monot5_base.py)
    - [monot5_large.py](monot5_large.py)
    - [MonoT5_Base.ipynb](MonoT5_Base.ipynb)
    - [MonoT5_Large.ipynb](MonoT5_Large.ipynb)
    - [requirements.txt](requirements.txt)
    - [483_Project_Report.pdf](483_Project_Report.pdf)

## 🛠️ Installation

To run the Python scripts locally:

1. **Clone this repository**:
   ```bash
   git clone https://github.com/mercedesortizz/483Project.git
   
2. **🛠️ Install Dependencies**:  
We recommend using a virtual environment (e.g. `venv` or `conda`).  
    ```bash
    pip install -r requirements.txt

3. **▶️ Run a Script**  
    ```bash
    python monot5_base.py
    #or  
    python monot5_large.py

## ☁️ Run on Google Colab

You can also run the Jupyter notebooks in the cloud using **Google Colab**:

1. Open either notebook from the repo:
   - `MonoT5_Base.ipynb`
   - `MonoT5_Large.ipynb`

2. Click **"Open in Colab"** at the top of the notebook (or manually upload it to [Google Colab](https://colab.research.google.com/)).

3. Set the runtime type to **GPU**:
   - Go to `Runtime` → `Change runtime type` → Select **GPU**

4. Follow the cells and run them step by step.