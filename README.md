# Unifying-DRO

This repository contains the official implementation of the paper 

**UNIFYING DISTRIBUTIONALLY ROBUST OPTIMIZATION VIA
OPTIMAL TRANSPORT THEORY**  
_by Jose Blanchet, Daniel Kuhn, Jiajin Li, and Bahar Tașkesen._

Arxiv: https://arxiv.org/abs/2308.05414

The code provides a unified implementation of several distributionally robust optimization (DRO) models, KL-DRO, Wasserstein-DRO, and the proposed **MOT-DRO** , together with scripts to reproduce all numerical experiments in the paper.

---

## Overview

This library implements DRO classifiers whose ambiguity sets are defined using:

- **KL divergence**
- **Wasserstein distance**
- **MOT discrepancy** (joint $\phi$-divergence + optimal transport distance)

All optimization problems are built using **CVXPY** with the **MOSEK** solver.  
The experiments focus on robust linear classification in high-dimensional, sparse settings.

---
## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BaharTaskesen/Unifying-DRO.git
cd Unifying-DRO
```

### 2. Install dependencies

Using Conda:
```bash
conda env create -f environment.yml
conda activate dro-env
```
or using pip:
```bash
pip install -r requirements.txt
```
### 3. Install MOSEK liscence
https://www.mosek.com/products/academic-licenses/

---
## Reproducing Experiments
Each main experiment corresponds to one Python script:
| Figure / Task           | Script / Notebook          | Description                                                              |
|-------------------------|----------------------------|--------------------------------------------------------------------------|
| **Fig. 1**              | `figure_1.py`              | SVM classification example and worst case distribution                   |
| **Fig. 2(a)**           | `figure_2a.py`             | CCR vs ambiguity radius \(r\)                                            |
| **Fig. 2(b)**           | `figure_2b.py`             | MOT DRO hyperparameter \(\theta_1\) sensitivity                          |
| **Fig. 2(c)**           | `figure_2c.py`             | CCR vs ratio \(n/d_x\)                                                   |
| **Result analysis**     | `analyse_results.ipynb`    | Loads saved `.npz` files, aggregates repetitions, and produces final CCR plots and summary figures used in the paper |



Example run of an experiment:
```bash
python figure_2a.py
```

