# Decision Tree Classifier - CART, ID3, and C4.5

This project implements decision tree classifiers using the CART, ID3, and C4.5 algorithms in Python with `scikit-learn`. Each algorithm is evaluated on the Iris dataset, with code to visualize each decision tree and assess model performance.

## Overview
This project demonstrates three popular decision tree algorithms:
1. **CART (Classification and Regression Tree)**: Uses Gini impurity for splitting.
2. **ID3 (Iterative Dichotomiser 3)**: Uses entropy for splitting.
3. **C4.5**: Similar to ID3 but incorporates post-pruning based on cost-complexity pruning (`ccp_alpha`).

## Requirements
- Python 3.x
- `pandas`
- `scikit-learn`
- `matplotlib`

Install the required packages with:
```bash
pip install -r requirements.txt
