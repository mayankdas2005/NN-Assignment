# Neural Networks Assignment: MLPs from scratch

| File | What it contains |
|---|---|
| `mlp_scratch.py` | The MLP implemented **from scratch in NumPy**: initialisation, forward pass, ReLU, softmax/sigmoid, CCE/BCE loss, backpropagation, gradient-descent update, numerical gradient check. Supports any number of hidden layers. |
| `helpers.py` | Data loading, stratified split + standardisation, metrics (sklearn), plotting. |
| `Q1_Iris_MLP.ipynb` | Question 1: Iris, architectures 4-6-3 / 4-10-3 / 4-20-3 × splits 70:30 / 80:20, plus the 2–5 hidden-layer extension. |
| `Q2_BreastCancer_MLP.ipynb` | Question 2: Breast Cancer, architectures 30-8-1 / 30-15-1 / 30-25-1 × splits 70:30 / 80:20, plus the 2–5 hidden-layer extension. |
| `Q3_PyTorch_Deep_MLP.ipynb` | Extension: deep MLPs (5, 7, 10 hidden layers) in PyTorch on both datasets, a check that PyTorch reproduces the scratch MLP, and a depth/optimiser study. |
| `results/` | All figures (PNG) and tables (CSV) produced by the notebooks. |
| `report/` | The written report (Word + PDF). |

## How to run

```bash
pip install numpy pandas matplotlib scikit-learn torch jupyter
jupyter notebook
```

Run the notebooks in order Q1 → Q2 → Q3 (Q3 reads the robustness CSVs written by Q1 and Q2).
Every run is seeded (`seed = 42`), so results are reproducible.
Approximate run times on a laptop CPU: Q1 ≈ 2 min, Q2 ≈ 5 min, Q3 ≈ 10–15 min.

**Windows note:** if `import torch` fails inside Jupyter with `WinError 1114 ... c10.dll` while it works
in plain Python, the installed `pyzmq` (used by Jupyter) is loading an older C++ runtime DLL first.
Fix: `pip install -U pyzmq`.
