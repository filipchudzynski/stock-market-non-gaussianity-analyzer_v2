
Wavelet Log-Energy Pipeline

Purpose
-------
Reference implementation of wavelet-based intermittency analysis.

Pipeline
--------
time series x(t)
→ CWT wavelet transform W(s,t)
→ energy E(s,t)=|W|^2
→ log-energy L(s,t)
→ intermittency I(s)=Var(L)
→ cascade memory C(Δt)
→ λ² estimation
→ scale coupling via mutual information

Dependencies
------------
numpy
pywavelets
scikit-learn

Install

pip install numpy pywavelets scikit-learn

Run example

python example_pipeline.py
