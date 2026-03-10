
LOG-ENERGY LAMBDA ESTIMATION REFERENCE IMPLEMENTATION

Pipeline:
returns
→ sliding window energy
→ log-energy
→ centered log-energy
→ covariance vs lag
→ slope vs log(lag)
→ λ² estimate

Expected sanity results:
White noise: λ² ≈ 0
Brownian motion: λ² ≈ 0
MRW-like models: λ² > 0

Run example:
python example_usage.py
