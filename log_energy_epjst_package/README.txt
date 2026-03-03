
LOG-ENERGY EPJST PACKAGE

This package implements a complete log-energy based framework for
multiscale intermittency and cascade analysis.

Structure:

log_energy/
    operators.py
    energy.py
    intermittency.py
    covariance.py
    mi_knn.py

models/
    white_noise.py
    brownian_motion.py

tests/
    test_white_noise.py

Workflow:

1) Generate model
2) Apply scale operator
3) Compute log-energy field
4) Estimate intermittency (variance)
5) Estimate covariance slope
6) Estimate MI using kNN

This framework avoids histogram fitting and is distribution-free.
