"""Mode solvers used by the built-in generator plugins.

These import ``fracture_utility`` (and therefore scipy / igl / mosek); they
are only imported lazily from inside plugin ``build`` methods.

Exception: ``exemplar.py`` (example-based plasticity) is numpy-only at import
time and imports scipy / igl lazily inside its precompute helpers.
"""
