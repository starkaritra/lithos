"""Grove D-008 cost-model spike.

An analytical cost model (NOT silicon, NOT a cycle-accurate sim) that screens
whether XGBoost tree-ensemble inference is control-bound with a small on-chip
working set, so an EDGE dataflow core could win. See spike-prereg.md.
"""
