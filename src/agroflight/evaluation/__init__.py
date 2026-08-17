"""Policy evaluation: baselines vs. trained checkpoints.

Intentionally does NOT re-export from `evaluate_policy` at import time: that
module is meant to be run with `python -m agroflight.evaluation.evaluate_policy`,
and eagerly importing it here makes Python load it twice (once as a package
attribute, once as __main__), which emits a RuntimeWarning. Import it directly:

    from agroflight.evaluation.evaluate_policy import evaluate
"""
