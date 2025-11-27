"""
Stub for logger utilities (training functionality removed).
Only includes functions needed for model code compatibility.
"""

from tqdm import tqdm
import sys

def def_tqdm(x):
    """Simple tqdm wrapper"""
    return tqdm(x, leave=True, file=sys.stdout,
                bar_format="{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

def get_range(x):
    """
    Simplified get_range for single-GPU.
    Returns tqdm wrapper for iterables.
    """
    return def_tqdm(x)

def average_metrics(metrics):
    """
    Stub for average_metrics - only used in training forward() method.
    """
    return metrics
