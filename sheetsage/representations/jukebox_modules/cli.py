"""
Command-line interface for Jukebox-Infer

This module provides the CLI entry point when installed as a package.
For standalone usage, use quick_infer.py directly.
"""

import sys
import os

def main():
    """CLI entry point - delegates to quick_infer.py"""
    # Import quick_infer from the parent directory
    quick_infer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "quick_infer.py")
    
    if not os.path.exists(quick_infer_path):
        print("Error: quick_infer.py not found. Please run from the package root.")
        sys.exit(1)
    
    # Execute quick_infer.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("quick_infer", quick_infer_path)
    quick_infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quick_infer)
    quick_infer.main()

if __name__ == "__main__":
    main()
