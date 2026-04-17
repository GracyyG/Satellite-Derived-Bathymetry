#!/usr/bin/env python3
"""
Streamlit launcher for the region selector
"""
import sys
import subprocess

if __name__ == "__main__":
    # Launch streamlit using python module
    cmd = [sys.executable, "-m", "streamlit", "run", "run_region_selector.py"] + sys.argv[1:]
    subprocess.run(cmd)