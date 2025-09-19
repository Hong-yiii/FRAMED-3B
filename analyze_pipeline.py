#!/usr/bin/env python3
"""
Pipeline Analysis Wrapper Script

Runs the pipeline analysis from any directory.
"""

import sys
import os

# Add the intermediateJsons directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
intermediate_dir = os.path.join(script_dir, "intermediateJsons")
sys.path.insert(0, intermediate_dir)

# Import and run the analysis
from analyze_pipeline import main

if __name__ == "__main__":
    main()
