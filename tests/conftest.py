import os
import sys

# Add the parent directory to the Python path
# This makes the main package importable in the tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
