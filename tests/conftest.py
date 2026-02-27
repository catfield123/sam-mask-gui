"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Ensure src is on the path when running tests from project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
