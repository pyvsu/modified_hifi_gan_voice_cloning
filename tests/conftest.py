"""
Pytest configuration for test discovery/imports.

Ensures the repository root is on sys.path so that imports like
`from thesis_models.resblock import ResBlock` work regardless of
the current working directory or how pytest is invoked.
"""

import os
import sys


def _add_repo_root_to_path():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(tests_dir)
    if repo_root not in sys.path:
        # Prepend to prioritize local sources over similarly named installed packages
        sys.path.insert(0, repo_root)


_add_repo_root_to_path()
