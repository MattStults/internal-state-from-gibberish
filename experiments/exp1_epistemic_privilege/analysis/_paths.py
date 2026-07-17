"""Shared path resolver for the offline analysis scripts. Imports the model registry from ../src so the
active model (env INTRO_MODEL, default qwen2.5-3b) selects which runs/<slug>/ bundle we read & write.
Usage in a script:  import _paths as P  ->  P.DATA / "covert_collect.pt",  P.FIGURES / "x.png",  P.RESULTS / "y.json"
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
import config as C  # noqa: E402

C.ensure_run_dirs()
DATA, RESULTS, FIGURES, STREAMS = C.DATA, C.RESULTS, C.FIGURES, C.STREAMS
ACTIVE = C.ACTIVE
