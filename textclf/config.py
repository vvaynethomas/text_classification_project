"""Repo-relative locations for input data and generated outputs."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_DATA = DATA_DIR / "story.data.csv"
