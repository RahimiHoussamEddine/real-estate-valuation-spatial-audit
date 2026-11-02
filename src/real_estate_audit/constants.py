"""Pinned source and deterministic analysis settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/477/"
    "real%2Bestate%2Bvaluation%2Bdata%2Bset.zip"
)
SOURCE_SHA256 = "aa437bdac3ca23200258a0a58d251c0af90651aeaf7c7a07aaa714f0f9f25e2c"
SOURCE_BYTES = 32_885
SOURCE_MEMBER = "Real estate valuation data set.xlsx"
RANDOM_STATE = 2026
N_SPATIAL_CLUSTERS = 6
N_CV_SPLITS = 6
SOURCE_ID = "No"
TARGET_COLUMN = "Y house price of unit area"
FEATURE_COLUMNS = [
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude",
]
COORDINATE_COLUMNS = ["X5 latitude", "X6 longitude"]
TARGET_UNIT = "10,000 New Taiwan dollars per ping"
