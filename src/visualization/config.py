"""
Configuration settings for BLiMP experiments
"""
import os
from dotenv import load_dotenv

# Load environment variables
ENV_FILE = ".env"
load_dotenv(ENV_FILE)

# Repository credentials
REPO_CODE = os.getenv("REPO_CODE")
TOKEN = os.getenv("TOKEN")

# Experiment settings
SEEDS = [0, 1, 2]
VOCAB_SIZES = [8000, 16000, 32000, 50257]
MILESTONES = [25, 50, 75, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, "final"]

# Default experiment parameters
DEFAULT_DATASET = "babylm3"
DEFAULT_MODEL_SIZE = "180m"
DEFAULT_MODE = "best"

# Data paths
DATA_DIR = "results/data/"
PLOTS_DIR = "results/plots/"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Color scheme for datasets and model sizes
# Each dataset gets a base color, with variations for different model sizes
DATASET_COLOR_SCHEME = {
    # 'babylm3': {
    #     '20m': '#5d4037',   # lighter brown
    #     '60m': '#372516',   # original coffee
    #     '180m': '#1a0f0a'   # darker brown
    # },
    'babylm3': {
        '20m': '#c4a484',  # soft beige-tan
        '60m': '#a37a4c',  # caramel brown
        '180m': '#7c532d'  # rich brown but more distinguishable than your current ones
    },
    'tinystories': {
        '20m': '#4a90c4',   # lighter blue
        '60m': '#0f4e77',   # original sea blue
        '180m': '#083655'   # darker blue
    },
    'composite': {
        '20m': '#a8c490',   # lighter green
        '60m': '#89a377',   # original matcha
        '180m': '#6b8259'   # darker green
    }
}

# Base colors for datasets (used for backward compatibility)
DATASET_BASE_COLORS = {
    'babylm3': '#372516',       # coffee
    'tinystories': '#0f4e77',   # sea blue
    'hybrid_3.7B': '#89a377'    # matcha
}