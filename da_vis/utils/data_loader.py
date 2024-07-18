# data_visualizer/utils/data_loader.py

import pandas as pd

def load_data(file_path):
    """Loads data from a CSV file into a Pandas DataFrame."""
    return pd.read_csv(file_path)
