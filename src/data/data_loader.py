import pandas as pd

from pathlib import Path

from config import (
    SUPERVISED_DATA_PATH,
    UNSUPERVISED_DATA_PATH
)

# helpers
def _load_excel_data(file_path: Path) -> pd.DataFrame:
    """ Load an Excel dataset from given file_path """
    # check whether the file_paths exists, return FileNotFoundError otherwise
    if not file_path.exists():
        raise FileNotFoundError(f'Data file not found: {file_path}')
    
    return pd.read_excel(file_path)

def load_supervised_data() -> pd.DataFrame:
    """ Load the supervised dataset for classification model """
    return _load_excel_data(file_path = SUPERVISED_DATA_PATH)

def load_unsupervised_data() -> pd.DataFrame:
    """ Load the unsupervised dataset for clustering model """
    return _load_excel_data(file_path = UNSUPERVISED_DATA_PATH)