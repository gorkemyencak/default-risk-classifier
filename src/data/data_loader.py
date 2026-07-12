import pandas as pd

from pathlib import Path

from config import (
    SUPERVISED_DATA_PATH,
    UNSUPERVISED_DATA_PATH,
    SUPERVISED_PROCESSED_DATA_PATH,
    UNSUPERVISED_PROCESSED_DATA_PATH
)

# helpers
def _load_excel_data(file_path: Path) -> pd.DataFrame:
    """ Load an Excel dataset from given file_path """
    # check whether the file_path exists, return FileNotFoundError otherwise
    if not file_path.exists():
        raise FileNotFoundError(f'Data file not found: {file_path}')
    
    return pd.read_excel(file_path)

def _save_features(
        df: pd.DataFrame,
        file_path: Path
) -> None:
    """ Save processed feature dataframe as CSV """
    file_path.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    df.to_csv(
        file_path,
        index = False
    )

def _load_features(
        file_path: Path,
        parse_dates: list[str] | None = None
) -> pd.DataFrame:
    """ Load processed feature dataframe from CSV """
    # check whether the file_path exists, return FileNotFoundError otherwise
    if not file_path.exists():
        raise FileNotFoundError(f'Data file not found: {file_path}')
    
    return pd.read_csv(
        file_path,
        parse_dates = parse_dates
    )

def load_supervised_data() -> pd.DataFrame:
    """ Load the supervised dataset for classification model """
    return _load_excel_data(file_path = SUPERVISED_DATA_PATH)

def load_unsupervised_data() -> pd.DataFrame:
    """ Load the unsupervised dataset for clustering model """
    return _load_excel_data(file_path = UNSUPERVISED_DATA_PATH)

def save_supervised_features(df: pd.DataFrame) -> None:
    """ Save processed supervised features """
    _save_features(
        df = df,
        file_path = SUPERVISED_PROCESSED_DATA_PATH
    )

def save_unsupervised_features(df: pd.DataFrame) -> None:
    """ Save processed unsupervised features """
    _save_features(
        df = df,
        file_path = UNSUPERVISED_PROCESSED_DATA_PATH
    )

def load_supervised_features() -> pd.DataFrame:
    """ Load processed supervised features for classification model """
    return _load_features(
        file_path = SUPERVISED_PROCESSED_DATA_PATH
    )

def load_unsupervised_features() -> pd.DataFrame:
    """ Load processed unsupervised features for clustering model """
    return _load_features(
        file_path = UNSUPERVISED_PROCESSED_DATA_PATH
    )