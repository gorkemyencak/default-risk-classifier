from pathlib import Path

def get_project_root(
        start_path: Path
) -> Path: 
    """
    Return the project root by searching upward for pyproject.toml

    To make the path robust when executing the code from scripts and notebooks    
    """
    current_path = start_path.resolve()

    for parent in [current_path, *current_path.parents]:
        if (parent / 'pyproject.toml').exists():
            return parent
    
    raise FileNotFoundError('Project root could not be found. Please make sure that pyproject.toml exists in the project folder!')

PROJECT_ROOT = get_project_root(Path(__file__))

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
METRICS_DIR = REPORTS_DIR / 'metrics'
MODELS_DIR = REPORTS_DIR / 'models'

SUPERVISED_DATA_PATH = RAW_DATA_DIR / 'supervised_data.xlsx'
UNSUPERVISED_DATA_PATH = RAW_DATA_DIR / 'unsupervised_data.xlsx'

SUPERVISED_PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / 'supervised_features.csv'
UNSUPERVISED_PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / 'unsupervised_features.csv'

TARGET_COLUMN = 'Target'