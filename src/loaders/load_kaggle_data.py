from pathlib import Path
import pandas as pd

KAGGLE_DATA_DIR = Path(
    "/kaggle/input/competitions/playground-series-s6e9"
)


def get_project_root() -> Path:
    """Return the project root directory."""
    
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return the appropriate data directory for the current environment."""

    if KAGGLE_DATA_DIR.exists():
        return KAGGLE_DATA_DIR

    return get_project_root() / "data" / "raw"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test and sample submission datasets."""

    data_dir = get_data_dir()

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(
        data_dir / "sample_submission.csv"
    )

    return train, test, sample_submission
