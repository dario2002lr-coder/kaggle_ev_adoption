from pathlib import Path
import pandas as pd


def create_submission(
    model,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    target: str,
    id_column: str = "id",
    output_filename: str = "submission.csv",
    output_dir: Path = Path("../data/outputs"),
) -> pd.DataFrame:
    """
    Generate and save a competition submission.

    Parameters
    ----------
    model
        Fitted classification model.
    test : pd.DataFrame
        Test dataset.
    sample_submission : pd.DataFrame
        Sample submission provided by the competition.
    target : str
        Target column name.
    id_column : str, default="id"
        Identifier column in the test dataset.
    output_filename : str, default="submission.csv"
        Name of the output submission file.
    output_dir : Path, default=Path("../data/outputs")
        Directory where the submission will be saved.

    Returns
    -------
    pd.DataFrame
        Generated submission.
    """
    X_test = test.drop(columns=[id_column])

    test_proba = model.predict_proba(X_test)[:, 1]

    submission = sample_submission.copy()
    submission[target] = test_proba

    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = output_dir / output_filename
    submission.to_csv(submission_path, index=False)

    print("Saved:", submission_path.resolve())

    return submission