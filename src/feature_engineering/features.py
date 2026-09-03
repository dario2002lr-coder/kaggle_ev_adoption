import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_charging_total(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that total charging station count is available."""
    if "Charging_Stations_Total" not in df.columns:
        df["Charging_Stations_Total"] = (
            df["Charging_Stations_Near_Home"]
            + df["Charging_Stations_Near_Work"]
        )

    return df


def _ensure_anxiety_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure range-anxiety indicator features are available."""
    if "Is_Medium_Range_Anxiety" not in df.columns:
        df["Is_Medium_Range_Anxiety"] = (
            df["Range_Anxiety_Level"] == "Medium"
        ).astype(int)

    if "Is_High_Range_Anxiety" not in df.columns:
        df["Is_High_Range_Anxiety"] = (
            df["Range_Anxiety_Level"] == "High"
        ).astype(int)

    return df


def _ensure_anxiety_score(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that the ordinal range-anxiety score is available."""
    if "Range_Anxiety_Score" not in df.columns:
        df["Range_Anxiety_Score"] = df["Range_Anxiety_Level"].map({
            "Low": 0,
            "Medium": 1,
            "High": 2,
        })

    return df


# ---------------------------------------------------------------------------
# Charging features
# ---------------------------------------------------------------------------

def add_charging_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features related to charging infrastructure."""
    df = df.copy()

    df = _ensure_charging_total(df)

    if "Charging_Stations_Home_Ratio" not in df.columns:
        total = df["Charging_Stations_Total"]

        df["Charging_Stations_Home_Ratio"] = np.divide(
            df["Charging_Stations_Near_Home"],
            total,
            out=np.zeros(len(df), dtype=float),
            where=total != 0,
        )

    if "Charging_Stations_Difference" not in df.columns:
        df["Charging_Stations_Difference"] = (
            df["Charging_Stations_Near_Home"]
            - df["Charging_Stations_Near_Work"]
        )

    return df


# ---------------------------------------------------------------------------
# Income features
# ---------------------------------------------------------------------------

def add_income_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add income-related features."""
    df = df.copy()

    if "Log_Income" not in df.columns:
        df["Log_Income"] = np.log1p(
            df["Annual_Income_USD"]
        )

    if "Income_per_Car" not in df.columns:
        df["Income_per_Car"] = (
            df["Annual_Income_USD"]
            / df["Number_of_Cars_Owned"]
        )

    if "Income_x_Cars" not in df.columns:
        df["Income_x_Cars"] = (
            df["Annual_Income_USD"]
            * df["Number_of_Cars_Owned"]
        )

    return df


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add selected interaction features."""
    df = df.copy()

    # Ensure dependencies exist.
    df = _ensure_charging_total(df)
    df = _ensure_anxiety_indicators(df)
    df = _ensure_anxiety_score(df)

    subsidy = (
        df["Subsidy_Available"] == "Yes"
    ).astype(int)

    home_charging = (
        df["Home_Charging_Possible"] == "Yes"
    ).astype(int)

    charging_total = df["Charging_Stations_Total"]
    medium_anxiety = df["Is_Medium_Range_Anxiety"]
    high_anxiety = df["Is_High_Range_Anxiety"]

    if "Income_x_Subsidy" not in df.columns:
        df["Income_x_Subsidy"] = (
            df["Annual_Income_USD"]
            * subsidy
        )

    if "Charging_x_Medium_Range_Anxiety" not in df.columns:
        df["Charging_x_Medium_Range_Anxiety"] = (
            charging_total
            * medium_anxiety
        )

    if "Charging_x_High_Range_Anxiety" not in df.columns:
        df["Charging_x_High_Range_Anxiety"] = (
            charging_total
            * high_anxiety
        )

    if "Commute_x_Medium_Range_Anxiety" not in df.columns:
        df["Commute_x_Medium_Range_Anxiety"] = (
            df["Daily_Commute_km"]
            * medium_anxiety
        )

    if "Commute_x_High_Range_Anxiety" not in df.columns:
        df["Commute_x_High_Range_Anxiety"] = (
            df["Daily_Commute_km"]
            * high_anxiety
        )

    if "Commute_x_Home_Charging" not in df.columns:
        df["Commute_x_Home_Charging"] = (
            df["Daily_Commute_km"]
            * home_charging
        )

    if "Commute_x_Cars" not in df.columns:
        df["Commute_x_Cars"] = (
            df["Daily_Commute_km"]
            * df["Number_of_Cars_Owned"]
        )

    if "Environmental_Concern_x_Subsidy" not in df.columns:
        df["Environmental_Concern_x_Subsidy"] = (
            df["Environmental_Concern_Level"]
            * subsidy
        )

    if "Environmental_Concern_x_Commute" not in df.columns:
        df["Environmental_Concern_x_Commute"] = (
            df["Environmental_Concern_Level"]
            * df["Daily_Commute_km"]
        )

    if "Environmental_Concern_x_Range_Anxiety" not in df.columns:
        df["Environmental_Concern_x_Range_Anxiety"] = (
            df["Environmental_Concern_Level"]
            * df["Range_Anxiety_Score"]
        )

    if "Charging_x_Home_Charging" not in df.columns:
        df["Charging_x_Home_Charging"] = (
            charging_total
            * home_charging
        )

    return df


# ---------------------------------------------------------------------------
# Feature-engineering pipeline
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "charging": add_charging_features,
    "income": add_income_features,
    "interactions": add_interaction_features,
}


def apply_feature_engineering(
    df: pd.DataFrame,
    groups: list[str],
) -> pd.DataFrame:
    """
    Apply selected feature-engineering groups.

    Parameters
    ----------
    df:
        Input dataframe.

    groups:
        List of feature groups to apply. Available groups are:
        "charging", "income", and "interactions".

    Returns
    -------
    pd.DataFrame
        Dataframe containing the selected engineered features.
    """
    df = df.copy()

    unknown_groups = set(groups) - set(FEATURE_GROUPS)

    if unknown_groups:
        raise ValueError(
            f"Unknown feature groups: {sorted(unknown_groups)}. "
            f"Available groups: {sorted(FEATURE_GROUPS)}"
        )

    for group in groups:
        df = FEATURE_GROUPS[group](df)

    return df
