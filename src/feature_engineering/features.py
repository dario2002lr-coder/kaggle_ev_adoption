from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# =============================================================================
# Feature definition
# =============================================================================

FeatureFunction = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class FeatureDefinition:
    """Definition of an engineered feature."""

    function: FeatureFunction
    requires: tuple[str, ...] = ()


# =============================================================================
# Charging features
# =============================================================================

def add_charging_total(df: pd.DataFrame) -> pd.DataFrame:
    """Add total charging stations near home and work."""
    df = df.copy()

    if "Charging_Stations_Total" not in df.columns:
        df["Charging_Stations_Total"] = (
            df["Charging_Stations_Near_Home"]
            + df["Charging_Stations_Near_Work"]
        )

    return df


def add_charging_difference(df: pd.DataFrame) -> pd.DataFrame:
    """Add the difference between charging stations near home and work."""
    df = df.copy()

    if "Charging_Stations_Difference" not in df.columns:
        df["Charging_Stations_Difference"] = (
            df["Charging_Stations_Near_Home"]
            - df["Charging_Stations_Near_Work"]
        )

    return df


def add_charging_home_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add the proportion of charging stations located near home."""
    df = df.copy()

    total = df["Charging_Stations_Total"]

    if "Charging_Stations_Home_Ratio" not in df.columns:
        df["Charging_Stations_Home_Ratio"] = np.divide(
            df["Charging_Stations_Near_Home"],
            total,
            out=np.zeros(len(df), dtype=float),
            where=total != 0,
        )

    return df


# =============================================================================
# Income features
# =============================================================================

def add_log_income(df: pd.DataFrame) -> pd.DataFrame:
    """Add logarithmically transformed annual income."""
    df = df.copy()

    if "Log_Income" not in df.columns:
        df["Log_Income"] = np.log1p(
            df["Annual_Income_USD"]
        )

    return df


def add_income_per_car(df: pd.DataFrame) -> pd.DataFrame:
    """Add annual income divided by the number of cars owned."""
    df = df.copy()

    if "Income_per_Car" not in df.columns:
        df["Income_per_Car"] = (
            df["Annual_Income_USD"]
            / df["Number_of_Cars_Owned"]
        )

    return df


def add_income_x_cars(df: pd.DataFrame) -> pd.DataFrame:
    """Add the interaction between income and number of cars owned."""
    df = df.copy()

    if "Income_x_Cars" not in df.columns:
        df["Income_x_Cars"] = (
            df["Annual_Income_USD"]
            * df["Number_of_Cars_Owned"]
        )

    return df


# =============================================================================
# Range-anxiety features
# =============================================================================

def add_medium_anxiety_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Add an indicator for medium range anxiety."""
    df = df.copy()

    if "Is_Medium_Range_Anxiety" not in df.columns:
        df["Is_Medium_Range_Anxiety"] = (
            df["Range_Anxiety_Level"] == "Medium"
        ).astype(int)

    return df


def add_high_anxiety_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Add an indicator for high range anxiety."""
    df = df.copy()

    if "Is_High_Range_Anxiety" not in df.columns:
        df["Is_High_Range_Anxiety"] = (
            df["Range_Anxiety_Level"] == "High"
        ).astype(int)

    return df


def add_range_anxiety_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ordinal score for range anxiety."""
    df = df.copy()

    if "Range_Anxiety_Score" not in df.columns:
        df["Range_Anxiety_Score"] = df["Range_Anxiety_Level"].map({
            "Low": 0,
            "Medium": 1,
            "High": 2,
        })

    return df


# =============================================================================
# Interaction features
# =============================================================================

def add_income_x_subsidy(df: pd.DataFrame) -> pd.DataFrame:
    """Add the interaction between income and subsidy availability."""
    df = df.copy()

    subsidy = (
        df["Subsidy_Available"] == "Yes"
    ).astype(int)

    if "Income_x_Subsidy" not in df.columns:
        df["Income_x_Subsidy"] = (
            df["Annual_Income_USD"]
            * subsidy
        )

    return df


def add_charging_x_medium_anxiety(df: pd.DataFrame) -> pd.DataFrame:
    """Add charging infrastructure × medium range anxiety."""
    df = df.copy()

    if "Charging_x_Medium_Range_Anxiety" not in df.columns:
        df["Charging_x_Medium_Range_Anxiety"] = (
            df["Charging_Stations_Total"]
            * df["Is_Medium_Range_Anxiety"]
        )

    return df


def add_charging_x_high_anxiety(df: pd.DataFrame) -> pd.DataFrame:
    """Add charging infrastructure × high range anxiety."""
    df = df.copy()

    if "Charging_x_High_Range_Anxiety" not in df.columns:
        df["Charging_x_High_Range_Anxiety"] = (
            df["Charging_Stations_Total"]
            * df["Is_High_Range_Anxiety"]
        )

    return df


def add_commute_x_medium_anxiety(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily commute × medium range anxiety."""
    df = df.copy()

    if "Commute_x_Medium_Range_Anxiety" not in df.columns:
        df["Commute_x_Medium_Range_Anxiety"] = (
            df["Daily_Commute_km"]
            * df["Is_Medium_Range_Anxiety"]
        )

    return df


def add_commute_x_high_anxiety(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily commute × high range anxiety."""
    df = df.copy()

    if "Commute_x_High_Range_Anxiety" not in df.columns:
        df["Commute_x_High_Range_Anxiety"] = (
            df["Daily_Commute_km"]
            * df["Is_High_Range_Anxiety"]
        )

    return df


def add_commute_x_home_charging(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily commute × home charging availability."""
    df = df.copy()

    home_charging = (
        df["Home_Charging_Possible"] == "Yes"
    ).astype(int)

    if "Commute_x_Home_Charging" not in df.columns:
        df["Commute_x_Home_Charging"] = (
            df["Daily_Commute_km"]
            * home_charging
        )

    return df


def add_commute_x_cars(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily commute × number of cars owned."""
    df = df.copy()

    if "Commute_x_Cars" not in df.columns:
        df["Commute_x_Cars"] = (
            df["Daily_Commute_km"]
            * df["Number_of_Cars_Owned"]
        )

    return df


def add_environmental_x_subsidy(df: pd.DataFrame) -> pd.DataFrame:
    """Add environmental concern × subsidy availability."""
    df = df.copy()

    subsidy = (
        df["Subsidy_Available"] == "Yes"
    ).astype(int)

    if "Environmental_Concern_x_Subsidy" not in df.columns:
        df["Environmental_Concern_x_Subsidy"] = (
            df["Environmental_Concern_Level"]
            * subsidy
        )

    return df


def add_environmental_x_commute(df: pd.DataFrame) -> pd.DataFrame:
    """Add environmental concern × daily commute."""
    df = df.copy()

    if "Environmental_Concern_x_Commute" not in df.columns:
        df["Environmental_Concern_x_Commute"] = (
            df["Environmental_Concern_Level"]
            * df["Daily_Commute_km"]
        )

    return df


def add_environmental_x_range_anxiety(df: pd.DataFrame) -> pd.DataFrame:
    """Add environmental concern × range anxiety score."""
    df = df.copy()

    if "Environmental_Concern_x_Range_Anxiety" not in df.columns:
        df["Environmental_Concern_x_Range_Anxiety"] = (
            df["Environmental_Concern_Level"]
            * df["Range_Anxiety_Score"]
        )

    return df


def add_charging_x_home_charging(df: pd.DataFrame) -> pd.DataFrame:
    """Add charging infrastructure × home charging availability."""
    df = df.copy()

    home_charging = (
        df["Home_Charging_Possible"] == "Yes"
    ).astype(int)

    if "Charging_x_Home_Charging" not in df.columns:
        df["Charging_x_Home_Charging"] = (
            df["Charging_Stations_Total"]
            * home_charging
        )

    return df


# =============================================================================
# Feature registry
# =============================================================================

FEATURES = {
    # Charging
    "charging_total": FeatureDefinition(
        function=add_charging_total,
    ),
    "charging_difference": FeatureDefinition(
        function=add_charging_difference,
    ),
    "charging_home_ratio": FeatureDefinition(
        function=add_charging_home_ratio,
        requires=("charging_total",),
    ),

    # Income
    "log_income": FeatureDefinition(
        function=add_log_income,
    ),
    "income_per_car": FeatureDefinition(
        function=add_income_per_car,
    ),
    "income_x_cars": FeatureDefinition(
        function=add_income_x_cars,
    ),

    # Range anxiety
    "medium_anxiety_indicator": FeatureDefinition(
        function=add_medium_anxiety_indicator,
    ),
    "high_anxiety_indicator": FeatureDefinition(
        function=add_high_anxiety_indicator,
    ),
    "range_anxiety_score": FeatureDefinition(
        function=add_range_anxiety_score,
    ),

    # Interactions
    "income_x_subsidy": FeatureDefinition(
        function=add_income_x_subsidy,
    ),
    "charging_x_medium_anxiety": FeatureDefinition(
        function=add_charging_x_medium_anxiety,
        requires=(
            "charging_total",
            "medium_anxiety_indicator",
        ),
    ),
    "charging_x_high_anxiety": FeatureDefinition(
        function=add_charging_x_high_anxiety,
        requires=(
            "charging_total",
            "high_anxiety_indicator",
        ),
    ),
    "commute_x_medium_anxiety": FeatureDefinition(
        function=add_commute_x_medium_anxiety,
        requires=("medium_anxiety_indicator",),
    ),
    "commute_x_high_anxiety": FeatureDefinition(
        function=add_commute_x_high_anxiety,
        requires=("high_anxiety_indicator",),
    ),
    "commute_x_home_charging": FeatureDefinition(
        function=add_commute_x_home_charging,
    ),
    "commute_x_cars": FeatureDefinition(
        function=add_commute_x_cars,
    ),
    "environmental_x_subsidy": FeatureDefinition(
        function=add_environmental_x_subsidy,
    ),
    "environmental_x_commute": FeatureDefinition(
        function=add_environmental_x_commute,
    ),
    "environmental_x_range_anxiety": FeatureDefinition(
        function=add_environmental_x_range_anxiety,
        requires=("range_anxiety_score",),
    ),
    "charging_x_home_charging": FeatureDefinition(
        function=add_charging_x_home_charging,
        requires=("charging_total",),
    ),
}


# =============================================================================
# Feature application
# =============================================================================

def apply_features(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Apply a selected set of engineered features.

    Dependencies are resolved automatically.

    Parameters
    ----------
    df:
        Input dataframe.

    features:
        List of feature identifiers from the FEATURES registry.

    Returns
    -------
    pd.DataFrame
        Dataframe with the selected engineered features.

    Raises
    ------
    ValueError
        If an unknown feature is requested or a dependency cycle is detected.
    """
    df = df.copy()

    unknown_features = set(features) - set(FEATURES)

    if unknown_features:
        raise ValueError(
            f"Unknown features: {sorted(unknown_features)}. "
            f"Available features: {sorted(FEATURES)}"
        )

    applied: set[str] = set()
    processing: set[str] = set()

    def apply_feature(feature_name: str) -> None:
        """Recursively apply a feature and its dependencies."""
        if feature_name in applied:
            return

        if feature_name in processing:
            raise ValueError(
                f"Circular feature dependency detected involving "
                f"'{feature_name}'."
            )

        processing.add(feature_name)

        definition = FEATURES[feature_name]

        for dependency in definition.requires:
            apply_feature(dependency)

        df_before = set(df.columns)
        result = definition.function(df)

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"Feature '{feature_name}' must return a pandas DataFrame."
            )

        df.update(result)

        # Add columns that did not previously exist.
        new_columns = set(result.columns) - df_before
        for column in new_columns:
            df[column] = result[column]

        applied.add(feature_name)
        processing.remove(feature_name)

    for feature in features:
        apply_feature(feature)

    return df
