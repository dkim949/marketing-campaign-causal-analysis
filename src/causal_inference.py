import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances


def load_data(file_path):
    return pd.read_csv(file_path)


def propensity_score_matching(df, treatment_col, outcome_col, covariates):
    """
    Perform propensity score matching.

    :param df: DataFrame containing the data
    :param treatment_col: Name of the treatment column
    :param outcome_col: Name of the outcome column
    :param covariates: List of covariate column names
    :return: Average Treatment Effect (ATE)
    """
    # Prepare the data
    X = df[covariates]
    y = df[treatment_col]

    # Standardize covariates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Estimate propensity scores
    pscore_model = LogisticRegression(random_state=42)
    pscore_model.fit(X_scaled, y)
    df["propensity_score"] = pscore_model.predict_proba(X_scaled)[:, 1]

    # Matching
    treated = df[df[treatment_col] == "ad"]
    control = df[df[treatment_col] == "psa"]

    def match(treated_unit):
        distances = pairwise_distances(
            treated_unit[covariates + ["propensity_score"]].values.reshape(1, -1),
            control[covariates + ["propensity_score"]].values,
        )
        return control.iloc[distances.argmin()]

    matched = treated.apply(match, axis=1)

    # Calculate ATE
    ate = treated[outcome_col].mean() - matched[outcome_col].mean()

    return ate


def difference_in_differences(
    df, time_col, treatment_col, outcome_col, pre_period, post_period
):
    """
    Perform Difference-in-Differences analysis.

    :param df: DataFrame containing the data
    :param time_col: Name of the time column
    :param treatment_col: Name of the treatment column
    :param outcome_col: Name of the outcome column
    :param pre_period: Value in time_col that represents the pre-treatment period
    :param post_period: Value in time_col that represents the post-treatment period
    :return: DiD estimate
    """
    # Filter data for pre and post periods
    pre_data = df[df[time_col] == pre_period]
    post_data = df[df[time_col] == post_period]

    # Calculate differences
    treated_diff = (
        post_data[post_data[treatment_col] == "ad"][outcome_col].mean()
        - pre_data[pre_data[treatment_col] == "ad"][outcome_col].mean()
    )

    control_diff = (
        post_data[post_data[treatment_col] == "psa"][outcome_col].mean()
        - pre_data[pre_data[treatment_col] == "psa"][outcome_col].mean()
    )

    # Calculate DiD estimate
    did_estimate = treated_diff - control_diff

    return did_estimate


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("../data/marketing_ab.csv")

    # Example usage of propensity score matching
    covariates = ["total ads", "most ads hour"]
    ate = propensity_score_matching(df, "test group", "converted", covariates)
    print(f"Average Treatment Effect (PSM): {ate}")

    # Note: The current dataset doesn't have a time column for DiD analysis
    # If you had time data, you could use the DiD function like this:
    # did_estimate = difference_in_differences(df, 'time', 'test group', 'converted', 'pre', 'post')
    # print(f"Difference-in-Differences Estimate: {did_estimate}")
