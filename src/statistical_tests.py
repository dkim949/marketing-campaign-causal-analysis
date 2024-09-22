import pandas as pd
import numpy as np
from scipy import stats


def t_test(df, group_col, value_col):
    """
    Performs a t-test to compare means between two groups.
    """
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError(
            "The group column must have exactly two unique values for a t-test."
        )

    group1 = df[df[group_col] == groups[0]][value_col]
    group2 = df[df[group_col] == groups[1]][value_col]

    t_stat, p_value = stats.ttest_ind(group1, group2)
    return t_stat, p_value


def chi_square_test(df, col1, col2):
    """
    Performs a chi-square test of independence between two categorical variables.
    """
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p_value


def anova_test(df, group_col, value_col):
    """
    Performs a one-way ANOVA test.
    """
    groups = [group for _, group in df.groupby(group_col)[value_col]]
    f_value, p_value = stats.f_oneway(*groups)
    return f_value, p_value


def correlation_test(df, col1, col2):
    """
    Performs a Pearson correlation test between two numerical variables.
    """
    r, p_value = stats.pearsonr(df[col1], df[col2])
    return r, p_value


if __name__ == "__main__":
    # 테스트를 위한 샘플 데이터 로드
    df = pd.read_csv("../data/marketing_ab_sampled.csv")

    # T-test for 'total ads' between test groups
    t_stat, p_value = t_test(df, "test group", "total ads")
    print(f"T-test results: t-statistic = {t_stat}, p-value = {p_value}")

    # Chi-square test between 'test group' and 'converted'
    chi2, p_value = chi_square_test(df, "test group", "converted")
    print(f"Chi-square test results: chi2 = {chi2}, p-value = {p_value}")

    # ANOVA test for 'total ads' across 'most ads day'
    f_value, p_value = anova_test(df, "most ads day", "total ads")
    print(f"ANOVA test results: F-value = {f_value}, p-value = {p_value}")

    # Correlation test between 'total ads' and 'most ads hour'
    r, p_value = correlation_test(df, "total ads", "most ads hour")
    print(f"Correlation test results: r = {r}, p-value = {p_value}")
