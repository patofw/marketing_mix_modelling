import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error


def get_median_filtered(signal: pd.Series, threshold: int = 3):
    """Filters outliers from a signal using the median absolute deviation method.

    Args:
        signal (pd.Series): The input signal to be filtered.
        threshold (int, optional): The threshold for outlier detection. Defaults to 3.

    Returns:
        pd.Series: The filtered signal with outliers replaced by the median.
    """

    outlier_check_signal = signal.copy()
    difference = np.abs(outlier_check_signal - np.median(outlier_check_signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    outlier_check_signal[mask] = np.median(outlier_check_signal)
    return outlier_check_signal


def plot_outliers_signal(
        signal: pd.Series | np.ndarray,
        threshold: int = 2,
        return_mask: bool = True
) -> None | np.ndarray:
    """Plots outliers after finding them using the `get_median_filtered`
    function.

    Args:
        signal (pd.Series | np.ndarray): Signal to check outliers for.
        threshold (int, optional):  Deviation threshold. Defaults to 2.
        return_mask (bool, optional): If true, returns the masked array. Defaults to True.

    Returns:
        None | np.ndarray: None or an array with boolean values as mask.
    """
    kw = dict(marker='o', linestyle='none', color='r', alpha=0.35)

    mds = get_median_filtered(signal, threshold=threshold)
    outlier_idx = np.where(mds != signal)[0]
    # make sure we use the series' index.
    outlier_idx = signal.index[outlier_idx]
    plt.figure(figsize=(10, 8))

    plt.plot(signal, color="darkblue")
    plt.plot(
        outlier_idx, signal[outlier_idx], **kw,
        label="Outliers"
    )
    plt.title("Outlier detection with cutoff {}".format(threshold))
    plt.legend()
    plt.show()
    if return_mask:
        return outlier_idx


def _is_between(x, low, high) -> bool:
    if x >= low and x <= high:
        return 1
    else:
        return 0


def add_flag(
    df: pd.DataFrame,
    low: int,
    high: int,
):
    """Adds a flag if a value is between a lower and upper bound.

    Using the index of the dataframe as reference.
    It's recommended to have the date as index if the DF has a date.

    Args:
        df (pd.DataFrame): DataFrame to add the flag to.
        low (int): lower bound.
        high (int): upper bound.

    Returns:
        np.ndarray: Array with bool values
    """
    # add flag
    flag = df.apply(
        lambda row:
            _is_between(
                row.name, low, high  # usig the index of the df.
            ),
        axis=1
    )
    return flag


def performance_metrics(actuals: np.ndarray, pred: np.ndarray) -> None:
    """Prints out the performance metrics of a regressive model.

    Wrapper for the SkLearn performance metrics `r2_score, mean_squared_error`
    functions.

    Args:
        actuals (np.ndarray): Array of true values.
        pred (np.ndarray): Array of predictions.
    """

    r2 = r2_score(
        actuals,
        pred
    )

    mape = mean_squared_error(
        actuals,
        pred
    )

    print("Performance Metrics:")
    print()
    print(f'MSE:{np.round(mape, 3)}')
    print(f'R2: {np.round(r2, 3)}')
    print()
