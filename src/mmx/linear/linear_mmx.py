
import pandas as pd
import numpy as np
import optuna

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.ERROR)


def carry_over_effect(
    input_variable: pd.Series, decay: float, lag: int
) -> pd.Series:
    """
    Applies a carry-over effect to a time series variable with a specified decay and lag.

    Args:
        input_variable (pd.Series): The input time series variable.
        decay (float): The decay factor applied to past values.
        lag (int): The number of past periods to consider for the carry-over effect.

    Returns:
        pd.Series: The time series with the carry-over effect applied.
    """
    output = []
    for row in range(len(input_variable)):
        # Take care of observations with less days than required for lags.
        # e.g. For the first day, we have a problem if lag=5.
        lag_aux = min(lag, row)

        # Apply decay effect to every row.
        output.append(
            sum(
                [
                    input_variable[row-i] * (decay ** i) for i in range(lag_aux + 1)
                ]
            )
        )
    return pd.Series(output)


def saturation_effect(
        input_variable: pd.Series,
        alpha: float
) -> pd.Series:
    """
    Applies a saturation effect to a time series variable using an exponential function.

    Args:
        input_variable (pd.Series): The input time series variable.
        alpha (float): The saturation rate parameter.

    Returns:
        pd.Series: The time series with the saturation effect applied.
    """

    return (1 - np.exp(-alpha * input_variable))


def adstock_tranformation(
    input_variable: pd.Series, alpha: float, decay: float, lag: int
) -> pd.Series:
    "Apply adstock transformation to a given pandas Series."
    return saturation_effect(
        carry_over_effect(
            input_variable, decay, lag
        ), alpha
    )


def add_adstock_to_pdf(
    pdf_input: pd.DataFrame,
    optimized_hyperparameters: dict,
    channels: list,
) -> pd.DataFrame:
    """Creates a set of new columns in a dataset with their adstocked values.

    Wrapper for the `apply_adstock_transformation` function.

    Args:
        pdf_input (pd.DataFrame): DataFrame to add the columns to.
        hyperparameters (dict): Optimized hyperparameters.
        channels (list): _description_

    Returns:
        pd.DataFrame: _description_
    """

    pdf = pdf_input.copy()
    pdf.reset_index(drop=True, inplace=True)
    # Add a column for each channel with the name adstock_channel.
    for c in channels:
        pdf[f'adstock_{c}'] = adstock_tranformation(
            pdf[c],
            optimized_hyperparameters[f'alpha_{c}'],
            optimized_hyperparameters[f'decay_{c}'],
            optimized_hyperparameters[f'lag_{c}']
        ).values
    return pdf


def sales_from(
    channel: str,
    features: list,
    model: LinearRegression,
    X: pd.DataFrame
) -> pd.DataFrame:
    """Compute sales attributed to a certain channel"""
    coef = model.coef_[features.index(channel)]
    obs = X.iloc[:, features.index(channel)]
    return coef * obs


def create_stack_df(
        _df, features: list, lr: LinearRegression
) -> pd.DataFrame:
    """Helper for plotting a contribution plot.

    Args:
        _df (pd.DataFrame): DataFrame with adstock variables.
        features (list): features
        lr (LinearRegression): Linear Model

    Returns:
        pd.DataFrame: A dataframe with the contributions to the target variable.
    """
    plot_df = pd.DataFrame()

    for f in _df.columns:
        if "adstock" in f:
            _series = sales_from(f, features, lr, _df)
        elif "index" in f:
            _series = sales_from(f, features, lr, _df)

        _series.name = f
        plot_df = pd.concat([plot_df, _series], axis=1)
    return plot_df


class AdstockHyperTuning:
    """Class that uses `optuna` for
    hyperparameter tuning to find the best fitting
    adstock parameters (alpha, decay, lag)
    """
    # This params dict will need to be updated
    # depending on the use case and the variables
    # to optimize. It should be located in a CONFIG file
    # when aiming to have a similar approach in production.

    def __init__(
        self,
        df: pd.DataFrame,
        hyperparameters: dict,
        features: tuple[str, ...],
        media_channels: tuple[str, ...],
        target: str,
        n_trials: int = 1_000,
        seed: int = 123456
    ):
        """Constructor of the class.

        Args:
            df (pd.DataFrame): DataFrame with channels, features and target.
            hyperparameters (dict): Dictionary of hyperparameters to pass to optuna.
            the dictionary needs to have the following format.
            {
                "alpha_<CHANNEL>: (lower_bound, upper_bound, typ["float" | "int"]),
                "decay_<CHANNEL>: (lower_bound, upper_bound, typ["float" | "int"]),
                "lag_<CHANNEL>: (lower_bound, upper_bound, typ["float" | "int"])
            } See `mmx_linear_model_example.ipynb` for more details.
            features (tuple[str, ...]): tuple of features (columns)
            media_channels (tuple[str, ...]): tuple of media_channels.
            target (str): Name of the target column.
            n_trials (int, optional): Number of simulations to optimize. Defaults to 1_000.
            seed (int, optional): Seed. Defaults to 123456.
        """

        self.df = df
        self.hyperparameters = hyperparameters
        self._features = features
        self._media_channels = media_channels
        self._target = target
        self.n_trials = n_trials
        self.seed = seed
        self.model = LinearRegression()  # use the same model to make the predictions!

    def _optimize_objective(self, trial):

        hyperparams_trial_dict: dict = {}

        for param in self.hyperparameters:
            # create the hyper param dictionary that
            # fits what Optuna is expecting
            _low = self.hyperparameters.get(param)[0]
            _high = self.hyperparameters.get(param)[1]
            typ = self.hyperparameters.get(param)[2]

            if typ.lower() == "float":
                hyperparams_trial_dict[param] = trial.suggest_float(
                    param, _low, _high
                )
            elif typ.lower() == "int":
                hyperparams_trial_dict[param] = trial.suggest_int(
                    param, _low, _high
                )
            else:
                raise ValueError(
                    "Wrong type parameter. Only 'int' and 'float' allowed"
                    f"{typ} received"
                )

        # Add adstock variables
        df_adstock = add_adstock_to_pdf(
            self.df, hyperparams_trial_dict, self._media_channels
        )
        X = df_adstock[self._features]
        Y = df_adstock[self._target]
        # fit model
        self.model.fit(X, Y)
        y_hat = self.model.predict(X)
        # we are going to optimize in mean squared error.
        return mean_squared_error(Y, y_hat)

    def hyperparam_tuning(self) -> optuna.Study:
        """Optimize the `self.hyperparameters` using `self.model`

        The optimal parameters are inside the object returned.
        Returns:
            optuna.Study: Result of the study (experiment).
        """
        study_tuned_params = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=self.seed)
        )
        print(f"Optimization under way with {self.n_trials}")
        study_tuned_params.optimize(
            self._optimize_objective, n_trials=self.n_trials
        )
        return study_tuned_params
