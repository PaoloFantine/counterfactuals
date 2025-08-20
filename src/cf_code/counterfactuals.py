import numpy as np

from src.cf_code import genetic_counterfactuals, prototypes


class Counterfactuals:
    def __init__(self, X, y, model):
        """
        Provides interactive functionality to investigate how predictions can be reversed to the desired outcome

        Parameters
        ----------
        X: pd.DataFrame
            The dataset to look for prototypes in. This should be the same dataset used to train the model or a dataframe with the same format
        y: pd.Series
            The outcome of the dataset. This should be the same outcome used to train the model or a series with the same format
        model:
            a model with a scikit-learn compatible .fit, .predict and .predict_proba methods
        """

        # Explainer inheritance
        self.y = y
        self.X = X
        self.model = model
        self.features = self.X.columns

        if self.y is None:
            if hasattr(model, "predict_proba"):
                self.problem_type = "classification"
            else:
                self.problem_type = "regression"
        else:
            if self.y.nunique() <= 8:
                self.problem_type = "classification"
            else:
                self.problem_type = "regression"

    def get_counterfactuals(
        self,
        base,
        n_counterfactuals,
        method=["genetic", "prototypes"],
        lower_limit=None,
        upper_limit=None,
        desired_class=None,
        fix_vars=None,
        one_hot_encoded=None,
        **kwargs,
    ):
        """
        base: pandas.DataFrame
            single-row dataframe for which counterfactuals are desired
        n_counterfactuals: int
            number of counterfactuals to generate
        method: str, ['genetic', 'prototypes']
            method to be used in generating counterfactuals. 'genetic' will use a genetic algorithm;
            'prototypes' will look for data instances with the desired outcome
        lower_limit: float, int
            for regression problems, the lower limit to the desired range of counterfactual predictions
        upper_limit: float, int
            for regression problems, the upper limit to the desired range of counterfactual predictions
        desired_class: int
            for classification problems, the desired class for counterfactual predictions
        fix_vars: list(str)
            features to keep fixed to base instance value when generating counterfactuals
        one_hot_encoded: list(str)
            prefixes of one-hot encoded features to guarantee co-mutation
            e.g. if one-hot encoded features are "cat__A", "cat__B", "cat__C", then one_hot_encoded=["cat__"]
            will guarantee that if "cat__A" is mutated, then "cat__B" and "cat__C" will
            be mutated as well, so that the one-hot encoding is preserved. Only needed for genetic algorithm counterfactuals.
        """
        # GA parameters
        self.base_instance = base.copy()

        # set limits if not given
        if self.problem_type == "regression":
            lower_limit = [lower_limit, -np.inf][lower_limit is None]
            upper_limit = [upper_limit, np.inf][upper_limit is None]
            if method == "prototypes":
                # make desired class str for kdtree dict
                desired_class = f"{lower_limit}_{upper_limit}"
        # set upper and lower limit to desired_class to correctly keep kdtree dict in prototype counterfactuals
        elif self.problem_type == "classification":
            lower_limit = [lower_limit, desired_class][lower_limit is None]
            upper_limit = [upper_limit, desired_class][upper_limit is None]

        # set attributes for testing
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        # check instance outcome != desired outcome
        if list(set(self.base_instance.columns) - set(self.X.columns)):
            self.outcome_column = list(
                set(self.base_instance.columns) - set(self.X.columns)
            )[0]
        else:
            self.outcome_column = "outcome"
            self.base_instance.loc[:, "outcome"] = self.model.predict(
                self.base_instance
            )[0]

        if (
            (self.problem_type == "regression")
            and (self.base_instance.iloc[0][self.outcome_column] >= self.lower_limit)
            and (self.base_instance.iloc[0][self.outcome_column] <= self.upper_limit)
        ):
            raise ValueError(
                f"instance outcome is {self.base_instance.iloc[0][self.outcome_column]}, which is already in the range of the desired outcome [{self.lower_limit},{self.upper_limit}]"
            )
        elif self.base_instance.iloc[0][self.outcome_column] == desired_class:
            raise ValueError(
                f"instance outcome is {self.base_instance.iloc[0][self.outcome_column]}, which is the same as\
                desired outcome ({desired_class})"
            )

        # generate counterfactuals using the chosen method
        if method == "genetic":
            GA = genetic_counterfactuals.GA_counterfactuals(
                y=self.y, X=self.X, model=self.model, one_hot_encoded=one_hot_encoded
            )

            counterfactual_df = GA.generate_counterfactuals(
                self.base_instance,
                n_counterfactuals=n_counterfactuals,
                lower_limit=self.lower_limit,
                upper_limit=self.upper_limit,
                desired_class=desired_class,
                fix_vars=fix_vars,
            )

            return counterfactual_df

        elif method == "prototypes":
            prototype = prototypes.Prototypes(
                y=self.y,
                X=self.X,
                model=self.model,
            )

            counterfactual = prototype.get_prototypes(
                self.base_instance,
                lower_limit=self.lower_limit,
                upper_limit=self.upper_limit,
                desired_class=desired_class,
                k=n_counterfactuals,
                fix_vars=fix_vars,
                **kwargs,
            )

            return counterfactual

        else:
            raise NameError(
                f"parameter 'method' should be either 'prototypes' or 'genetic' but '{method}' was given"
            )
