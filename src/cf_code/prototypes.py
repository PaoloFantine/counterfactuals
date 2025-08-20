import warnings

# Imports
import numpy as np
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler


class Prototypes:
    def __init__(self, X, y, model):
        """
        Provides functionality to look up prototypes in the dataset showcasing the desired outcome from the model

        Parameters
        ----------
        X: pd.DataFrame
            The dataset to look for prototypes in. This should be the same dataset used to train the model or a dataframe with the same format
        y: pd.Series
            The outcome of the dataset. This should be the same outcome used to train the model or a series with the same format
        model:
            a model with a scikit-learn compatible .fit, .predict and .predict_proba methods
        """

        self.model = model

        self.y = y
        self.X = X

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

    def get_prototypes(
        self,
        base,
        desired_class=None,
        lower_limit=None,
        upper_limit=None,
        fix_vars=None,
        k=4,
        distance_range=4,
        p=1,
        workers=1,
    ):
        """
        Query kdtree with base to find k closest counterfactuals

        Parameters
        ----------
        base: pd.DataFrame
            The data instance for which the counterfactual is desired. This should be the row
            of the dataframe for which the counterfactual should be generated, including the original
            outcome.
        desired_class: int, float
            Desired class for classification problems. In regression problems, it is generated as a str
            of shape {lower_limit}_{upper_limit} as a key for kdtrees
        lower_limit: int, float
            The lower limit desired in case of regression problems
        upper_limit: int, float
            The upper limit desired in case of regression problems
        fix_vars: list (str), optional
            list of *base* columns whose value should be kept fixed when generating counterfactuals
        k: int or Sequence[int], optional
            the number of counterfactuals to return
        distance_range: nonnegative float, optional
            Return approximate nearest neighbors; the kth returned value is guaranteed to be
            no further than (1+distance_range) times the distance to the real kth nearest neighbor.
            It is advised to leave this value to default (4), as distances in k-dimensional space might just be weird
            for most humans to interpret.
        p: float, 1<=p<=infinity, optional
            Which Minkowski p-norm to use. 1 is the sum-of-absolute-values distance (“Manhattan” distance).
            2 is the usual Euclidean distance. infinity is the maximum-coordinate-difference distance.
            A large, finite p may cause a ValueError if overflow can occur.
            Manhattan distance is recommended for most use cases, especially as dimensions increase.
            Default: 1.
        workers: int, optional
            Number of workers to use for parallel processing. If -1 is given all CPU threads are used. Default: 1.

        Returns
        ------
        Dict with structure {'base': pd.DataFrame , 'counterfactuals':pd.DataFrame} giving the instance and
        its k nearest neighbors ('counterfactuals')
        """

        # check inputs
        if self.problem_type == "regression" and (
            (lower_limit is None) & (upper_limit is None)
        ):
            raise ValueError(
                "one of [lower_limit, upper_limit] should be set for regression problems"
            )

        if self.problem_type == "classification" and desired_class is None:
            raise ValueError("desired_class should be set for classification problems")

        # Drop outcome column
        base_features = base.copy()[self.X.columns]

        # check base shape
        if len(base_features.index) > 1:
            raise ValueError(
                f"'base' should be a single row, got {len(base_features.index)}"
            )

        if not hasattr(self, "kdtrees"):
            # initialize kdtrees dict
            self.kdtrees = {
                desired_class: self._kdtree(
                    base, desired_class, lower_limit, upper_limit, fix_vars
                )
            }
        elif desired_class not in self.kdtrees.keys():
            # if desired_class is not in kdtrees dict, add it
            self.kdtrees[desired_class] = self._kdtree(
                base, desired_class, lower_limit, upper_limit, fix_vars
            )

        # get kdtree, scaler and data for desired_class
        kd, scaler, data = self.kdtrees[desired_class]

        # scale base features to search for nearest neighbors in the scaled space
        base_scaled = scaler.transform(base_features)

        # query kdtree for k nearest neighbors
        if len(data.index) < k:
            idx = kd.query(
                base_scaled,
                k=len(data.index),
                eps=distance_range,
                p=p,
                workers=workers,
            )[1][0]
            warnings.warn(
                f"only {len(data.index)} counterfactuals found. k was set to {k}, returning available counterfactuals:"
            )
        else:
            idx = kd.query(base_scaled, k=k, eps=distance_range, p=p, workers=workers)[
                1
            ][0]

        return data.iloc[idx]

    def _kdtree(
        self,
        base,
        desired_class=None,
        lower_limit=None,
        upper_limit=None,
        fix_vars=None,
    ):
        """
        Initialize KDtree to compute distances between data instances of desired_class
        """

        df = self.X.copy()
        df["outcome"] = self.y

        # Restrict the search to oucomes we want to see as counterfactuals
        if self.problem_type == "regression":
            # if lower_/upper_limit are defined, take their values, otherwise set it to +/- infinity
            lower_limit = [lower_limit, -np.inf][lower_limit is None]
            upper_limit = [upper_limit, np.inf][upper_limit is None]
            data = df[(df["outcome"] >= lower_limit) & (df["outcome"] <= upper_limit)]
        else:
            data = df[df["outcome"] == desired_class]

        # Restrict search to fixed feature values whenever some features should be fixed
        if fix_vars:
            for col, value in base.iloc[0][fix_vars].items():
                data = data[data[col] == value]

        data_features = data.copy().drop("outcome", axis=1)
        # scale for computing distance evenly (not scaling might lead to high-scale features dominating the distance)
        scaler = MinMaxScaler()
        if data_features.shape[0] > 0:  # only scale if there is data to search from
            scaler.fit(data_features)
            df_scaled = scaler.transform(data_features)

            # return kdtree for later search
            return KDTree(df_scaled), scaler, data
        elif self.problem_type == "classification":
            raise ValueError(
                f"No counterfactual found for desired outcome {desired_class} with fix_vars {fix_vars}"
            )
        elif self.problem_type == "regression":
            raise ValueError(
                f"No counterfactual found for desired outcome in range [{lower_limit}, {upper_limit}] with fix_vars {fix_vars}"
            )
