import numpy as np
import numpy.random as npr
import pandas as pd
import scipy.stats as stats
from pandas.api.types import is_numeric_dtype
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler
import gower


class GA_counterfactuals:
    """
    Provides interactive functionality to investigate how predictions can be reversed to the desired outcome
    Makes new data instances based on a Non-dominated sorting genetic algorithm (NSGA-II) approach

    Parameters
    ----------
    X: pd.DataFrame
        The dataset to look for prototypes in. This should be the same dataset used to train the model or a dataframe with the same format
    y: pd.Series
        The outcome of the dataset. This should be the same outcome used to train the model or a series with the same format
    model:
        a model with a scikit-learn compatible .fit, .predict and .predict_proba methods
    one_hot_encoded: list(str)
            prefixes of one-hot encoded features to guarantee co-mutation
            e.g. if one-hot encoded features are "cat__A", "cat__B", "cat__C", then one_hot_encoded=["cat__"]
            will guarantee that if "cat__A" is mutated, then "cat__B" and "cat__C" will
            be mutated as well, so that the one-hot encoding is preserved
    """

    def __init__(self, X, y, model, one_hot_encoded=None):

        self.y = y
        self.X = X
        self.model = model
        self.features = self.X.columns
        
        if one_hot_encoded is not None:
            self.one_hot_encoded = [
                col for col in self.X.columns if col.startswith(tuple(one_hot_encoded))
            ]
        else:
            self.one_hot_encoded = []
            
        self._ohe_prefixes = one_hot_encoded
        
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
        
        # set predict proba
        if self.problem_type == "classification":
            self.predict_proba = True
        else:
            self.predict_proba = False
            
    def generate_counterfactuals(
        self,
        base,
        n_counterfactuals,
        lower_limit=None,
        upper_limit=None,
        desired_class=None,
        fix_vars=None,
    ):
        """
        evolve the initialized data until the desired amount of counterfactuals is achieved

        Parameters
        ----------
        base: pandas.DataFrame
            single-row dataframe for which counterfactuals are desired
        n_counterfactuals: int
            number of counterfactuals to generate
        lower_limit: float, int
            for regression problems, the lower limit to the desired range of counterfactual predictions
        upper_limit: float, int
            for regression problems, the upper limit to the desired range of counterfactual predictions
        desired_class: int
            for classification problems, the desired class for counterfactual predictions
        fix_vars: list(str)
            features to keep fixed to base instance value when generating counterfactuals
        """
        # note down the base instance; needed for fitness evaluation
        self.base = base
        
        if fix_vars is not None:
            self.features_to_vary = list(set(self.X.columns) - set(fix_vars))
        else:
            self.features_to_vary = self.X.columns.tolist()
        
        # initialize parents
        parents = self._initialize_parents(
            base=base,
            n_counterfactuals=n_counterfactuals,
        )
        
        # Dataframe to track found solutions
        solutions = pd.DataFrame(columns=list(self.X.columns) + ["outcome"])
        
        while len(solutions) < 2*n_counterfactuals:  # finding more counterfactuals than needed to allow for selection of the fittest ones     
            # -- evaluate parents fitness
            if self.problem_type == "classification":
                parents['outcome'] = self.model.predict_proba(parents[self.X.columns])[:, desired_class]
                selected_parents = parents[parents['outcome'] > 0.5]
            elif self.problem_type == "regression":
                parents['outcome'] = self.model.predict(parents[self.X.columns])
                selected_parents = parents[
                    (parents['outcome'] >= lower_limit) & (parents['outcome'] <= upper_limit)
                ]
            
            # track the counterfactuals generated so far
            if not selected_parents.empty:
                if solutions.empty:
                    solutions = selected_parents.copy()
                else:
                    solutions = pd.concat([solutions, selected_parents], ignore_index=True)
            
            current_generation = self._evaluate_selection_probability(
                parents=parents,
                desired_class=desired_class,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            )
        
            # -- pair parents according to their selection probability
            couples = [
                npr.choice(
                    list(range(len(current_generation.index))),
                    2,
                    p=list(current_generation.selection_probability),
                    replace=False,
                )
                for i in range(len(current_generation.index))
            ]
        
            children = []
        
            # -- reproduction
            for idx1, idx2 in couples:
                parent1 = current_generation[self.X.columns].iloc[[idx1]].copy()
                parent2 = current_generation[self.X.columns].iloc[[idx2]].copy()

                child = pd.DataFrame(index=[0], columns=self.X.columns)
            
                for col in self.X.columns:
                    if parent1[col].dtype in ["int64", "int32", "object", "category"] and col not in self.one_hot_encoded:
                        # for categorical or int values, choose the value of one of the parents
                        # with equal probability
                        child[col] = npr.choice(
                            [parent1[col].values[0], parent2[col].values[0]], 1, p=[0.5, 0.5]
                        )
                    
                    elif parent1[col].dtype in ["float64", "float32"]:
                        # for float values, choose the average of the parents
                        child[col] = (parent1[col].values[0] + parent2[col].values[0]) / 2
                    
                # for one-hot encoded columns, choose one of the parents' beforehand and assign
                # the whole array of one-hot encoded values to the child. This ensure co-mutation
                for prefix in self._ohe_prefixes:
                    ohe_parent = npr.choice(["parent1", "parent2"], 1, p=[0.5, 0.5])[0]
                    ohe_cols = [col for col in self.X.columns if col.startswith(prefix)]
                    for col in ohe_cols:
                        if ohe_parent == "parent1":
                            child[col] = parent1[col].values[0]
                        else:
                            child[col] = parent2[col].values[0]
            
                
                # append the child to the list of children and allow for mutation to explore the feature space better       
                children.append(self._data_mutation(child, self.features_to_vary))
            
            parents = pd.concat(children, ignore_index=True)
            
        solutions = self._evaluate_selection_probability(
            parents=solutions,
            desired_class=desired_class,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        
        solutions = solutions.nlargest(n_counterfactuals, "selection_probability")
            
        return solutions[list(self.X.columns)+["outcome"]].reset_index(drop=True)
    
    def _initialize_parents(self, base, n_counterfactuals):
        '''
        Generate enough parents to start the evolution process from the base instance
        '''

        # loop _data_mutation to have enough parents
        
        mutation_batch = pd.concat(
            [self._data_mutation(to_be_mutated=base, features_to_vary=self.features_to_vary) for _ in range(2*n_counterfactuals)]
        )
        
        return mutation_batch
    
    def _data_mutation(self, to_be_mutated, features_to_vary):
        """
        Mutate data instance before next generation
        """
        gene_dict = dict(to_be_mutated)
        
        regular_features = [col for col in features_to_vary if col not in self.one_hot_encoded]
        
        for col in regular_features:
            if self.X[col].dtype == "float": #numerical features
                
                # choose a random value from the distribution of the feature
                var_distribution = stats.truncnorm(
                        self.X[col].min(), self.X[col].max(), scale=self.X[col].std()
                    )
                res = var_distribution.rvs(len(self.X.index))
                
            elif self.X[col].dtype in [
                    "int",
                    "object",
                    "category",
                    "bool",
                ]:
                    # choose one of the available int/categorical values
                    res = list(self.X[col])
                    
            mutation = npr.choice(res, 1)

            # 20% chance to mutate the feature
            gene_dict[col] = npr.choice([mutation[0], gene_dict[col][0]], 1, p=[0.2, 0.8])
            
        # co-mutate one-hot encoded features
        for prefix in self._ohe_prefixes:
            ohe_cols = [col for col in self.X.columns if col.startswith(prefix)]
            res = list(self.X[ohe_cols].values)
            
            mutation_idx = npr.choice(range(len(res)), 1)[0]
            
            mutation_choice = npr.choice([True, False], 1, p=[0.2, 0.8])[0]
            
            for i, col in enumerate(ohe_cols):
                if mutation_choice:
                    gene_dict[col] = res[mutation_idx][i]
                else:
                    continue
                
            
            
            
        gene_val = list(gene_dict.values())[0]
        
        if isinstance(gene_val, (str, int, float)):
            return pd.DataFrame(gene_dict, index=[0])
        else:
            return pd.DataFrame(gene_dict)
                

    def _evaluate_fitness(self, current_generation, desired_class=None, lower_limit=None, upper_limit=None):
        """
        Evaluate the fitness of the current generation
        """
        
        # solution range
        # -- verify how close we are to the desired solution
        current_generation['outcome_fitness'] = self._solution_fitness(
            current_generation, 
            desired_class=desired_class, 
            lower_limit=lower_limit, 
            upper_limit=upper_limit
        )
        
        # point-likelihood
        # -- check how likely it is for the point to be in the feature space
        current_generation['point_likelihood_fitness'] = self._point_likelihood_fitness(current_generation)
        
        # sparsity
        # -- check the amount of features changed is minimal
        # len(self.X.columns) - count of features that have changed gives higher fitness for solutions that have changed the least features
        current_generation["sparsity_fitness"] = len(self.X.columns) - current_generation[self.X.columns].apply(
            lambda x: list(np.not_equal(x.values, self.base[self.X.columns].values)[0]).count(
                True
            ),
            axis=1,
        )
        
        # closeness to base instance
        # -- check how close the mutated instance is to the base instance
        # 1 - gower distance to have higher fitness for solutions that are closer to the base instance
        base_current = pd.concat([self.base, current_generation[self.X.columns]], axis=0, ignore_index=True)
        current_generation["closeness_fitness"] = 1 - gower.gower_matrix(base_current[self.X.columns])[0, 1:].mean()
        
        return current_generation.reset_index(drop=True)
    
    def _solution_fitness(
        self, 
        current_generation, 
        desired_class=None, 
        lower_limit=None, 
        upper_limit=None
    ):
        """
        Calculate the fitness of a single solution
        """
        
        # check how close the solution is to the desired class/range
        if self.problem_type == "classification":
            # 'outcome' is the predicted probability of the desired class, so the probability is the fitness in this case
            outcome_fitness = current_generation['outcome']
        elif self.problem_type == "regression":
            outcome_fitness = np.min(abs(current_generation['outcome'] - lower_limit), abs(upper_limit - current_generation['outcome']))
            
        return outcome_fitness
    
    def _point_likelihood_fitness(self, current_generation):
        """
        Calculate the fitness of a point based on its likelihood in the feature space.
        This means finding the closest points in feature space to each of the points in the current generation. 
        This step uses a KDTree, which is not perfect in the case we are dealing with categorical features, but simplifies the search a lot.
        The final distance is then computed using the gower distance to account for the categorical features. 
        The idea is to balance quality and speed, as the gower distance scales like O(n^2) and would not be feasible for large datasets.
        """
        
        # scaling the data to have all features in the same range
        scaler = MinMaxScaler() 
        scaled_X = scaler.fit_transform(self.X)
        scaled_generation = scaler.transform(current_generation[self.X.columns])
        
        # use KDTree to find 5 closest points in the feature space
        # 5 is rather arbitrary. One could experiment with different values to see how it affects speed and quality
        kd = KDTree(scaled_X)
        
        nearest_neighbors = [
            self.X.iloc[kd.query(scaled_generation[i], k=5, eps=3, p=1, workers=1)[1]]
            for i in range(len(scaled_generation))
        ]
        
        # concatenate the points in current generation to their nearest neighbors
        point_distribution_dfs = [pd.concat([current_generation[self.X.columns].iloc[i:i+1], nearest_neighbors[i]], axis=0, ignore_index=True) for i in range(len(current_generation))]
        
        # convert "category" columns to "object" to avoid gower distance errors
        for i in range(len(point_distribution_dfs)):
            for col in point_distribution_dfs[i].select_dtypes(include=["category"]).columns:
                point_distribution_dfs[i][col] = point_distribution_dfs[i][col].astype("object")
        
        # find average gower distance between the current generation and its nearest neighbors
        # The slicing [0, 1:] takes the row of the distance matrix corresponding to the current point in point_distribution_dfs [0]
        # and its distances to the rest of te points [1:]; the diagonal of the distance matrix is identically 0
        # Returning 1- gower distance because fitness is higher when the distance is lower
        gower_distance = [1 - gower.gower_matrix(point_distribution_dfs[i])[0, 1:].mean() for i in range(len(point_distribution_dfs))]

        
        return gower_distance
    
    def _nondominance_sorting(self, current_fitness):
        """
        Determine non-dominance fronts in the current generation based on its individuals' fitness 
        """
        
        # Establish dominance in the current generation
        dominates = {i:[] for i in current_fitness.index}
        dominated_by_count = {i:0 for i in current_fitness.index}
        
        for i in current_fitness.index:
            for j in current_fitness.index:
                # dominance is established by being fitter in at least one fitness metric and not worse in any metric
                # fitness metrics are such that lower is worse so we check that at least one is higher and none is lower
                if (list(current_fitness.iloc[i] > current_fitness.iloc[j]).count(True) > 0) and (list(current_fitness.iloc[i] < current_fitness.iloc[j]).count(True) == 0):
                    # i dominates j
                    # track the indices row i dominates
                    dominates[i].append(j)
                        
                    # track how many solutions j is dominated by
                    dominated_by_count[j] += 1
    
        # Assign domination fronts
        front_counter=1  
        current_fitness.loc[:, "front"] = 0     
        while dominates:
            # find indices dominating the current front (not dominated by any solution)
            front_idx = [i for i in dominated_by_count.keys() if dominated_by_count[i] == 0]
            
            current_fitness.loc[front_idx, "front"] = front_counter
            front_counter+=1
            
            # remove current front from further competition
            reduce_dominated_by_count = [dominates[i] for i in front_idx]
            reduce_dominated_by_count_flattened = [item for sublist in reduce_dominated_by_count for item in sublist]
            
            for i in reduce_dominated_by_count_flattened:
                dominated_by_count[i] -= 1
                
            for idx in front_idx:
                # remove dominated solutions from the current generation
                del dominates[idx]
                del dominated_by_count[idx]

            
        return current_fitness['front']
    
    def _crowding_distance(self, current_fitness):
        """
        calculate crowding distance of found solutions to encourage diversity
        in selection across generations
        """

        fitness_metrics = current_fitness.copy()[
            [col for col in current_fitness.columns if "_fitness" in col]
        ]

        for col in fitness_metrics.columns:

            fitness_metrics = fitness_metrics.sort_values(col)

            # in case the column has no variance (max=min) distance should amount to 0, not nan
            # this also avoids a divide by 0
            column_range = max(fitness_metrics[col]) - min(fitness_metrics[col])
            if column_range == 0:
                range_ = 1
            else:
                range_ = column_range

            # partial crowding distance calculation:
            distance = [
                (fitness_metrics[col].iloc[i + 1] - fitness_metrics[col].iloc[i - 1])
                / range_
                for i in range(1, len(fitness_metrics.index) - 1)
            ]

            column = np.zeros(len(fitness_metrics.index))
            column[1:-1] = distance
            column[0] = column[-1] = float("inf")
            fitness_metrics[f"meta_{col}"] = column

        # crowding distance calculation
        current_fitness["crowding_distance"] = fitness_metrics[
            [col for col in fitness_metrics.columns if col.startswith("meta_")]
        ].sum(axis=1)

        return current_fitness
    
    def _evaluate_selection_probability(self, parents, desired_class=None, lower_limit=None, upper_limit=None):
        """
        Evaluate the selection probability of the current generation
        """
        
        current_generation = self._evaluate_fitness(
                current_generation=parents, 
                desired_class=desired_class, 
                lower_limit=lower_limit, 
                upper_limit=upper_limit
            )  
        
        # -- Nondominance sorting    
        current_generation['front'] = self._nondominance_sorting(current_generation[['outcome_fitness', 'point_likelihood_fitness', 'sparsity_fitness', 'closeness_fitness']].copy())
        
        # -- crowding distance
        current_generation = self._crowding_distance(current_generation)
        
        # -- selection probability
        # find combined overall ranking of nondominance+crowding distance
        current_generation["crowding_rank"] = current_generation.crowding_distance.rank(
            ascending=True, pct=True
        )

        # rank fronts by
        current_generation["front_rank"] = current_generation["front"].rank(ascending=False, pct=True)

        current_generation["combined_rank"] = 0.5*(
            current_generation["front_rank"] + current_generation["crowding_rank"]
        )

        # compute reproduction probability
        current_generation["selection_probability"] = current_generation["combined_rank"] / sum(
            current_generation["combined_rank"]
        )
        
        return current_generation.reset_index(drop=True)