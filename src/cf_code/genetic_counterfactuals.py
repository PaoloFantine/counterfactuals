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
    """

    def __init__(self, X, y, model):

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
        
        # initialize parents
        parents = self._initialize_parents(
            base=base,
            n_counterfactuals=n_counterfactuals,
            fix_vars=fix_vars
        )
        
        solutions = pd.DataFrame(columns=self.X.columns)
        
        # -- evaluate parents fitness
        if self.problem_type == "classification":
            parents['outcome'] = self.model.predict_proba(parents)[:, desired_class]
        elif self.problem_type == "regression":
            parents['outcome'] = self.model.predict(parents)
            
        current_generation = self._evaluate_fitness(
            current_generation=parents, 
            desired_class=desired_class, 
            lower_limit=lower_limit, 
            upper_limit=upper_limit
        )  
        
        check=0     
              
        
        # -- pair parents randomly
        
        # -- reproduction
        
        # -- evaluate fitness
        
        # -- select best parents
        # -- -- rank based on nondominance+distance
        
        # stop when enough counterfactuals are generated
        
        # select best children
        
        # reproduce
        pass
    
    def _initialize_parents(self, base, n_counterfactuals, fix_vars):
        '''
        Generate enough parents to start the evolution process from the base instance
        '''

        if fix_vars is not None:
            features_to_vary = list(set(self.X.columns) - set(fix_vars))
        else:
            features_to_vary = self.X.columns.tolist()

        # loop _data_mutation to have enough parents
        
        mutation_batch = pd.concat(
            [self._data_mutation(to_be_mutated=base, features_to_vary=features_to_vary) for _ in range(2*n_counterfactuals)]
        )
        
        return mutation_batch
    
    def _data_mutation(self, to_be_mutated, features_to_vary):
        """
        Mutate data instance before next generation
        """
        gene_dict = dict(to_be_mutated)
        
        for col in features_to_vary:
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

            gene_dict[col] = npr.choice(res, 1)
            
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
        current_generation = self._solution_fitness(
            current_generation, 
            desired_class=desired_class, 
            lower_limit=lower_limit, 
            upper_limit=upper_limit
        )
        
        # point-likelihood
        # -- check how likely it is for the point to be in the feature space
        current_generation = self._point_likelihood_fitness(current_generation)
        
        # sparsity
        # -- check the amount of features changed is minimal
        
        # closeness to base instance
        # -- check how close the mutated instance is to the base instance
        pass
    
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
            current_generation['outcome_fitness'] = abs(current_generation['outcome']- desired_class)
        elif self.problem_type == "regression":
            current_generation['outcome_fitness'] = np.min(abs(current_generation['outcome'] - lower_limit), abs(upper_limit - current_generation['outcome']))
            
        return current_generation
    
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
        
        # use KDTree to find closest points in the feature space
        kd = KDTree(scaled_X)
        
        nearest_neighbors = [
            self.X.iloc[kd.query(scaled_generation[i], k=5, eps=3, p=1, workers=1)[1]]
            for i in range(len(scaled_generation))
        ]
        
        gower = pd.DataFrame(
            [
                stats.gower_distance(current_generation.iloc[i], nearest_neighbors[i])
                for i in range(len(current_generation))
            ]
        )
        
        return gower.mean(axis=1).to_frame(name='point_likelihood_fitness')
        
    
            
            
        