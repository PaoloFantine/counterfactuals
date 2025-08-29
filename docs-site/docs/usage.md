# Using the library

Assuming you already have data split in `X` a pandas dataframe containing predictor variables and `y` a pandas series containing the outcomes of the rows in `X`, using the library is rather easy.

In general, `X` is used only to determine existing distributions and data types, and, in the case of **prototypes**, to search for existing solutions so there is no fear of data leakage and the data could be the training data or a larger set if needed.

A model is also needed. The code has been developed with `scikit-learn` in mind, so, as long as a model has methods `.fit`, `.predict` and, in the case of classification, `predict_proba`, it should work easily.

We just need to initialize a counterfactuals class:

```python
from src import Counterfactuals

cf = Counterfactuals(X=X, y=y, model=model)
```

We should then have an instance for which a counterfactual is desired. Say we want to take it from `X`:

```python
instance = X.iloc[0:1].copy
```

but it could be taken by any dataframe, as long as it contains the columns `X` contains.

Now, generating a counterfactual only requires a single call. There is a slight difference depending on whether Prototypes or genetic counterfactuals are needed.

## Prototypes

For prototypes, an instance in `X` is searched that has the desired outcome and is close in feature space to the `instance` we pass:

```python
prototypes = cf.get_counterfactuals(instance, 
                                    n_counterfactuals=5, 
                                    method="prototypes", 
                                    desired_class=1)
```

We just need to specify how many counterexamples we want and what outcome class we want.

Sometimes though, a counterfactual will suggest changing features we cannot change (like our age, sex or height). This would point to a bias in the model probably, but we can get around this by fixing some feature values:

```python
cf.get_counterfactuals(instance, 
                       n_counterfactuals=5, 
                       method="prototypes", 
                       desired_class=1, 
                       fix_vars=['relationship', 'race'])
```
Where, in this case, we fixed features 'relationship' and 'race' from the `adult` dataset.

It may happen that not enough counterfactuals exist in the data. All the more if we restrict the search field by fixing some feature values. In that case, `get_counterfactuals` will only return as many as it finds in the existing data.

## Genetic algorithm

For genetic counterfactuals, existence of the solution is not a problem as they are made to generate solutions and return the fittest ones. 
The call is very similar, we just need to specify the different method:

```python
cf.get_counterfactuals(instance, 
                       n_counterfactuals=5, 
                       method="genetic", 
                       desired_class=1, 
                       fix_vars=['relationship', 'race'])
```

Genetic counterfactuals are based on mutating existing data though. Seems straightforward, but some care needs to be taken whenever one-hot encoded features exist. A mutation could make those two-, three- or n-hot encoded, which, in most cases, would not even mean anything. This is also accounted for by adding the prefix of one-hot encoded variables to ensure co-mutation:

```python
cf.get_counterfactuals(instance, 
                       n_counterfactuals=5, 
                       method="genetic", 
                       desired_class=1, 
                       fix_vars=['relationship', 'race'], 
                       one_hot_encoded = ['sex_'])
```

# Regression counterfactuals

Finally, for regression problems, we might need an interval of outcome values or, at the very list, a lower or upper bound. In this case, we would need to specify at least one of `lower_limit`, `upper_limit`, the code automatically recognizes classification or regression models:

```python
cf.get_counterfactuals(instance, 
                       n_counterfactuals=3, 
                       method='prototypes', 
                       upper_limit=4, 
                       lower_limit=3)
```

Obviously, what was said in the classification case about choosing the method, fixing variables values and specifying one-hot encoded feature prefixes holds for regression as well.
