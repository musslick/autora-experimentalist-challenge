"""
Example Experimentalist
"""
import numpy as np
import pandas as pd

from typing import Union, List


import itertools
import warnings
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from autora.utils.deprecation import deprecated_alias


def score_sample(
    conditions: Union[pd.DataFrame, np.ndarray],
    models: List,
    num_samples: Optional[int] = None,
):


    if isinstance(conditions, Iterable) and not isinstance(conditions, pd.DataFrame) and not isinstance(conditions, list):
        conditions = np.array(list(conditions))

    condition_pool_copy = conditions.copy()

    if isinstance(conditions, list):
        X_predict = conditions
    else:
        conditions = np.array(conditions)
        X_predict = np.array(conditions)
        if len(X_predict.shape) == 1:
            X_predict = X_predict.reshape(-1, 1)

    model_disagreement = list()

    # collect diagreements for each model pair
    for model_a, model_b in itertools.combinations(models, 2):

        # determine the prediction method
        if hasattr(model_a, "predict_proba") and hasattr(model_b, "predict_proba"):
            model_a_predict = model_a.predict_proba
            model_b_predict = model_b.predict_proba
        elif hasattr(model_a, "predict") and hasattr(model_b, "predict"):
            model_a_predict = model_a.predict
            model_b_predict = model_b.predict
        else:
            raise AttributeError(
                "Models must both have `predict_proba` or `predict` method."
            )

        if isinstance(X_predict, list):
            disagreement_part_list = list()
            for element in X_predict:
                if not isinstance(element, np.ndarray):
                    raise ValueError("X_predict must be a list of numpy arrays if it is a list.")
                else:
                    disagreement_part = compute_disagreement(model_a_predict, model_b_predict, element)
                    disagreement_part_list.append(disagreement_part)
            disagreement = np.sum(disagreement_part_list, axis=1)
        else:
            disagreement = compute_disagreement(model_a_predict, model_b_predict, X_predict)

        model_disagreement.append(disagreement)

    assert len(model_disagreement) >= 1, "No disagreements to compare."

    # sum up all model disagreements
    summed_disagreement = np.sum(model_disagreement, axis=0)

    if isinstance(condition_pool_copy, pd.DataFrame):
        conditions = pd.DataFrame(conditions, columns=condition_pool_copy.columns)
    elif isinstance(condition_pool_copy, list):
        conditions = pd.DataFrame({'X': conditions})
    else:
        conditions = pd.DataFrame(conditions)

    # normalize the distances
    scaler = StandardScaler()
    score = scaler.fit_transform(summed_disagreement.reshape(-1, 1)).flatten()

    # order rows in Y from highest to lowest
    conditions["score"] = score
    conditions = conditions.sort_values(by="score", ascending=False)

    if num_samples is None:
        return conditions
    else:
        return conditions.head(num_samples)


##################################################################################################

def compute_disagreement(model_a_predict, model_b_predict, X_predict,w_disagreement=1,w_distance=1):
    # get predictions from both models
    y_a = model_a_predict(X_predict)
    y_b = model_b_predict(X_predict)

    assert y_a.shape == y_b.shape, "Models must have same output shape."

    # determine the disagreement between the two models in terms of mean-squared error
    if len(y_a.shape) == 1:
        disagreement = (.2*(y_a - y_b)**2 + (X_predict-np.median(X_predict))**2)/np.sqrt(((y_a - y_b)**2 + (X_predict-np.median(X_predict))**2))
        # disagreement = w_disagreement*np.median(X_predict)*(y_a - y_b) ** 2 + w_distance*(X_predict-np.median(X_predict))**2
    else:
        disagreement = np.power(np.sqrt(.2*np.power((y_a - y_b),2) + np.power(X_predict-np.median(X_predict),2)),-1)*(np.power((y_a - y_b),2) + np.power(X_predict-np.median(X_predict),2)) #np.mean((y_a - y_b) ** 2, axis=1)+(X_predict-np.median(X_predict))**2

    if np.isinf(disagreement).any() or np.isnan(disagreement).any():
        warnings.warn('Found nan or inf values in model predictions, '
                      'setting disagreement there to 0')
    disagreement[np.isinf(disagreement)] = 0
    disagreement = np.nan_to_num(disagreement)
    return disagreement
##################################################################################################
def sample(
    conditions: Union[pd.DataFrame, np.ndarray], models: List, num_samples: int = 1):
   

    selected_conditions = score_sample(conditions, models, num_samples)
    selected_conditions.drop(columns=["score"], inplace=True)

    return selected_conditions

##################################################################################################
model_disagreement_sample = sample
model_disagreement_score_sample = score_sample
model_disagreement_sampler = deprecated_alias(
    model_disagreement_sample, "model_disagreement_sampler"
)




# def sample(
#         conditions: Union[pd.DataFrame, np.ndarray],
#         models: List,
#         reference_conditions: Union[pd.DataFrame, np.ndarray],
#         num_samples: int = 1) -> pd.DataFrame:
#     """
#     Add a description of the sampler here.

#     Args:
#         conditions: The pool to sample from.
#             Attention: `conditions` is a field of the standard state
#         models: The sampler might use output from the theorist.
#             Attention: `models` is a field of the standard state
#         reference_conditions: The sampler might use reference conditons
#         num_samples: number of experimental conditions to select

#     Returns:
#         Sampled pool of experimental conditions

#     *Optional*
#     Examples:
#         These examples add documentation and also work as tests
#         >>> example_sampler([1, 2, 3, 4])
#         1
#         >>> example_sampler(range(3, 10))
#         3

#     """
#     if num_samples is None:
#         num_samples = conditions.shape[0]

#     new_conditions = conditions

#     return new_conditions[:num_samples]
