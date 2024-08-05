"""
SAME Experimentalist
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
import math
# experimentalist
from autora.experimentalist.grid import grid_pool
from autora.experimentalist.random import random_pool, random_sample
from autora.experimentalist.falsification import falsification_sample
from autora.experimentalist.model_disagreement import model_disagreement_sample
from autora.experimentalist.uncertainty import uncertainty_sample
from autora.experimentalist.falsification import falsification_pool
from autora.experimentalist.novelty import novelty_sample



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

def SAME_sample_type_alpha(conditions: Union[pd.DataFrame, np.ndarray], 
                models: List,
                reference_conditions: Union[pd.DataFrame, np.ndarray],
                present_cycle: int,
                total_no_of_cycles: int,
                num_samples: int = 1):

    limit_val_1 = math.ceil(total_no_of_cycles*0.5)

    if present_cycle <= limit_val_1:
        if isinstance(conditions, pd.DataFrame):
            sorted_df = conditions.sort_values(by=list(conditions.columns), kind='mergesort').reset_index(drop=True)
            if present_cycle>0:
              for i in sorted_df.columns:
                if sorted_df[i].nunique()>present_cycle:
                  sorted_df = sorted_df[~sorted_df[i].isin(reference_conditions[i])]
            if num_samples == 1:
              return sorted_df.iloc[math.ceil(len(sorted_df)/(limit_val_1)) * present_cycle:math.ceil(len(sorted_df)/(limit_val_1)) * present_cycle + 1]
            else:
              return sorted_df.iloc[math.ceil(len(sorted_df)/(limit_val_1)) * present_cycle:(math.ceil(len(sorted_df)/(limit_val_1)) * (present_cycle)) + num_samples]
        
        # Ensure conditions is a NumPy array
        elif not isinstance(conditions, np.ndarray):
            raise ValueError("conditions must be a pandas DataFrame or a NumPy array")
        
        # Get the number of columns
        num_columns = conditions.shape[1]
        # Create a tuple of column arrays for lexsort, in reverse order for proper sorting
        sort_keys = tuple(conditions[:, i] for i in reversed(range(num_columns)))
        
        # Get sorted indices
        sorted_indices = np.lexsort(sort_keys)
        
        # Return the sorted array
        sorted_array = conditions[sorted_indices]
        if num_samples == 1:
          return sorted_array[limit_val_1 * present_cycle:limit_val_1 * present_cycle + 1]
        else:
          return sorted_array[limit_val_1 * present_cycle:(limit_val_1 * (present_cycle)) + num_samples]

    else:
      if isinstance(conditions, pd.DataFrame) and isinstance(reference_conditions, pd.DataFrame):
        merged_df = conditions.merge(reference_conditions[list(conditions.columns)], how='left', indicator=True)
        conditions = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        return novelty_sample(conditions=conditions, reference_conditions=reference_conditions[list(conditions.columns)], num_samples=num_samples)
      else:
        df1_df = pd.DataFrame(conditions)
        df2_df = pd.DataFrame(reference_conditions[:-1])
        
        # Perform the same anti-join logic
        merged_df = df1_df.merge(df2_df, how='left', indicator=True)
        result_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        conditions = result_df.values
        return novelty_sample(conditions=conditions, reference_conditions=reference_conditions[:-1], num_samples=num_samples)

def SAME_sample_type_beta(conditions: Union[pd.DataFrame, np.ndarray], 
                models: List,
                reference_conditions: Union[pd.DataFrame, np.ndarray],
                present_cycle: int,
                total_no_of_cycles: int,
                num_samples: int = 1):

    limit_val_1 = math.ceil(total_no_of_cycles*0.5)

    if present_cycle <= limit_val_1:
        if isinstance(conditions, pd.DataFrame):
            sorted_df = conditions.sort_values(by=list(conditions.columns), kind='mergesort').reset_index(drop=True)
            if present_cycle>0:
              for i in sorted_df.columns:
                unique_counts = conditions[i].unique()
                result = unique_counts > limit_val_1
                if result.all():
                  sorted_df = sorted_df[~sorted_df[i].isin(reference_conditions[i])]
            if num_samples == 1:
              return sorted_df.iloc[math.ceil(len(sorted_df)/(limit_val_1)) * present_cycle:math.ceil(len(sorted_df)/(limit_val_1)) * present_cycle + 1]
            else:
              return sorted_df.iloc[math.ceil(len(sorted_df)/(limit_val_1)) * present_cycle:(math.ceil(len(sorted_df)/(limit_val_1)) * (present_cycle)) + num_samples]
        
        # Ensure conditions is a NumPy array
        elif not isinstance(conditions, np.ndarray):
            raise ValueError("conditions must be a pandas DataFrame or a NumPy array")
        
        # Get the number of columns
        num_columns = conditions.shape[1]
        # Create a tuple of column arrays for lexsort, in reverse order for proper sorting
        sort_keys = tuple(conditions[:, i] for i in reversed(range(num_columns)))
        
        # Get sorted indices
        sorted_indices = np.lexsort(sort_keys)
        
        # Return the sorted array
        sorted_array = conditions[sorted_indices]
        if num_samples == 1:
          return sorted_array[limit_val_1 * present_cycle:limit_val_1 * present_cycle + 1]
        else:
          return sorted_array[limit_val_1 * present_cycle:(limit_val_1 * (present_cycle)) + num_samples]

    else:
      if isinstance(conditions, pd.DataFrame) and isinstance(reference_conditions, pd.DataFrame):
        for i in conditions.columns:
          unique_counts = conditions[i].unique()
          result = unique_counts > total_no_of_cycles
          if result.all():
            conditions = conditions[~conditions[i].isin(reference_conditions[i])]
          else:
            merged_df = conditions.merge(reference_conditions[list(conditions.columns)], how='left', indicator=True)
            conditions = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        return novelty_sample(conditions=conditions, reference_conditions=reference_conditions[list(conditions.columns)], num_samples=num_samples)
      else:
        df1_df = pd.DataFrame(conditions)
        df2_df = pd.DataFrame(reference_conditions[:-1])
        
        # Perform the same anti-join logic
        merged_df = df1_df.merge(df2_df, how='left', indicator=True)
        result_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        conditions = result_df.values
        return novelty_sample(conditions=conditions, reference_conditions=reference_conditions[:-1], num_samples=num_samples)


def SAME_sample_type_gamma(conditions: Union[pd.DataFrame, np.ndarray], 
                models: List,
                reference_conditions: Union[pd.DataFrame, np.ndarray],
                present_cycle: int,
                total_no_of_cycles: int,
                num_samples: int = 1):

    limit_val_1 = math.ceil(total_no_of_cycles*0.33)
    limit_val_2 = math.ceil(total_no_of_cycles*0.66)
    limit_val_3 = int(total_no_of_cycles*0.83)
    if present_cycle <= limit_val_1:
        if isinstance(conditions, pd.DataFrame):
            sorted_df = conditions.sort_values(by=list(conditions.columns), kind='mergesort').reset_index(drop=True)
            if present_cycle>0:
              for i in sorted_df.columns:
                sorted_df = sorted_df[~sorted_df[i].isin(reference_conditions[i])]
            if num_samples == 1:
              return sorted_df.iloc[math.ceil(len(sorted_df)/(limit_val_1)) * present_cycle:math.ceil(len(sorted_df)/(limit_val_1)) * present_cycle + 1]
            else:
              return sorted_df.iloc[math.ceil(len(sorted_df)/(limit_val_1)) * present_cycle:(math.ceil(len(sorted_df)/(limit_val_1)) * (present_cycle)) + num_samples]
        
        # Ensure conditions is a NumPy array
        elif not isinstance(conditions, np.ndarray):
            raise ValueError("conditions must be a pandas DataFrame or a NumPy array")
        
        # Get the number of columns
        num_columns = conditions.shape[1]
        # Create a tuple of column arrays for lexsort, in reverse order for proper sorting
        sort_keys = tuple(conditions[:, i] for i in reversed(range(num_columns)))
        
        # Get sorted indices
        sorted_indices = np.lexsort(sort_keys)
        
        # Return the sorted array
        sorted_array = conditions[sorted_indices]
        if num_samples == 1:
          return sorted_array[limit_val_1 * present_cycle:limit_val_1 * present_cycle + 1]
        else:
          return sorted_array[limit_val_1 * present_cycle:(limit_val_1 * (present_cycle)) + num_samples]

    elif present_cycle > limit_val_1 and present_cycle <= limit_val_2:
      if isinstance(conditions, pd.DataFrame) and isinstance(reference_conditions, pd.DataFrame):
        merged_df = conditions.merge(reference_conditions[list(conditions.columns)], how='left', indicator=True)
        conditions = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        return novelty_sample(conditions=conditions, reference_conditions=reference_conditions[list(conditions.columns)], num_samples=num_samples)
      else:
        df1_df = pd.DataFrame(conditions)
        df2_df = pd.DataFrame(reference_conditions[:-1])
        
        # Perform the same anti-join logic
        merged_df = df1_df.merge(df2_df, how='left', indicator=True)
        result_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        conditions = result_df.values
        return novelty_sample(conditions=conditions, reference_conditions=reference_conditions[:-1], num_samples=num_samples)
    else:
      if len(models) == 2:
        return model_disagreement_sample(conditions, models, num_samples)
      elif len(models) == 3:
        if present_cycle > limit_val_3:
          return model_disagreement_sample(conditions, models[::2], num_samples)
        else:
          return model_disagreement_sample(conditions, models[:-1], num_samples)
