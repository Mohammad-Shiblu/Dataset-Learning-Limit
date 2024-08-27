import pandas as pd
import numpy as n
# Function for discrete case ambiguity 
def calculate_discrete_ambiguity(df_i, class_column):
    """
    Calculate ambiguity for categorical featured datasets.
    Args:
        df (pandas dataframe): Dataset
        class_column (string): class column name
    returns:
        float: ambiguity for categorical dataset.
    """
    df = df_i.copy()
    d = df[class_column].nunique() # no of classes
    # combinaion generators
    feature_columns = [col for col in df.columns if col != class_column]
    df["feature_combination"] = df[feature_columns].astype(str).agg('_'.join, axis=1)
    
    # total no of combinations
    combination_counts = df.groupby(['feature_combination', class_column]).size().reset_index(name='count')
    
    # no of combination
    total_combination_counts = df['feature_combination'].value_counts().reset_index(name='total_count').rename(columns={'index': 'feature_combination'})
    
    combination_probs = pd.merge(combination_counts, total_combination_counts, on='feature_combination')
    combination_probs['probability'] = combination_probs['count'] / combination_probs['total_count']
    
    
    combination_probs_pivot = combination_probs.pivot(index='feature_combination', columns=class_column, values='probability').fillna(0)
    combination_probs_pivot.columns = [f'P(class={col})' for col in combination_probs_pivot.columns]
    
    df = pd.merge(df, combination_probs, on=['feature_combination', class_column], how='left')

    ambiguity = ((1-combination_probs_pivot.max(axis=1))*(d/(d-1))).mean() # d
    # df, combination_probs_pivot can be used for debugging the code
    return ambiguity


def calculate_discrete_error(data, class_column):
    """
    Calculate error for categorical featured datasets.
    Args:
        df (pandas dataframe): Dataset
        class_column (string): class column name
    returns:
        float: error for categorical dataset.
    """
    
    feature_columns = [col for col in data.columns if col != class_column]
    feature_combinations = data.groupby(feature_columns + [class_column]).size().reset_index(name='count')

    pivot_table = feature_combinations.pivot_table(index=feature_columns, columns=class_column, values='count', fill_value=0)

    pivot_table['total_count'] = pivot_table.sum(axis=1)
    class_columns = pivot_table.columns[pivot_table.columns != 'total_count']

    pivot_table['max_count'] = pivot_table[class_columns].max(axis=1)

    pivot_table['error'] = (pivot_table['total_count'] - pivot_table['max_count']) / pivot_table['total_count']

    mean_error = pivot_table['error'].mean()
    # pivot_table can be used fpr debugging the code
    return mean_error
