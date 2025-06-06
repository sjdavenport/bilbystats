#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A series of functions for generating data!
"""
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import bilbystats as bs


def sentence_gen(df: pd.DataFrame, combined_indices: list[int]) -> pd.DataFrame:
    """
    Generate a new DataFrame with extracted and translated sentences from specified indices.

    This function extracts a random sentence from the 'text' column of the input DataFrame 
    at given indices, translates it using a custom translation utility, and builds a new 
    DataFrame containing the original 'title', extracted 'text', and 'translated_text'.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    df : pd.DataFrame
        Input DataFrame containing at least 'title' and 'text' columns.
    combined_indices : list of int
        List of row indices from which to extract sentences and generate outputs.

    ---------------------------------------------------------------------------
    OUTPUT:
    new_df : pd.DataFrame
        A DataFrame containing the selected 'title', extracted 'text' (a sentence), 
        and its 'translated_text'.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    df = pd.DataFrame(
        columns=['title', 'text', 'translated_text'])
    for i in np.arange(len(combined_indices)):
        bs.modul(i, 10)
        idx2use = combined_indices[i]
        title = df['title'].iloc[idx2use]
        text = df['text'].iloc[idx2use]
        random_sentence = bs.get_random_sentence(text, 15)
        # random_sentence = sentence_extractor(text)
        translated_text = bs.translate(random_sentence)

        new_row = {
            'title': title,
            'text': random_sentence,
            'translated_text': translated_text,
        }

        # Append using loc
        df.loc[len(df)] = new_row

    return df


def heatmap(df, column_names, metric='cityblock', normalize=False, normalize_factor=None,
            figsize=(10, 8), annot=True, cmap="viridis", title=None, return_matrix=False):
    """
    Create a heatmap showing distances between columns in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data
    column_names : list
        List of column names to include in the distance calculation
    metric : str, default 'euclidean'
        Distance metric to use. Options: 'euclidean', 'cityblock', 'cosine', 
        'correlation', 'hamming', 'jaccard', etc.
    normalize : bool, default False
        Whether to normalize the distance matrix
    normalize_factor : float or None
        Factor to divide distances by. If None and normalize=True, 
        uses the length of the vectors
    figsize : tuple, default (10, 8)
        Figure size for the heatmap
    annot : bool, default True
        Whether to annotate the heatmap with numerical values
    cmap : str, default "viridis"
        Colormap for the heatmap
    title : str or None
        Title for the heatmap
    return_matrix : bool, default False
        Whether to return the distance matrix DataFrame

    Returns:
    --------
    pandas.DataFrame (optional)
        The distance matrix if return_matrix=True
    """

    # Extract vectors from the specified columns
    vectors = []
    available_columns = []

    for col_name in column_names:
        if col_name in df.columns:
            vectors.append(df[col_name].values)
            available_columns.append(col_name)
        else:
            print(
                f"Warning: Column '{col_name}' not found in DataFrame. Skipping.")

    if len(vectors) < 2:
        raise ValueError(
            "At least 2 valid columns are required to compute distances.")

    # Stack vectors
    vectors = np.vstack(vectors)

    # Compute distance matrix
    if metric == 'cos_sim':
        dist_matrix = create_cosine_similarity_matrix(
            vectors, model_name='spacy')
    else:
        dist_matrix = squareform(pdist(vectors, metric=metric))

    # Normalize if requested
    if normalize:
        if normalize_factor is None:
            # Use the length of the vectors as default normalization factor
            normalize_factor = len(vectors[0])
        dist_matrix = dist_matrix / normalize_factor

    # Create DataFrame
    df_distances = pd.DataFrame(
        dist_matrix, index=available_columns, columns=available_columns)

    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df_distances, annot=annot, cmap=cmap, cbar=True, square=True)

    # Set title
    if title is None:
        metric_name = metric.replace('_', ' ').title()
        title = f'{metric_name} Distance Matrix'
        if normalize:
            title += ' (Normalized)'
    plt.title(title)

    plt.tight_layout()
    plt.show()

    if return_matrix:
        return df_distances


def create_cosine_similarity_matrix(texts, model_name='distilbert-base-uncased'):
    """
    Create a cosine similarity matrix using bs.cos_sim for text data
    """
    n = len(texts)
    similarity_matrix = np.zeros((n, n))

    for k in range(len(texts[0])):
        bs.loader(k, len(texts[0]))
        for i in range(n):
            for j in range(n):
                if i < j:  # Only compute upper triangle to avoid redundancy
                    try:
                        sim = bs.cos_sim(texts[i][k], texts[j][k], model_name)
                    except:
                        print(texts[i][k])
                        print(texts[j][k])
                    similarity_matrix[i, j] += sim
                    similarity_matrix[j, i] += sim  # Symmetric matrix

    similarity_matrix = similarity_matrix/float(len(texts[0]))
    for i in range(n):
        similarity_matrix[i, i] = 1

    return similarity_matrix
