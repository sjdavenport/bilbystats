"""
Functions for performing inference and prediction using transformer models
"""
from transformers import AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import bilbystats as bs
import numpy as np
import pandas as pd


def predict(data: Dataset, model_path: str, model_name: str) -> dict:
    """
    Generate predictions from a fine-tuned transformer model on input data.

    This function tokenizes the input data using a specified tokenizer, loads a pre-trained 
    sequence classification model from a given path, and runs inference using Hugging Face's 
    Trainer. It returns predicted logits, predicted class labels, and (if available) true labels.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    data : Dataset or str
        The dataset to generate predictions for, containing at least a 'text' column 
        and optionally a 'label' column. If a string is given then that string is 
        converted into a Dataset format and evaluated in the same way.
    model_path : str
        Path to the directory containing the fine-tuned transformer model.
    model_name : str
        Name or path of the model used to load the corresponding tokenizer.

    ---------------------------------------------------------------------------
    OUTPUT:
    output : dict
        A dictionary containing:
            - 'logits': np.ndarray, the raw output scores from the model.
            - 'pred_labels': np.ndarray, the predicted class labels.
            - 'true_labels': np.ndarray (optional), the true labels if present in the dataset.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    # If the datatype is a string convert to a dataset for evaluation
    if type(data) == str:
        data = Dataset.from_dict({'text': [data]})

    # Tokenize the input data
    data2predict_tk = bs.tokenize(data, model_name)

    # Load in the model and the trainer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(model=model)

    # Calculate the model predictions
    predictions = trainer.predict(data2predict_tk)

    pred_labels = np.argmax(predictions.predictions, axis=1)

    # Save the output as a dictionary
    output = {
        'logits': predictions.predictions,
        'pred_labels': pred_labels
    }

    if hasattr(predictions, 'label_ids') and predictions.label_ids is not None:
        output['true_labels'] = predictions.label_ids

    return output


def predict_df(df: pd.DataFrame, covariate: str, model_path: str, model_name: str,
               indices: list[int] = None, target: str = None) -> dict:
    """
    Generate predictions from a model for a DataFrame subset based on specified covariate.

    This function selects a subset of the input DataFrame (if indices are provided),
    converts the data into a format suitable for model inference, and uses a fine-tuned
    transformer model to compute predictions. Optionally includes the true target labels.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    df : pd.DataFrame
        Input DataFrame containing the covariate column used for prediction.
    covariate : str
        Name of the column in `df` containing the input text data.
    model_path : str
        Path to the directory containing the fine-tuned transformer model.
    model_name : str
        Name or path of the pre-trained model for tokenizer loading.
    indices : list of int, optional
        Indices specifying the subset of rows from `df` to predict on.
        If None, the entire DataFrame is used.
    target : str, optional
        Name of the column in `df` containing target labels, if available.

    ---------------------------------------------------------------------------
    OUTPUT:
    output : dict
        A dictionary containing:
            - 'logits': np.ndarray, raw model output scores.
            - 'pred_labels': np.ndarray, predicted class labels.
            - 'true_labels': np.ndarray (optional), ground-truth labels if provided.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    if indices is not None:
        subset_df = df.iloc[indices].copy()
    else:
        subset_df = df.copy()

    data2predict = bs.df2dict(subset_df, covariate, target)

    output = predict(data2predict, model_path, model_name)

    return output
