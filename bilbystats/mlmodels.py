"""
    ML models for classification
"""
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
import numpy as np


def data_idx_split(idx, ratio=0.2, valratio=0.5, random_state=42):
    """
    Split indices into training, validation, and test sets.

    This function splits a set of indices into training, validation, and test 
    sets using specified ratios. First, it divides the indices into a training 
    set and a temporary set (for validation and testing). Then it further splits 
    the temporary set into validation and test sets.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    idx : array-like
        The indices to split (e.g., DataFrame indices).
    ratio : float, optional (default=0.2)
        The proportion of the total indices to allocate to the combined 
        validation and test sets.
    valratio : float, optional (default=0.5)
        The proportion of the temporary (validation + test) set to allocate to 
        the test set.
    random_state : int, optional (default=42)
        The random seed to ensure reproducibility of the splits.

    ---------------------------------------------------------------------------
    OUTPUT:
    indices : dict
        A dictionary with keys 'train', 'valid', and 'test', each containing the 
        corresponding index arrays.

    ---------------------------------------------------------------------------
    Copyright (C) - 2025 - Samuel Davenport
    ---------------------------------------------------------------------------
    """
    # Step 1: Split into train and temporary (valid+test)
    train_indices, temp_indices = train_test_split(
        idx, test_size=ratio, random_state=random_state
    )

    # Step 2: Split temporary data into validation and test sets
    valid_indices, test_indices = train_test_split(
        temp_indices, test_size=valratio, random_state=random_state
    )

    indices = {
        'train': train_indices,
        'valid': valid_indices,
        'test': test_indices
    }

    return indices


def train_val_test_split(df, covariate, target, indices, ratio=0.3, valratio=0.5):
    """
    Split a DataFrame into train, validation, and test datasets.

    This function splits the provided DataFrame into train, validation, and test 
    sets based on indices and specified ratios. It extracts the covariate and 
    target columns for each set and converts them into Hugging Face `Dataset` 
    objects.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    df : pandas.DataFrame
        The DataFrame containing the data to split.
    covariate : str
        The name of the column to be used as the input feature (e.g., text).
    target : str
        The name of the column to be used as the label.
    indices : dict
        A dictionary with keys 'train', 'valid', and 'test' containing index 
        lists for each split.
    ratio : float, optional (default=0.3)
        The proportion of the dataset to allocate to the test and validation 
        sets combined.
    valratio : float, optional (default=0.5)
        The proportion of the non-training set to allocate to the validation 
        set (the rest is test data).

    ---------------------------------------------------------------------------
    OUTPUT:
    train_data : Dataset
        A Hugging Face Dataset object containing the training data.
    valid_data : Dataset
        A Hugging Face Dataset object containing the validation data.
    test_data : Dataset
        A Hugging Face Dataset object containing the test data.

    ---------------------------------------------------------------------------
    Copyright (C) - 2025 - Samuel Davenport
    ---------------------------------------------------------------------------
    """
    # Split the indices of df into train, validation and test sets
    train_indices, valid_indices, test_indices = data_idx_split(
        df.index, ratio=ratio, valratio=valratio)

    # Get the corresponding data from the indices
    train_texts = df.loc[indices['train'], covariate].tolist()
    train_labels = df.loc[indices['train'], target].tolist()

    valid_texts = df.loc[indices['valid'], covariate].tolist()
    valid_labels = df.loc[indices['valid'], target].tolist()

    test_texts = df.loc[indices['test'], covariate].tolist()
    test_labels = df.loc[indices['test'], target].tolist()

    # Convert into Hugging Face dataset format
    train_data = Dataset.from_dict(
        {'text': train_texts, 'label': train_labels})
    valid_data = Dataset.from_dict(
        {'text': valid_texts, 'label': valid_labels})
    test_data = Dataset.from_dict({'text': test_texts, 'label': test_labels})

    return train_data, valid_data, test_data


def logreg_fit(X, y, train_idx):
    """
    Fit a Logistic Regression model on a subset of data.

    This function trains a Logistic Regression model using a subset of the 
    input data, where the subset is defined by the indices provided in 
    `train_idx`. It returns the trained model.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    X : array-like, shape (n_samples, n_features)
        Feature matrix for training the model.
    y : array-like, shape (n_samples,)
        Target labels corresponding to the feature matrix X.
    train_idx : array-like, shape (n_samples_subset,)
         Indices used to restrict the input data (X and y) for training.

    ---------------------------------------------------------------------------
    OUTPUT:
    model : LogisticRegression
         A trained Logistic Regression model fitted on the restricted data.

    ---------------------------------------------------------------------------
    Copyright (C) - 2025 - Samuel Davenport
    ---------------------------------------------------------------------------
    """
    X_restricted = X[train_idx]
    y_restricted = y[train_idx]

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_restricted, y_restricted)

    return model


def logreg_metrics(X, y, model, test_indices, doprint=0):
    """
    Compute and optionally display the accuracy of a Logistic Regression model.

    This function evaluates the performance of a given Logistic Regression model 
    on a test subset of the data, specified by `test_indices`. It computes the 
    accuracy score and, if requested, prints both the accuracy and a detailed 
    classification report.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    X : array-like, shape (n_samples, n_features)
        Feature matrix containing the input data.
    y : array-like, shape (n_samples,)
        Target labels corresponding to the feature matrix X.
    model : LogisticRegression
        A trained Logistic Regression model that implements the `predict` method.
    test_indices : array-like, shape (n_samples_subset,)
        Indices specifying which samples to use as the test set.
    doprint : int, optional (default=0)
        If set to 1, prints the accuracy and the classification report.

    ---------------------------------------------------------------------------
    OUTPUT:
    accuracy : float
        The accuracy of the model on the test subset.

    ---------------------------------------------------------------------------
    Copyright (C) - 2025 - Samuel Davenport
    ---------------------------------------------------------------------------
    """
    X_test = X[test_indices]
    y_test = y[test_indices]
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Print accuracy and classification report if doprint is 1
    if doprint == 1:
        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred))

    return accuracy


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for model predictions.

    This function calculates accuracy, precision, recall, and F1-score for the 
    provided predictions and labels. The metrics are computed using a macro-average 
    to account for class imbalance across multiple classes.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    eval_pred : tuple
        A tuple containing two elements:
        - predictions (array-like): The raw prediction scores or logits output by the model.
        - labels (array-like): The true labels corresponding to the predictions.

    ---------------------------------------------------------------------------
    OUTPUT:
    metrics : dict
        A dictionary containing the computed metrics:
        - 'accuracy' (float): The overall accuracy of the predictions.
        - 'precision' (float): The macro-averaged precision score.
        - 'recall' (float): The macro-averaged recall score.
        - 'f1' (float): The macro-averaged F1 score.

    ---------------------------------------------------------------------------
    Copyright (C) - 2025 - Samuel Davenport
    ---------------------------------------------------------------------------
    """
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")[
        "f1"
    ]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
