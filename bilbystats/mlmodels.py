"""
    ML models for classification
"""
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate


def data_idx_split(idx, ratio=0.2, valratio=0.5, random_state=42):
    """
    Splits a given list of indices into training, validation, and test sets.

    The function first divides the indices into a training set and a temporary set (which will contain both validation and test data). Then, the temporary set is further split into validation and test sets.

    Parameters:
    -----------
    - idx (list or array-like): A list or array of indices to be split.
    - ratio (float, optional): The proportion of the data to be used for the temporary set (valid + test). Default is 0.2 (20% of the data).
    - valratio (float, optional): The proportion of the temporary set to be used for the validation set. Default is 0.5 (50% of the temporary set).
    - random_state: a means of initializing randomness

    Returns:
    -----------
    - tuple: A tuple containing three elements:
      - train_indices: The indices for the training set.
      - valid_indices: The indices for the validation set.
      - test_indices: The indices for the test set.
    """
    # Step 1: Split into train and temporary (valid+test)
    train_indices, temp_indices = train_test_split(
        idx, test_size=ratio, random_state=random_state
    )

    # Step 2: Split temporary data into validation and test sets
    valid_indices, test_indices = train_test_split(
        temp_indices, test_size=valratio, random_state=random_state
    )

    return train_indices, valid_indices, test_indices


def train_val_test_split(df, covariate, target, ratio=0.2, valratio=0.5, random_state=42):
    """
    Splits a DataFrame into training, validation, and test sets and formats them as Hugging Face Datasets.

    Parameters:
    -----------
    df (pd.DataFrame): Input DataFrame containing the data.
    covariate (str): Name of the column to be used as the input text.
    target (str): Name of the column to be used as the label.
    ratio (float, optional): Proportion of the data to reserve for validation and test sets combined. Default is 0.2.
    valratio (float, optional): Proportion of the reserved data (from `ratio`) to assign to the validation set.
                                The remainder is assigned to the test set. Default is 0.5.
    random_state (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
    --------
    tuple: A tuple containing three Hugging Face `Dataset` objects:
        - train_data (Dataset): Training set.
        - valid_data (Dataset): Validation set.
        - test_data (Dataset): Test set.
"""
    # Split the indices of df into train, validation and test sets
    train_indices, valid_indices, test_indices = data_idx_split(
        df.index, ratio=ratio, valratio=valratio, random_state=random_state)

    # Get the corresponding data from the indices
    train_texts = df.loc[train_indices, covariate].tolist()
    train_labels = df.loc[train_indices, target].tolist()

    valid_texts = df.loc[valid_indices, covariate].tolist()
    valid_labels = df.loc[valid_indices, target].tolist()

    test_texts = df.loc[test_indices, covariate].tolist()
    test_labels = df.loc[test_indices, target].tolist()

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

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix for training the model.
    y : array-like, shape (n_samples,)
        Target labels corresponding to the feature matrix X.
    train_idx : array-like, shape (n_samples_subset,)
        Indices used to restrict the input data (X and y) for training.

    Returns:
    --------
    model : LogisticRegression
        A trained Logistic Regression model fitted on the restricted data.
    """
    X_restricted = X[train_idx]
    y_restricted = y[train_idx]

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_restricted, y_restricted)

    return model


def logreg_metrics(X, y, model, test_indices, doprint=0):
    """
    Evaluate and print metrics for a Logistic Regression model.

    This function evaluates the performance of a Logistic Regression model on 
    the test data by calculating accuracy and generating a classification report. 
    Optionally, it can print these metrics to the console.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix for the test set.
    y : array-like, shape (n_samples,)
        True labels for the test set.
    model : LogisticRegression
        A trained Logistic Regression model used to make predictions.
    test_indices: the indices on which to test the model
    doprint : int, optional, default=0
        If set to 1, the accuracy and classification report will be printed.

    Returns:
    --------
    accuracy : float
        The accuracy of the model on the test set.
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
