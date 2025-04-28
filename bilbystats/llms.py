"""
    Transformer based llm models and analysis
"""
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


def train_tf():
    return 5


def load_tf_model(model_name="distilbert-base-uncased", num_labels):
    """
    Loads a pre-trained Hugging Face transformer model and its tokenizer for sequence classification.

    Parameters:
    -----------
    model_name (str, optional): The name or path of the pre-trained model to load. 
        Defaults to "distilbert-base-uncased".
    num_labels (int): The number of labels for the classification task.

    Returns:
    -----------
    tuple: A tuple (tokenizer, model) where:
        - tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the pre-trained model.
        - model (PreTrainedModel): The model configured for sequence classification with the specified number of labels.
"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels)

    return tokenizer, model


def tokenize_function(batch, max_length=512):
    """
    Tokenizes a batch of text examples using a pre-defined tokenizer.

    Args:
    batch (dict): A dictionary containing a "text" key with a list of text strings to tokenize.
    max_length (int, optional): The maximum sequence length after tokenization. 
            Sequences longer than this will be truncated, and shorter ones will be padded. Defaults to 512.

    Returns:
    dict: A dictionary of tokenized outputs suitable for model input (e.g., input_ids, attention_mask).
"""
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)


def tokenize_data(train_data, valid_data, test_data):
    """
    Tokenizes and formats training, validation, and test datasets for PyTorch models.

    Each dataset is tokenized using a predefined `tokenize_function`, and the outputs are
    converted to PyTorch tensors with the relevant input columns.

    Args:
    train_data (Dataset): The training dataset containing text and labels.
    valid_data (Dataset): The validation dataset containing text and labels.
    test_data (Dataset): The test dataset containing text and labels.

    Returns:
    tuple: A tuple (train_data_tk, valid_data_tk, test_data_tk) where each element is:
        - A tokenized and PyTorch-formatted dataset with columns "input_ids", "attention_mask", and "label".
"""
    # Tokenize all datasets
    train_data_tk = train_data.map(tokenize_function, batched=True)
    valid_data_tk = valid_data.map(tokenize_function, batched=True)
    test_data_tk = test_data.map(tokenize_function, batched=True)

    # Convert datasets to PyTorch tensors
    train_data_tk.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"])
    valid_data_tk.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"])
    test_data_tk.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"])

    return train_data_tk, valid_data_tk, test_data_tk


def get_sentiment_score(text):
    """
    Analyze the sentiment of a given text using FinBERT for financial sentiment analysis.

    This function uses the pre-trained FinBERT model to evaluate the sentiment of the input 
    text. The sentiment is returned as a continuous score, with the following transformations:
    - Negative sentiment: returns a negative score (negative of the model's score).
    - Neutral sentiment: returns a score shifted by 10.
    - Positive sentiment: returns the raw model score.

    The function automatically detects whether a GPU is available and uses it for faster processing 
    if possible.

    Parameters:
    -----------
    text : str
        The text for which the sentiment score is to be calculated.

    Returns:
    --------
    out : float
        A continuous sentiment score representing the text's sentiment. The score is modified as:
        - Negative sentiment: negative of the raw score.
        - Neutral sentiment: score shifted by +10.
        - Positive sentiment: raw score.
    """
    # Check if a GPU is available, otherwise default to CPU
    device = 0 if torch.cuda.is_available() else -1

    # Load FinBERT for financial sentiment analysis, using GPU if available
    sentiment_analyzer = pipeline(
        'sentiment-analysis',
        model='yiyanghkust/finbert-tone',
        device=device
    )

    result = sentiment_analyzer(text)
    # Return the continuous sentiment score
    if result[0]['label'] == 'Negative':
        out = -result[0]['score']
    elif result[0]['label'] == 'Neutral':
        out = result[0]['score'] + 10
    else:
        out = result[0]['score']
    return out
