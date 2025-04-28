"""
    Transformer based llm models and analysis
"""
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os


def trainTFmodel(train_data, valid_data, model_name, savename=None, savedir="./", num_labels=2, training_args=None):
    """
    Trains a sequence classification model using the Hugging Face Trainer API.

    This function initializes a model for sequence classification, prepares the training setup,
    and starts the training process. It allows for the use of custom or default training arguments
    and automatically disables WandB logging.

    Args:
        train_data (Dataset): The training dataset, already tokenized and padded.
        val_data (Dataset): The validation dataset, already tokenized and padded.
        model_name (str): The name or path of the pre-trained model to be used (e.g., "bert-base-uncased").
        savedir (str): Directory where the trained model and checkpoints will be saved.
        num_labels (int, optional): The number of labels for the classification task (default is 2 for binary classification).
        training_args (TrainingArguments, optional): Custom training arguments. If None, default training arguments are used.

    Returns:
        Trainer: The Hugging Face Trainer object, which contains the trained model and additional training details.
        model: The trained model after the training process.
        training_args: The training arguments used for the training process.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels)

    if not training_args:
        training_args = default_training_args(model_name, savename, savedir)
    else:
        training_args = training_args

    # Turn WandB logging off
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "offline"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer, model, training_args


def default_training_args(model_name, savename=None, savedir='./'):
    if not savename:
        savename = model_name

    training_args = TrainingArguments(
        output_dir=savedir + savename,
        run_name=f"{model_name}",
        learning_rate=2e-5,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        auto_find_batch_size=True,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_steps=300,  # how often to log to W&B
        load_best_model_at_end=True,
        # save_total_limit=1,
        seed=42,
        data_seed=42,
        save_safetensors=False,
    )
    return training_args


def tokenize_data(train_data, valid_data, test_data, model_name):
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch, tokenizer, max_length=512):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

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
