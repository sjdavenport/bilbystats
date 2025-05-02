"""
    Transformer based llm models and analysis
"""
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from tqdm import tqdm
import os
import bilbystats as bs


def trainTFmodel(train_data, valid_data, model_name, savename=None, savedir="./", num_labels=2, label2id=None, training_args=None):
    """
    Trains a sequence classification model using the Hugging Face Transformers library.

    Parameters:
        - train_data (Dataset): The training dataset in Hugging Face format (e.g., `datasets.Dataset`).
        - valid_data (Dataset): The validation dataset in Hugging Face format (e.g., `datasets.Dataset`).
        - model_name (str): The name or path to a pre-trained model from Hugging Face.
        - savename (str, optional): The name to save the trained model as (default is None).
        - savedir (str, optional): Directory where the trained model will be saved (default is './').
        - num_labels (int, optional): The number of labels in the classification task (default is 2).
        - label2id (dict, optional): A dictionary mapping label names to label IDs. If None, labels are automatically handled (default is None).
        - training_args (TrainingArguments, optional): The training arguments (default is None, in which case defaults are used).

    Returns:
        - trainer (Trainer): The trained `Trainer` object.
        - model (PreTrainedModel): The trained model.
        - training_args (TrainingArguments): The training arguments used for the training process.
"""
    if not label2id:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
    else:
        if len(label2id) != num_labels:
            raise ValueError(
                "The number of label ids does not match the number of labels")

        id2label = {v: k for k, v in label2id.items()}

        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id)

    # Turn WandB logging off
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "offline"

    # Obtain the training arguments
    if not training_args:
        training_args = default_training_args(model_name, savename, savedir)
    else:
        training_args = training_args

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        compute_metrics=bs.compute_metrics,
    )

    # Train the model
    trainer.train()

    return trainer, model, training_args


def evalTFmodel(texts, model_dir, batch_size=32):
    """
    Evaluates a text classification model on a list of input texts.

    Loads a pre-trained HuggingFace `AutoModelForSequenceClassification` model from a local directory
    and applies it to the given list of input texts in batches. The function returns predicted class 
    labels and their associated probabilities.

    Parameters:
        texts (list of str or numpy array): List or array of text inputs to be classified.
        model_dir (str): Path to the directory containing the locally saved pre-trained model.
        batch_size (int, optional): Number of samples to process per batch. Defaults to 32.

    Returns:
        predictions (list of lists of int): List containing predicted class indices for each input.
        probabilities (list of numpy arrays): List containing probability distributions over classes 
                                          for each input.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True)
    model.eval()

    if isinstance(my_var, list):
        texts2use = texts2use.tolist()
    else:
        texts2use = texts

    probabilities = []
    predictions = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
        batch_texts = texts[i:i+batch_size]
        batch_inputs = tokenizer(batch_texts, padding=True,
                                 truncation=True, max_length=512, return_tensors="pt")

        with torch.no_grad():
            logits = model(**batch_inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            probabilities.append(probs.numpy())
            predictions.append(preds.tolist())  # <- extend, not append

    return predictions, probabilities


def default_training_args(model_name, savename=None, savedir='./'):
    """
    Generates default training arguments for fine-tuning a model using Hugging Face's `TrainingArguments`.

    Parameters:
        - model_name (str): The name or path of the pre-trained model to be fine-tuned.
        - savename (str, optional): The name for saving the model. If not provided, the `model_name` will be used.
        - savedir (str, optional): The directory where the trained model will be saved. Default is './'.

    Returns:
        - training_args (TrainingArguments): A `TrainingArguments` object with default configuration for model training.

    This function sets various training hyperparameters such as the learning rate, the number of epochs,
    evaluation strategy, and logging steps. The arguments are set for training the model for 4 epochs with 
    weight decay and automatic batch size determination, saving the best model at the end of training, 
    and logging to W&B every 300 steps.
"""
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


def load_tokenizer(model_name):
    """
    Loads a tokenizer from a pretrained model.

    Parameters:
    model_name (str): The name or path of the pretrained model from which to load the tokenizer.
                      This can be a model identifier from the Hugging Face Model Hub
                      (e.g., 'bert-base-uncased') or a local path.

    Returns:
    transformers.PreTrainedTokenizer: An instance of the tokenizer associated with the specified model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


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

    def tokenize_function(batch, max_length=512):
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


def tokenize_data_chunks(train_data, valid_data, test_data, model_name, chunk_size=512, stride=128):
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

    def tokenize(batch):
        return chunked_tokenize_function(batch, tokenizer, chunk_size, stride)

    # Tokenize and chunk datasets
    train_data_tk = train_data.map(
        tokenize, batched=True, remove_columns=train_data.column_names)
    valid_data_tk = valid_data.map(
        tokenize, batched=True, remove_columns=valid_data.column_names)
    test_data_tk = test_data.map(
        tokenize, batched=True, remove_columns=test_data.column_names)

    # Set torch format
    for ds in [train_data_tk, valid_data_tk, test_data_tk]:
        ds.set_format("torch", columns=[
                      "input_ids", "attention_mask", "label"])

    return train_data_tk, valid_data_tk, test_data_tk


def chunked_tokenize_function(example, tokenizer, chunk_size=512, stride=128):
    """
    Tokenizes input text into overlapping chunks, aligning labels with the resulting tokenized segments.

    Parameters:
        example (dict): A dictionary containing at least two keys:
                        - "text" (str): The input text to be tokenized.
                        - "label" (list): A list of labels corresponding to the text samples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenizing the text.
        chunk_size (int, optional): The maximum length of each chunk (in tokens). Defaults to 512.
        stride (int, optional): The number of tokens to overlap between consecutive chunks. Defaults to 128.

    Returns:
        dict: A dictionary of tokenized outputs, including input IDs, attention masks, offset mappings,
              and the aligned "label" list for each chunk. The output will include all keys typically
              returned by the tokenizer, with an added "label" key that maps the original labels to
              the correct chunks.
    """
    tokenized = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=chunk_size,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )

    # We need to align the label with the right chunks
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    labels = [example["label"][i] for i in sample_mapping]

    tokenized["label"] = labels
    return tokenized
