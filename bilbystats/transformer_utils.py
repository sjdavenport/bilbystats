"""
    Transformer based llm models and analysis
"""
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
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


def tokenize_data(train_data, valid_data, test_data, model_name, chunk_size=512, stride=128):
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
    labels = []
    for i in sample_mapping:
        # same label for all chunks from a single input
        labels.append(example["label"])

    tokenized["label"] = labels
    return tokenized
