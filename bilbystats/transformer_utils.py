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
    Train a Hugging Face Transformers model for sequence classification.

    This function initializes and trains a sequence classification model using 
    Hugging Face Transformers. It supports optional custom label mappings and 
    training arguments. If no `training_args` are provided, default training 
    arguments are generated.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    train_data : Dataset
        The Hugging Face Dataset object containing training data.
    valid_data : Dataset
        The Hugging Face Dataset object containing validation data.
    model_name : str
        The name or path of the pretrained model to fine-tune.
    savename : str, optional (default=None)
        Optional name to use when saving the model.
    savedir : str, optional (default="./")
        Directory where the model and checkpoints will be saved.
    num_labels : int, optional (default=2)
        The number of output labels for classification.
    label2id : dict, optional (default=None)
        A dictionary mapping label names to integer IDs.
    training_args : TrainingArguments, optional (default=None)
        Hugging Face `TrainingArguments` object. If not provided, default 
        arguments are generated.

    ---------------------------------------------------------------------------
    OUTPUT:
    trainer : Trainer
        The Hugging Face Trainer object used for training.
    model : AutoModelForSequenceClassification
        The trained sequence classification model.
    training_args : TrainingArguments
        The training arguments used during training.

    ---------------------------------------------------------------------------
    Copyright (C) - 2025 - Samuel Davenport
    ---------------------------------------------------------------------------
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
    Evaluate a Hugging Face Transformers model on a list of texts.

    This function loads a pretrained sequence classification model from the 
    specified directory and performs batch-wise evaluation on the provided 
    texts. It returns the predicted labels and corresponding probability 
    distributions.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    texts : list of str or array-like
        The input texts to classify.
    model_dir : str
        Path to the directory containing the pretrained model.
    batch_size : int, optional (default=32)
        The batch size to use during evaluation.

    ---------------------------------------------------------------------------
    OUTPUT:
    predictions : list
        A list of predicted class labels for each input text.
    probabilities : list
        A list of probability distributions for each input text.

    ---------------------------------------------------------------------------
    Copyright (C) - 2025 - Samuel Davenport
    ---------------------------------------------------------------------------
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
    Create default Hugging Face TrainingArguments for model training.

    This function generates a set of default training arguments suitable for 
    fine-tuning a Hugging Face Transformers model. It configures basic 
    parameters such as evaluation strategy, saving strategy, and logging, and 
    allows specifying custom save names and directories.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    model_name : str
        The name of the model, used for logging and directory naming.
    savename : str, optional (default=None)
        The name to use when saving the model. Defaults to `model_name` if not 
        provided.
    savedir : str, optional (default='./')
        Directory where model checkpoints and logs will be saved.

    ---------------------------------------------------------------------------
    OUTPUT:
    training_args : TrainingArguments
        A Hugging Face `TrainingArguments` object with default settings.

    ---------------------------------------------------------------------------
    Copyright (C) - 2025 - Samuel Davenport
    ---------------------------------------------------------------------------
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
    Load a Hugging Face tokenizer.

    This function loads a pretrained tokenizer from Hugging Face Transformers 
    based on the provided model name.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    model_name : str
        The name or path of the pretrained model whose tokenizer will be loaded.

    ---------------------------------------------------------------------------
    OUTPUT:
    tokenizer : AutoTokenizer
        The loaded tokenizer object.

    ---------------------------------------------------------------------------
    Copyright (C) - 2025 - Samuel Davenport
    ---------------------------------------------------------------------------
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def tokenize_data(train_data, valid_data, test_data, model_name):
    """
    Tokenize datasets for Hugging Face Transformers.

    This function tokenizes the provided train, validation, and test datasets 
    using a pretrained tokenizer. It applies padding and truncation to ensure 
    inputs fit within the model's maximum sequence length, and formats the 
    datasets for use with PyTorch.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    train_data : Dataset
        The Hugging Face Dataset object containing training data.
    valid_data : Dataset
        The Hugging Face Dataset object containing validation data.
    test_data : Dataset
        The Hugging Face Dataset object containing test data.
    model_name : str
        The name or path of the pretrained model whose tokenizer will be used.

    ---------------------------------------------------------------------------
    OUTPUT:
    train_data_tk : Dataset
        Tokenized and formatted training dataset.
    valid_data_tk : Dataset
        Tokenized and formatted validation dataset.
    test_data_tk : Dataset
        Tokenized and formatted test dataset.

    ---------------------------------------------------------------------------
    Copyright (C) - 2025 - Samuel Davenport
    ---------------------------------------------------------------------------
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
    Tokenize and chunk datasets for Hugging Face Transformers.

    This function tokenizes and chunks the provided datasets using a pretrained 
    tokenizer. The data is split into overlapping chunks based on the specified 
    chunk size and stride, which is useful for handling long documents.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    train_data : Dataset
        The Hugging Face Dataset object containing training data.
    valid_data : Dataset
        The Hugging Face Dataset object containing validation data.
    test_data : Dataset
        The Hugging Face Dataset object containing test data.
    model_name : str
        The name or path of the pretrained model whose tokenizer will be used.
    chunk_size : int, optional (default=512)
        The maximum sequence length for each chunk.
    stride : int, optional (default=128)
        The number of overlapping tokens between consecutive chunks.

    ---------------------------------------------------------------------------
    OUTPUT:
    train_data_tk : Dataset
        Tokenized, chunked, and formatted training dataset.
    valid_data_tk : Dataset
        Tokenized, chunked, and formatted validation dataset.
    test_data_tk : Dataset
        Tokenized, chunked, and formatted test dataset.

    ---------------------------------------------------------------------------
    Copyright (C) - 2025 - Samuel Davenport
    ---------------------------------------------------------------------------
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
    Tokenize input text into overlapping chunks with aligned labels.

    This function tokenizes long input text into overlapping chunks using the 
    specified tokenizer. It ensures that labels are aligned with the resulting 
    tokenized segments by mapping them according to the overflow token mapping.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    example : dict
        A dictionary containing at least:
        - "text" (str): The input text to tokenize.
        - "label" (list): A list of labels corresponding to the text samples.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use for tokenizing the text.
    chunk_size : int, optional (default=512)
        The maximum length of each chunk (in tokens).
    stride : int, optional (default=128)
        The number of tokens to overlap between consecutive chunks.

    ---------------------------------------------------------------------------
    OUTPUT:
    tokenized : dict
        A dictionary containing tokenized outputs, including input IDs, 
        attention masks, offset mappings, and aligned "label" entries for each 
        chunk.

    ---------------------------------------------------------------------------
    Copyright (C) - 2025 - Samuel Davenport
    ---------------------------------------------------------------------------
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
