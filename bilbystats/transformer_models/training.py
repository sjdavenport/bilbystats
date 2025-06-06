#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 11:43:28 2025

@author: samd
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from tqdm import tqdm
import os
import bilbystats as bs
from datasets import Dataset
import numpy as np

from importlib import resources
from dotenv import load_dotenv
import os


def trainTFmodel(
    train_data: Dataset,
    valid_data: Dataset,
    model_name: str,
    savename: str = None,
    savedir: str = "./",
    num_labels: int = 2,
    label2id: dict = None,
    training_args: TrainingArguments = None
) -> tuple[Trainer, AutoModelForSequenceClassification, TrainingArguments]:
    """
    Train a transformer model for sequence classification using Hugging Face Trainer.

    This function initializes and trains a transformer-based model using the Hugging Face 
    `Trainer` API. It supports optional custom label mappings and training arguments, 
    and disables Weights & Biases logging by default.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    train_data : Dataset
        Tokenized training dataset compatible with Hugging Face models.
    valid_data : Dataset
        Tokenized validation dataset for evaluation during training.
    model_name : str
        Name or path of the pre-trained model to fine-tune.
    savename : str, optional
        Optional name for saving the trained model outputs.
    savedir : str, optional (default="./")
        Directory path where model outputs will be saved.
    num_labels : int, optional (default=2)
        Number of unique class labels for classification.
    label2id : dict, optional
        Dictionary mapping label names to integer IDs.
    training_args : TrainingArguments, optional
        Hugging Face TrainingArguments object. If not provided, default settings are used.

    ---------------------------------------------------------------------------
    OUTPUT:
    trainer : Trainer
        The Hugging Face Trainer instance used for model training.
    model : AutoModelForSequenceClassification
        The trained transformer classification model.
    training_args : TrainingArguments
        The training arguments used during training.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
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


def default_training_args(model_name: str, savename: str = None,
                          savedir: str = './', argfile: str = None) -> TrainingArguments:
    """
    Generate default training arguments for a Hugging Face model.

    This function sets up a default configuration of `TrainingArguments` used 
    for model training. If an `argfile` is not specified, it loads training 
    defaults from a predefined `.env` file. The model's output directory is 
    configured using the provided `savename` or falls back to `model_name`.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    model_name : str
        Name of the model to be used in the training configuration and run name.
    savename : str, optional
        Custom name for the training run. Defaults to `model_name` if not provided.
    savedir : str, optional
        Directory in which to save the training outputs. Defaults to current directory './'.
    argfile : str, optionals
        Path to a file containing training arguments (currently unused but 
        triggers loading defaults from environment if not specified).
    ---------------------------------------------------------------------------
    OUTPUT:
    training_args : TrainingArguments
        An instance of Hugging Face's `TrainingArguments` initialized with default settings.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    if not savename:
        savename = model_name

    if not argfile:
        # Use importlib to get a temporary path to the installed .env file
        env_path = resources.files(
            "bilbystats.defaults").joinpath("training_defaults.env")

        # Load environment variables from the .env file
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv(dotenv_path=argfile)

    def str_to_bool(value):
        return value.lower() in ("true", "1", "yes", "on")

    training_args = TrainingArguments(
        output_dir=savedir + savename,
        run_name=f"{model_name}",
        learning_rate=float(os.getenv('LEARNING_RATE')),
        do_eval=True,
        eval_strategy=os.getenv('EVAL_STRATEGY'),
        save_strategy=os.getenv('SAVE_STRATEGY'),
        save_total_limit=int(os.getenv('SAVE_TOTAL_LIMIT')),
        auto_find_batch_size=str_to_bool(os.getenv('AUTO_FIND_BATCH_SIZE')),
        num_train_epochs=int(os.getenv('NUM_TRAIN_EPOCHS')),
        weight_decay=float(os.getenv('WEIGHT_DECAY')),
        # how often to log to W&B
        logging_steps=int(os.getenv('LOGGING_STEPS')),
        load_best_model_at_end=True,
        seed=int(os.getenv('SEED')),
        data_seed=int(os.getenv('DATA_SEED')),
        save_safetensors=str_to_bool(os.getenv('SAVE_SAFETENSORS')),
        per_device_eval_batch_size=int(
            os.getenv('PER_DEVICE_EVAL_BATCH_SIZE')),
        per_device_train_batch_size=int(
            os.getenv('PER_DEVICE_TRAIN_BATCH_SIZE'))
    )
    return training_args


def default_training_args_dep(model_name: str, savename: str = None,
                              savedir: str = './') -> TrainingArguments:
    """
    Generate default training arguments for Hugging Face Trainer.

    This function creates a `TrainingArguments` object with sensible defaults 
    for fine-tuning a transformer model using the Hugging Face `Trainer` API. 
    It supports optional customization of output directory naming.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    model_name : str
        Name or identifier of the model, used for naming the run.
    savename : str, optional
        Custom name to use for the saved model output directory. 
        Defaults to `model_name` if not specified.
    savedir : str, optional (default='./')
        Base directory where training outputs will be saved.

    ---------------------------------------------------------------------------
    OUTPUT:
    training_args : TrainingArguments
        A configured TrainingArguments object ready for use with Hugging Face Trainer.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
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
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=300,  # how often to log to W&B
        load_best_model_at_end=True,
        # save_total_limit=1,
        seed=42,
        data_seed=42,
        save_safetensors=False,
    )
    return training_args
