#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:00:42 2025

@author: samd
"""
from transformers import AutoTokenizer, BertTokenizer, BertModel, AutoModel
import torch.nn.functional as F
import torch
from datasets import Dataset
import numpy as np
import spacy
from numpy.linalg import norm


def load_tokenizer(model_name: str) -> BertTokenizer:
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


def tokenize(data: Dataset, model_name: str, max_length: int = 512) -> Dataset:
    """
    Tokenize a dataset using a pre-trained tokenizer for transformer models.

    This function tokenizes the "text" field of the input dataset using the specified 
    pre-trained tokenizer. It applies padding and truncation to ensure uniform input 
    lengths. The output is formatted for use with PyTorch. If the dataset includes 
    a "label" column, it is preserved in the output format.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    data : Dataset
        The dataset containing a "text" field and optionally a "label" field.
    model_name : str
        Name or path of the pre-trained model to load the tokenizer from.

    ---------------------------------------------------------------------------
    OUTPUT:
    data_tk : Dataset
        Tokenized dataset in PyTorch tensor format, with "input_ids", 
        "attention_mask", and optionally "label".

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch, max_length=max_length):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    # Tokenize all datasets
    data_tk = data.map(tokenize_function, batched=True)

    # Convert datasets to PyTorch tensors
    # Dynamically set format columns based on existence
    columns_to_set = ["input_ids", "attention_mask"]
    if "label" in data.column_names:
        columns_to_set.append("label")

    data_tk.set_format("torch", columns=columns_to_set)
    # data_tk.set_format(
    #    "torch", columns=["input_ids", "attention_mask", "label"])

    return data_tk


def tokenize_data(train_data: Dataset, valid_data: Dataset, test_data: Dataset,
                  model_name: str, max_length: int = 512) -> tuple[Dataset, Dataset, Dataset]:
    """
    Tokenize datasets using a pre-trained tokenizer for transformer models.

    This function applies a tokenizer to the "text" field of each dataset 
    (train, validation, and test), ensuring each example is padded and 
    truncated to a fixed maximum length. The output datasets are formatted 
    for PyTorch usage.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    train_data : Dataset
        The training dataset containing "text" and "label" fields.
    valid_data : Dataset
        The validation dataset with the same format as train_data.
    test_data : Dataset
        The test dataset with the same format as train_data.
    model_name : str
        Name or path of the pre-trained model to load the tokenizer from.

    ---------------------------------------------------------------------------
    OUTPUT:
    train_data_tk : Dataset
        Tokenized training dataset in PyTorch tensor format.
    valid_data_tk : Dataset
        Tokenized validation dataset in PyTorch tensor format.
    test_data_tk : Dataset
        Tokenized test dataset in PyTorch tensor format.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch, max_length=max_length):
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


def tokenize_data_chunks(train_data: Dataset, valid_data: Dataset, test_data: Dataset, model_name: str,
                         chunk_size: int = 512, stride: int = 128) -> tuple[Dataset, Dataset, Dataset]:
    """
    Tokenize and chunk input datasets using a pre-trained tokenizer with overlapping windows.

    This function tokenizes and chunks training, validation, and test datasets into overlapping
    sequences compatible with transformer models. It uses a sliding window approach for handling
    long texts and converts the datasets into PyTorch format for model training.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    train_data : Dataset
        The training dataset containing "text" and "label" fields.
    valid_data : Dataset
        The validation dataset with the same format as train_data.
    test_data : Dataset
        The test dataset with the same format as train_data.
    model_name : str
        Name or path of the pre-trained model to load the tokenizer from.
    chunk_size : int, optional (default=512)
        The maximum length of each tokenized chunk.
    stride : int, optional (default=128)
        The number of overlapping tokens between successive chunks.

    ---------------------------------------------------------------------------
    OUTPUT:
    train_data_tk : Dataset
        Tokenized and chunked training dataset in PyTorch format.
    valid_data_tk : Dataset
        Tokenized and chunked validation dataset in PyTorch format.
    test_data_tk : Dataset
        Tokenized and chunked test dataset in PyTorch format.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
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


def chunked_tokenize_function(example: dict, tokenizer: BertTokenizer,
                              chunk_size: int = 512, stride: int = 128) -> dict:
    """
    Tokenize input text into overlapping chunks for transformer-based models.

    This function tokenizes long input text into fixed-size overlapping chunks suitable
    for models like BERT. It ensures that text exceeding the maximum sequence length is 
    split appropriately, and aligns the original labels with the generated chunks.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    example : dict
        A dictionary containing:
            - "text": str, the input text to tokenize.
            - "label": list, labels corresponding to the input text.
    tokenizer : BertTokenizer
        A tokenizer compatible with transformer models.
    chunk_size : int, optional (default = 512)
        The maximum length of each tokenized chunk.
    stride : int, optional (default=128)
        The number of overlapping tokens between successive chunks.

    ---------------------------------------------------------------------------
    OUTPUT:
    tokenized : dict
        A dictionary containing the tokenized chunks and corresponding labels.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
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


def cos_sim(text1: str, text2: str, model_name: str = 'spacy') -> float:
    if model_name == 'spacy':
        nlp = spacy.load("en_core_web_md")
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        vec1 = doc1.vector
        vec2 = doc2.vector
        cos_sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    else:
        distil_tokenizer = AutoTokenizer.from_pretrained(model_name)
        distil_model = AutoModel.from_pretrained(model_name)
        vec1_distil = get_embedding(distil_model, distil_tokenizer, text1)
        vec2_distil = get_embedding(distil_model, distil_tokenizer, text2)
        cos_sim = F.cosine_similarity(vec1_distil, vec2_distil, dim=0).item()

    return cos_sim


def cos_sim_dep(text1: str, text2: str, model_name: str) -> float:
    """
    Compute cosine similarity between two texts using a BERT-based model.

    This function encodes two input text strings into embeddings using a 
    pre-trained BERT model and computes the cosine similarity between them.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    text1 : str
        The first input text to be compared.
    text2 : str
        The second input text to be compared.
    model_name : str
        Name or path of the pre-trained BERT model to use for generating embeddings.

    ---------------------------------------------------------------------------
    OUTPUT:
    cos_sim : float
        Cosine similarity score between the embeddings of the two input texts.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    emb1 = get_embedding(text1, model, tokenizer)
    emb2 = get_embedding(text2, model, tokenizer)

    # Compute cosine similarity
    cos_sim = F.cosine_similarity(emb1, emb2).item()

    return cos_sim


def get_embedding(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    # Mean pooling
    return outputs.last_hidden_state.mean(dim=1).squeeze()


def get_embedding_dep(text: str, model: torch.nn.Module, tokenizer: BertTokenizer) -> torch.Tensor:
    """
    Generate the embedding of a text string using the [CLS] token from a BERT-based model.

    This function tokenizes the input text and uses a pre-trained BERT model to extract the 
    embedding corresponding to the [CLS] token, which is often used to represent the entire sentence.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    text : str
        The input text to encode.
    model : torch.nn.Module
        A pre-trained BERT model used to generate embeddings.
    tokenizer : BertTokenizer
        Tokenizer corresponding to the BERT model, used to preprocess the input text.

    ---------------------------------------------------------------------------
    OUTPUT:
    cls_embedding : torch.Tensor, shape (1, hidden_size)
        Embedding vector for the input text, extracted from the [CLS] token representation.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding
