#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:26:10 2025

@author: samd
"""

import spacy
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import torch
import torch.nn.functional as F


def get_embedding(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    # Mean pooling
    return outputs.last_hidden_state.mean(dim=1).squeeze()


text1 = "I love machine learning"
text2 = "I like machine learning"

# DistilBERT
distil_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
distil_model = AutoModel.from_pretrained("distilbert-base-uncased")
vec1_distil = get_embedding(distil_model, distil_tokenizer, text1)
vec2_distil = get_embedding(distil_model, distil_tokenizer, text2)
cos_distil = F.cosine_similarity(vec1_distil, vec2_distil, dim=0).item()

# BERT
bert_tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = BertModel.from_pretrained("distilbert-base-uncased")
bert_model.eval()
vec1_bert = get_embedding(bert_model, bert_tokenizer, text1)
vec2_bert = get_embedding(bert_model, bert_tokenizer, text2)
cos_bert = F.cosine_similarity(vec1_bert, vec2_bert, dim=0).item()

print(f"DistilBERT similarity: {cos_distil:.4f}")
print(f"BERT similarity: {cos_bert:.4f}")


# %%


# Load a model with word vectors
nlp = spacy.load("en_core_web_md")

# Get word vectors
text1 = "man"
text2 = "men"
man = nlp(text1)
woman = nlp(text2)

# Compute similarity
bs.tic()
similarity = man.similarity(woman)
bs.toc()
print(f"spaCy cosine similarity: {similarity}")

bs.tic()
bs.cos_sim(text1, text2, 'distilbert-base-uncased')
bs.toc()
