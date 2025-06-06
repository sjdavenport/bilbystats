#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:20:20 2025

@author: samd
"""

# import spacy
import torch.nn.functional as F
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# %%% Note that this approach
# Example texts
text1 = "this is a not a very good man"
text2 = "this is a not a very good woman"

# Vectorize the texts
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text1, text2])

# Compute cosine similarity
cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"Cosine Similarity: {cos_sim[0][0]:.4f}")


# %%

# Load model and tokenizer (Chinese RoBERTa)
model_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Your texts
# text1 = "我喜欢机器学习"
# text2 = "我喜欢机器学习a"

text1 = '19日下午，习近平首先来到洛阳轴承集团股份有限公司考察。该公司前身为“一五”期间建成的洛阳轴承厂。在智能工厂，习近平了解企业发展历程，听取不同类型轴承产品用途和性能介绍，走近生产线察看生产流程。他对围拢过来的企业职工说，制造业是国民经济的重要支柱，推进中国式现代化必须保持制造业优势合理比重。现代制造业离不开科技赋能，要大力加强技术攻关，走自主创新的发展路子。他勉励职工发扬主人翁精神，在企业发展中奋发有为、多作贡献。'
text1 = '19日下午，习近平首先来到洛阳轴承集团股份有限公司考察。'
llm = 'gemma3:1b'
bs.tic()
text1_english = bs.translate(text1,  llm)
text2 = bs.translate(text1_english, llm, 'Chinese')
bs.toc()
# Tokenize and encode


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding


emb1 = get_embedding(text1)
emb2 = get_embedding(text2)

# Compute cosine similarity
cos_sim = F.cosine_similarity(emb1, emb2)
print(f"Cosine Similarity: {cos_sim.item():.4f}")

# %%

# Load model and tokenizer (Chinese RoBERTa)
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text1 = 'man'
text1 = 'woman'


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding


emb1 = get_embedding(text1)
emb2 = get_embedding(text2)

# Compute cosine similarity
cos_sim = F.cosine_similarity(emb1, emb2)
print(f"Cosine Similarity: {cos_sim.item():.4f}")

# %%
bs.cos_sim('19日下午，习', '19日下午，习3', 'hfl/chinese-roberta-wwm-ext')

# %%
text1 = 'The phrase "经济增速将有所放缓" indicates a deceleration in economic growth, which is typically viewed as a concern or a challenge, hence conveying a slightly negative sentiment.'
text2 = 'The label "neutral" applies because the phrase reports on an economic prediction from the IMF without expressing a personal opinion or emotion, simply conveying factual information about the forecast for Asia''s economic growth. The tone is objective and informative, characteristic of neutral language.'
bs.cos_sim(text1, text2, 'roberta-large')

# %%
bs.cos_sim('king', 'man', 'roberta-large')

# %%
bs.cos_sim(bs.translate('man', 'gpt-4o-mini', 'Chinese'), bs.translate('woman',
           'gpt-4o-mini', 'Chinese'), 'hfl/chinese-roberta-wwm-ext')

# %%
llm = 'gemini'
bs.tic()
text1_english = bs.translate(text1,  llm)
bs.toc()

# %%

# Load a model with word vectors
nlp = spacy.load("en_core_web_md")

# Get word vectors
man = nlp("man")
woman = nlp("woman")

# Compute similarity
similarity = man.similarity(woman)
print(f"spaCy cosine similarity: {similarity}")
