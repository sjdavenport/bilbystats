#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example implementation of the bilbystats pipeline
"""
import pandas as pd
import bilbystats as bs

df = bs.read_data('gold-dataset-sinha-khandait.csv')
df_first_100 = df.head(100)

covariate = 'News'
target = 'Price Direction Up'

indices = bs.data_idx_split(df_first_100.index)
train_data, valid_data, test_data = bs.train_val_test_split(
    df_first_100, covariate, target, indices)
model_name = "distilbert-base-uncased"
train_data_tk, valid_data_tk, test_data_tk = bs.tokenize_data(
    train_data, valid_data, test_data, model_name)

# %%
savedir = "./modelcheckpoints/"
savename = "bs_training_example"
label2id = {"NEUTRAL": 0, "UP": 1}
trainer, model, training_args = bs.trainTFmodel(
    train_data_tk, valid_data_tk, model_name, savename=savename, savedir=savedir, num_labels=2, label2id=label2id)

# %% Evaluate the trained model
