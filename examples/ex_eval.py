"""
Example model evaluation
"""
from sklearn.metrics import accuracy_score
import bilbystats as bs
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer

datadir = "/Users/samd/Documents/Packages/bilbystats/bilbystats/data/"
filename = "gold-dataset-sinha-khandait.csv"
df = pd.read_csv(datadir+filename)
df_first_1000 = df.head(1000)

covariate = 'News'
target = 'Price Direction Up'

indices = bs.data_idx_split(df_first_1000.index)
train_data, valid_data, test_data = bs.train_val_test_split(
    df_first_1000, covariate, target, indices)
model_name = "distilbert-base-uncased"
train_data_tk, valid_data_tk, test_data_tk = bs.tokenize_data(
    train_data, valid_data, test_data, model_name)

# Load the model
model_path = "/Users/samd/Documents/General_code/ModelCheckpoints/bs_training_example/checkpoint-200"

# %%
model = AutoModelForSequenceClassification.from_pretrained(model_path)
trainer = Trainer(model=model)

# Calculate the model predictions
predictions = trainer.predict(valid_data_tk)

# %%
predictions = bs.predict(valid_data, model_path, model_name)
accuracy_score(predictions['true_labels'], predictions['pred_labels'])

# %%
true_labels = np.array(valid_data['label'])
valid_data_nolabel = valid_data.remove_columns('label')
predictions = bs.predict(valid_data_nolabel, model_path, model_name)
accuracy_score(true_labels, predictions['pred_labels'])

# %%
predictions = bs.predict(test_data, model_path, model_name)
accuracy_score(predictions['true_labels'], predictions['pred_labels'])

# %% DF option without including the target covariates as a label
predictions = bs.predict_df(
    df_first_1000, covariate, model_path, model_name, indices['test'])
accuracy_score(
    df_first_1000['Price Direction Up'].loc[indices['test']], predictions['pred_labels'])

# %% DF option with including the target covariates as a label
predictions = bs.predict_df(df_first_1000, covariate,
                            model_path, model_name, indices['test'], target)
accuracy_score(predictions['true_labels'], predictions['pred_labels'])
