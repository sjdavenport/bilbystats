"""
A collection of examples for the functions form llmapicalls.py
"""
import bilbystats as bs
import pandas as pd

# %% Example of applying the api to the entries of a dataframe
bs.tic()
colname = 'text'
datadir = "/Users/samd/Documents/Packages/bilbystats/bilbystats/data/"
dataloc = datadir + "example_21_sentences.parquet"
saveloc = datadir + "example_21_sentences_labelled.parquet"
promptloc = datadir + 'Prompts/simple_sentiment_prompt.txt'
model_name = "gemma3:1b"

bs.api2data(colname, promptloc, dataloc, saveloc, model_name)

# %%
df = pd.read_parquet(saveloc)
print(df.head(5)['gpt-4o_label'])
print(df.head(5)['gpt-4o_explanation'])

# %%
bs.tic()
colname = 'text'
datadir = "/Users/samd/Documents/Packages/bilbystats/bilbystats/data/"
dataloc = datadir + "example_21_sentences.parquet"
saveloc = datadir + "example_21_sentences_just_label.parquet"
promptloc = datadir + 'Prompts/sentiment_classification.txt'
model_name = "llama"

bs.api2data(colname, promptloc, dataloc, saveloc, model_name,
            labels=['Label:'], names=['label'], lowercase=[True])

# %%
df = pd.read_parquet(saveloc)
print(df.head(5)['gpt-4o_label'])

# %%
bs.translate('ti amo', "gpt-4o")

# %%
bs.llm_api('translate this:', 'ti amo', "gpt-4o")
