#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 18:24:24 2025

@author: samd
"""
llm = 'gemini'
out = bs.sentiment_xAI('商场更创下单日近12万人次的客流高峰', 'positive',
                       'slightly positive', llm)
print(out)

# %%
model = 'gemini'
out = bs.sentiment_xAI('并在行风管理经办部门设立办公室', 'negative',
                       'neutral', model)
print(out)

# %%
instructions = bs.read_data('simple_sentiment_prompt.txt')

llm = 'gemini'
out = bs.llm_api('并在行风管理经办部门设立办公室', instructions, llm)
print(out)


# %%
bs.tic()
instructions = bs.read_data('simple_sentiment_prompt.txt')

llm = 'gemini'
out = bs.llm_api('商场更创下单日近12万人次的客流高峰', instructions, llm)
print(out)
bs.toc()

# %%
bs.tic()
instructions = bs.read_data('sentiment_classification.txt')

llm = 'gemini'
out = bs.llm_api('商场更创下单日近12万人次的客流高峰', instructions, llm)
print(out)
bs.toc()

# %%
bs.tic()
instructions = bs.read_data('simple_sentiment_prompt.txt')

llm = 'gemma3:4b'
out = bs.llm_api(
    'Classify and explain the sentiment of the following sentence: 商场更创下单日近12万人次的客流高峰', instructions, llm)
print(out)
bs.toc()
