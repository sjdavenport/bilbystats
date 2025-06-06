#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples of generating data
"""
import bilbystats as bs
import pandas as pd
datadir = '/Users/samd/Documents/Packages/bilbystats/bilbystats/data/'
dfname = 'ex_article_sentiment.parquet'
df = pd.read_parquet(datadir + dfname)
indices = np.array([1, 2, 3, 4, 5])
df_aug = sentence_gen(df, indices)
