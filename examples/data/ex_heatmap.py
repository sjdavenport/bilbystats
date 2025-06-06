# -*- coding: utf-8 -*-

text1 = 'my brothers and my sisters and me'
text2 = 'we are family'
print(bs.cos_sim(text1, text2, 'distilbert-base-uncased'))
print(bs.cos_sim(text1, text2, 'spacy'))

# %%
text1 = '我爱机器学习'
text2 = '我喜欢机器学习'
bs.cos_sim(text1, text2, 'hfl/chinese-roberta-wwm-ext')

# %%
df = bs.read_data('example_21_sentences_labelled.parquet')
texts2map = [df['title'], df['text'], df['gpt-4o_explanation']]
bs.create_cosine_similarity_matrix(
    texts2map, model_name='hfl/chinese-roberta-wwm-ext')

# %%
df = bs.read_data('example_21_sentences_labelled.parquet')
texts2map = [df['translated_text'], df['gpt-4o_explanation']]
out = bs.create_cosine_similarity_matrix(
    texts2map, model_name='spacy')

# %%
df = df.head(3)
bs.heatmap(df, ['translated_text', 'gpt-4o_explanation'], metric='cos_sim')
