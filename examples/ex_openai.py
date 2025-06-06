#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 16:46:37 2025

@author: samd
"""
import openai
from openai import OpenAI
import bilbystats as bs

openai.api_key = bs.read_api_key("openai")

# Now you can use the OpenAI client
client = openai

instructions = 'Give a single name answer.'
content = 'Who is the president of France?'
completion = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {"role": "system",
         "content": instructions},
        {
            "role": "user",
            "content": content
        }
    ],
    logprobs=True,
    # top_logprobs = 5
)
response = completion.choices[0].message
output = response.content
print(output)

# %%
instructions = 'Give a lower case one word answer which is either true or false'
content = '1+1 = 2'
completion = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {"role": "system",
         "content": instructions},
        {
            "role": "user",
            "content": content
        }
    ],
    logprobs=True,
    top_logprobs=20
)
response = completion.choices[0].message
output = response.content
print(output)
print(completion.choices[0].logprobs.content[0].logprob)
print(completion.choices[0].logprobs.content[0].top_logprobs)

# %%
instructions = 'To 1 decimal place what is the percentage probability that the following is true. Only return a percentage.'
content = 'Churchill is the prime minister of England'
completion = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {"role": "system",
         "content": instructions},
        {
            "role": "user",
            "content": content
        }
    ],
    logprobs=True,
    top_logprobs=5
)
response = completion.choices[0].message
output = response.content
print(output)
print(completion.choices[0].logprobs.content[0].logprob)
print(completion.choices[0].logprobs.content[0].top_logprobs)
