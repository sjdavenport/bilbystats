#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:55:00 2025

@author: samd
"""
import google.generativeai as genai
from google import genai
from openai import OpenAI
import openai
import bilbystats as bs

openai.api_key = bs.read_api_key("openai")

response = openai.Completion.create(
    model="gpt-4o",
    prompt="The quick brown fox jumps over the lazy dog.",
    max_tokens=0,
    logprobs=5,
    echo=True
)

for token, logprob in zip(response['choices'][0]['logprobs']['tokens'],
                          response['choices'][0]['logprobs']['token_logprobs']):
    print(f"Token: {token} | Logprob: {logprob}")

# %%

client = OpenAI(api_key=bs.read_api_key("openai"))

response = client.completions.create(
    model="text-davinci-002",
    prompt="The quick brown fox jumps over the lazy dog.",
    max_tokens=0,
    logprobs=5,
    echo=True
)

for token, logprob in zip(response.choices[0].logprobs.tokens,
                          response.choices[0].logprobs.token_logprobs):
    print(f"Token: {token} | Logprob: {logprob}")

# %%

client = OpenAI(api_key=bs.read_api_key("openai"))

models = client.models.list()

for model in models.data:
    print(model.id)

# %%

client = OpenAI(api_key=bs.read_api_key("deepseek"),
                base_url="https://api.deepseek.com")

response = client.completions.create(
    model="deepseek-chat",
    prompt="The quick brown fox jumps over the lazy dog.",
    max_tokens=0,
    logprobs=5,
    echo=True
)

for token, logprob in zip(response.choices[0].logprobs.tokens,
                          response.choices[0].logprobs.token_logprobs):
    print(f"Token: {token} | Logprob: {logprob}")

# %%

client = genai.Client(api_key=bs.read_api_key("gemini"))

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)

# %%

model_name = "gemini-2.0-flash"

instructions = 'translate'
content = 'ti amo'

# Initialize Client with API key
genai.configure(api_key=bs.read_api_key("gemini"))

# Initialize model
model = genai.GenerativeModel(model_name)

# Prepare chat history (mimicking OpenAI's messages format)
chat_history = [
    {"role": "system", "parts": [instructions]},
    {"role": "user", "parts": [content]}
]

# Generate content based on chat history
response = model.generate_content(chat_history)

# Extract response text
output = response.text
