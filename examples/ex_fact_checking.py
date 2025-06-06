#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 19:55:04 2025

@author: samd
"""
import openai
from openai import OpenAI
import bilbystats as bs

openai.api_key = bs.read_api_key("openai")

# Now you can use the OpenAI client
client = openai
text = '''
The shingles vaccine is typically
recommended for adults aged 50 and
older. The vaccine is given in two
doses, with the second dose
administered 2 to 6 months after the
first dose. It is currently
recommended that individuals receive
the shingles vaccine once in their
lifetime. However, it is always best
to consult with a healthcare provider
for personalized recommendations.'''

instructions = '''
You will be given a text which makes a number of different claims. 
Your task will be to extract the claims and return them as follows:
    
- <claim 1>
- <claim 2> 
etc

Make sure to preserve the context so that each claim is self contained.
'''
out = bs.llm_api('Extract the different claims made in the following text:' +
                 text, instructions, 'gpt-4o-mini')
print(out)

# %%
instructions = 'Give a lower case one word answer which is either true or false'
content = 'The shingles vaccine is given in 2 doses'
completion = client.chat.completions.create(
    model='gpt-4o',
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

# %%
instructions = 'Give a lower case one word answer which is either true or false'
content = '''Given the following context from the NHS website: You're eligible for the shingles vaccine if you're aged 50 or over and you're at higher risk from shingles because you have a severely weakened immune system.

This includes:

some people with blood cancer (such as leukaemia or lymphoma)
some people with HIV or AIDS
some people who've recently had a stem cell transplant, radiotherapy, chemotherapy or an organ transplant
people taking certain medicines that severely weaken the immune system
You'll be given 2 doses of the shingles vaccine. These are given between 8 weeks and 6 months apart.

Ask your GP or care team if you're not sure if you're eligible for the shingles vaccine. 

Evaluate the following claim: The shingles vaccine is given in two doses.'''
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

# %%
text = bs.llm_api('How often is a shingles vaccine required', '', 'gpt-4o')

out = bs.llm_api('Extract the different claims made in the following text:' +
                 text, instructions, 'gpt-4o-mini')
