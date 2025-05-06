import openai
import pandas as pd

def read_api_key(api_key_name="openai", key_dir="/Users/samd/Documents/Packages/othercode/api_keys/"):
    filename = key_dir + api_key_name + "_api_key.txt"
    api_key = open(filename).read().strip()
    return api_key


def openai_api(instructions, content, model_name="gpt-4o"):

    openai.api_key = read_api_key("openai")

    # Now you can use the OpenAI client
    client = openai

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "developer",
                "content": instructions},
            {
                "role": "user",
                "content": content
            }
        ]
    )
    reponse = completion.choices[0].message
    output = reponse.content
    return output


def translator(text, model_name="gpt-4o"):
    instructions = "You are an expert in translation of Chinese texts."
    content = "For the following text return its translation to English and just that: " + text
    translation = openai_api(instructions, content, model_name=model_name)
    return translation


def sentiment_detector(text, model_name="gpt-4o"):
    instructions = "You are an expert in sentiment analysis."
    content = "For the following text return positive/neutral/negative depending on the sentiment of the text. Just give the sentiment as an answer. Text: "
    content = content + text
    sentiment = openai_api(instructions, content, model_name=model_name)
    return sentiment