import openai
import pandas as pd

def read_api_key(api_key_name="openai", key_dir="/Users/samd/Documents/Packages/othercode/api_keys/"):
    filename = key_dir + api_key_name + "_api_key.txt"
    api_key = open(filename).read().strip()
    return api_key