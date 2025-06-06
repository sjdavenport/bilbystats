"""
A collection of unit for the functions form llmapicalls.py
"""
import bilbystats as bs

instructions = bs.read_data('simple_sentiment_prompt.txt')

bs.llm_api('并在行风管理经办部门设立办公室', instructions, 'gpt-4o-mini')

bs.llm_api('What is Bilby?',
           instructions="Give a 1 sentence explanation", model_name="llama")

bs.translate('Bilby è fantastico!', 'gpt-4o-mini')
