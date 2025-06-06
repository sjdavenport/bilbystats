from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='bilbystats',
    version='0.0.6',
    license='MIT',
    author='Samuel DAVENPORT',
    author_email='12sdavenport@gmail.com',
    url='https://github.com/sjdavenport/bilbystats/',
    download_url='https://github.com/bilbyai/bilbystats/',
    description='Python Packages of functions for performing stats for bilby',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='LLMs, Transformers',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "bilbystats": ["data/*.csv", "data/*.json", "data/*.parquet", "data/prompts/*.txt"],
        "bilbystats.defaults": ["config.yaml", "*.env"],
        "bilbystats.defaults.training": ["*.env"],
    },
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'scipy',
        'transformers',
        'datasets',
        'evaluate',
        'torch',
        'pandas',
        'openai',
        'dotenv',
        'anthropic',
        'ollama',
        'google-genai',
        'tiktoken',
        'nltk',
        'transformers[torch]',
        'seaborn',
        'spacy'
    ],
    python_requires='>=3',
)
