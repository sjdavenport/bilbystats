# BilbyStats 
## A collection of  statistical and machine learning functions for use in the Bilby pipeline

### Clone the package
To pull the repo run
```bash
git clone --depth=1 https://github.com/bilbyai/bilbystats/
```

### Set up API keys
(Note that if you are just interested in the functionality without the api calls then you can skip this step 
and proceed directly to package installation.)

Navigate to the root directory of the package and run 
 ```bash
cp bilbystats/defaults/apikeys_example.env bilbystats/defaults/apikeys.env
```
This file looks like 

 ```bash
OPENAI_API_KEY=exampleapikey
BILBY_API_KEY=exampleapikey
DEEPSEEK_API_KEY=exampleapikey
CLAUDE_API_KEY=exampleapikey
```
You need to edit this file to include the true values of one or more of API keys that you have available.
Each key must be stroed with the key name in capitals followed by _API_KEY. If you don't want to use the LLM
functionality of the package then this step is optional. Note that the file bilbystats/defaults/apikeys.env is
.gitignored so that it is not uploaded to github. The next step is to install the package as shown below - 
if you would like to modify/add to the api keys you will need to edit the bilbystats/defaults/apikeys.env file
and then reinstall the package.

### Install the package

#### Optional: Create a conda environment
To create a conda environment for this repo, just run 
```bash
conda create --name bilbystats python=3.13
conda activate bilbystats
```
#### Installation using uv
Navigate to the root directory of the package and run 
```bash
uv pip install .
pip install -r requirements-pip.txt
```
Note that the requirements-pip.txt is required to deal with dependencies which uv struggles to install.
#### Alternative: Installation using just pip
Finally to install the package navigate to the root directory of the package and run
```bash
pip install .
pip install -r requirements-pip.txt
```
#### Importing the package
Then the package can be imported from within python via e.g. 

```bash
import bilbystats as bs
```

### Run local LLMs using Ollama
If you'd like to use the Ollama functions which allow you to call LLMs on your local machine you'll need to install [Ollama](https://ollama.com/).
Once you've installed ollama you can download LLMs such as llama3.2

```bash
ollama run llama3.2
```
or deepseek-r1:7b

```bash
ollama run deepseek-r1:7b
```

See https://ollama.com/search for a full list of the available models.

To call the LLM programmatically using bilbystats you can run

```
bs.llm_api('test call', 'you are an llm', 'llama3.2')
```
or 
```
bs.llm_api('test call', 'you are an llm', 'deepseek-r1:7b')
```
Or in general use the model name in any llm related function such as bs.translate. 


