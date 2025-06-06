"""
A set of functions for calling langchain
"""
import subprocess
from ollama import ChatResponse
from ollama import chat

from importlib import resources
from dotenv import load_dotenv
import os

# Use importlib to get a temporary path to the installed .env file
env_path = resources.files("bilbystats.defaults").joinpath("other.env")

# Load environment variables from the .env file
load_dotenv(dotenv_path=env_path)


def ollama(content: str, instructions: str = '',
           model_name: str = "llama3.2", stream: bool = False) -> str:
    """
    Send a prompt to a specified language model and return its response.

    This function wraps a call to a chat-based language model, allowing for
    optional instruction customization and support for multiple model aliases.
    It constructs the message payload and retrieves the generated response
    content.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    content : str
        The main prompt or user message to be sent to the model.
    instructions : str, optional
        Additional system-level instructions to guide the model's behavior.
        Defaults to an empty string.
    model_name : str, optional
        The name or alias of the model to be used. Supports aliases like 
        "llama", "llama8b", and "ds". Defaults to "llama3.2".
    stream: bool, optional
        Whether to respond as a stream or not.
    ---------------------------------------------------------------------------
    OUTPUT:
    output : str
        The content of the model's response message.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    if model_name == "llama":
        model_name = "llama3.2"
    elif model_name == "llama8b":
        model_name = "llama3.1:8b"
    elif model_name == "ds":
        model_name = 'deepseek-r1:7b'
    elif model_name == 'gemma':
        model_name = 'gemma3:4b'

    if stream:
        stream = chat(
            model=model_name,
            messages=[{'role': 'system', 'content': instructions},
                      {
                'role': 'user',
                'content': content,
            }],
            stream=True,
        )

        output = ""

        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)  # live print
            output += content

        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
    else:
        response: ChatResponse = chat(model=model_name, messages=[
            {'role': 'system', 'content': instructions},
            {
                'role': 'user',
                'content': content,
            },
        ])
        output = response.message.content

    return output


def get_ollama_models() -> list:
    """
    Retrieve a list of available Ollama models.

    This function attempts to execute the `ollama list` command to fetch a 
    list of installed Ollama models. It first checks for a custom path to the 
    Ollama binary via the `OLLAMA_PATH` environment variable. If unavailable, 
    it defaults to invoking `ollama` directly from the system path. 

    It parses the command output to extract and return model names.

    ---------------------------------------------------------------------------
    OUTPUT:
    models : list
        A list of model names (as strings) available through the Ollama CLI.

    ---------------------------------------------------------------------------
    RAISES:
    Exception
        If the Ollama binary cannot be found via either the environment 
        variable or system path, an exception is raised instructing the user 
        to redefine the OLLAMA_PATH.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    ollama_path = os.getenv("OLLAMA_PATH")
    try:
        result = subprocess.run([ollama_path, 'list'],
                                capture_output=True, text=True)
    except:
        try:
            result = subprocess.run(['ollama', 'list'],
                                    capture_output=True, text=True)
        except:
            raise Exception(
                "Ollama path not found, please redefine it appropriately in bilbystats/defaults/other.env")

    lines = result.stdout.strip().split('\n')
    if len(lines) < 2:
        return []

    models = []

    for line in lines[1:]:
        parts = line.split()
        # Basic parsing assuming format: name tag size modified
        if len(parts) >= 4:
            models.append(parts[0])

    return models
