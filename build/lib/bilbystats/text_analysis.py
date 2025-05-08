"""
A collection of functions for performing text analysis.
"""
from snownlp import SnowNLP
import random
import nltk
#nltk.download('punkt')  # only once
from nltk.tokenize import sent_tokenize

def get_sentences(text, minlen=1, language='Chinese'):
    """
    Split input text into sentences, with optional language-specific processing and length filtering.

    This function tokenizes text into sentences. For Chinese text, it uses the `SnowNLP` library; 
    for other languages, it falls back to NLTK's `sent_tokenize`. It can also filter out sentences 
    shorter than a specified minimum length.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    text : str
        The input text to be split into sentences.
    minlen : int, optional (default=1)
        The minimum length a sentence must have to be included in the output. 
        Sentences shorter than this will be discarded.
    language : str, optional (default='Chinese')
        The language of the input text. Determines the sentence tokenizer to use.
        'Chinese' uses SnowNLP; all other values use NLTK's `sent_tokenize`.

    ---------------------------------------------------------------------------
    OUTPUT:
    output : list of str
        A list of sentences tokenized from the input text, filtered by minimum length.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    language_lower = language.lower()
    if language_lower == 'chinese':
        s = SnowNLP(text)
        sentences = s.sentences
    else:
        sentences = sent_tokenize(text)

    if minlen > 1:
        output = [sent for sent in sentences if len(sent) >= minlen]
    else:
        output = sentences
    return output

    
def get_random_sentence(text, minlen):
    """
    Select a random sentence from text after filtering by minimum length.

    This function extracts sentences from the input text using `get_sentences`,
    filters them based on a minimum length, and returns one randomly selected
    sentence from the filtered list.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    text : str
        The input text from which to extract and select a sentence.
    minlen : int
        Minimum length a sentence must have to be considered for selection.

    ---------------------------------------------------------------------------
    OUTPUT:
    random_sentence : str
        A randomly chosen sentence from the list of filtered sentences.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    sentences = get_sentences(text, minlen)
    random_sentence = random.choice(sentences)
    return random_sentence