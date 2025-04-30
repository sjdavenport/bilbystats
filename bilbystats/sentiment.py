#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of functions for studying sentiment
"""

def get_sentiment_score(text):
    """
    Analyze the sentiment of a given text using FinBERT for financial sentiment analysis.

    This function uses the pre-trained FinBERT model to evaluate the sentiment of the input 
    text. The sentiment is returned as a continuous score, with the following transformations:
    - Negative sentiment: returns a negative score (negative of the model's score).
    - Neutral sentiment: returns a score shifted by 10.
    - Positive sentiment: returns the raw model score.

    The function automatically detects whether a GPU is available and uses it for faster processing 
    if possible.

    Parameters:
    -----------
    text : str
        The text for which the sentiment score is to be calculated.

    Returns:
    --------
    out : float
        A continuous sentiment score representing the text's sentiment. The score is modified as:
        - Negative sentiment: negative of the raw score.
        - Neutral sentiment: score shifted by +10.
        - Positive sentiment: raw score.
    """
    # Check if a GPU is available, otherwise default to CPU
    device = 0 if torch.cuda.is_available() else -1

    # Load FinBERT for financial sentiment analysis, using GPU if available
    sentiment_analyzer = pipeline(
        'sentiment-analysis',
        model='yiyanghkust/finbert-tone',
        device=device
    )

    result = sentiment_analyzer(text)
    # Return the continuous sentiment score
    if result[0]['label'] == 'Negative':
        out = -result[0]['score']
    elif result[0]['label'] == 'Neutral':
        out = result[0]['score'] + 10
    else:
        out = result[0]['score']
    return out
