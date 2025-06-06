#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 18:56:29 2025

@author: samd
"""


def api2data(colname, promptloc, dataloc, saveloc, task='Classify the following text: ',
             model_name="gpt-4o", labels=['Label:', 'Explanation:'], names=['label', 'explanation'], lowercase=[True, False]):
    """
    Apply an LLM API to a dataset column and save results with multiple extractable fields.
    This function reads a dataset from a Parquet file, applies a large language model (LLM) API
    to each entry in a specified column, and extracts multiple labeled fields from the model's
    response. The results are saved back into the dataset under dynamically generated column
    names based on the model name, unless explicitly specified.
    ---------------------------------------------------------------------------
    ARGUMENTS:
    colname : str
        Name of the column in the dataset whose entries will be passed as input to the LLM.
    promptloc : str
        Path to the file containing instructions or prompts for the LLM.
    dataloc : str
        Path to the input Parquet file containing the dataset.
    saveloc : str
        Path where the resulting Parquet file with model outputs will be saved.
    task : str, optional (default='Classify the following text: ')
        Task prefix to prepend to each input text before sending to the LLM.
    model_name : str, optional (default="gpt-4o")
        Name of the LLM model to be used for inference.
    names : list, optional (default=[None, None])
        List of column names where the extracted fields will be stored in the dataset.
        If None for any element, defaults to "<model_name>_<label_suffix>" where
        label_suffix is derived from the corresponding label (e.g., "label", "explanation").
    labels : list, optional (default=['Label:', 'Explanation:'])
        List of label prefixes to search for in the model's response. Each line of the
        response should start with one of these prefixes for proper extraction.
    ---------------------------------------------------------------------------
    OUTPUT:
    None
        The function saves the updated dataset with new columns for each extracted field
        to the specified output Parquet file but does not return any object.
    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    # Ensure names and labels have the same length
    if len(names) != len(labels):
        raise ValueError("Length of 'names' and 'labels' lists must be equal")

    # Load the dataset
    df = pd.read_parquet(dataloc)

    final_names = []
    for i in np.arange(len(names)):
        name = names[i]
        final_names.append(f"{model_name}_{name}")

    # Initialize new columns with NaN
    for name in final_names:
        df[name] = np.nan

    # Read instructions from prompt file
    with open(promptloc, 'r') as file:
        instructions = file.read()

    # Process each row
    for i in np.arange(len(df)):
        bs.loader(i, len(df))
        # Apply the LLM and get response
        output = bs.llm_api(task + df[colname][i], instructions, model_name)
        lines = output.strip().splitlines()

        # Extract each labeled field
        for j, label_prefix in enumerate(labels):
            # Find the line that starts with this label
            for line in lines:
                if line.startswith(label_prefix):
                    extracted_value = line.replace(
                        label_prefix, "").strip()
                    if lowercase[j]:
                        extracted_value = extracted_value.lower()
                    df.loc[i, final_names[j]] = extracted_value

    # Save the updated dataset
    df.to_parquet(saveloc)
