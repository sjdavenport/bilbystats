import pandas as pd
from typing import Optional
import numpy as np
import bilbystats as bs


def translate(text: str, model_name: str = "gpt-4o", languageout: str = "English") -> str:
    """
    Translate input text into a specified language using a language model.

    This function sends the input text to a language model API with an 
    instruction prompt that requests translation into the desired output 
    language. The translation result is returned as plain text.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    text : str
        The input text to be translated.
    model_name : str, optional (default="gpt-4o")
        The name of the language model to be used for translation.
    languageout : str, optional (default="English")
        The target language for translation.

    ---------------------------------------------------------------------------
    OUTPUT:
    translation : str
        The translated text in the target language.
    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    instructions = "You are a translator. Only return the " + \
        languageout+" translation. Do not add anything else."
    translation = bs.llm_api(text, instructions, model_name=model_name)
    if model_name == 'gemini':
        translation = translation.replace('\n, ')
        
    return translation


def translate_df(df: pd.DataFrame, covariate: str, languageout: str, model_name: str = "gpt-4o") -> pd.DataFrame:
    """
    Translate a column in a DataFrame to a specified language and add as a new column.

    This function takes a DataFrame column containing text, translates each entry
    to the specified output language using a language model, and adds the translations
    as a new column with the naming convention '{original_column}_{language}'.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    df : pd.DataFrame
        The input DataFrame containing the column to be translated.
    covariate : str
        The name of the column in the DataFrame to translate.
    languageout : str
        The target language for translation (e.g., "Chinese", "Spanish", "French").
    model_name : str, optional (default="gpt-4o")
        The name of the language model to use for translation.

    ---------------------------------------------------------------------------
    OUTPUT:
    df : pd.DataFrame
        The original DataFrame with an additional column containing the translations,
        named '{covariate}_{languageout}'.

    ---------------------------------------------------------------------------
    RAISES:
    ValueError:
        If the specified covariate column does not exist in the DataFrame.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    # Check if the covariate column exists
    if covariate not in df.columns:
        raise ValueError(f"Column '{covariate}' not found in DataFrame")

    # Create the new column name
    new_column_name = f"{covariate}_{languageout}"

    # Initialize the new column
    df[new_column_name] = None

    print(f"Starting translation of '{covariate}' column to {languageout}...")
    print(f"Processing {len(df)} entries using {model_name}")

    # Translate each entry
    for i in range(len(df)):
        # Show progress using bilbystats loader function
        bs.loader(i, len(df))

        try:
            # Get the original text
            original_text = df.loc[i, covariate]

            # Skip if the text is null or empty
            if pd.isna(original_text) or original_text == "":
                df.loc[i, new_column_name] = None
                continue

            # Translate using bilbystats translate function
            translated_text = bs.translate(
                original_text, model_name=model_name, languageout=languageout)

            # Store the translation
            df.loc[i, new_column_name] = translated_text

        except Exception as e:
            print(f"\nError translating entry {i}: {e}")
            df.loc[i, new_column_name] = None

    # Final progress update
    bs.loader(len(df), len(df))
    print(
        f"\nTranslation complete! New column '{new_column_name}' added to DataFrame.")

    return df


# Example usage:
# Load your dataframe
# df = pd.read_csv('golddatasetsinhakhandait.csv')

# Translate the News column to Chinese
# df = translate_df(df, covariate='News', languageout='Chinese', model_name='gpt-4o')

# Translate to other languages
# df = translate_df(df, covariate='News', languageout='Spanish', model_name='gpt-4o')
# df = translate_df(df, covariate='News', languageout='French', model_name='claude')

# Display some examples
# print("\nFirst 3 translations:")
# for i in range(min(3, len(df))):
#     print(f"\nEntry {i}:")
#     print(f"English: {df.loc[i, 'News']}")
#     print(f"Chinese: {df.loc[i, 'News_Chinese']}")

# Save the updated dataframe
# df.to_csv('translated_dataset.csv', index=False)
