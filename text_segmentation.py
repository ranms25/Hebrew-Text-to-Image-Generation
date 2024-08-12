"""
text_segmentation.py

This script uses spaCy and Sentence Transformers to segment text into paragraphs. It employs natural language processing
to identify sentence boundaries and utilizes pre-trained models for sentence embeddings to determine paragraph breaks.

Author: Ran Moshe
Date: November 19, 2023
"""

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# Load the English language model from spaCy
nlp = spacy.load("en_core_web_sm")

# Load the pre-trained SentenceTransformer model for embedding sentences
model = SentenceTransformer("all-mpnet-base-v2")


def dynamic_threshold(scores, percentile=75):
    """
    Calculates a dynamic threshold based on the given percentile of the similarity scores.

    Parameters:
        - scores (list or np.array): The list of similarity scores.
        - percentile (int): The percentile to be used for threshold calculation (default is 75).

    Returns:
        float: The calculated threshold value.
    """
    return np.percentile(scores, percentile)


def clean_text_spaces(text):
    """
    Cleans the text by removing unnecessary tabs, newlines, and extra spaces.

    Parameters:
        - text (str): The original text to be cleaned.

    Returns:
        str: The cleaned text with normalized spaces.
    """
    cleaned_text = " ".join(text.split())  # Join words with a single space
    return cleaned_text


def segment_text(text, threshold_percentile=75):
    """
    Segments text into paragraphs based on sentence embeddings and similarity scores.

    Parameters: - text (str): The input text to be segmented. - threshold_percentile (int): The percentile to
    determine the similarity threshold for paragraph breaks (default is 75).

    Returns:
        list: A list of segmented paragraphs.
    """
    # Clean the text before processing
    text = clean_text_spaces(text)

    # Process the text using spaCy to identify sentences
    doc = nlp(text)
    sentences = list(doc.sents)  # Convert generator to a list for iteration

    # Initialize an empty list to store paragraphs
    paragraphs = []
    current_paragraph = []

    # Initialize an empty list to store similarity scores
    similarity_scores = []

    # Encode sentences to obtain their embeddings using SentenceTransformer
    sentence_embeddings = model.encode(
        [sent.text for sent in sentences], convert_to_tensor=True
    )

    # Iterate through the sentences to determine paragraph breaks
    for i in range(len(sentences)):
        if i == 0:
            # For the first sentence, start a new paragraph
            current_paragraph.append(sentences[i].text)
        else:
            # Calculate the similarity score with the previous sentence
            similarity_previous = util.pytorch_cos_sim(
                sentence_embeddings[i - 1].unsqueeze(0),
                sentence_embeddings[i].unsqueeze(0),
            )[0][0]

            if i < len(sentences) - 1:
                # Calculate the similarity score with the next sentence
                similarity_next = util.pytorch_cos_sim(
                    sentence_embeddings[i + 1].unsqueeze(0),
                    sentence_embeddings[i].unsqueeze(0),
                )[0][0]

                # Append the similarity scores to the list
                similarity_scores.append(similarity_previous)
                similarity_scores.append(similarity_next)

                # Check if the current sentence is a good candidate for a new paragraph
                if (
                    sentences[i].text[0].isupper()  # Starts with a capital letter
                    and sentences[i - 1].text[-1] in [".", "!", "?"]  # Ends with a punctuation mark
                    and (similarity_previous + similarity_next) / 2  # Average similarity is below the threshold
                    < dynamic_threshold(similarity_scores, threshold_percentile)
                ):
                    # Finalize the current paragraph and start a new one
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = [sentences[i].text]
                    similarity_scores = []  # Reset similarity scores for the new paragraph
                else:
                    # Continue adding to the current paragraph
                    current_paragraph.append(sentences[i].text)
            else:
                # Handle the last sentence, only considering similarity with the previous one
                similarity_scores.append(similarity_previous)
                current_paragraph.append(sentences[i].text)

    # Append the last paragraph to the list
    paragraphs.append(" ".join(current_paragraph))

    # Return the segmented paragraphs
    return paragraphs
