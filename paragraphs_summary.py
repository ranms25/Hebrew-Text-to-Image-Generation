"""
paragraphs_summary.py

This script utilizes the Falconsai summarization pipeline and tokenizer to summarize a collection of paragraphs.
The summarization is performed on each paragraph individually.

Author: Ran Moshe
Date: November 23, 2023
"""

import streamlit as st
from transformers import pipeline, AutoTokenizer

# Load falconsai summarization pipeline and tokenizer
# The pipeline is used for summarization, and the tokenizer is for tokenizing the input paragraphs.
falcon_summarizer = pipeline("summarization", model="Falconsai/text_summarization")
tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")


def paragraph_summary(paragraphs):
    """
    Summarizes a list of paragraphs using the Falconsai summarization model.

    Parameters:
        - paragraphs (list): A list of paragraphs (strings) to be summarized.

    Returns:
        list: A list of summarized paragraphs (strings).
    """
    summary_of_paragraphs = []

    # Iterate through each paragraph and summarize it individually
    for i, paragraph in enumerate(paragraphs):
        # Tokenize the paragraph to get the input IDs
        tokens = tokenizer(paragraph, return_tensors="pt")["input_ids"]

        # Set max_length based on the token count to ensure that the summary is neither too long nor too short.
        max_length = min(
            tokens.size(1), 150  # Max length for the summary; adjust if needed.
        )
        min_length = min(max_length, 20)  # Ensure that min_length is not greater than max_length.

        # Generate the summary using the Falconsai model
        summary = falcon_summarizer(
            paragraph, max_length=max_length, min_length=min_length, do_sample=False
        )

        # Append the summary of the current paragraph to the list
        summary_of_paragraphs.append(summary[0]["summary_text"])

    # Return the list of summaries
    return summary_of_paragraphs
