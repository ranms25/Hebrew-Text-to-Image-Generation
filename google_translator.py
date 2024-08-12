"""
This script uses the GoogleTranslator from the deep_translator library to automatically translate a given text
from Hebrew to English. Additionally, it aligns translated sentences with the original Hebrew text based on
semantic similarity using Sentence Transformers.

Author: Ran Moshe
Date: November 19, 2023
"""

# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
import streamlit as st

# Initialize the Sentence Transformer model for multilingual embeddings
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
model = SentenceTransformer(model_name)


def translate_text(text):
    """
    Translates the given Hebrew text to English.

    Parameters:
        - text (str): The Hebrew text to translate.

    Returns:
        str: The translated English text.
    """
    target_language = "en"  # Set the target language to English
    translation = GoogleTranslator(source="iw", target=target_language).translate(text)
    return translation


def translate_to_hebrew(sentences):
    """
    Translates a list of sentences to Hebrew.

    Parameters:
        - sentences (list): List of tuples where each tuple contains sentence information.

    Returns:
        list: List of translated Hebrew sentences.
    """
    target_language = "iw"  # Set the target language to Hebrew

    translations = []
    for sentence in sentences:
        translated_text = GoogleTranslator(source="iw", target=target_language).translate(sentence[1])
        translations.append(translated_text)

    return translations


def find_most_similar_hebrew(original_text, translated_sentence):
    """
    Finds the most semantically similar Hebrew sentence in the original text to the given translated sentence.

    Parameters:
        - original_text (str): The original Hebrew text.
        - translated_sentence (str): The sentence to compare against.

    Returns:
        tuple: The most similar Hebrew sentence and its end index in the original text.
    """
    max_similarity = 0
    most_similar_sentence = None

    for sentence in original_text.split('.'):
        if not sentence.strip():
            continue

        # Compute embeddings for both sentences
        sentence_embedding = model.encode(sentence, convert_to_tensor=True).numpy().reshape(1, -1)
        translated_embedding = model.encode(translated_sentence, convert_to_tensor=True).numpy().reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(sentence_embedding, translated_embedding).item()

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_sentence = sentence.strip()
            end_index = original_text.find(sentence) + len(sentence)  # Update end index

    return most_similar_sentence, end_index


def align_hebrew_best_sentences(best_sentences, hebrew_base_text):
    """
    Aligns the best sentences identified in English with their corresponding Hebrew sentences.

    Parameters:
        - best_sentences (list): List of best sentences identified in English.
        - hebrew_base_text (str): The original Hebrew text.

    Returns:
        list: List of tuples containing the most similar Hebrew sentence and its index in the original text.
    """
    alignment_hebrew_best_sentences = []
    for idx, sentence_info in enumerate(best_sentences):
        # Translate the sentence back to Hebrew
        translated_sentence = translate_to_hebrew([sentence_info])[0]
        # Find the most similar Hebrew sentence in the original text
        most_similar_hebrew, index = find_most_similar_hebrew(hebrew_base_text, translated_sentence)
        alignment_hebrew_best_sentences.append((most_similar_hebrew, index))
    return alignment_hebrew_best_sentences
