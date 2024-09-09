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


def precompute_hebrew_embeddings(original_text):
    """
    Precompute the embeddings for all Hebrew sentences in the original text using batch processing.

    Parameters:
        - original_text (str): The original Hebrew text.

    Returns:
        dict: A dictionary mapping each sentence to its precomputed embedding.
    """
    sentences = [sentence.strip() for sentence in original_text.split('.') if sentence.strip()]
    embeddings = model.encode(sentences, convert_to_tensor=True)  # Batch processing for efficiency
    return dict(zip(sentences, embeddings))


def find_most_similar_hebrew(precomputed_embeddings, translated_sentence, matched_sentences):
    """
    Finds the most semantically similar Hebrew sentence using precomputed embeddings.

    Parameters:
        - precomputed_embeddings (dict): A dictionary of precomputed embeddings for Hebrew sentences.
        - translated_sentence (str): The translated sentence to match.
        - matched_sentences (set): A set of already matched sentences to avoid duplicates.

    Returns:
        tuple: The most similar Hebrew sentence and its index in the original text.
    """
    max_similarity = 0
    most_similar_sentence = None
    end_index = -1

    # Compute the embedding for the translated sentence
    translated_embedding = model.encode(translated_sentence, convert_to_tensor=True).numpy().reshape(1, -1)
    for sentence, embedding in precomputed_embeddings.items():
        if sentence in matched_sentences:
            continue
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding.reshape(1, -1), translated_embedding).item()
        # Debugging: Concise similarity check
        # st.write(f"Hebrew: {sentence[:30]}... | Similarity: {similarity:.3f}")
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_sentence = sentence
            end_index = sentence
    # st.write(f"Best match: {most_similar_sentence[:30]}... | Score: {max_similarity:.3f}")
    return most_similar_sentence, end_index


def align_hebrew_best_sentences(best_sentences, hebrew_base_text):
    """
    Aligns the best sentences identified in English with their corresponding Hebrew sentences,
    ensuring each Hebrew sentence is aligned only once.

    Parameters:
        - best_sentences (list): A list of best sentences identified in English.
        - hebrew_base_text (str): The original Hebrew text.

    Returns:
        list: A list of tuples containing the most similar Hebrew sentence and its index in the original text.
    """
    # Precompute embeddings for all Hebrew sentences
    precomputed_embeddings = precompute_hebrew_embeddings(hebrew_base_text)
    matched_sentences = set()  # Track already aligned sentences
    alignment_hebrew_best_sentences = []
    for idx, sentence_info in enumerate(best_sentences):
        # Translate the sentence back to Hebrew
        translated_sentence = translate_to_hebrew([sentence_info])[0]
        # st.write(f"Aligning English sentence: {sentence_info[1][:30]}...")
        # Find the most similar Hebrew sentence excluding already matched sentences
        most_similar_hebrew, index = find_most_similar_hebrew(precomputed_embeddings, translated_sentence, matched_sentences)
        if most_similar_hebrew:
            matched_sentences.add(most_similar_hebrew)  # Avoid duplicates
        alignment_hebrew_best_sentences.append((most_similar_hebrew, index))
    return alignment_hebrew_best_sentences
