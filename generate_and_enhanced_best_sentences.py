# -*- coding: utf-8 -*-

"""
generate_and_enhanced_best_sentences.py

This script leverages the power of spaCy, Transformers, and a BERT-based model for part-of-speech tagging and named
entity recognition (NER). It intelligently evaluates a set of paragraphs, meticulously scoring and selecting the most
apt sentences by combining NER and POS criteria. The identified elite sentences are not only cataloged but also
enhanced by seamlessly integrating descriptive words using a cutting-edge text generation model.

Author: Ran Moshe
Date: December 5, 2023
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import requests
import re
import nltk
from collections import Counter
import streamlit as st

nltk.download("punkt")  # Download the Punkt sentence tokenizer

# Load spaCy English model for linguistic processing
nlp = spacy.load("en_core_web_sm")

# Load BERT-based model specifically trained for part-of-speech tagging
model_name_ner = "QCRI/bert-base-multilingual-cased-pos-english"
tokenizer_ner = AutoTokenizer.from_pretrained(model_name_ner)
model_ner = AutoModelForTokenClassification.from_pretrained(
    model_name_ner, ignore_mismatched_sizes=True
)

# Instantiate the NER pipeline using the loaded model and tokenizer
ner_pipeline = pipeline(task="ner", model=model_ner, tokenizer=tokenizer_ner, device=-1)

# API endpoint for making requests to the Hugging Face model
# API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b"
# API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b"
# API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b-it"
# API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder2-15b"
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
# API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"


# Headers including the authorization token for accessing the API
HEADERS = {"Authorization": f"Bearer {st.secrets.hf_credentials.header}"}

# Define scoring weights for each category in NER (Named Entity Recognition)
category_weights_ner = {
    "NNP": 3,  # Proper noun, singular
    "NN": 2,  # Noun, singular or mass
    "NNS": 3,  # Noun, plural
    "VBD": 2,  # Verb, past tense
    "VBG": 12,  # Verb, gerund or present participle
    "RB": 3,  # Adverb
    "JJ": 8,  # Adjective
    "CC": 1,  # Coordinating conjunction
    "PRP": 2,  # Personal pronoun
    "PRP$": 2,  # Possessive pronoun
    "IN": 1,  # Preposition or subordinating conjunction
    "PDT": 1,  # Predeterminer
}

# Define scoring weights for each category in POS (Part of Speech)
category_weights_pos = {
    "ADJ": 6,  # Adjective
    "ADV": 6,  # Adverb
    "PRONP": 4,  # Proper noun (assumed typo; possibly "PROPN")
    "NOUN": 3,  # Noun
    "VERB": 3,  # Verb
    "PRON": 1,  # Pronoun
    "DET": 1,  # Determiner
    "CCONJ": 1,  # Coordinating conjunction
    "SCONJ": 1,  # Subordinating conjunction
    "ADP": 1,  # Adposition
    "NER": 1,  # Named Entity Recognition
}

# Define weights for each scoring aspect
weight_ner = 0.3  # Weight for NER score
weight_pos = 0.3  # Weight for POS score
weight_descriptive = 0.3  # Weight for descriptive word density
weight_length = 0.1  # Weight for sentence length


def query(payload, characters=None):
    """
    Make a request to the Hugging Face model API with the given payload.

    Parameters:
        - payload (dict): The payload to be sent in the API request.
        - characters (list): Optional. List of characters to include in the payload.

    Returns:
        dict: The JSON response from the API.
    """
    try:
        if characters:
            payload["characters"] = characters

        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()

    except requests.exceptions.HTTPError as errh:
        print("Http Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        print("Oops: Something Else", err)


def generate_best_sentence(paragraphs_summary):
    """
    Generate the best sentence for each paragraph based on NER, POS, and other criteria.

    Parameters:
        - paragraphs_summary (list): List of summarized paragraphs.

    Returns: list: List of tuples containing the paragraph number, best sentence, its score, and the index of the
    selected sentence.
    """
    # List to store the best sentences along with their paragraph numbers and scores
    best_sentences = []

    # Assess and select the most suitable sentence for each paragraph
    for i, paragraph in enumerate(paragraphs_summary):
        # Tokenize the paragraph into sentences using spaCy
        sentences = [sent.text for sent in nlp(paragraph).sents]
        # Variables to store the best sentence and its score
        best_sentence = ""
        best_score = float("-inf")

        # Evaluate each sentence using NER and POS criteria
        for j, sentence in enumerate(sentences):
            # Use the NER pipeline for part-of-speech tagging
            predictions = ner_pipeline(sentence)

            # Extract POS tags for each token from NER predictions
            pos_tags = [(token["word"], token["entity"]) for token in predictions]

            # Score the sentence based on criteria from NER
            score_ner = sum(category_weights_ner.get(tag, 0) for _, tag in pos_tags)

            # Score the sentence based on criteria from POS using spaCy
            score_pos = sum(
                category_weights_pos.get(token.pos_, 0) for token in nlp(sentence)
            )

            # Calculate the density of descriptive words (adjectives and adverbs)
            descriptive_word_density = sum(
                1 for token in nlp(sentence) if token.pos_ in {"ADJ", "ADV"}
            )

            # Calculate the overall length of the sentence in words
            sentence_length = len(sentence.split())

            # Combine scores with additional factors and apply predefined weights
            total_score = (
                    weight_ner * score_ner
                    + weight_pos * score_pos
                    + weight_descriptive * descriptive_word_density
                    + weight_length * sentence_length
            )

            # Update the best sentence if the current one has a higher total score
            if total_score > best_score:
                best_score = total_score
                best_sentence = sentence

        # Append the best sentence, its corresponding paragraph number, and its score to the list
        best_sentences.append((i, best_sentence.capitalize(), best_score))

    return best_sentences


def chunk_text(text, max_chars=325):
    """
    Split the text into chunks based on a maximum character limit.

    Parameters:
        - text (str): The input text to be split.
        - max_chars (int): The maximum number of characters per chunk.

    Returns:
        list: List of text chunks.
    """
    chunks = []
    current_chunk = ""
    joined_sentences = " ".join(text)
    # Split the text into sentences using NLTK's sentence tokenizer
    sentences = nltk.sent_tokenize(joined_sentences)
    for sentence in sentences:
        # Check if adding the current sentence to the current chunk exceeds the limit
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            # If adding the sentence exceeds the limit, start a new chunk
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    # Add the last chunk if present
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def generate_enhanced_text(sentences_list, progress_bar):
    """
    Generate enhanced text by seamlessly integrating descriptive words into each sentence.

    Parameters:
        - sentences_list (list): List of tuples containing the paragraph number, best sentence, and its score.
        - progress_bar (streamlit.Progress): Streamlit progress bar to visualize processing.

    Returns:
        tuple: Combined enhanced text and a mapping between original and enhanced sentences.
    """
    try:
        enhanced_sentences = []
        sentence_mapping = {}  # Mapping between original and enhanced sentences
        # Loop through each sentence in the original list
        for sentence_info in sentences_list:
            paragraph_number, original_sentence, _ = sentence_info
            print(
                f"\nProcessing Paragraph {paragraph_number + 1}:\n{original_sentence}"
            )
            # Enhancement of sentences is currently disabled; using the original sentences
            # output = query(
            #     {
            #         "inputs": f"""text-generation: Add to each sentence adjectives and descriptions for nouns,
            #     keep the original meaning intact. text: "{original_sentence}" EndEnhancedText"""
            #     }
            # )
            output = original_sentence
            print("generate_enhanced_text-output:", output)
            # Check if the response contains the expected information
            # Matching part is disabled
            # match = re.search(r'EndEnhancedText:(.*?)(?=\s*(?:text:|$))',
            #                   output[0]["generated_text"].replace("\n", " "), re.DOTALL)
            match = output
            print(f'generate_enhanced_text - match:{match}')
            if match:
                # Enhancement extraction is disabled; using the match directly
                # enhanced_text = match.group(1)
                enhanced_text = match
                # Replace newline characters with spaces
                enhanced_text = enhanced_text.replace("\n", " ")
                # print(f'enhanced_text:{enhanced_text}')
                enhanced_sentences.append((paragraph_number, enhanced_text))

                # Update the sentence mapping
                sentence_mapping[original_sentence] = enhanced_text
                # print(f'sentence_mapping:{sentence_mapping}')
                progress_percent = 55 + paragraph_number * (55 / len(sentences_list))
                progress_value = min(int(progress_percent), 100)
                progress_bar.progress(progress_value, text="תכף מסיימים!")  # Hebrew for "Almost done!"
                print(f"Debug: Progress for Paragraph {paragraph_number + 1}: {progress_percent}%")
                print("len(sentences_list):", len(sentences_list))
            else:
                # Handle the case where the match is not found
                print(f"Error: Unable to extract enhanced text for the given sentence")

        # Combine the enhanced texts for all paragraphs
        combined_enhanced_text = " ".join(
            enhanced_text for _, enhanced_text in enhanced_sentences
        )
        return combined_enhanced_text, sentence_mapping

    except Exception as e:
        return f"Error: Unable to generate enhanced text. Details: {str(e)}"

    # If any check fails, return an error message
    return (
        "Error: Unable to generate enhanced text. Details: Invalid response format.",
        None,
    )


def extract_characteristics(generated_text, characters):
    """
    Extract characteristics of given characters from the generated text.

    Parameters:
        - generated_text (str): The text generated with integrated descriptive words.
        - characters (list): List of characters to extract characteristics for.

    Returns:
        dict: Dictionary mapping characters to their extracted descriptions.
    """
    characteristics = {}
    print(f"extract_characteristics - generated_text:{generated_text}")
    print(f"extract_characteristics - characters:{characters}")
    # If only one character is provided
    if len(characters) == 1:
        character = characters[0].lower()  # Convert to lowercase for case-insensitive comparison
        description = generated_text.strip()  # Strip leading/trailing whitespace

        # Capitalize the character for consistent formatting
        characteristics[characters[0].capitalize()] = [description]

        return characteristics

    # Create a regex pattern to find text between each character
    pattern = re.compile(
        rf"({'|'.join(characters)})(.*?)(?=(?:{'|'.join(characters)}|$))",
        flags=re.IGNORECASE | re.DOTALL,
    )
    print("pattern:", pattern)
    # Find matches in the generated text
    matches = pattern.findall(generated_text)
    print("matches:", matches)
    for match in matches:
        character, description = match[0].strip(), match[1].strip()

        # Convert the character to lowercase for case-insensitive comparison
        character_lower = character.lower()

        # Check if the character is already in the dictionary
        if character_lower not in characteristics:
            characteristics[character_lower] = [description]

    # Convert the keys back to the original case
    characteristics = {
        key.capitalize(): value for key, value in characteristics.items()
    }

    return characteristics


def generate_general_descriptions(characters, text):
    """
    Generate general descriptions for given characters based on external characteristics.

    Parameters:
        - characters (list): List of characters to generate descriptions for.
        - text (str): The input text used for context in generating descriptions.

    Returns:
        dict: Dictionary mapping characters to their generated general descriptions.
    """
    # Construct the payload with the characters and text
    payload = {
        "inputs": f"Please provide 5 words of external descriptive characteristics for the following objects if they "
                  f"are missing. "
                  f"If specific situations or actions are mentioned, please disregard them."
                  f"Objects: {', '.join(characters)}."
                  f"Text: {text} generate:",
    }

    # Make the API call to generate general descriptions
    output = query(payload)
    print(f'generate_general_descriptions-output:{output}')
    try:
        # Extract the "text" section from the output
        generated_text = output[0].get("generated_text", "")
        print(f'generate_general_descriptions-generated_text:{generated_text}')
        print(f'generated_text.split("generate:")[1]: {generated_text.split("generate:")[1]}')
        # Extract character descriptions using the modified function
        character_descriptions = extract_characteristics(
            generated_text.split("generate:")[1], characters
        )
        print(f'generate_general_descriptions-character_descriptions:{character_descriptions}')
        return character_descriptions

    except Exception as e:
        print(f"Error: {str(e)}")
        return {}


def extract_main_characters(sentence):
    """
    Extract main characters from a sentence based on proper nouns and nouns.

    Parameters:
        - sentence (str): The sentence to extract main characters from.

    Returns:
        list: List of main characters extracted from the sentence.
    """
    # Tokenize the sentence using spaCy
    tokens = nlp(sentence)

    # Extract proper nouns and nouns
    words = [token.text.lower() for token in tokens if token.pos_ in ["PROPN", "NOUN"]]

    # Count occurrences of each word
    counts = Counter(words)

    # Normalize by text length
    total_words = len(words)
    normalized_counts = {word: count / total_words for word, count in counts.items()}

    # Set a dynamic threshold (e.g., top 3% of frequencies)
    threshold_index = min(int(0.03 * total_words), len(normalized_counts) - 1)
    threshold_value = sorted(normalized_counts.values(), reverse=True)[threshold_index]

    # Filter out words below the threshold
    main_characters = [
        word for word, count in normalized_counts.items() if count >= threshold_value
    ]
    return main_characters
