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
from text_segmentation import nlp

nltk.download("punkt")  # Download the Punkt sentence tokenizer

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
# API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
# API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
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


def generate_best_sentence(paragraphs_summary):
    """
    Generate the best sentence for each paragraph based on NER, POS, and other criteria.

    Parameters:
        - paragraphs_summary (list): List of summarized paragraphs.

    Returns:
        list: List of tuples containing the paragraph number, best sentence, its score, and the index of the selected sentence.
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
            # print(
            #     f"\nProcessing Paragraph {paragraph_number + 1}:\n{original_sentence}"
            # )
            # Enhancement of sentences is currently disabled; using the original sentences
            # output = query(
            #     {
            #         "inputs": f"""text-generation: Add to each sentence adjectives and descriptions for nouns,
            #     keep the original meaning intact. text: "{original_sentence}" EndEnhancedText"""
            #     }
            # )
            output = original_sentence
            # print("generate_enhanced_text-output:", output)
            # Check if the response contains the expected information
            # Matching part is disabled
            # match = re.search(r'EndEnhancedText:(.*?)(?=\s*(?:text:|$))',
            #                   output[0]["generated_text"].replace("\n", " "), re.DOTALL)
            match = output
            # print(f'generate_enhanced_text - match:{match}')
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
                # print(f"Debug: Progress for Paragraph {paragraph_number + 1}: {progress_percent}%")
                # print("len(sentences_list):", len(sentences_list))
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


def query_and_extract(character, context_text, retries=2):
    """
    Query the Hugging Face API to generate a description for a character.
    If the original model fails after `retries` attempts, switch to a backup model for another `retries` attempts.

    Parameters:
        - character (str): The name of the character to describe.
        - context_text (str): The context that can influence the character's description.
        - retries (int): The number of attempts for each model (original and backup).

    Returns:
        dict: The generated character description, or an empty dictionary if all attempts fail.
    """
    # URL for the original model
    original_model_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    # URL for the backup model in case the original model fails
    backup_model_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

    def make_request(model_url, character, context_text):
        """
        Sends a POST request to the given model URL to generate a description based on the provided character and context.

        Parameters:
            - model_url (str): The API URL of the model to send the request to.
            - character (str): The name of the character to describe.
            - context_text (str): The context influencing the description.

        Returns:
            dict: The JSON response from the model.
        """
        # Create a prompt with specific instructions for generating a description
        prompt = (
            f"Describe the physical appearance of '{character}' in simple terms that are easy to draw for children's storybook. "
            f"Focus on features like size (e.g., small), height (e.g., tall), shape (e.g., broad), form (e.g., muscular), "
            f"color (e.g., blue eyes), and texture (e.g., rough). "
            f"The description must be limited to one sentence and include exactly four descriptive words, "
            f"structured in the format: "
            f"'CharacterName is an EntityType with first_description_word, second_description_word, "
            f"third_description_word, fourth_description_word' "
            f"\nCharacter: {character}\n"
            f"\nContext: {context_text}\n"
            f"\nGenerate the description here below:"
        )
        # Payload contains the input for the model
        payload = {"inputs": prompt}
        # Sending the request to the model API with the appropriate headers (e.g., API key)
        response = requests.post(model_url, headers=HEADERS, json=payload)
        # Return the model's JSON response
        return response.json()

    # Try to generate the description with the original model for the specified number of attempts
    for attempt in range(retries):
        # Request description from the original model
        output = make_request(original_model_url, character, context_text)
        if output:
            # Extract the description from the output using the existing extraction function
            description = extract_characteristics(output[0].get("generated_text", ""))
            if description:
                # Return the description if successfully generated
                return description

    # If the original model fails, try the backup model for the specified number of attempts
    for attempt in range(retries):
        # Request description from the backup model
        output = make_request(backup_model_url, character, context_text)
        if output:
            # Extract the description from the output
            description = extract_characteristics(output[0].get("generated_text", ""))
            if description:
                # Return the description if successfully generated from the backup model
                return description

    # If all attempts fail, print an error message and return an empty dictionary
    print(f"Failed to generate description for {character} after {2 * retries} attempts.")
    return {}


def extract_characteristics(generated_text):
    """
    Extracts character descriptions from the generated text based on a specific pattern.

    Parameters:
        - generated_text (str): The generated text containing character descriptions.

    Returns:
        list: List of extracted character descriptions, or None if not found.
    """
    # Find all occurrences of "Generate the description here below:"
    pattern = r"Generate the description here below:\s*['\"]?([^'\"]+)"
    matches = re.findall(pattern, generated_text, re.DOTALL)

    # If matches are found, return the list of descriptions
    if matches:
        return matches
    return None


# Modified `generate_general_descriptions` function to handle retries and model switching
def generate_general_descriptions(characters, text, max_retries=2):
    """
    Generates general descriptions for a list of characters based on their context within the provided text.
    If the original model fails, it switches to a backup model for two additional attempts.

    Parameters:
        - characters (list): List of character names to generate descriptions for.
        - text (str): The context text that may influence the character descriptions.
        - max_retries (int): Maximum number of attempts per model (both original and backup) to generate a description.

    Returns:
        dict: A dictionary mapping each character to its generated description.
    """
    # Dictionary to store descriptions for each character
    character_descriptions = {}
    # Loop through each character to generate a description
    for character in characters:
        # Use the query_and_extract function to generate the description with retries and model switching
        character_description = query_and_extract(character, text, retries=max_retries)
        if character_description:
            # Store the generated description in the dictionary
            character_descriptions[character] = character_description
    # Return the dictionary with generated descriptions
    return character_descriptions


def extract_main_characters(sentence, proper_noun_weight=0.3, boost_factor=1.2):
    """
    Extract main characters from a sentence based on proper nouns and nouns.
    """
    tokens = nlp(sentence)
    words = []
    original_words = []
    full_names = []  # Store full names like 'Dr. Henry Jekyll'
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.pos_ == "PROPN":
            combined_name = token.text

            # Combine with preceding titles
            if i > 0 and tokens[i - 1].text in {"Mr.", "Mrs.", "Dr.", "Sir"}:
                combined_name = tokens[i - 1].text + " " + combined_name

            # Combine with succeeding proper nouns
            while i < len(tokens) - 1 and tokens[i + 1].pos_ == "PROPN":
                combined_name += " " + tokens[i + 1].text
                i += 1

            # Handle "of" between proper nouns (e.g., "Queen of Hearts")
            if i < len(tokens) - 2 and tokens[i + 1].text.lower() == "of" and tokens[i + 2].pos_ == "PROPN":
                combined_name += " of " + tokens[i + 2].text
                i += 2

            words.extend([combined_name.lower()] * int(proper_noun_weight * 10))
            original_words.append(combined_name)
            full_names.append(combined_name.lower())  # Store the full name

        elif token.pos_ == "NOUN":
            words.append(token.text.lower())
            original_words.append(token.text)

        i += 1

    counts = Counter(words)

    boosted_counts = {word: count ** boost_factor for word, count in counts.items()}

    total_words = sum(boosted_counts.values())
    normalized_counts = {word: count / total_words for word, count in boosted_counts.items()}

    # Adjust counts for shorter names that are part of full names
    final_counts = {}
    for word, count in normalized_counts.items():
        matched = False
        for full_name in full_names:
            full_name_parts = full_name.split()
            word_parts = word.split()

            # Ensure all parts of the short name are within the full name
            if all(part in full_name_parts for part in word_parts):
                final_counts[full_name] = final_counts.get(full_name, 0) + count
                matched = True
                break

        if not matched:
            final_counts[word] = count

    # Dynamic threshold based on the top 3% frequencies
    threshold_index = min(int(0.03 * total_words), len(final_counts) - 1)
    threshold_value = sorted(final_counts.values(), reverse=True)[threshold_index]

    main_characters = [
        word for word, count in final_counts.items() if count >= threshold_value
    ]

    # Sort main characters by their final counts and select top 6
    sorted_main_characters = sorted(main_characters, key=lambda x: final_counts[x], reverse=True)[:6]

    capitalized_characters = [
        next((orig for orig in original_words if orig.lower() == character), character.capitalize())
        for character in sorted_main_characters
    ]

    # Post-processing to combine related names
    combined_characters = []
    for char in capitalized_characters:
        if any(existing in char or char in existing for existing in combined_characters):
            continue
        close_matches = [existing for existing in capitalized_characters if char in existing or existing in char]
        best_match = max(close_matches, key=len)
        combined_characters.append(best_match)

    capitalized_characters = [character.capitalize() for character in combined_characters]
    # print(capitalized_characters)
    return capitalized_characters
