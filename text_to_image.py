"""
This script facilitates the generation of images from text prompts with integrated character descriptions. It
enhances sentences by inserting character descriptions, generates prompts suitable for image generation models,
and asynchronously retrieves images from a primary model or a secondary API if necessary. The script is designed
for use in applications where detailed and contextually accurate images are needed, particularly in scenarios
requiring the visual representation of characters and their attributes.

Key Features:
- Text enhancement by incorporating character descriptions.
- Prompt generation for selected art styles.
- Asynchronous API requests for efficient image retrieval.
- Fallback mechanism to a secondary API when the primary model fails.

Author: Ran Moshe
Date: December 20, 2023
"""

import asyncio
import re
import base64
import streamlit as st


def insert_description(sentence, character, description):
    """
    Integrates the character and its description at the beginning of the sentence if the character is mentioned.

    Parameters:
        - sentence (str): The original sentence where the description is to be inserted.
        - character (str): The character whose description is to be added.
        - description (str): The descriptive text to be inserted.

    Returns:
        str: The modified sentence with the character's description if the character is present; otherwise, returns the original sentence.
    """
    # Convert character name to lowercase for case-insensitive matching
    character = character.lower()
    # Clean the description by removing any text after a newline character
    cleaned_description = re.sub(r'\n.*', '', description)
    # Further clean the description by removing non-alphabetic characters and quotes
    cleaned_description = re.sub(r'[^a-zA-Z\s,]', '', cleaned_description).replace("'", '').replace('"', '')
    # Check if the character is mentioned in the sentence
    if character in sentence.lower():
        # Insert the character and its cleaned description at the beginning of the sentence
        modified_sentence = f"{character}: {cleaned_description.strip()}. {sentence}"
        return modified_sentence
    else:
        # If the character isn't mentioned, return the sentence unchanged
        return sentence


def process_text(sentence, character_dict):
    """
    Enhances the given sentence by incorporating descriptions for each mentioned character.

    Parameters:
        - sentence (str): The original sentence to be processed.
        - character_dict (dict): A dictionary mapping characters to their respective descriptions.

    Returns:
        str: The sentence modified to include character descriptions where applicable.
    """
    # Start with the original sentence
    modified_sentence = sentence

    # Iterate through each character and their descriptions
    for character, descriptions in character_dict.items():
        for description in descriptions:
            # Update the sentence with the character's description if the character is present
            modified_sentence = insert_description(modified_sentence, character, description)
    return modified_sentence


def generate_prompt(text, sentence_mapping, character_dict, selected_style):
    """
    Generates a prompt for image generation models based on the text, character descriptions, and selected art style.

    Parameters:
        - text (str): The original text or sentence.
        - sentence_mapping (dict): A mapping from original sentences to their enhanced versions.
        - character_dict (dict): Dictionary containing character descriptions.
        - selected_style (str): The chosen art style for image generation.

    Returns:
        tuple: A tuple containing the generated prompt and a negative prompt to guide image generation.
    """
    # Retrieve the enhanced version of the sentence; if not found, use the original
    enhanced_sentence = sentence_mapping.get(text, text)
    # Process the text to incorporate character descriptions
    image_descriptions = process_text(enhanced_sentence, character_dict)
    # Construct the prompt with the desired art style
    prompt = f"Make an illustration in {selected_style} style from: {image_descriptions}"
    # Define aspects to be avoided in the generated image
    negative_prompt = (
        "lowres, bad anatomy, bad hands, text, chat box, words, error, missing fingers, extra digit, "
        "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, "
        "watermark, username, blurry "
    )
    # For debugging: print the generated prompt
    # print(f"Generated prompt: {prompt}")
    return prompt, negative_prompt


async def query_model(session, model_name, prompt, negative_prompt):
    """
    Sends an asynchronous request to the specified image generation model with the given prompts.

    Parameters:
        - session (aiohttp.ClientSession): The asynchronous HTTP session for making requests.
        - model_name (str): The name of the Hugging Face model to query.
        - prompt (str): The positive prompt guiding image generation.
        - negative_prompt (str): The negative prompt specifying aspects to avoid.

    Returns:
        bytes: The content of the response, typically image bytes.
    """
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {
        "Authorization": f"Bearer {st.secrets.hf_credentials.header}"
    }  # Replace with your actual API key

    # Define the payload with additional parameters for image generation
    payload = {
        "inputs": prompt,
        "negative_prompt": negative_prompt,
        "width": 512,
        "height": 512,
        "guidance_scale": 12,
        "num_inference_steps": 50
    }
    # Make the POST request to the model's API endpoint
    async with session.post(api_url, headers=headers, json=payload) as response:
        response_content = await response.content.read()
        # For debugging: print response status
        print(f"Response status: {response.status}")
        if response.status != 200:
            # If there's an error, print the content for debugging
            print(f"Error: {response_content}")
        return response_content


async def get_image_from_primary(session, model_name, text, paragraph_number, sentence_mapping, character_dict,
                                 selected_style, hebrew_best_sentences_each_par):
    """
    Attempts to fetch an image from the primary image generation model based on the provided text and parameters.

    Parameters:
        - session (aiohttp.ClientSession): The asynchronous HTTP session for making requests.
        - model_name (str): The name of the primary Hugging Face model to query.
        - text (str): The text or sentence guiding image generation.
        - paragraph_number (int): The index of the current paragraph.
        - sentence_mapping (dict): Mapping from original to enhanced sentences.
        - character_dict (dict): Dictionary containing character descriptions.
        - selected_style (str): The chosen art style for image generation.
        - hebrew_best_sentences_each_par (list): List of best sentences in Hebrew for each paragraph.

    Returns:
        bytes or None: The image bytes if a valid image is received; otherwise, None.
    """
    # For debugging: print the paragraph being processed
    # print(f"Fetching primary image for paragraph {paragraph_number}")
    # Generate the prompts based on the text and other parameters
    prompt, negative_prompt = generate_prompt(text, sentence_mapping, character_dict, selected_style)
    # For debugging: print the prompt
    # print(f"Prompt for paragraph {paragraph_number}: {prompt}")
    # Query the model to get the image bytes
    image_bytes = await query_model(session, model_name, prompt, negative_prompt)
    # Check if the response starts with JPEG magic bytes indicating a valid image
    if image_bytes and image_bytes.startswith(b"\xff\xd8"):
        # For debugging: confirm valid image reception
        # print(f"Valid image received for paragraph {paragraph_number}")
        return image_bytes
    # For debugging: indicate invalid image reception
    # print(f"Invalid image received for paragraph {paragraph_number}")
    return None


async def get_image_from_space(client, paragraph_number, sentences, character_dict, selected_style):
    """
    Fetches an image from an alternative space API when the primary model doesn't yield results.

    Parameters:
        - client (GradioClient): The client for interacting with the Gradio API.
        - paragraph_number (int): The index of the current paragraph.
        - sentences (list): List of sentences guiding image generation.
        - character_dict (dict): Dictionary containing character descriptions.
        - selected_style (str): The chosen art style for image generation.

    Returns:
        bytes or None: The image bytes if successfully retrieved; otherwise, None.
    """
    try:
        # For debugging: indicate the initiation of image fetching from space API
        # print(f"Fetching image from space API for paragraph {paragraph_number}")
        # print(f"Sentences: {sentences}")
        # print(f"Character dict: {character_dict}")

        # Since Gradio API doesn't support async, use asyncio's to_thread to call it synchronously
        result = await asyncio.to_thread(client.predict, sentences, character_dict, selected_style, api_name="/predict")

        # For debugging: print the result type
        # print(f"Space API result for paragraph {paragraph_number}")
        # print(f"Result type: {type(result)}")

        if isinstance(result, dict):
            # For debugging: confirm result is a dictionary
            # print(f"Result is a dict")
            image_bytes_base64 = result.get(str(paragraph_number))
            if image_bytes_base64:
                # For debugging: indicate decoding of base64 image bytes
                # print(f"Decoding base64 image bytes for paragraph {paragraph_number}")
                # Decode the base64 string to get image bytes
                image_bytes = base64.b64decode(image_bytes_base64)
                return image_bytes
            else:
                print(f"No image found for paragraph {paragraph_number} in the result")
        else:
            print(f"Result is not a dict: {type(result)}")
            return None
    except Exception as e:
        print(f"Error fetching image from space API for paragraph {paragraph_number}: {str(e)}")
        return None
