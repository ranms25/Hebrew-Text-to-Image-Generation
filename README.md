# Automated Text Illustration

This repository contains a comprehensive set of scripts designed for the automated generation of text illustrations, with a particular focus on Hebrew language processing. The pipeline integrates various Natural Language Processing (NLP) tools, translation services, and advanced models to enhance and visualize textual content seamlessly.

## Scripts

### 1. `streamlit_app.py`

**Author:** Ran Moshe  
**Date:** December 28, 2023

This script serves as the main application for the project. It utilizes multiple functionalities to process Hebrew text, including translation, summarization, text segmentation, and text-to-image generation. The script is built using the Streamlit framework to create an interactive web application. Key features include:

- **Translation:** Converts Hebrew text to English for processing.
- **Coreference Resolution:** Focuses on replacing specified pronouns with the corresponding character names or references. It leverages spaCy's advanced coreference using an external Docker Space of HF.
- **Summarization:** Summarizes paragraphs to extract the most relevant content.
- **Text Segmentation:** Breaks down the text into paragraphs using NLP techniques.
- **Image Generation:** Generates illustrations based on text using models from Hugging Face's API.

### 2. `generate_and_enhanced_best_sentences.py`

**Author:** Ran Moshe  
**Date:** December 5, 2023

This script intelligently evaluates and enhances textual content by leveraging spaCy, Transformers, and a BERT-based model. It performs the following functions:
- **Sentence Evaluation:** Scores and selects the best sentences from paragraphs based on Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.
- **Sentence Enhancement:** Integrates descriptive words into the best sentences using a text generation model, improving the richness of the content.

### 3. `paragraphs_summary.py`

**Author:** Ran Moshe  
**Date:** November 23, 2023

This script uses the Falconsai summarization pipeline and tokenizer to summarize a collection of paragraphs. Each paragraph is individually summarized, ensuring that the most important information is retained. It is particularly useful for condensing large texts while preserving essential content.

### 4. `text_segmentation.py`

**Author:** Ran Moshe  
**Date:** November 19, 2023

Utilizing spaCy and Sentence Transformers, this script segments text into logical paragraphs. It employs natural language processing to:

- **Identify Sentence Boundaries:** Detects where one sentence ends and another begins.
- **Determine Paragraph Breaks:** Uses pre-trained models to analyze sentence embeddings and determine the best points to divide the text into paragraphs.

### 5. `google_translator.py`

**Author:** Ran Moshe  
**Date:** November 19, 2023

This script employs the GoogleTranslator from the deep_translator library to automatically translate Hebrew text into English. It is essential for preparing Hebrew content for further processing, such as summarization and image generation, which require English input.

### 6. `text_to_image.py`

**Author:** Ran Moshe  
**Date:** December 12, 2023

This script handles the generation of illustrations from textual descriptions. Key features include:
- **Streamlit Integration:** Creates a user-friendly web application for inputting text and generating images.
- **Hugging Face API Integration:** Connects to Hugging Face models to generate images based on processed text.
- **Character Descriptions:** Incorporates character descriptions directly into the image generation prompts, ensuring that the generated images are closely aligned with the narrative.
