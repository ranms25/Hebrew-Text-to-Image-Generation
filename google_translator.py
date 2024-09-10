# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

# Initialize the Sentence Transformer model for multilingual embeddings
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
model = SentenceTransformer(model_name)


def translate_text(text):
    """
    Translates the given Hebrew text to English.
    """
    target_language = "en"
    translation = GoogleTranslator(source="iw", target=target_language).translate(text)
    return translation


def translate_to_hebrew(sentences):
    """
    Translates a list of sentences to Hebrew.
    """
    target_language = "iw"
    translations = []
    for sentence in sentences:
        translated_text = GoogleTranslator(source="iw", target=target_language).translate(sentence[1])
        translations.append(translated_text)

    return translations


def precompute_hebrew_embeddings(original_text):
    """
    Precompute the embeddings for all Hebrew sentences in the original text using batch processing.
    """
    sentences = [sentence.strip() for sentence in original_text.split('.') if sentence.strip()]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return dict(zip(sentences, embeddings))


def find_most_similar_hebrew_with_penalty(precomputed_embeddings, translated_sentence, matched_sentences, idx,
                                          distance_threshold=3, similarity_threshold=0.5, penalty_factor=0.05):
    """
    Finds the top 3 most semantically similar Hebrew sentences for a given translated sentence,
    and penalizes distant unrelated sentences based on a distance threshold and similarity score.

    Parameters:
        - precomputed_embeddings (dict): Precomputed embeddings for Hebrew sentences.
        - translated_sentence (str): Translated English sentence to find a match for.
        - matched_sentences (set): Sentences that have already been matched to avoid duplicates.
        - idx (int): Index of the current sentence being matched.
        - distance_threshold (int): Number of sentence indices after which penalty applies.
        - similarity_threshold (float): Minimum similarity score below which penalty is applied for distant sentences.
        - penalty_factor (float): The factor by which the similarity score is reduced for distant unrelated sentences.

    Returns:
        tuple: (most similar sentence, max similarity, second-best sentence, second-best similarity, third-best sentence, third-best similarity).
    """
    top_sentences = []
    translated_embedding = model.encode(translated_sentence, convert_to_tensor=True).numpy().reshape(1, -1)

    # Get a list of all sentences and their embeddings
    all_sentences = list(precomputed_embeddings.keys())

    for sentence, embedding in precomputed_embeddings.items():
        if sentence in matched_sentences:
            continue

        similarity = cosine_similarity(embedding.reshape(1, -1), translated_embedding).item()

        # Get the index of the current Hebrew sentence
        sentence_idx = all_sentences.index(sentence)

        # Calculate the distance between the current sentence and the matched sentence
        distance = abs(sentence_idx - idx)

        # Apply penalty if the sentence is "distant" and has a low similarity score
        if distance > distance_threshold and similarity < similarity_threshold:
            penalty = penalty_factor * distance
            similarity -= penalty

        # Track the top 3 sentences by similarity
        top_sentences.append((sentence, similarity, sentence_idx))
        top_sentences = sorted(top_sentences, key=lambda x: x[1], reverse=True)[:3]

    if len(top_sentences) >= 3:
        most_similar_sentence, max_similarity, _ = top_sentences[0]
        second_best_sentence, second_best_similarity, _ = top_sentences[1]
        third_best_sentence, third_best_similarity, _ = top_sentences[2]
    elif len(top_sentences) == 2:
        most_similar_sentence, max_similarity, _ = top_sentences[0]
        second_best_sentence, second_best_similarity, _ = top_sentences[1]
        third_best_sentence, third_best_similarity = None, 0
    elif len(top_sentences) == 1:
        most_similar_sentence, max_similarity, _ = top_sentences[0]
        second_best_sentence, second_best_similarity = None, 0
        third_best_sentence, third_best_similarity = None, 0
    else:
        return None, -1, 0, 0, 0, None, None

    return most_similar_sentence, max_similarity, second_best_sentence, second_best_similarity, third_best_sentence, third_best_similarity


def align_hebrew_best_sentences(best_sentences, hebrew_base_text, similarity_threshold=0.05, distance_threshold=3,
                                penalty_factor=0.05):
    """
    Aligns sentences using a deferred matching strategy. Defers uncertain matches and resolves them
    when a certain match is found. Penalizes distant unrelated sentences based on a distance threshold.

    Parameters:
        - best_sentences (list): A list of best sentences identified in English.
        - hebrew_base_text (str): The original Hebrew text.
        - similarity_threshold (float): Threshold for considering scores as 'close'.
        - distance_threshold (int): Number of sentence indices after which penalty applies.
        - penalty_factor (float): Penalty factor for distant unrelated sentences.

    Returns:
        list: A list of tuples containing the most similar Hebrew sentence and its index in the original text.
    """
    precomputed_embeddings = precompute_hebrew_embeddings(hebrew_base_text)
    matched_sentences = set()  # Track already aligned sentences
    alignment_hebrew_best_sentences = []

    deferred_match = None  # Store deferred matches

    for idx, sentence_info in enumerate(best_sentences):
        translated_sentence = translate_to_hebrew([sentence_info])[0]

        # Get the top 3 most similar sentences with penalty logic applied
        most_similar_hebrew, max_similarity, second_best_hebrew, second_best_similarity, third_best_hebrew, third_best_similarity = find_most_similar_hebrew_with_penalty(
            precomputed_embeddings, translated_sentence, matched_sentences, idx, distance_threshold=distance_threshold,
            penalty_factor=penalty_factor
        )

        # Case: Handle deferred matching first (if we have a deferred sentence to resolve)
        if deferred_match:
            # If the next sentence shows certainty, resolve the deferred sentence
            if max_similarity - second_best_similarity > similarity_threshold:
                # Lock in the deferred match to its second-best
                alignment_hebrew_best_sentences.append((deferred_match['second_best_hebrew'], deferred_match['index']))
                matched_sentences.add(deferred_match['second_best_hebrew'])

                # Lock in the current certain match
                alignment_hebrew_best_sentences.append((most_similar_hebrew, idx))
                matched_sentences.add(most_similar_hebrew)

                deferred_match = None  # Clear deferred match
            else:
                # Continue deferring if no clear certainty
                deferred_match['max_similarity'] = max_similarity
                deferred_match['most_similar_hebrew'] = most_similar_hebrew
                deferred_match['second_best_similarity'] = second_best_similarity
                deferred_match['second_best_hebrew'] = second_best_hebrew
                deferred_match['index'] = idx
                continue

        # If the best and second-best scores are close, defer the decision
        elif abs(max_similarity - second_best_similarity) <= similarity_threshold:
            deferred_match = {
                'most_similar_hebrew': most_similar_hebrew,
                'index': idx,
                'second_best_hebrew': second_best_hebrew,
                'second_best_similarity': second_best_similarity,
                'max_similarity': max_similarity,
            }
        else:
            # No conflict or close match, finalize the assignment immediately
            alignment_hebrew_best_sentences.append((most_similar_hebrew, idx))
            matched_sentences.add(most_similar_hebrew)

    # If there's any remaining deferred match after the loop, finalize it:
    if deferred_match:
        alignment_hebrew_best_sentences.append((deferred_match['most_similar_hebrew'], deferred_match['index']))

    return alignment_hebrew_best_sentences
