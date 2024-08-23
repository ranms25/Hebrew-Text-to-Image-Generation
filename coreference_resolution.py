import spacy
import re

# Load the advanced coreference model
nlp_coref = spacy.load("en_coreference_web_trf")
nlp = spacy.load("en_core_web_sm")  # Use this for pronoun density calculation

# Define the pronouns to replace, focusing on specific ones
REPLACE_PRONOUNS = {"he", "she", "they", "He", "She", "They"}


def extract_core_name(mention_text, main_characters):
    """Extracts the core name from the mention text with priority to main characters."""
    words = mention_text.split()
    for character in main_characters:
        if character.lower() in mention_text.lower():
            print(f"DEBUG: Main character '{character}' found in '{mention_text}'.")
            return character
    print(f"DEBUG: Extracting core name from '{mention_text}' -> '{words[-1]}'")
    return words[-1]


def calculate_pronoun_density(text):
    """Calculates the density of pronouns in the text."""
    doc = nlp(text)
    pronoun_count = sum(1 for token in doc if token.pos_ == "PRON" and token.text in REPLACE_PRONOUNS)
    named_entity_count = sum(1 for ent in doc.ents if ent.label_ == "PERSON")
    print(f"DEBUG: Pronoun count = {pronoun_count}, Named entity count = {named_entity_count}")
    return pronoun_count / max(named_entity_count, 1), named_entity_count


def resolve_coreferences_across_text(text, main_characters):
    # Process the text with the coreference model
    doc = nlp_coref(text)
    print("DEBUG: Processing coreference resolution.")

    # Create a mapping of coreference clusters
    coref_mapping = {}
    for key, cluster in doc.spans.items():
        if re.match(r"coref_clusters_*", key):  # Ensure it's a coreference cluster
            main_mention = cluster[0]  # The first mention is considered the main reference
            core_name = extract_core_name(main_mention.text, main_characters)
            print(f"DEBUG: Found cluster '{key}' with main mention '{main_mention.text}'. Core name: '{core_name}'")
            if core_name in main_characters:  # Only map if the core name is in main characters
                for mention in cluster:
                    for token in mention:
                        if token.text in REPLACE_PRONOUNS:  # Replace only if the token is one of the specified pronouns
                            # Handle case sensitivity
                            core_name_final = core_name if token.text.istitle() else core_name.lower()
                            coref_mapping[token.i] = core_name_final
                            print(f"DEBUG: Mapping token '{token.text}' at index {token.i} to '{core_name_final}'.")

    # Reconstruct the text with resolved coreferences
    resolved_tokens = []
    current_sentence_characters = set()
    current_sentence = []

    for i, token in enumerate(doc):
        if token.is_sent_start and current_sentence:
            # Reset for the new sentence
            resolved_tokens.extend(current_sentence)
            current_sentence_characters.clear()
            current_sentence = []

        if i in coref_mapping:
            core_name = coref_mapping[i]
            if core_name not in current_sentence_characters and core_name.lower() not in [t.lower() for t in
                                                                                          current_sentence]:
                current_sentence.append(core_name)
                current_sentence_characters.add(core_name)
                print(f"DEBUG: Replaced '{token.text}' with '{core_name}' at token index {i}.")
            else:
                print(
                    f"DEBUG: Skipping replacement of '{token.text}' with '{core_name}' as it already exists in the sentence.")
                current_sentence.append(token.text)
        else:
            current_sentence.append(token.text)

    # Add the last sentence if any
    resolved_tokens.extend(current_sentence)

    # Join tokens back into text
    resolved_text = " ".join(resolved_tokens)

    # Remove consecutive duplicate phrases
    resolved_text = remove_consecutive_duplicate_phrases(resolved_text)
    print(f"DEBUG: Final resolved text: {resolved_text}")
    return resolved_text


def remove_consecutive_duplicate_phrases(text):
    words = text.split()
    i = 0
    while i < len(words) - 1:
        j = i + 1
        while j < len(words):
            if words[i:j] == words[j:j + (j - i)]:
                del words[j:j + (j - i)]
            else:
                j += 1
        i += 1
    return " ".join(words)


def check_if_coreferences_across(text, main_characters):
    """Determines whether to apply coreference resolution based on pronoun density and named entity count."""
    pronoun_density, named_entity_count = calculate_pronoun_density(text)
    min_named_entities = len(main_characters)  # Set min_named_entities to the length of main characters
    print(f"DEBUG: Pronoun density: {pronoun_density}, Named entity count: {named_entity_count}")

    if pronoun_density > 0:
        print("Applying coreference resolution due to pronoun density.")
        return resolve_coreferences_across_text(text, main_characters)
    else:
        print("Skipping coreference resolution. Text is clear enough.")
        return text

