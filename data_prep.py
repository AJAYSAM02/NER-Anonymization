import csv
import json
import requests
import ast
import tqdm

MODEL_NAME = "mistral"
import requests
import re

OLLAMA_URL = "http://localhost:11434/api/chat"

VALID_CLASSES = {0, 1, 2, 3, 6}

def call_ollama_mistral(tokens):
    prompt = f"""
        You are an expert Named Entity Recognition (NER) annotator. Assign ONE NER label to each token. 

        **Rules:**
        - Output MUST be in the format `{'entity_group': '{class}', 'word': '{token}'}` (one per line)
        - Use ONLY these class IDs:
        0 = O (not an entity)
        1 = Person (e.g., "John")
        2 = Location (e.g., "Paris")
        3 = Organization (e.g., "NASA")
        6 = Miscellaneous (other entities)

        **Example Input:** ["Apple", "CEO", "Tim", "Cook", "said", "on", "Monday"]
        **Example Output:**
        [{'entity_group': 'ORG', 'word': 'Apple'},{'entity_group': 'O', 'word': 'CEO'},{'entity_group': 'PER', 'word': 'Tim'}]

        **Now process these tokens (ONE LABEL PER TOKEN):**
        {'\n'.join(tokens)}
        """ 

    payload = {
        "model": "mistral",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.1,  # Reduces randomness
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    output_text = response.json()["message"]["content"]

    # Parse the `token:class` format
    label_map = {}
    for line in output_text.strip().split('\n'):
        if ':' not in line:
            continue  # Skip malformed lines
        token, class_str = line.split(':', 1)
        token = token.strip()
        class_str = class_str.strip()
        
        if class_str.isdigit() and int(class_str) in VALID_CLASSES:
            label_map[token] = int(class_str)
        else:
            label_map[token] = 0  # Fallback to "O"

    # Ensure ALL tokens are labeled (in correct order)
    labels = []
    for token in tokens:
        labels.append(label_map.get(token, 0))  # Default to 0 if missing

    if len(labels) != len(tokens):
        raise ValueError(f"Label count mismatch (expected {len(tokens)}, got {len(labels)})")

    return labels


def generate_ner_csv(input_csv_path, output_csv_path):
    rows = []

    with open(input_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Tagging"):
            tokens = ast.literal_eval(row["tokenized_words"])
            ner_labels = call_ollama_mistral(tokens)

            if len(tokens) != len(ner_labels):
                raise ValueError(f"Length mismatch: {len(tokens)} tokens vs {len(ner_labels)} labels")

            row["Tagged_pos"] = json.dumps(ner_labels)
            rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(output_csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


generate_ner_csv("new_data.csv", "df.csv")