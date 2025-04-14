# 🕵️ Privacy-Preserving Text Anonymization using Knowledge Distillation & LLMs
This project explores lightweight, scalable, and ethical Named Entity Recognition (NER) for text anonymization. It leverages Large Language Models for data annotation, and distills their knowledge into efficient transformer-based NER models (e.g., BERT, RoBERTa).

## 🧠 Project Motivation
In domains like healthcare, law, and finance, protecting sensitive information is paramount. Large Language Models have revolutionized NLP but raise concerns about data privacy. This project implements a resource-efficient NER pipeline that aligns with regulations such as GDPR, HIPAA, and CCPA.

## 📌 Key Features
🧪 LLM-powered data annotation using token-level prompts

⚙️ Knowledge distillation into smaller transformer models

🔐 Text anonymization compliant with privacy regulations

## 📂 Project Structure
``` bash
├── data_prep.py              # Script to annotate tokens using a local LLM (Mistral via Ollama API)
├── ETHICS_PROJECT_CODE.ipynb # Main notebook for training & evaluation
├── new_data.csv              # Input: tokenized data for annotation
├── ner_with_tags.csv         # Output: Annotated dataset with NER tags
```

## 🔄 Pipeline Overview
Data Collection: Public documents are preprocessed and tokenized.

Annotation via LLM: Local LLM labels tokens using a structured prompt via the Ollama API.

Training NER Model: Lightweight NER models (e.g., BERT) are trained on the annotated data.

Evaluation: Performance is tested on a hold-out set and compared against benchmarks.

📜 Sample Prompt (NER via LLM)
```
You are an expert Named Entity Recognition (NER) annotator...
Assign ONE NER label to each token using this format:
{'entity_group': '{class}', 'word': '{token}'}
Classes:
0 = O (not an entity)
1 = Person
2 = Location
3 = Organization
6 = Miscellaneous
```

## 🛠 Dependencies
Python 3.8+

tqdm, requests, csv, json, ast

Ollama for serving the LLM

HuggingFace Transformers (for NER model training)

Install with:

``` bash
pip install tqdm requests
```
## 🚀 Running the Code
Start your LLM locally via Ollama.

Prepare your new_data.csv with a tokenized_words column.

Run the data preparation script:

``` bash
python data_prep.py
```
Train and evaluate your NER model in ETHICS_PROJECT_CODE.ipynb.

## 👨‍👩‍👧‍👦 Team

Tanishk Nandal

Rushil Kumar

Bhavay Malhotra

Ajay Samuel Victor

## For any queries 
Email: avict019@uottawa.ca
