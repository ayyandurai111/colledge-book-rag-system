#!/bin/bash

pip install -e ".[ui]"

pip install --upgrade gradio==4.44.1

python -c "
from sentence_transformers import SentenceTransformer
print('Pre-downloading embedding model...')
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print('Model cached.')
"
