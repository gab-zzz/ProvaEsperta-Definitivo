#create_faiss_index.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_documents(file_path="documents.json"):
    """Carica i documenti da un file JSON, se esiste."""
    if not os.path.exists(file_path):
        print(f"File {file_path} non trovato.")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Errore durante il caricamento dei documenti: {e}")
        return []

def create_faiss_index(documents=None, index_path="faiss_index.index", ids_path="document_ids.json"):
    """Crea e salva un indice FAISS a partire da una lista di documenti."""
    if documents is None:
        documents = load_documents()
        if not documents:
            print("Nessun documento trovato. Creando un indice vuoto.")
            documents = []

    if not documents:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        dummy = model.encode(["test"]).astype('float32')
        index = faiss.IndexFlatL2(dummy.shape[1])
        faiss.write_index(index, index_path)
        with open(ids_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print("Indice FAISS vuoto creato.")
        return

    texts = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Creazione degli embedding...")
    embeddings = model.encode(texts, show_progress_bar=True).astype('float32')

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(ids_path, 'w', encoding='utf-8') as f:
        json.dump(ids, f)

    print(f"Indice FAISS creato con {len(texts)} documenti.")
    print(f"Salvato: {index_path}, Mappatura ID: {ids_path}")

if __name__ == "__main__":
    if os.path.exists("faiss_index.index"):
        os.remove("faiss_index.index")
    if os.path.exists("document_ids.json"):
        os.remove("document_ids.json")
    if os.path.exists("documents.json"):
        os.remove("documents.json")

    create_faiss_index([])
