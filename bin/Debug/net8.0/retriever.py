#retriever.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pubmed import search_pubmed
import logging

# Configurazione del logging per tracciare le operazioni
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Caricamento del modello pre-addestrato per l'encoding delle query e dei documenti
model = SentenceTransformer('all-MiniLM-L6-v2')

# Costanti per i file necessari
FAISS_INDEX_FILE = "faiss_index.index"
DOCS_FILE = "documents.json"
ID_MAP_FILE = "document_ids.json"

def get_query_embedding(query):
    """
    Calcola l'embedding (vettore) di una query.
    
    Args:
        query (str): La query da trasformare in embedding.

    Returns:
        numpy.ndarray: Vettore di embedding della query.
    """
    return model.encode([query]).astype('float32')

def ensure_faiss_index():
    """
    Verifica se esiste un indice FAISS salvato. Se non esiste o è corrotto, ne crea uno nuovo.
    
    Returns:
        faiss.Index: L'indice FAISS.
    """
    if os.path.exists(FAISS_INDEX_FILE):
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
            logger.info(f"Indice FAISS caricato con {index.ntotal} vettori.")
            return index
        except Exception as e:
            logger.warning(f"Indice FAISS corrotto, verrà ricreato: {e}")
    
    logger.info("Creazione nuovo indice FAISS vuoto.")
    dummy = model.encode(["test"]).astype('float32')
    index = faiss.IndexFlatL2(dummy.shape[1])
    faiss.write_index(index, FAISS_INDEX_FILE)
    return index

def load_json(file_path, default):
    """
    Carica un file JSON o restituisce un valore predefinito se il file non esiste.
    
    Args:
        file_path (str): Il percorso del file JSON da caricare.
        default (any): Il valore predefinito da restituire se il file non esiste.
    
    Returns:
        any: Il contenuto del file JSON o il valore predefinito.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default

def save_json(data, file_path):
    """
    Salva i dati in formato JSON su un file.
    
    Args:
        data (any): I dati da salvare.
        file_path (str): Il percorso del file dove salvare i dati.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_document_by_index(idx, documents, id_mapping):
    """
    Recupera un documento in base all'indice.

    Args:
        idx (int): L'indice del documento da recuperare.
        documents (list): La lista di documenti.
        id_mapping (list): La mappatura degli ID dei documenti.

    Returns:
        dict: Il documento recuperato o un messaggio di errore se non trovato.
    """
    if idx >= len(id_mapping):
        return {"text": "Documento non trovato", "title": "N/A"}
    doc_id = id_mapping[idx]
    return next((doc for doc in documents if doc["id"] == doc_id), {"text": "Documento non trovato", "title": "N/A"})

def filtra_risultati_per_rilevanza(query, results, threshold=0.5):
    """
    Filtra i risultati in base alla similarità con la query.

    Args:
        query (str): La query per la quale filtrare i risultati.
        results (list): I risultati da filtrare.
        threshold (float): La soglia di similarità per filtrare i risultati.

    Returns:
        list: I risultati filtrati per rilevanza.
    """
    logger.info(f"Ricerca con query: {query}")
    logger.info(f"Soglia di similarità: {threshold}")
    
    # Calcola l'embedding della query
    query_emb = model.encode([query])[0]
    
    # Filtra i documenti
    results_list = list(results)
    filtered_results = []
    
    for doc in results_list:
        testo = doc.get("text", "").strip()
        
        if not testo or testo in ["no abstract available", "no abstract", ""]:
            continue
        
        # Calcola l'embedding del documento
        doc_emb = model.encode([testo])[0]
        
        # Calcola la similarità tra query e documento
        dot_product = np.dot(query_emb, doc_emb)
        norm_query = np.linalg.norm(query_emb)
        norm_doc = np.linalg.norm(doc_emb)
        similarity = dot_product / (norm_query * norm_doc) if norm_query > 0 and norm_doc > 0 else 0
        
        # Se la similarità è sopra la soglia, aggiungi il documento
        if similarity >= threshold:
            doc['similarity'] = float(similarity)
            filtered_results.append(doc)
    
    logger.info(f"→ {len(results_list)} documenti trovati, {len(filtered_results)} dopo filtro di rilevanza.")
    
    # Ordina i risultati per similarità in ordine decrescente
    filtered_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    
    return filtered_results

def cerca_documenti(query, k=3, max_search=50, similarity_threshold=0.5):
    """
    Cerca i documenti più pertinenti per la query, prima in FAISS e poi su PubMed se necessario.

    Args:
        query (str): La query da cercare.
        k (int): Numero di risultati da restituire.
        max_search (int): Numero massimo di documenti da cercare.
        similarity_threshold (float): Soglia di similarità per il filtro.

    Returns:
        tuple: I documenti trovati, un flag che indica se sono stati trovati documenti in FAISS, e un flag per l'aggiornamento di FAISS.
    """
    logger.info(f"Ricerca documenti per la query: '{query}' (cercando {max_search} documenti, top {k} restituiti)")
    query_emb = get_query_embedding(query)
    index = ensure_faiss_index()
    id_mapping = load_json(ID_MAP_FILE, [])
    documents = load_json(DOCS_FILE, [])

    faiss_results = []
    if index.ntotal > 0:
        D, I = index.search(query_emb, min(max_search, index.ntotal))
        results = [get_document_by_index(i, documents, id_mapping) for i in I[0] if i < len(id_mapping)]
        valid_results = [r for r in results if "text" in r and "Documento non trovato" not in r["text"]]
        
        if valid_results:
            # Filtra per rilevanza semantica
            faiss_results = filtra_risultati_per_rilevanza(query, valid_results, similarity_threshold)
            logger.info(f"→ Trovati {len(valid_results)} documenti da FAISS, {len(faiss_results)} dopo filtro di rilevanza.")
            
            # Se i risultati sono sufficienti, restituisci
            if len(faiss_results) >= min(3, k):
                top_results = faiss_results[:k]
                logger.info("→ Contenuti finali inviati al Reasoner:")
                for i, doc in enumerate(top_results):
                    logger.info(f"[{i+1}] TITOLO: {doc.get('title', '')[:80]}...\nScore: {doc.get('similarity', 0):.3f}")
                return top_results, True, False
            else:
                logger.info("→ Risultati FAISS insufficienti, cerco anche su PubMed...")
    else:
        logger.info("→ Indice FAISS vuoto. Ricerca su PubMed...")

    # Cerca su PubMed
    nuovi_documenti = search_pubmed(query, max_results=max_search)

    if not nuovi_documenti:
        logger.info("→ Nessun risultato da PubMed.")
        return faiss_results[:k] if faiss_results else [], len(faiss_results) > 0, False

    # Filtra i risultati di PubMed per rilevanza
    nuovi_documenti_filtrati = filtra_risultati_per_rilevanza(query, nuovi_documenti, similarity_threshold)
    logger.info(f"→ {len(nuovi_documenti)} documenti trovati su PubMed, {len(nuovi_documenti_filtrati)} dopo filtro di rilevanza.")

    if not nuovi_documenti_filtrati:
        logger.info("→ Nessun documento rilevante da PubMed.")
        return faiss_results[:k] if faiss_results else [], len(faiss_results) > 0, False

    # Aggiorna l'indice FAISS con i nuovi documenti
    nuovi_embeddings = model.encode([d["text"] for d in nuovi_documenti_filtrati]).astype('float32')
    index.add(nuovi_embeddings)

    # Aggiungi i nuovi documenti e la mappatura degli ID
    aggiunti = 0
    for doc in nuovi_documenti_filtrati:
        if doc.get("id") and doc["id"] not in id_mapping and doc.get("text", "").strip():
            documents.append(doc)
            id_mapping.append(doc["id"])
            aggiunti += 1

    save_json(documents, DOCS_FILE)
    save_json(id_mapping, ID_MAP_FILE)
    faiss.write_index(index, FAISS_INDEX_FILE)

    logger.info(f"→ FAISS aggiornato con {aggiunti} nuovi documenti. Totale vettori: {index.ntotal}")

    # Combina i risultati da FAISS e PubMed, eliminando duplicati
    combined_results = faiss_results.copy()
    existing_ids = {doc["id"] for doc in combined_results}
    
    for doc in nuovi_documenti_filtrati:
        if doc["id"] not in existing_ids:
            combined_results.append(doc)
            existing_ids.add(doc["id"])
    
    # Ordina i risultati combinati per similarità
    combined_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    top_results = combined_results[:k]
    
    # Log dei risultati finali
    logger.info("→ Contenuti finali inviati al Reasoner:")
    for i, doc in enumerate(top_results):
        logger.info(f"[{i+1}] TITOLO: {doc.get('title', '')[:80]}...\nScore: {doc.get('similarity', 0):.3f}")

    return top_results, len(faiss_results) > 0, True

# Alias per compatibilità
search = cerca_documenti
