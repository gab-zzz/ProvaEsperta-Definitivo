#search_faiss.py
from retriever import search

def search_faiss(query, top_k=5):
    """
    Cerca nei documenti indicizzati con FAISS.

    Args:
        query (str): La query da cercare.
        top_k (int): Numero massimo di risultati da restituire.

    Returns:
        tuple: (risultati, trovato_in_faiss, aggiornato_da_pubmed)
    """
    try:
        # Esegui la ricerca nel retriever con la query e il limite top_k
        results, trovato_faiss, aggiornato_pubmed = search(query, top_k=top_k)
        
        # Restituisce i risultati, insieme ai flag che indicano
        # se sono stati trovati risultati in FAISS e se è stato aggiornato PubMed
        return results, trovato_faiss, aggiornato_pubmed
    except Exception as e:
        # Gestione degli errori in caso di problemi durante la ricerca FAISS
        print(f"Errore durante la ricerca FAISS: {e}")
        return [], False, False
