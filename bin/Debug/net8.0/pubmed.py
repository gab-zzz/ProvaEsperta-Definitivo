#pubmed.py
import requests
import xml.etree.ElementTree as ET
import logging
import re

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pre_elabora_query(query):
    # Porta tutto in minuscolo e rimuove le parole comuni
    query_lower = query.lower()

    stop_words = [
        'what', 'are', 'is', 'the', 'how', 'when', 'why', 'which', 'does', 'do', 
        'can', 'could', 'would', 'should', 'a', 'an', 'of', 'in', 'on', 'for', 
        'to', 'with', 'and', 'from', 'i', 'you', 'have'
    ]

    words = re.findall(r'\b\w+\b', query_lower)
    keywords = [word for word in words if word not in stop_words]

    # Aggiungi termini clinici se necessari
    expanded_keywords = keywords.copy()
    if 'pregnancy' in keywords:
        expanded_keywords += ['symptoms', 'clinical', 'manifestations']

    return ' '.join(expanded_keywords)

def search_pubmed(query, max_results=3):
    query_elaborata = pre_elabora_query(query)
    logger.info(f"[PubMed] Query elaborata: {query_elaborata}")

    # Step 1: Cerca ID articoli su PubMed
    base_url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params_search = {
        "db": "pubmed",
        "term": query_elaborata,
        "retmax": 100,  # cerca più ID per evitare risultati errati
        "retmode": "xml",
        "sort": "relevance"
    }

    try:
        response_search = requests.get(base_url_search, params=params_search, timeout=10)
        response_search.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"[PubMed] Errore nella ricerca: {e}")
        return []

    root_search = ET.fromstring(response_search.content)
    ids = [id_elem.text for id_elem in root_search.findall(".//Id")]

    if not ids:
        logger.info("[PubMed] Nessun articolo trovato.")
        return []

    logger.info(f"[PubMed] Trovati {len(ids)} ID da esaminare")

    # Step 2: Recupera i dettagli degli articoli
    base_url_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    results = []
    batch_size = 20
    for i in range(0, len(ids), batch_size):
        id_batch = ids[i:i+batch_size]
        params_fetch = {
            "db": "pubmed",
            "id": ",".join(id_batch),
            "retmode": "xml"
        }

        try:
            response_fetch = requests.get(base_url_fetch, params=params_fetch, timeout=10)
            response_fetch.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"[PubMed] Errore nel recupero dettagli: {e}")
            continue

        root_fetch = ET.fromstring(response_fetch.content)

        for article in root_fetch.findall(".//PubmedArticle"):
            title_elem = article.find(".//ArticleTitle")
            abstract_elem = article.find(".//Abstract/AbstractText")

            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
            abstract = abstract_elem.text.strip() if abstract_elem is not None and abstract_elem.text else ""

            # Aggiungi solo articoli con abstract significativo
            if abstract and len(abstract) > 60:
                results.append({
                    "id": title[:50] if title else "No title",
                    "title": title if title else "No title",
                    "text": f"{title}\n\n{abstract}"
                })

            if len(results) >= max_results:
                break

        if len(results) >= max_results:
            break

    logger.info(f"[PubMed] Restituiti {len(results)} articoli con abstract validi")
    return results
