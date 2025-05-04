#reasoning.py
import os
import glob
import logging
import multiprocessing
import time
import traceback
from gpt4all import GPT4All

# Configurazione del logging per tracciare il flusso del programma
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variabili globali per il modello
model = None
model_path_global = None  # Variabile globale per memorizzare il percorso del modello

def get_available_models():
    """
    Funzione per ottenere tutti i modelli GPT4All disponibili nella cartella dell'utente.
    Cerca in tre posizioni standard e restituisce i percorsi ai file di modello.
    """
    user_home = os.path.expanduser("~")  # Percorso home dell'utente
    model_dirs = [
        os.path.join(user_home, "AppData", "Local", "nomic.ai", "GPT4All"),
        os.path.join(user_home, ".cache", "gpt4all"),
        os.path.join("models")  # Percorso per modelli locali
    ]

    available_models = []
    for model_dir in model_dirs:
        if os.path.exists(model_dir):  # Se la cartella esiste
            for ext in [".gguf", ".bin", ".ggml"]:  # Estensioni dei modelli supportati
                available_models.extend(glob.glob(os.path.join(model_dir, f"*{ext}")))
    return available_models

def initialize_model(model_name=None):
    """
    Inizializza il modello GPT4All selezionato. Se il modello è già caricato, restituisce quello esistente.
    Se non viene trovato nessun modello valido, solleva un'eccezione.
    """
    global model, model_path_global
    if model is not None:
        return model

    models = get_available_models()
    if not models:
        raise Exception("Nessun modello GPT4All trovato localmente.")

    # Lista di modelli preferiti, vengono cercati prima
    preferred_models = ["qwen2.5", "mistral", "orca", "gpt4all-j", "replit"]
    excluded = ["falcon"]

    candidates = []
    for name in preferred_models:
        for m in models:
            if name in os.path.basename(m).lower() and not any(x in m.lower() for x in excluded):
                candidates.append(m)
    if not candidates:
        candidates = [m for m in models if not any(x in m.lower() for x in excluded)]

    for model_path in candidates:
        try:
            logger.info(f"Caricamento modello: {model_path}")
            model = GPT4All(model_path, allow_download=False)  # Carica il modello senza download
            model_path_global = model_path  # Salva il percorso del modello
            return model
        except Exception as e:
            logger.warning(f"Errore nel caricamento modello {model_path}: {e}")

    raise Exception("Nessun modello valido disponibile.")

def _reasoner_worker(domanda, contesti, storia, model_path, queue):
    """
    Funzione interna che esegue la generazione della risposta in un processo separato.
    Si occupa di caricare il modello e rispondere alla domanda considerando contesti e storia.
    """
    try:
        from gpt4all import GPT4All

        logger.info(f"Processo worker: Caricamento modello da {model_path}")
        local_model = GPT4All(model_path, allow_download=False)  # Carica il modello nel worker
        blocchi = []  # Lista per contenere i blocchi di contesto

        # Gestione del contesto (lista o stringa)
        if isinstance(contesti, list):
            for doc in contesti[:5]:  # Limita i documenti al massimo di 5
                if isinstance(doc, dict):
                    titolo = doc.get("title", "").strip()
                    testo = doc.get("text", "").strip()
                    if titolo and testo:
                        blocchi.append(f"Documento: {titolo}\nContenuto: {testo}")
                    elif testo:
                        blocchi.append(f"Contenuto: {testo}")
                elif isinstance(doc, str):
                    blocchi.append(f"Contenuto: {doc}")
        elif isinstance(contesti, str):  # Caso in cui contesti sia una singola stringa
            blocchi.append(f"Contenuto: {contesti}")
        else:
            blocchi.append(str(contesti))
            
        contesto_testo = "\n\n".join(blocchi)  # Combina i blocchi separati da due newline
    
        storia_testo = ""
        if storia:
            parti = []
            for turn in storia:
                dom_prec = turn.get("domanda", "").strip()
                risp_prec = turn.get("risposta", "").strip()
                if dom_prec and risp_prec:
                    parti.append(f"Domanda: {dom_prec}\nRisposta: {risp_prec}")
            storia_testo = "\n".join(parti)

        # Creazione del prompt per il modello
        prompt = f"""<|im_start|>system
Sei un assistente medico. Rispondi sempre in italiano, in modo chiaro, semplice e comprensibile da un paziente.
IMPORTANTE: Utilizza le informazioni fornite nel contesto clinico per rispondere alla domanda.
Se il contesto contiene la risposta ed è completa, basati esclusivamente su quelle informazioni.
<|im_start|>user
Domanda dell'utente:
{domanda.strip()}

Contesto clinico (utilizza queste informazioni per rispondere):
{contesto_testo.strip()}

{f"Storia della conversazione:\n{storia_testo.strip()}" if storia_testo else ""}
<|im_start|>"""

        logger.info("Prompt inviato al Reasoner:\n" + prompt)

        # Generazione della risposta dal modello
        risposta = local_model.generate(
            prompt,
            max_tokens=250,
            temp=0.7,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.2,
            streaming=False
        )

        logger.info("Risposta grezza dal Reasoner:\n" + risposta)

        # Pulizia e validazione della risposta
        risposta_pulita = risposta.strip()
        token_da_rimuovere = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        for token in token_da_rimuovere:
            risposta_pulita = risposta_pulita.replace(token, "")
            
        # Rilassare i criteri di accettazione delle risposte
        if len(risposta_pulita) > 30:  # Ridotto da 50 a 30
            trovato = True
        else:
            pattern_possibili = [
            "### Risposta dettagliata in italiano",
            "Risposta dettagliata in italiano",
            "Risposta in italiano:",
            "Risposta:"
            ]
            trovato = False
            for pattern in pattern_possibili:
                if pattern in risposta_pulita:
                    parti = risposta_pulita.split(pattern, 1)
                    if len(parti) > 1:
                        risposta_pulita = parti[1].strip()
                        trovato = True
                        break

        # Se la risposta non è sufficientemente dettagliata, tenta una seconda generazione
        if not trovato:
            prompt_markers = [
                "### Istruzioni:",
                "### Domanda dell'utente:",
                "### Contesto clinico"
            ]
            contains_prompt = any(marker in risposta_pulita for marker in prompt_markers)

            if not contains_prompt and len(risposta_pulita) > 20:
                trovato = True
            else:
                if ":" in risposta_pulita:
                    last_part = risposta_pulita.split(":")[-1].strip()
                    if len(last_part) > 20:
                        risposta_pulita = last_part
                        trovato = True

        if len(risposta_pulita) > 50:
            trovato = True

        risposta_pulita = risposta_pulita.strip()

        # Se la risposta è troppo breve o malformata, tenta una risposta alternativa
        if len(risposta_pulita) < 20 or not trovato:
            logger.warning("Risposta troppo breve o malformata. Rigenero la risposta basandosi sulle conoscenze generali del modello.")

            prompt_generale = f"""### Istruzioni:
Sei un assistente medico. Rispondi sempre in italiano, in modo chiaro, semplice e comprensibile da un paziente.
Rispondi basandoti esclusivamente sulle tue conoscenze mediche generali, senza fare riferimento a documenti esterni.

### Domanda dell'utente:
{domanda.strip()}

### Risposta in italiano, semplice e utile per un paziente:"""

            logger.info("Prompt alternativo inviato al Reasoner:\n" + prompt_generale)

            # Genera una risposta alternativa usando conoscenze generali
            risposta_generale = local_model.generate(
                prompt_generale,
                max_tokens=500,
                temp=0.7,
                top_k=40,
                top_p=0.9,
                repeat_penalty=1.2,
                streaming=False
            )

            logger.info("Risposta alternativa grezza dal Reasoner:\n" + risposta_generale)

            # Pulizia della risposta alternativa
            risposta_finale = risposta_generale.strip()
            for token in token_da_rimuovere:
                risposta_finale = risposta_finale.replace(token, "")

            risposta_finale = risposta_finale.strip()
            queue.put(risposta_finale if len(risposta_finale) > 20 else "Mi dispiace, non sono riuscito a trovare una risposta adeguata.")
        else:
            queue.put(risposta_pulita)

    except Exception as e:
        traceback_str = traceback.format_exc()
        queue.put(f"Errore interno nel Reasoner:\n{traceback_str}")


def genera_risposta(domanda, contesti, timeout=1500):
    """
    Funzione che avvia la generazione della risposta in un processo separato, con timeout.
    """
    return genera_risposta_con_storia(domanda, contesti, storia=[], timeout=timeout)

def genera_risposta_con_storia(domanda, contesti, storia=[], timeout=1500):
    """
    Funzione che avvia la generazione della risposta in un processo separato considerando anche la storia delle domande.
    """
    try:
        # Log per verificare il tipo di contesti e la loro struttura
        logger.info(f"Tipo di contesti: {type(contesti)}")
        if isinstance(contesti, list):
            logger.info(f"Numero di documenti nel contesto: {len(contesti)}")
            for i, doc in enumerate(contesti[:2]):  # Log dei primi 2 documenti come esempio
                logger.info(f"Documento {i} - tipo: {type(doc)}")
                if isinstance(doc, dict):
                    logger.info(f"Documento {i} - chiavi: {doc.keys()}")
        
        global model, model_path_global
        # Inizializzazione del modello (avviene solo una volta)
        if model is None:
            try:
                initialize_model()
            except Exception as e:
                logger.error(f"Errore nell'inizializzazione del modello: {e}")
                return f"Errore nell'inizializzazione del modello: {e}"
        
        # Verifica che il percorso del modello sia stato impostato
        if model_path_global is None:
            logger.error("Percorso del modello non disponibile")
            return "⚠️ Errore: percorso del modello non disponibile"
            
        logger.info(f"Processo principale: Usando modello da {model_path_global}")
                
        # Coda per la comunicazione tra processi
        queue = multiprocessing.Queue()
        
        # Creazione e avvio del processo separato per la generazione della risposta
        processo = multiprocessing.Process(
            target=_reasoner_worker, 
            args=(domanda, contesti, storia, model_path_global, queue)
        )
        processo.start()
        processo.join(timeout)

        # Gestione del timeout
        if processo.is_alive():
            logger.warning(f"Timeout raggiunto dopo {timeout} secondi")
            processo.terminate()
            processo.join()
            return "⚠️ Timeout: la generazione della risposta ha superato il tempo massimo consentito."

        # Recupero della risposta dalla coda
        if not queue.empty():
            risposta = queue.get()
            return risposta

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Errore nella generazione della risposta: {traceback_str}")
        return f"Errore nella generazione della risposta: {traceback_str}"
