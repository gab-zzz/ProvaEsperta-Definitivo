# server.py
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import cerca_documenti
from reasoning import genera_risposta, initialize_model
from create_faiss_index import create_faiss_index
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from mistral_inference import genera_risposta_mistral
import uvicorn
import logging
import traceback
import os
import re
import torch
import sys

sys.stdout.reconfigure(line_buffering=True)


# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI(title="Medical AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modello semantico multilingua
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Frasi di esempio per classificazione - ESPANSIONE SIGNIFICATIVA
medical_examples = [
    # Sintomi generali
    "What are the symptoms of a heart attack?",
    "How long does a migraine usually last?",
    "What are the causes of cancer?",
    "What should I do if I have a fever?",
    "What are the side effects of antibiotics?",
    
    # Condizioni specifiche
    "What are the symptoms of an aneurysm?",
    "How do I recognize diabetes?",
    "What causes hypertension?",
    "What are the signs of stroke?",
    "How is pneumonia diagnosed?",
    "What are the types of leukemia?",
    "What are the symptoms of a concussion?",
    
    # Trattamenti
    "How is chemotherapy administered?",
    "What antibiotics treat bronchitis?",
    "How long should I take these medications?",
    "What is the treatment for arthritis?",
    "Is surgery necessary for herniated disc?",
    
    # Anatomia
    "Where is the appendix located?",
    "What does the pancreas do?",
    "How does the heart pump blood?",
    "What is the function of the thyroid?",
    
    # Malattie ed esami
    "What does high cholesterol mean?",
    "How to interpret blood test results?",
    "What is a normal blood pressure?",
    "What causes elevated liver enzymes?",
    
    # Salute mentale
    "How to recognize depression?",
    "What are ADHD symptoms?",
    "How is anxiety treated?",
    "What is bipolar disorder?",
    
    # Emergenze
    "What to do in case of seizure?",
    "How to perform CPR?",
    "What are the signs of anaphylaxis?",
    "How to stop bleeding?",
    
    # Altri termini medici
    "What is an aneurysm?",
    "What causes arrhythmia?",
    "What is a benign tumor?",
    "What is the difference between CT and MRI?"
]

# Frasi di esempio non mediche per migliorare la classificazione
non_medical_examples = [
    "What is a scalene triangle?",
    "How do I cook pasta?",
    "What is the capital of France?",
    "How do I change a tire?",
    "Who won the World Cup in 2018?",
    "What are the best tourist spots in Italy?",
    "How to solve this math equation?",
    "What is photosynthesis?",
    "How much does a Tesla cost?",
    "Who wrote Romeo and Juliet?",
    "Who invented Coca Cola?",
    "When was the Eiffel Tower built?",
    "How to make a chocolate cake?",
    "What is the height of Mount Everest?",
    "How does a car engine work?",
    "What is the meaning of this emoji?",
    "How to play chess?",
    "What are the rules of football?",
    "Who painted the Mona Lisa?",
    "What is the boiling point of water?",
    "How to grow tomatoes?",
    "What is the diameter of the Earth?",
    "How to tie a tie?",
    "What is the square root of 144?",
    "Who is the current president of France?",
    "What is the function of a capacitor?",  # Termine tecnico non medico
    "How does a refrigerator work?",
    "What is the difference between a crocodile and an alligator?",
    "What happened in World War II?",
    "How to solve a Rubik's cube?"
]

# Combina gli esempi con etichette
labeled_examples = [(esempio, 1) for esempio in medical_examples] + [(esempio, 0) for esempio in non_medical_examples]

# Crea gli embedding per tutti gli esempi
all_examples = [example for example, _ in labeled_examples]
all_embeddings = embedder.encode(all_examples, convert_to_tensor=True)

# Salva le etichette per recuperarle facilmente
example_labels = [label for _, label in labeled_examples]

# Stato conversazione per utente
user_context = {}

class DomandaRequest(BaseModel):
    domanda: str
    num_results: int = 5

class RispostaResponse(BaseModel):
    risposta: str
    documenti_utilizzati: list

# Traduzione
def traduci_testo(text, src='auto', target='en'):
    try:
        return GoogleTranslator(source=src, target=target).translate(text)
    except Exception as e:
        logger.warning(f"Errore nella traduzione: {e}")
        return text

# Pulizia risposta
import re

import re

def pulisci_risposta(text, domanda_utente=None):
    """
    Pulisce la risposta generata dal modello rimuovendo markup, prefissi,
    e la domanda ripetuta. Mantiene solo la parte utile della risposta.
    
    Args:
        text (str): Testo grezzo generato.
        domanda_utente (str, optional): La domanda originale, per rimuovere duplicazioni.

    Returns:
        str: Testo pulito.
    """
    # Rimuovi token speciali e prefissi comuni
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("Assistant:", "").strip()

    # Se c'è "Risposta:", taglia tutto prima
    match = re.search(r"(?i)\bRisposta:\s*", text)
    if match:
        text = text[match.end():].strip()

    # Se la risposta inizia con la domanda ripetuta, rimuovila
    if domanda_utente:
        domanda_norm = domanda_utente.strip().lower().rstrip("?!.")
        risposta_norm = text.strip().lower()
        if risposta_norm.startswith(domanda_norm):
            text = text[len(domanda_utente):].lstrip(" \n:.").strip()

    return text



# Correzione grammaticale in italiano
def correggi_risposta_italiana(testo):
    try:
        if re.search(r'[àèéìòù]', testo.lower()):
            return testo
        return GoogleTranslator(source='auto', target='it').translate(testo)
    except Exception as e:
        logger.warning(f"Errore durante la correzione/traduzione in italiano: {e}")
        return testo

# Contesto utente
def get_user_context(user_id):
    if user_id not in user_context:
        user_context[user_id] = []
    return user_context[user_id]

def update_user_context(user_id, domanda, risposta):
    contesto = get_user_context(user_id)
    contesto.append({"domanda": domanda, "risposta": risposta})

# Classificazione migliorata - considera anche esempi non medici e usa voto di maggioranza
def classifica_domanda_con_storia(translated_question, history, soglia=0.65, k=5):
    """
    Classifica una domanda come medica o non medica usando un approccio semantico.
    
    Args:
        translated_question: La domanda tradotta in inglese
        history: Storico delle conversazioni
        soglia: Soglia di similarità per la classificazione diretta
        k: Numero di esempi simili da considerare per il voto di maggioranza
        
    Returns:
        Boolean: True se la domanda è medica, False altrimenti
    """
    # Calcola embedding della domanda attuale per classificazione generale
    question_embedding = embedder.encode([translated_question], convert_to_tensor=True)
    
    # Calcola similarità con tutti gli esempi (medici e non)
    cos_scores = util.pytorch_cos_sim(question_embedding, all_embeddings)[0]
    
    # Trova i k esempi più simili
    top_k_scores, top_k_indices = torch.topk(cos_scores, k=min(k, len(all_examples)))
    
    # Estrai le etichette dei k esempi più simili
    top_k_labels = [example_labels[idx] for idx in top_k_indices]
    
    # Calcola punteggio medico come media ponderata delle etichette
    weighted_sum = sum(score.item() * label for score, label in zip(top_k_scores, top_k_labels))
    total_weight = sum(score.item() for score in top_k_scores)
    medical_score = weighted_sum / total_weight if total_weight > 0 else 0
    
    # Ottieni l'esempio più simile per logging
    max_idx = torch.argmax(cos_scores).item()
    max_example = all_examples[max_idx]
    max_label = example_labels[max_idx]
    max_similarity = cos_scores[max_idx].item()
    
    # Logging dettagliato
    logger.info(f"Esempio più simile: '{max_example}' (etichetta: {'medica' if max_label == 1 else 'non medica'}, similarità: {max_similarity:.3f})")
    logger.info(f"Classificazione diretta: {'medical' if medical_score >= 0.5 else 'non-medical'} (score: {medical_score:.3f})")
    
    # Verifica semantica tra domanda attuale e precedente se esiste una storia
    if history:
        last = history[-1]
        prev_question = last["domanda"]
        prev_embedding = embedder.encode([prev_question], convert_to_tensor=True)
        
        # Calcola similarità semantica tra domanda corrente e precedente
        semantic_similarity = util.pytorch_cos_sim(question_embedding, prev_embedding)[0][0].item()
        logger.info(f"Similarità semantica con domanda precedente: {semantic_similarity:.3f}")
        
        # Verifica se la domanda attuale è semanticamente più vicina alla categoria medica o non medica
        # Utilizziamo gli esempi medici e non medici già classificati per creare centroidi semantici
        medical_examples_indices = [i for i, label in enumerate(example_labels) if label == 1]
        non_medical_examples_indices = [i for i, label in enumerate(example_labels) if label == 0]
        
        # Calcoliamo la similarità media con gli esempi medici
        medical_similarities = [cos_scores[i].item() for i in medical_examples_indices]
        avg_medical_similarity = sum(medical_similarities) / len(medical_similarities) if medical_similarities else 0
        
        # Calcoliamo la similarità media con gli esempi non medici
        non_medical_similarities = [cos_scores[i].item() for i in non_medical_examples_indices]
        avg_non_medical_similarity = sum(non_medical_similarities) / len(non_medical_similarities) if non_medical_similarities else 0
        
        logger.info(f"Similarità media con esempi medici: {avg_medical_similarity:.3f}")
        logger.info(f"Similarità media con esempi non medici: {avg_non_medical_similarity:.3f}")
        
        # Se la domanda è chiaramente più simile agli esempi non medici
        # e c'è una differenza significativa, la consideriamo non medica
        if avg_non_medical_similarity > avg_medical_similarity + 0.1:
            logger.info("Domanda semanticamente più vicina agli esempi non medici")
            is_new_topic = True
            # Se siamo sicuri che è un argomento non medico, restituisci False
            if avg_non_medical_similarity > 0.5 and avg_medical_similarity < 0.4:
                logger.info("Domanda chiaramente non medica in base al confronto semantico")
                return False
        else:
            is_new_topic = False
        
        # Se la domanda corrente ha bassa similarità semantica con la precedente
        # ma non siamo sicuri della classificazione, procedi con altri controlli
        if semantic_similarity < 0.3:
            logger.info("Domanda semanticamente diversa dalla precedente - potrebbe essere un nuovo argomento")
            
            # Se la similarità con la domanda precedente è molto bassa
            # e la similarità con esempi non medici è maggiore, considerala un cambio argomento
            if semantic_similarity < 0.2 and avg_non_medical_similarity > avg_medical_similarity:
                logger.info("Probabile cambio di argomento verso topic non medico")
                return False
    
    # Gestione di domande molto brevi e ambigue (follow-up)
    if len(translated_question.split()) <= 4 and history:
        is_semantically_similar = semantic_similarity >= 0.3 if 'semantic_similarity' in locals() else False
        
        if is_semantically_similar:
            logger.info("Domanda molto breve rilevata e semanticamente simile - potrebbe essere un follow-up")
            
            # Frasi comuni di follow-up in inglese
            followup_patterns = [
                "what about", "how about", "and now", "what if", "what is", "what does", 
                "how does", "why does", "when did", "where is", "who is", "what function", 
                "what purpose", "what use", "how many", "how much", "what are",
                "what was", "why is", "can you", "could you", "please explain", "tell me more",
                "more info", "more information", "give me", "what happens", "how can", "why would"
            ]
            
            # Verifica se ci sono pattern di follow-up
            is_followup = any(pattern in translated_question.lower() for pattern in followup_patterns)
            
            # Se sembra un follow-up e la domanda precedente era chiaramente in una categoria
            if is_followup or len(translated_question.split()) <= 3:
                # Calcola punteggio medico per la domanda precedente
                prev_embedding = embedder.encode([prev_question], convert_to_tensor=True)
                prev_cos_scores = util.pytorch_cos_sim(prev_embedding, all_embeddings)[0]
                prev_top_k_scores, prev_top_k_indices = torch.topk(prev_cos_scores, k=min(k, len(all_examples)))
                prev_top_k_labels = [example_labels[idx] for idx in prev_top_k_indices]
                
                prev_weighted_sum = sum(score.item() * label for score, label in zip(prev_top_k_scores, prev_top_k_labels))
                prev_total_weight = sum(score.item() for score in prev_top_k_scores)
                prev_medical_score = prev_weighted_sum / prev_total_weight if prev_total_weight > 0 else 0
                
                logger.info(f"Rilevato probabile follow-up - la domanda precedente era {'MEDICA' if prev_medical_score >= 0.6 else 'NON MEDICA'} con score {prev_medical_score:.3f}")
                
                # Per domande molto brevi come "perché?", "come?", "e poi?", ecc.
                if len(translated_question.split()) <= 3:
                    logger.info("Domanda estremamente corta - probabilmente è un follow-up diretto")
                    return prev_medical_score >= 0.5
                    
                # Per domande che sembrano follow-up ma sono un po' più lunghe
                if is_followup and prev_medical_score >= 0.6:
                    logger.info("Domanda precedente era chiaramente medica e questa è un follow-up - mantengo classificazione MEDICA")
                    return True
                elif is_followup and prev_medical_score <= 0.4:
                    logger.info("Domanda precedente era chiaramente NON medica e questa è un follow-up - mantengo classificazione NON MEDICA")
                    return False
        else:
            logger.info("Domanda breve ma non semanticamente simile alla precedente - probabile nuovo argomento")
    
    # Se la similarità massima è molto bassa, potrebbe essere una domanda ambigua
    if max_similarity < 0.3:
        logger.info("Bassa similarità con tutti gli esempi - classificazione potenzialmente incerta")
    
    # Ottiene una risposta binaria basata sul punteggio calcolato all'inizio
    is_medical = medical_score >= 0.5
    
    # Considera il contesto solo se la risposta è ambigua (vicino alla soglia)
    # e se c'è alta similarità semantica con la domanda precedente
    if 0.4 <= medical_score <= 0.6 and history and 'semantic_similarity' in locals() and semantic_similarity >= 0.3:
        logger.info("Controllo contesto per classificazione ambigua...")
        
        # Se la domanda precedente era chiaramente medica, aumenta la probabilità
        if prev_medical_score >= 0.7:
            # Combina le due domande
            combined = prev_question + " " + translated_question
            combined_embedding = embedder.encode([combined], convert_to_tensor=True)
            
            # Calcola similarità della combinazione con tutti gli esempi
            combined_cos_scores = util.pytorch_cos_sim(combined_embedding, all_embeddings)[0]
            
            # Trova i k esempi più simili per la combinazione
            combined_top_k_scores, combined_top_k_indices = torch.topk(combined_cos_scores, k=min(k, len(all_examples)))
            combined_top_k_labels = [example_labels[idx] for idx in combined_top_k_indices]
            
            # Calcola punteggio medico per la combinazione
            combined_weighted_sum = sum(score.item() * label for score, label in zip(combined_top_k_scores, combined_top_k_labels))
            combined_total_weight = sum(score.item() for score in combined_top_k_scores)
            combined_medical_score = combined_weighted_sum / combined_total_weight if combined_total_weight > 0 else 0
            
            logger.info(f"Classificazione con contesto: {'medical' if combined_medical_score >= 0.5 else 'non-medical'} (score: {combined_medical_score:.3f})")
            return combined_medical_score >= 0.5
        else:
            logger.info("Contesto precedente non chiaramente medico: mantengo classificazione diretta.")
    
    return is_medical

# Esecuzione asincrona
async def esegui_in_background(funzione, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, funzione, *args)

@app.post("/generate", response_model=RispostaResponse)
async def generate(request: DomandaRequest):
    try:
        domanda_originale = request.domanda.strip()
        logger.info(f"Domanda ricevuta: {domanda_originale}")
        user_id = "user_1"
        contesto_utente = get_user_context(user_id)

        domanda_tradotta = await esegui_in_background(traduci_testo, domanda_originale, 'it', 'en')
        is_medica = classifica_domanda_con_storia(domanda_tradotta, contesto_utente)
        
        # Log la decisione finale
        logger.info(f"Decisione finale: La domanda '{domanda_originale}' è {'MEDICA' if is_medica else 'NON MEDICA'}")

        contesto_storico = ""
        if contesto_utente:
            ultimi = contesto_utente[-3:]
            contesto_storico = "\n".join([f"Domanda: {x['domanda']}\nRisposta: {x['risposta']}" for x in ultimi])

        if is_medica:
            logger.info("Avvio ricerca FAISS/PubMed...")
            documenti, da_faiss, aggiornato = await esegui_in_background(cerca_documenti, domanda_tradotta, request.num_results)

            if not documenti and not os.path.exists("faiss_index.index"):
                logger.info("Indice FAISS mancante. Lo creo...")
                await esegui_in_background(create_faiss_index)
                documenti, da_faiss, aggiornato = await esegui_in_background(cerca_documenti, domanda_tradotta, request.num_results)

            if documenti:
                logger.info(f"Trovati {len(documenti)} documenti - Fonte: {'FAISS' if da_faiss else 'PubMed'}")
                if aggiornato:
                    logger.info("Indice FAISS aggiornato con nuovi dati da PubMed.")
                prompt = f"{contesto_storico}\nDomanda: {domanda_originale}\nRisposta:"
                risposta = await esegui_in_background(genera_risposta, prompt, documenti)
                risposta = pulisci_risposta(risposta)
                if not risposta or risposta == "La risposta è stata:":
                    risposta = "Mi scuso, non sono riuscito a trovare una risposta adeguata. Ti consiglio di consultare un esperto."
                update_user_context(user_id, domanda_originale, risposta)
                return {
                    "risposta": risposta,
                    "documenti_utilizzati": [{"id": d.get("id", ""), "title": d.get("title", "")} for d in documenti]
                }
            else:
                logger.warning("Nessun documento rilevante trovato.")
                risposta = "Non ho trovato informazioni mediche rilevanti. Ti consiglio di consultare un medico."
                update_user_context(user_id, domanda_originale, risposta)
                return {"risposta": risposta, "documenti_utilizzati": []}
        else:
            logger.info("Uso modello Mistral per domanda non medica.")
            prompt = f"{contesto_storico}\nDomanda: {domanda_originale}\nRisposta:"
            risposta_raw = await esegui_in_background(genera_risposta_mistral, prompt)
            risposta_tradotta = await esegui_in_background(correggi_risposta_italiana, pulisci_risposta(risposta_raw))
            if not risposta_tradotta or risposta_tradotta == "La risposta è stata:":
                risposta_tradotta = "Mi dispiace, non sono riuscito a generare una risposta adeguata."
            update_user_context(user_id, domanda_originale, risposta_tradotta)
            return {
                "risposta": risposta_tradotta,
                "documenti_utilizzati": []
            }

    except Exception as e:
        logger.error(f"Errore nella generazione: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Errore durante l'elaborazione della domanda.")

@app.post("/search")
async def search_only(request: DomandaRequest):
    try:
        domanda_tradotta = await esegui_in_background(traduci_testo, request.domanda, 'it', 'en')
        documenti, _, _ = await esegui_in_background(cerca_documenti, domanda_tradotta, request.num_results)
        return {"documenti": documenti}
    except Exception as e:
        logger.error(f"Errore nella ricerca: {e}")
        raise HTTPException(status_code=500, detail="Errore durante la ricerca dei documenti.")

@app.get("/")
async def root():
    return {
        "name": "Medical Retrieval & Reasoning API",
        "status": "online",
        "endpoints": [
            {"path": "/generate", "method": "POST", "description": "Genera una risposta"},
            {"path": "/search", "method": "POST", "description": "Cerca documenti"}
        ]
    }

if __name__ == "__main__":
    logger.info("Avvio del server FastAPI...")
    uvicorn.run("server:app", host="127.0.0.1", port=5000)