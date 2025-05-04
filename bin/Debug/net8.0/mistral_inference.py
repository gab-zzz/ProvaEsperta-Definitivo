#mistral_inference.py
import os
from llama_cpp import Llama

# Percorso del modello Mistral in formato GGUF, relativo alla directory corrente
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mistral-7b-instruct-v0.1.Q4_K_M.gguf")

llm = None  # Il modello viene inizializzato una sola volta

def initialize_mistral():
    global llm
    if llm is None:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=35,
            verbose=False
        )

def genera_risposta_mistral(domanda: str) -> str:
    return genera_risposta_mistral_con_storia(domanda, storia=[])

def genera_risposta_mistral_con_storia(domanda: str, storia: list) -> str:
    global llm
    if llm is None:
        initialize_mistral()

    storia_testo = ""
    if storia:
        blocchi = []
        for turn in storia:
            dom_prec = turn.get("domanda", "").strip()
            risp_prec = turn.get("risposta", "").strip()
            if dom_prec and risp_prec:
                blocchi.append(f"Domanda precedente: {dom_prec}\nRisposta precedente: {risp_prec}")
        storia_testo = "\n\n".join(blocchi)

    prompt = (
        f"[INST] Rispondi in italiano in modo chiaro, conciso e naturale anche se la domanda non è medica.\n\n"
        f"Domanda: {domanda}\n"
        f"{storia_testo}\n"
        f"[/INST]"
    )

    risposta = llm.create_completion(
        prompt=prompt,
        max_tokens=400,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop=["END"]
    )
    return risposta['choices'][0]['text'].strip()
