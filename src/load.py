import os
import json
import pandas as pd
import pdfplumber
import logging

# Silenzia i warning di pdfminer
logging.getLogger('pdfminer').setLevel(logging.ERROR)


def load_dish_mapping(path: str) -> dict:
    """
    Carica il file JSON che mappa i nomi dei piatti ai loro ID.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_questions(path: str) -> pd.DataFrame:
    """
    Carica le domande da un CSV. Se manca 'row_id', lo genera sequenzialmente.
    Restituisce DataFrame con colonne ['row_id', 'question'].
    """
    df = pd.read_csv(path, skip_blank_lines=True)
    # Normalizza nomi colonne
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
    # Identifica colonna domanda
    question_col = next((col for col in df.columns if col.lower() in ('question','domanda','text','query')), None)
    if question_col is None:
        raise KeyError(f"Nessuna colonna domanda trovata in: {list(df.columns)}")
    df.rename(columns={question_col: 'question'}, inplace=True)
    # Identifica o genera row_id
    if 'row_id' not in df.columns:
        alt = next((col for col in df.columns if col.lower() in ('row','id')), None)
        if alt:
            df.rename(columns={alt: 'row_id'}, inplace=True)
        else:
            df.insert(0, 'row_id', range(1, len(df) + 1))
    return df[['row_id','question']]


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Estrae testo da tutte le pagine di un PDF, gestendo errori.
    """
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
    except Exception as e:
        logging.getLogger('hackapizza').warning(f"Errore in {pdf_path}: {e}")
    return "\n".join(pages)


def load_all_menus(folder_path: str) -> dict:
    """
    Carica tutti i PDF di menu (estensione .pdf/.PDF) in una cartella.
    Ritorna dict {filename: text}. Restituisce anche lista di PDF vuoti.
    """
    menus = {}
    empty = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.pdf'):
            path = os.path.join(folder_path, fname)
            text = extract_text_from_pdf(path)
            if text.strip():
                menus[fname] = text
            else:
                empty.append(fname)
    if empty:
        logging.getLogger('hackapizza').warning(f"PDF senza testo: {empty}")
    return menus