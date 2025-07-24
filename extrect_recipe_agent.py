import os
import csv
import json
import re
from glob import glob
from unstructured.partition.pdf import partition_pdf
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("⚠️ Variabile OPENAI_API_KEY mancante nel .env")

client = OpenAI(api_key=api_key)

def extract_text_from_pdf(pdf_file):
    """
    Estrae TUTTO il testo grezzo da un PDF.
    """
    elements = partition_pdf(
        filename=pdf_file,
        strategy="fast",
        infer_table_structure=False,
        extract_images_in_pdf=False
    )
    texts = [el.text.strip() for el in elements if el.text.strip()]
    return "\n\n".join(texts)

def extract_json_from_response(text):
    """
    Estrae la parte JSON eventualmente incapsulata tra ```json ... ```
    """
    pattern = r"```json(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
        return json_str
    else:
        return text.strip()

def call_gpt_extract_recipes(text):
    """
    Chiede a GPT di identificare le ricette e restituirle in JSON.
    """
    prompt = f"""
Il testo seguente proviene da un menu in PDF. 
Analizza attentamente e restituisci una lista JSON di ricette.
Ogni ricetta DEVE avere questi campi:
- nome (stringa): nome completo del piatto
- ingredienti (stringa): elenco degli ingredienti principali

Restituisci SOLO il JSON.

Testo:
\"\"\"
{text}
\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Sei un assistente culinario che estrae dati strutturati da menu PDF."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    content = response.choices[0].message.content.strip()

    clean_json = extract_json_from_response(content)

    try:
        recipes = json.loads(clean_json)
        if not isinstance(recipes, list):
            raise ValueError("Il JSON restituito non è una lista.")
    except Exception as e:
        print("⚠️ Errore nel parsing JSON:", e)
        print("Risposta grezza GPT:\n", content)
        return []

    return recipes

def main():
    pdf_folder = "Hackapizza Dataset/Menu/"
    output_csv = "ricette_estratte_agentico.csv"

    pdf_files = glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("❌ Nessun PDF trovato nella cartella 'Menu'.")

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["ristorante", "nome_ricetta", "ingredienti"])

        for pdf_file in pdf_files:
            ristorante = os.path.splitext(os.path.basename(pdf_file))[0]
            print(f"📄 Elaboro '{ristorante}'...")

            text = extract_text_from_pdf(pdf_file)
            recipes = call_gpt_extract_recipes(text)

            if not recipes:
                print(f"⚠️ Nessuna ricetta trovata in '{ristorante}'.")
                continue

            for r in recipes:
                nome = r.get("nome", "").strip()
                ingredienti = r.get("ingredienti", "").strip()
                writer.writerow([ristorante, nome, ingredienti])

            print(f"✅ Estratte {len(recipes)} ricette da '{ristorante}'.")

    print(f"✅ File creato: {output_csv}")

if __name__ == "__main__":
    main()