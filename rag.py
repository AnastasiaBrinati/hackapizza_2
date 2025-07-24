import os
import re
import csv
import json
import numpy as np
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# -------------------------------
# 1) Carica .env e la chiave
# -------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("⚠️ La variabile OPENAI_API_KEY non è stata trovata nel file .env")

client = OpenAI(api_key=api_key)

# -------------------------------
# 2) Parsing PDF e creazione embeddings
# -------------------------------
def parse_pdfs(pdf_files):
    records = []
    model = SentenceTransformer("all-mpnet-base-v2")
    idx = 0

    for pdf_file in pdf_files:
        elements = partition_pdf(
            filename=pdf_file,
            strategy="fast",
            infer_table_structure=False,
            extract_images_in_pdf=False
        )

        chunks = [el.text for el in elements if el.text.strip()]
        embeddings = model.encode(chunks, normalize_embeddings=True)

        for text, emb in zip(chunks, embeddings):
            idx += 1
            records.append({
                "id": f"ricetta_{idx:04d}",
                "nome": extract_title(text),
                "contenuto": text,
                "embedding": emb,
                "file": pdf_file
            })

    return records

def extract_title(text):
    lines = text.strip().splitlines()
    if lines:
        return lines[0][:50]
    return "Ricetta"

# -------------------------------
# 3) Estrazione keyword semplice
# -------------------------------
def extract_keywords(question):
    tokens = re.findall(r"\b\w{4,}\b", question.lower())
    return tokens

# -------------------------------
# 5) Retrieval combinato keyword + semantico
# -------------------------------
def retrieve_records(question, keywords, records, k=5):
    filtered = [
        r for r in records
        if any(kw in r["contenuto"].lower() or kw in r["file"].lower() for kw in keywords)
    ]
    if not filtered:
        filtered = records

    model = SentenceTransformer("all-mpnet-base-v2")
    query_emb = model.encode([question], normalize_embeddings=True)

    filtered_embeddings = np.stack([r["embedding"] for r in filtered])
    sub_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
    sub_index.add(filtered_embeddings)

    scores, indices = sub_index.search(query_emb, k)
    selected = [filtered[i] for i in indices[0]]

    return selected

# -------------------------------
# 6) Costruzione prompt per GPT-3.5
# -------------------------------
def build_prompt(question, retrieved):
    chunks = []
    for r in retrieved:
        snippet = r['contenuto'][:300].replace("\n", " ")
        chunks.append(f"[ID: {r['id']}] {r['nome']} - {snippet}")
    context = "\n".join(chunks)

    prompt = f"""
Seleziona SOLO le ricette che rispondono alla domanda qui sotto.
Restituisci una lista Python con SOLO gli ID delle ricette pertinenti (es. ["ricetta_0001","ricetta_0002"]).

Domanda:
{question}

Contesto:
{context}
"""
    return prompt

# -------------------------------
# 7) Chiamata a GPT-3.5-Turbo
# -------------------------------
def call_gpt35(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Sei un assistente culinario che restituisce solo liste di ID."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# -------------------------------
# 8) Pipeline completa
# -------------------------------
def process_question(question, records):
    keywords = extract_keywords(question)
    retrieved = retrieve_records(question, keywords, records, k=5)
    prompt = build_prompt(question, retrieved)
    ids = call_gpt35(prompt)
    return ids

# -------------------------------
# 9) Caricamento domande da CSV con id domanda e testo
# -------------------------------
def load_questions(csv_file):
    questions = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                questions.append((row[0], row[1]))
            elif len(row) == 1:
                questions.append((f"id_{len(questions)+1}", row[0]))
    return questions

# -------------------------------
# 10) Caricamento dish_mappings.json
# -------------------------------
def load_dish_mappings(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        dish_mapping = json.load(f)
    # NON normalizzare le chiavi!
    return {k: str(v) for k, v in dish_mapping.items()}

# -------------------------------
# 11) Esecuzione principale
# -------------------------------
if __name__ == "__main__":
    from glob import glob

    pdf_files = glob("Hackapizza Dataset/Menu/*.pdf")
    if not pdf_files:
        raise FileNotFoundError("❌ Nessun PDF trovato nella cartella Dataset/Menu/")

    print("📄 Parsing e indicizzazione PDF...")
    records = parse_pdfs(pdf_files)
    print(f"✅ Indicizzate {len(records)} ricette.")

    questions = load_questions("Hackapizza Dataset/domande.csv")
    dish_mappings = load_dish_mappings("Hackapizza Dataset/Misc/dish_mapping.json")

    ricetta_lookup = {r["id"]: r for r in records}

    output_file = "risposte.csv"
    with open(output_file, "w", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["row_id", "result"])

        for idx, (id_domanda, question) in enumerate(questions, start=1):
            print(f"🔍 Elaboro la domanda [{id_domanda}]: {question}")
            result_ids_str = process_question(question, records)

            try:
                ricetta_ids = json.loads(result_ids_str.replace("'", '"'))
            except Exception as e:
                print(f"⚠️ Errore nel parsing della risposta GPT per domanda {id_domanda}: {e}")
                ricetta_ids = []

            piatti_ids = []
            for rid in ricetta_ids:
                record = ricetta_lookup.get(rid)
                if record:
                    nome_piatto = record["nome"]
                    id_piatto = dish_mappings.get(nome_piatto)
                    if id_piatto is not None:
                        piatti_ids.append(str(id_piatto))
                    else:
                        print(f"⚠️ ID piatto NON trovato per nome esatto: '{nome_piatto}'")

            piatti_str = ",".join(piatti_ids)
            writer.writerow([idx, piatti_str])
            print(f"✅ Scritta risposta per {id_domanda}: {piatti_str}")
            print("-" * 40)

    print(f"✅ Risultati scritti su {output_file}")
