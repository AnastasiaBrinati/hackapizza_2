import csv
from src.load import load_dish_mapping, load_questions, load_all_menus
from src.preprocessor import split_dishes
from src.embedder import Embedder
from src.query_parser import parse_query
from src.retriever import retrieve
from src.response_builder import build_response


def main():
    # Percorsi dataset
    mapping_path = 'HackaPizza Dataset/Misc/dish_mapping.json'
    questions_path = 'HackaPizza Dataset/domande.csv'
    menu_folder = 'HackaPizza Dataset/Menu'

    # Carica dati
    dish_mapping = load_dish_mapping(mapping_path)
    questions_df = load_questions(questions_path)
    menu_texts = load_all_menus(menu_folder)

    # Prepara catalogo
    catalog = []
    for menu in menu_texts.values():
        catalog.extend(split_dishes(menu))
    catalog = list(dict.fromkeys(catalog))  # deduplica

    # Embedding catalogo
    embedder = Embedder()
    print('Computing catalog embeddings...')
    catalog_embs = embedder.encode(catalog)
    print(f"[DEBUG] Numero di piatti in catalogo: {len(catalog_embs)}")

    # Processa query
    output = []
    for _, row in questions_df.iterrows():
        raw_q = row['question']
        q_clean = parse_query(raw_q)
        q_emb = embedder.encode([q_clean])[0]
        idxs = retrieve(q_emb, catalog_embs)
        ids = build_response(catalog, idxs, dish_mapping)
        result = ','.join(ids)
        output.append((row['row_id'], f'"{result}"'))

    # Scrivi submission
    with open('submission.csv','w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id','result'])
        for rid, res in output:
            writer.writerow([rid, res])
    print('Submission file written: submission.csv')

if __name__ == '__main__':
    main()