from sklearn.metrics.pairwise import cosine_similarity


def retrieve(query_emb, catalog_embs, top_k=5):
    """
    Restituisce gli indici dei piatti più simili al vettore di query descritto da query_emb.
    Se catalog_embs è vuoto, restituisce una lista vuota.
    """
    # Gestione caso in cui non ci siano embeddings da confrontare
    if catalog_embs is None or len(catalog_embs) == 0:
        return []

    # Calcola similarità coseno
    sims = cosine_similarity([query_emb], catalog_embs)[0]
    # Ordina gli indici in base alla similarità decrescente
    idxs = sims.argsort()[::-1][:top_k]
    return idxs.tolist()