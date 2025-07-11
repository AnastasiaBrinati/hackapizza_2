from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def find_similar(query_embedding, item_embeddings, top_k=5):
    sims = cosine_similarity([query_embedding], item_embeddings)[0]
    indices = np.argsort(sims)[::-1][:top_k]
    return indices, sims[indices]
