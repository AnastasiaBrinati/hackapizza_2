from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wrapper per il modello di embedding.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list, batch_size: int = 32) -> list:
        """
        Calcola embedding per una lista di testi.
        """
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)