import re


def parse_query(raw: str) -> str:
    """
    Normalizza query: lowercase, rimozione punteggiatura.
    """
    q = raw.lower()
    q = re.sub(r"[\.,;!?]", "", q)
    return q.strip()