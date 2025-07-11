import re


def clean_text(text: str) -> str:
    """
    Rimuove caratteri speciali e normalizza spazi.
    """
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"–|—", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_dishes(menu_text: str, min_length: int = 5, max_length: int = 100) -> list:
    """
    Divide un menu in linee che rappresentano piatti.
    Filtra righe troppo corte o troppo lunghe per escludere descrizioni estese.
    """
    cleaned = clean_text(menu_text)
    lines = [
        line.strip()
        for line in cleaned.split('\n')
        if min_length < len(line.strip()) <= max_length
    ]
    return lines