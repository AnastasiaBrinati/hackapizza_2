from difflib import get_close_matches
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('hackapizza')


def fuzzy_match(name: str, mapping: dict, cutoff: float = 0.7) -> str:
    """
    Restituisce ID della chiave mapping più simile a name.
    """
    names = list(mapping.keys())
    matches = get_close_matches(name, names, n=1, cutoff=cutoff)
    if matches:
        matched = matches[0]
        nid = mapping.get(matched)
        logger.info(f"Fuzzy match '{name}' -> '{matched}' (ID {nid})")
        return nid
    logger.warning(f"No fuzzy match for '{name}'")
    return None