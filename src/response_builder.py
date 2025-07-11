from src.utils import fuzzy_match


def build_response(catalog: list, idxs: list, dish_mapping: dict) -> list:
    """
    Converte indici catalog in ID piatti unici, con fuzzy match.
    """
    ids = []
    for i in idxs:
        name = catalog[i]
        dish_id = dish_mapping.get(name) or fuzzy_match(name, dish_mapping)
        if dish_id:
            ids.append(str(dish_id))
    return sorted(set(ids), key=int)