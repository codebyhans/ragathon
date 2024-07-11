import uuid
from typing import List


def generate_id(parts: List[str]) -> str:
    """Generate an ID from a list of parts.

    Args:
        parts (List[str]): List of parts to generate the ID from.

    Returns:
        str: Generated ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, " ".join(parts)))
