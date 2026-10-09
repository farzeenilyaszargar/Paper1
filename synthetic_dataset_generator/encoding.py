"""Shared character IDs for familiar and held-out names; no UNK fallback."""

from .generator import LABELS, VOCABULARY

CHARACTER_IDS = {char: index for index, char in enumerate(VOCABULARY)}
PAD_ID, BOS_ID, EOS_ID = range(len(VOCABULARY), len(VOCABULARY) + 3)
TARGET_IDS = {label: index for index, label in enumerate(LABELS)}


def encode(text: str, *, add_boundaries: bool = True) -> list[int]:
    try:
        ids = [CHARACTER_IDS[char] for char in text]
    except KeyError as exc:
        raise ValueError(f"Character outside shared vocabulary: {exc.args[0]!r}") from exc
    return [BOS_ID, *ids, EOS_ID] if add_boundaries else ids


def decode(ids: list[int]) -> str:
    result = []
    for index in ids:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= EOS_ID:
            raise ValueError(f"Invalid character ID: {index!r}")
        if index < len(VOCABULARY):
            result.append(VOCABULARY[index])
    return "".join(result)
