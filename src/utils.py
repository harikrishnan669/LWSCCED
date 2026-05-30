from __future__ import annotations

import re

# Dictionary storing ASR mistakes
TYPO_CORRECTIONS: dict[str, str] = {
    "dowm": "down",
    "wolume": "volume",
    "musics": "music",
    "vehical": "vehicle",
    "prewious": "previous",
}

# Filler words mostly used while talking
FILLER_WORDS: list[str] = [
    "um",
    "uh",
    "like",
    "actually",
    "please",
    "ya",
]

# Normalize noisy ASR-style text for using in evaluate
def normalize_text(text: str) -> str:

    text = text.lower().strip()

    for typo, correction in TYPO_CORRECTIONS.items():
        text = text.replace(typo, correction)

    tokens: list[str] = text.split()

    tokens = [
        token
        for token in tokens
        if token not in FILLER_WORDS # keeps only useful words
    ]

    text = " ".join(tokens)

    text = re.sub(r"\s+", " ", text)

    return text
