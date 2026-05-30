from __future__ import annotations # Treat type hint as strings

import csv
import json
import random
import re # remove extra spaces
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import nlpaug.augmenter.char as nac # Character Augmentation (to simulate typing and ASR mistakes)
import nlpaug.augmenter.word as naw # Word Augmentation (can be used in future for used for checking the word completely)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Seed is used to get same output at every run.

SEED: Final[int] = 42
random.seed(SEED)
np.random.seed(SEED)

# Stores all the core command sets

CORE_LABELS: Final[list[str]] = [
    "activate_dnd",
    "deactivate_dnd",
    "decline_call",
    "pick_up_call",
    "play_music",
    "pause_music",
    "next_song",
    "previous_song",
    "increase_volume",
    "decrease_volume",
]
# Stores the extension command sets

EXTENSION_LABELS: Final[list[str]] = [
    "increase_brightness",
    "decrease_brightness",
    "start_vehicle",
    "stop_vehicle",
]

# Out of Scope label described here

OOS_LABEL: Final[str] = "out_of_scope"

ALL_LABELS: Final[list[str]] = CORE_LABELS + EXTENSION_LABELS + [OOS_LABEL]

# The seed phrases help in the data set generations that include English words, Indian Ascent Variants (Automated Speech Recognition)

SEED_PHRASES: Final[dict[str, list[str]]] = {

    "activate_dnd": [
        "activate do not disturb",
        "turn on do not disturb",
        "enable do not disturb mode",
        "switch on do not disturb",
        "please activate do not disturb",
        "put on dnd mode",
        "set do not disturb",
        "i want do not disturb on",
        "activate dnd",
        "put do not disturb on please",
        "activate the do not disturb",
        "on the do not disturb mode",
        "please to activate dnd",
        "do not disturb on karo",
        "dnd on kar do",
    ],
    "deactivate_dnd": [
        "deactivate do not disturb",
        "turn off do not disturb",
        "disable do not disturb mode",
        "switch off do not disturb",
        "please deactivate do not disturb",
        "remove dnd mode",
        "cancel do not disturb",
        "i want do not disturb off",
        "deactivate dnd",
        "put do not disturb off please",
        "off the do not disturb mode",
        "please to deactivate dnd",
        "do not disturb off karo",
        "dnd band karo",
        "deactivate the dnd mode",
    ],

    "decline_call": [
        "decline the call",
        "reject the call",
        "ignore the call",
        "do not answer the call",
        "hang up the incoming call",
        "dismiss the call",
        "send the call to voicemail",
        "cut the call",
        "i don't want to take this call",
        "refuse the incoming call",
        "call cut karo",
        "decline karo call",
        "reject the calling",
        "don't pick the call",
        "call reject kar do",
    ],
    "pick_up_call": [
        "pick up the call",
        "answer the call",
        "accept the call",
        "take the call",
        "receive the call",
        "answer the incoming call",
        "pick the call",
        "i want to answer this call",
        "attend the call",
        "connect the call",
        "call uthao",
        "pick the calling",
        "attend the calling",
        "answer it please",
        "call attend karo",
    ],

    "play_music": [
        "play the music",
        "start playing music",
        "play some music",
        "resume the music",
        "start the music",
        "play audio",
        "turn on the music",
        "play songs",
        "i want to listen to music",
        "get the music going",
        "music chalao",
        "play the musics",
        "start music please",
        "on the music",
        "play song please",
    ],
    "pause_music": [
        "pause the music",
        "stop the music",
        "pause the song",
        "hold the music",
        "pause playback",
        "mute the music",
        "stop playing music",
        "pause audio",
        "pause for a moment",
        "please pause the music",
        "music band karo",
        "pause the musics",
        "stop the song please",
        "music ruko",
        "pause it please",
    ],
    "next_song": [
        "play the next song",
        "skip to the next track",
        "next song please",
        "go to the next track",
        "skip this song",
        "play the next track",
        "forward to the next song",
        "change the song",
        "next please",
        "skip song",
        "agla song chalao",
        "next the song",
        "next track please",
        "skip the current song",
        "move to next song",
    ],
    "previous_song": [
        "play the previous song",
        "go back to the previous track",
        "previous song please",
        "play the last song",
        "go to the previous track",
        "play the song before this",
        "rewind to the previous song",
        "back to the last track",
        "previous please",
        "play again from start",
        "pichla song chalao",
        "previous the song",
        "back track please",
        "last song please",
        "go back song",
    ],

    "increase_volume": [
        "increase the volume",
        "turn up the volume",
        "raise the volume",
        "make it louder",
        "volume up please",
        "boost the volume",
        "i need more volume",
        "louder please",
        "crank up the volume",
        "increase sound",
        "volume badhao",
        "volume up karo",
        "increase the sounds",
        "zyada loud karo",
        "thoda aur loud karo",
        "turn it up",
        "make it louder",
        "too quiet",
        "raise it",
        "bring the sound up",
        "sound is too low",
        "louder please",
        "turn it up",
        "up the sound",
        "make it loud",
        "more volume",
        "sound louder",
        "raise sound",
        "up",
        "volume up",
        "sound up",
        "make louder",
        "raise it",
        "higher volume",
        "more sound",

    ],
    "decrease_volume": [
        "decrease the volume",
        "turn down the volume",
        "lower the volume",
        "make it quieter",
        "volume down please",
        "reduce the volume",
        "i need less volume",
        "quieter please",
        "soften the volume",
        "decrease sound",
        "volume kam karo",
        "volume down karo",
        "decrease the sounds",
        "thoda kam karo",
        "soft karo please",
        "turn it down",
        "make it quieter",
        "too loud",
        "reduce it",
        "lower it",
        "sound is too high",
        "bring the sound down",
        "quiet please",
        "turn it down",
        "down the sound",
        "make it soft",
        "less volume",
        "sound quieter",
        "reduce sound",
        "down",
        "volume down",
        "sound down",
        "make quieter",
        "lower it",
        "less sound",
        "lower volume",

    ],

# The extension command set data for training

    "increase_brightness": [
        "increase the brightness",
        "turn up the brightness",
        "make it brighter",
        "raise the screen brightness",
        "brightness up please",
        "boost brightness",
        "more brightness please",
        "increase display brightness",
        "make screen brighter",
        "crank up the brightness",
        "brightness badhao",
        "screen bright karo",
        "increase the bright",
        "aur bright karo",
        "thoda zyada brightness karo",
    ],
    "decrease_brightness": [
        "decrease the brightness",
        "turn down the brightness",
        "make it dimmer",
        "lower the screen brightness",
        "brightness down please",
        "reduce brightness",
        "less brightness please",
        "decrease display brightness",
        "dim the screen",
        "soften the brightness",
        "brightness kam karo",
        "screen dim karo",
        "decrease the bright",
        "thoda kam bright karo",
        "dim kar do screen",
    ],

    "start_vehicle": [
        "start the vehicle",
        "start the car",
        "turn on the engine",
        "ignite the engine",
        "fire up the car",
        "please start the engine",
        "start my car",
        "engine on",
        "turn the car on",
        "start engine please",
        "gaadi start karo",
        "engine on karo",
        "start the gadi",
        "car start kar do",
        "vehicle start please",
    ],
    "stop_vehicle": [
        "stop the vehicle",
        "stop the car",
        "turn off the engine",
        "kill the engine",
        "please stop the engine",
        "stop my car",
        "engine off",
        "turn the car off",
        "stop engine please",
        "shut down the vehicle",
        "gaadi band karo",
        "engine off karo",
        "stop the gadi",
        "car band kar do",
        "vehicle stop please",
    ],

    # Out of Scope data set

    OOS_LABEL: [
        "what is the weather today",
        "tell me a joke",
        "how far is the nearest petrol station",
        "call mom",
        "send a text to John",
        "set an alarm for six am",
        "navigate to the airport",
        "what time is it",
        "open google maps",
        "how long until i reach home",
        "book a restaurant table",
        "remind me to buy milk",
        "read my notifications",
        "translate hello to french",
        "turn on the air conditioning",
        "set the temperature to twenty two degrees",
        "find parking nearby",
        "play a podcast",
        "read the news",
        "convert dollars to rupees",
        "who is the prime minister of india",
        "how much fuel is left",
        "check my schedule for tomorrow",
        "open spotify",
        "search for nearby hospitals",
        "turn on the headlights",
        "add meeting to calendar",
        "call office number",
        "decrease the fan speed",
        "turn on hazard lights",
        "kya time ho gaya",
        "weather batao",
        "rastha dikhao",
        "petrol station kahan hai",
        "mujhe ghar le chalo",
        "abhi kitna time lagega",
        "mummy ko call karo",
        "ac on karo",
        "traffic kaisa hai",
        "gaana sunao",
        "call john",
    ],
}


# Train and Test Data set represented as Text | label | Source
@dataclass
class Sample:
    text: str
    label: str
    source: str   # "clean" | "paraphrase" | "asr_noise" | "char_noise" | "filler" | "dropped"


# Stores summary information

@dataclass
class DatasetStats:
    total: int
    per_label: dict[str, int]
    source_counts: dict[str, int]
    train_size: int
    test_size: int


# Noise filler tokens

FILLER_TOKENS: Final[list[str]] = [
    "um", "uh", "like", "you know", "err", "hmm",
    "basically", "actually", "so", "well", "kind of",
]

# Common ASR substitution pairs (phonetically similar confusion)
ASR_SUBSTITUTIONS: Final[list[tuple[str, str]]] = [
    ("increase", "in crease"),
    ("decrease", "de crease"),
    ("activate", "act of ate"),
    ("deactivate", "de activate"),
    ("volume", "volum"),
    ("brightness", "bright ness"),
    ("previous", "previus"),
    ("music", "musics"),
    ("vehicle", "vehical"),
    ("pause", "paws"),
    ("decline", "de cline"),
    ("disturb", "dis turb"),
    ("the", ""),           # article drop — common in Indian ASR
    ("please", "plz"),
    ("engine", "engeen"),
    ("brighter", "brightter"),
    ("quieter", "quieter"),
    ("louder", "loudder"),
]

# Regional phonetic confusions specific to Indian accent ASR
INDIAN_ASR_SUBSTITUTIONS: Final[list[tuple[str, str]]] = [
    ("v", "w"),            #Example: "vehicle" → "wehicle"
    ("w", "v"),
    ("th", "d"),
    ("th", "t"),
    ("z", "j"),
    ("ph", "f"),
    ("volume", "wolume"),
    ("vehicle", "wehicle"),
    ("previous", "prewious"),
]

# Adds filler words before actual statements as people sometimes talk like "um play the music"
def insert_fillers(text: str, n: int = 1) -> str:
    tokens: list[str] = text.split()
    for _ in range(n):
        idx: int = random.randint(0, len(tokens))
        tokens.insert(idx, random.choice(FILLER_TOKENS))
    return " ".join(tokens)

# Helps randomly remove words
def drop_random_words(text: str, drop_prob: float = 0.20) -> str:
    tokens: list[str] = text.split()
    kept: list[str] = [t for t in tokens if random.random() > drop_prob]
    return " ".join(kept) if kept else text

# Simulates speech recognition mistakes
def apply_asr_substitutions(text: str) -> str:
    for original, substituted in ASR_SUBSTITUTIONS:
        if original and original in text:
            if random.random() < 0.4:
                text = text.replace(original, substituted, 1)
    return text.strip()

# Helps in simulating indian pronunciation effects
def apply_indian_asr(text: str) -> str:
    for original, substituted in INDIAN_ASR_SUBSTITUTIONS:
        if original in text and random.random() < 0.30:
            text = text.replace(original, substituted, 1)
    return text.strip()

# Helps in adding keywords typos as if any mistakes happened while typing the context
def apply_char_noise(text: str, aug: nac.KeyboardAug) -> str:
    try:
        result: list[str] = aug.augment(text)
        return result[0] if result else text
    except Exception:
        return text

# Text cleaning function as if it is lowercase ro uppercase easily understand it
def lowercase_and_strip(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


# Paraphrases that help to maintain the exact core meaning

PARAPHRASE_TRANSFORMS: Final[list[tuple[str, str]]] = [
    # volume synonyms
    (r"\bincrease the volume\b",  "turn up the volume"),
    (r"\bdecrease the volume\b",  "turn down the volume"),
    (r"\bincrease the volume\b",  "make it louder"),
    (r"\bdecrease the volume\b",  "make it quieter"),
    (r"\bturn up the volume\b",   "raise the volume"),
    (r"\bturn down the volume\b", "lower the volume"),
    # brightness synonyms
    (r"\bincrease the brightness\b",  "turn up the brightness"),
    (r"\bdecrease the brightness\b",  "turn down the brightness"),
    (r"\bincrease the brightness\b",  "make the screen brighter"),
    (r"\bdecrease the brightness\b",  "dim the screen"),
    # music synonyms
    (r"\bplay the music\b",   "start playing music"),
    (r"\bpause the music\b",  "stop the music"),
    (r"\bplay the next song\b",     "skip to the next track"),
    (r"\bplay the previous song\b", "go back to the previous track"),
    # calls synonyms
    (r"\bpick up the call\b",  "answer the call"),
    (r"\bdecline the call\b",  "reject the call"),
    # dnd synonyms
    (r"\bactivate do not disturb\b",   "turn on do not disturb"),
    (r"\bdeactivate do not disturb\b", "turn off do not disturb"),
    # vehicle synonyms
    (r"\bstart the vehicle\b",  "start the car"),
    (r"\bstop the vehicle\b",   "stop the car"),
    (r"\bstart the vehicle\b",  "turn on the engine"),
    (r"\bstop the vehicle\b",   "turn off the engine"),
]

# Checks if paraphrase matches any rules
def paraphrase(text: str) -> str:
    random.shuffle(PARAPHRASE_TRANSFORMS)
    for pattern, replacement in PARAPHRASE_TRANSFORMS:
        if re.search(pattern, text, re.IGNORECASE):
            return re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Fallback: add a prefix before the context
    prefixes: list[str] = ["please ", "hey, ", "can you ", "i want to "]
    if not any(text.startswith(p) for p in prefixes):
        return random.choice(prefixes) + text
    return text

# Generate the data set in a proper structured manner Text | Label | Source
def generate_clean_samples(
    phrases: dict[str, list[str]]
) -> list[Sample]:
    samples: list[Sample] = []
    for label, texts in phrases.items():
        for t in texts:
            samples.append(
                Sample(
                    text=lowercase_and_strip(t),
                    label=label,
                    source="clean",
                )
            )
    return samples

# Helps in generating upto 3 multi paraphrased versions
def generate_paraphrase_samples(
    phrases: dict[str, list[str]],
    multiplier: int = 3,
) -> list[Sample]:
    samples: list[Sample] = []
    for label, texts in phrases.items():
        for t in texts:
            seen: set[str] = {lowercase_and_strip(t)}
            for _ in range(multiplier):
                candidate: str = lowercase_and_strip(paraphrase(t))
                if candidate not in seen:
                    seen.add(candidate)
                    samples.append(Sample(text=candidate, label=label, source="paraphrase"))
    return samples

# For every clean sample produce `multiplier` noisy variants by randomly mixing: filler insertion, word dropping, ASR substitution, Indian-ASR phonetics, and keyboard-character augmentation.
def generate_noise_samples(
    clean_samples: list[Sample],
    char_aug: nac.KeyboardAug,
    multiplier: int = 2,
) -> list[Sample]:
    noise_strategies: list[str] = [
        "filler", "drop", "asr_sub", "indian_asr", "char_noise", "combined"
    ]
    samples: list[Sample] = []

    for s in clean_samples:
        for _ in range(multiplier):
            strategy: str = random.choice(noise_strategies)
            noisy: str = s.text

            if strategy == "filler":
                noisy = insert_fillers(noisy, n=random.randint(1, 2))
                src = "filler"
            elif strategy == "drop":
                noisy = drop_random_words(noisy, drop_prob=0.25)
                src = "dropped"
            elif strategy == "asr_sub":
                noisy = apply_asr_substitutions(noisy)
                src = "asr_noise"
            elif strategy == "indian_asr":
                noisy = apply_indian_asr(noisy)
                src = "asr_noise"
            elif strategy == "char_noise":
                noisy = apply_char_noise(noisy, char_aug)
                src = "char_noise"
            else:  # combined
                noisy = insert_fillers(noisy, n=1)
                noisy = apply_asr_substitutions(noisy)
                noisy = apply_indian_asr(noisy)
                if random.random() < 0.4:
                    noisy = drop_random_words(noisy, drop_prob=0.15)
                src = "asr_noise"

            noisy = lowercase_and_strip(noisy)
            if noisy and noisy != s.text:
                samples.append(Sample(text=noisy, label=s.label, source=src))
    return samples

# Helps to remove the duplicate samples
def deduplicate_samples(samples: list[Sample]) -> list[Sample]:
    seen: set[tuple[str, str]] = set()
    unique: list[Sample] = []
    for s in samples:
        key: tuple[str, str] = (s.text, s.label)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# Hard Negative Injections: Teaching the model between every similar commands

HARD_NEGATIVES: Final[list[tuple[str, str]]] = [
    ("increase the volume all the way up",    "increase_volume"),
    ("increase the brightness all the way up","increase_brightness"),
    ("turn the volume down completely",       "decrease_volume"),
    ("turn the brightness down completely",   "decrease_brightness"),
    ("make it louder now",                    "increase_volume"),
    ("make it brighter now",                  "increase_brightness"),
    ("lower the volume a bit",                "decrease_volume"),
    ("lower the brightness a bit",            "decrease_brightness"),
    ("please start the car engine now",       "start_vehicle"),
    ("please stop the car engine now",        "stop_vehicle"),
    ("get the engine going",                  "start_vehicle"),
    ("kill the engine now",                   "stop_vehicle"),
    ("start playing the song",                "play_music"),
    ("stop playing the song",                 "pause_music"),
    ("answer the incoming call please",       "pick_up_call"),
    ("reject the incoming call please",       "decline_call"),
    ("go to the next song in the playlist",   "next_song"),
    ("go to the previous song in the playlist","previous_song"),
    ("switch do not disturb on",              "activate_dnd"),
    ("switch do not disturb off",             "deactivate_dnd"),
    ("turn it down",                          "decrease_volume"),
    ("turn it up",                            "increase_volume"),
    ("make it quieter",                       "decrease_volume"),
    ("make it louder",                        "increase_volume"),
    ("lower the sound",                       "decrease_volume"),
    ("raise the sound",                       "increase_volume"),
]

# Hard negatives into samples
def get_hard_negative_samples() -> list[Sample]:
    return [
        Sample(text=lowercase_and_strip(text), label=label, source="hard_negative")
        for text, label in HARD_NEGATIVES
    ]

# Actual data set generation
def build_dataset(
    output_dir: Path = Path("data"),
    paraphrase_multiplier: int = 3,
    noise_multiplier: int = 6,
    test_size: float = 0.15,
) -> DatasetStats:

    output_dir.mkdir(parents=True, exist_ok=True)

    # Augmenter (keyboard-typo, low action probability)
    char_aug: nac.KeyboardAug = nac.KeyboardAug(
        aug_char_p=0.10, # 10% of characters in a selected word can be modified (means less character mistake in a word, 50% change won't occur
        aug_word_p=0.15, # 15% probability means usually only one word gets modified.
        include_special_char=False, # No insertion of special characters
        include_upper_case=False, # Speech-to-text systems usually produce lowercase text, so uppercase noise is unnecessary.
    )

    # creates clean samples
    print("[1/6] Generating clean samples …")
    clean: list[Sample] = generate_clean_samples(SEED_PHRASES)

    # creates paraphrase samples
    print("[2/6] Generating paraphrase samples …")
    paraphrased: list[Sample] = generate_paraphrase_samples(
        SEED_PHRASES, multiplier=paraphrase_multiplier
    )

    # creates noisy or asr samples
    print("[3/6] Generating noisy / ASR samples …")
    noisy: list[Sample] = generate_noise_samples(
        clean, char_aug, multiplier=noise_multiplier
    )

    # Hard negatives samples
    print("[4/6] Injecting hard-negative samples …")
    hard_neg: list[Sample] = get_hard_negative_samples()

    # Removes duplicate samples
    print("[5/6] Merging and deduplicating …")
    all_samples: list[Sample] = deduplicate_samples(
        clean + paraphrased + noisy + hard_neg
    )
    random.shuffle(all_samples)

    # Split data set between train and test data for class balance
    print("[6/6] Stratified train/test split …")
    texts: list[str]  = [s.text   for s in all_samples]
    labels: list[str] = [s.label  for s in all_samples]
    sources: list[str]= [s.source for s in all_samples]

    (
        train_texts, test_texts,
        train_labels, test_labels,
        train_sources, test_sources,
    ) = train_test_split(
        texts, labels, sources,
        test_size=test_size,
        stratify=labels,
        random_state=SEED,
    )

    # Writing CSV files and saving the files
    def write_csv(path: Path, rows_texts: list[str], rows_labels: list[str], rows_sources: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["text", "label", "source"])
            writer.writeheader()
            for t, l, s in zip(rows_texts, rows_labels, rows_sources):
                writer.writerow({"text": t, "label": l, "source": s})

    train_csv = output_dir / "train.csv"
    test_csv  = output_dir / "test.csv"
    write_csv(train_csv, train_texts, train_labels, train_sources)
    write_csv(test_csv,  test_texts,  test_labels,  test_sources)

    label_map: dict[str, int] = {lbl: idx for idx, lbl in enumerate(ALL_LABELS)}
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as fh:
        json.dump(label_map, fh, indent=2)

    # -- Stats -----------------------------------------------------------------
    from collections import Counter
    per_label: dict[str, int] = dict(Counter(labels))
    source_counts: dict[str, int] = dict(Counter(sources))

    stats = DatasetStats(
        total=len(all_samples),
        per_label=per_label,
        source_counts=source_counts,
        train_size=len(train_texts),
        test_size=len(test_texts),
    )

    # Pint the dataset Summary
    print("\n" + "═" * 60)
    print(f"  DATASET SUMMARY")
    print("═" * 60)
    print(f"  Total samples    : {stats.total}")
    print(f"  Train / Test     : {stats.train_size} / {stats.test_size}")
    print(f"\n  Samples by source:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src:<18} {cnt:>5}")
    print(f"\n  Samples by label:")
    for lbl, cnt in sorted(per_label.items(), key=lambda x: -x[1]):
        print(f"    {lbl:<28} {cnt:>5}")
    print("═" * 60)
    print(f"  Files written to : {output_dir.resolve()}")
    print("═" * 60 + "\n")

    return stats


# Sanity Check for verifying the dataset quality
def _sanity_check(data_dir: Path) -> None:
    for split in ("train", "test"):
        path: Path = data_dir / f"{split}.csv"
        df: pd.DataFrame = pd.read_csv(path)
        print(f"\n── {split.upper()} ({len(df)} rows) ─────────────────────────────")
        # Show 2 rows per label for first 5 labels
        for label in list(df["label"].unique())[:5]:
            subset = df[df["label"] == label].sample(min(2, len(df[df["label"]==label])), random_state=0)
            for _, row in subset.iterrows():
                print(f"  [{row['label']}]  [{row['source']}]  \"{row['text']}\"")


if __name__ == "__main__":
    DATA_DIR: Path = Path("data")
    build_dataset(output_dir=DATA_DIR)
    _sanity_check(DATA_DIR)
