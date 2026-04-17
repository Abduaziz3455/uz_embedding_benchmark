"""Uzbek Latin -> Cyrillic transliteration for models trained on Cyrillic-only corpora.

Rules follow the official 1995 Uzbek Latin alphabet. Multi-character sequences
(sh, ch, ng, o', g') are handled before single-character substitution.
"""

from typing import Iterable, List

_DIGRAPHS = [
    ("o'", "ў"), ("O'", "Ў"), ("o‘", "ў"), ("O‘", "Ў"), ("oʻ", "ў"), ("Oʻ", "Ў"),
    ("o’", "ў"), ("O’", "Ў"), ("oʼ", "ў"), ("Oʼ", "Ў"),
    ("g'", "ғ"), ("G'", "Ғ"), ("g‘", "ғ"), ("G‘", "Ғ"), ("gʻ", "ғ"), ("Gʻ", "Ғ"),
    ("g’", "ғ"), ("G’", "Ғ"), ("gʼ", "ғ"), ("Gʼ", "Ғ"),
    ("sh", "ш"), ("Sh", "Ш"), ("SH", "Ш"),
    ("ch", "ч"), ("Ch", "Ч"), ("CH", "Ч"),
    ("ng", "нг"), ("Ng", "Нг"), ("NG", "НГ"),
    ("yo", "ё"), ("Yo", "Ё"), ("YO", "Ё"),
    ("yu", "ю"), ("Yu", "Ю"), ("YU", "Ю"),
    ("ya", "я"), ("Ya", "Я"), ("YA", "Я"),
    ("ye", "е"), ("Ye", "Е"), ("YE", "Е"),
    ("ts", "ц"), ("Ts", "Ц"), ("TS", "Ц"),
]

_SINGLES = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "э", "f": "ф", "g": "г", "h": "ҳ",
    "i": "и", "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о",
    "p": "п", "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в",
    "w": "в", "x": "х", "y": "й", "z": "з",
    "A": "А", "B": "Б", "C": "К", "D": "Д", "E": "Э", "F": "Ф", "G": "Г", "H": "Ҳ",
    "I": "И", "J": "Ж", "K": "К", "L": "Л", "M": "М", "N": "Н", "O": "О",
    "P": "П", "Q": "Қ", "R": "Р", "S": "С", "T": "Т", "U": "У", "V": "В",
    "W": "В", "X": "Х", "Y": "Й", "Z": "З",
    "'": "ъ", "‘": "ъ", "’": "ъ", "ʻ": "ъ", "ʼ": "ъ",
}


def latin_to_cyrillic(text: str) -> str:
    for src, dst in _DIGRAPHS:
        text = text.replace(src, dst)
    return "".join(_SINGLES.get(ch, ch) for ch in text)


def batch_latin_to_cyrillic(texts: Iterable[str]) -> List[str]:
    return [latin_to_cyrillic(t) for t in texts]


_TRANSLITERATORS = {
    "latin2cyrillic": batch_latin_to_cyrillic,
}


def apply(mode: str, texts: Iterable[str]) -> List[str]:
    fn = _TRANSLITERATORS.get(mode)
    if fn is None:
        raise ValueError(f"unknown transliterate mode: {mode!r}")
    return fn(texts)
