"""Shared rules for what counts as hate (used by scan_hate.py and chat_to_board.py)."""
import re

from better_profanity import profanity

# hater words to cover; when several match, exact and longer matches win
HATE_KEYS = ["DISGUSTING", "PATHETIC", "FUCKING", "GARBAGE", "FUCKED", "FUCKER", "HATERS", "LOSERS", "PHUCK", "SUCKS",
             "HATER", "HATES", "HATED", "DEATH", "LOSER", "TRASH", "FUCK", "SUCK", "HATE", "KILL", "STFU", "SCUM",
             "SHIT", "FCK", "FUK", "SUX", "DIE", "KYS"]
HOSTILE = [r"\bshit", r"\bgarbage\b", r"\bsux\b",
    r"\bhate[sd]?\b", r"\bhating\b", r"\bhaters?\b", r"\bsucks?\b", r"\bdeath\s+to\b", r"\bkill\b",
    r"\bdie\b", r"\bf+[\W_]*[u*]*[\W_]*c+[\W_]*k+", r"\bstfu\b", r"\bkys\b", r"\bgo\s+home\b",
    r"\bscum\b", r"\btrash\b", r"\bpathetic\b", r"\bloser?s?\b", r"\bdisgusting\b",
]
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "|": "l"})
SCALE = 3
# identity words are not hate on their own (and may well be pride messages): never cover them
IDENTITY = ["gay", "gays", "lesbian", "lesbians", "queer", "queers", "trans", "transgender", "bi", "bisexual",
            "lgbt", "lgbtq", "homosexual", "pride", "nonbinary"]
profanity.load_censor_words(whitelist_words=IDENTITY)


def norm(text):
    return re.sub(r"[^a-z\s*]", " ", text.lower().translate(LEET))


def hateful(text):
    t = norm(text)
    return profanity.contains_profanity(t) or any(re.search(p, t) for p in HOSTILE)
