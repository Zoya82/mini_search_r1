import re
import string

ARTICLES = {"a", "an", "the"}

def normalize(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    words = [w for w in text.split() if w not in ARTICLES]
    return " ".join(words)

def em(pred, gold):
    return 1 if normalize(pred) == normalize(gold) else 0