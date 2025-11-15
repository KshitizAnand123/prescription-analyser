# ai_models/preprocessing.py
import re

NUM_RE = re.compile(r'(\d+(\.\d+)?)')
UNIT_MAP = {
    r'\bmg/dl\b': ' mg/dl',
    r'\bbp\b': ' blood_pressure',
    r'\bhba1c\b': ' hba1c',
    r'\bmmhg\b': ' mmhg'
}

def normalize_text(s: str) -> str:
    """Simple normalizer for medical text: lowercases, expands units, separates numbers/letters."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    for pat, rep in UNIT_MAP.items():
        s = re.sub(pat, rep, s)
    s = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', s)
    s = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', s)
    s = re.sub(r'\s+', ' ', s)
    return s
