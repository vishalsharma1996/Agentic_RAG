import re
import pandas as pd
import numpy as np
from docling.document_converter import DocumentConverter

def clean_text_content(text):
    """
    Minimal structural cleaning for research-paper text.

    Cleaning steps:
    1: Remove table 1: table 1 : table-1: table 1. and any other format
    1. Remove citation markers like [29], [12], [3,4]
    2. Normalize whitespace
    3. Fix punctuation spacing
    4. Normalize parenthesis spacing
    5. Preserve mathematical tokens
    """
    text = text.lower()

    text = re.sub(r"table\s*[-]?\s*\d+\s*[:\.]\s*", "", text)

    text = text.replace('-','')

    text = text.replace('@',' at ').replace('%',' percent ').replace('$',' dollar ')
    # ---------------------------------------------------------
    # 1. Remove citation markers like [29], [3], [12,13]
    # ---------------------------------------------------------
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)

    # ---------------------------------------------------------
    # 2. Normalize whitespace
    # Convert multiple spaces/newlines → single space
    # ---------------------------------------------------------
    text = re.sub(r"\s+", " ", text)

    # ---------------------------------------------------------
    # 3. Remove spaces before punctuation
    # Example: "z , scoring" → "z, scoring"
    # ---------------------------------------------------------
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    # ---------------------------------------------------------
    # 4. Fix spacing inside parentheses
    # Example: "p ( y | x )" → "p(y | x)"
    # ---------------------------------------------------------
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    # ---------------------------------------------------------
    # 5. Normalize pipe spacing in formulas
    # Example: "y|x" → "y | x"
    # ---------------------------------------------------------
    text = re.sub(r"\s*\|\s*", " | ", text)

    return text.strip()

def normalize_header_token(token):
    token = token.lower().strip()
    token = token.replace('..','.').replace('.',' ')
    token = re.sub(r'\s+',' ',token)
    token = token.replace('#',' number of')
    token = token.replace("%", " percent ")
    token = token.replace("$", " dollar ")
    words = token.split()
    deduped = []
    for w in words:
      if not deduped or deduped[-1]!=w:
        deduped.append(w)
    token = ' '.join(deduped)
    return token.strip()

def normalize_column_values(token):
    token = token.lower().strip()
    # Remove citation markers and stars
    token = re.sub(r"\[\d+\]", "", token)
    token = token.replace("*", "")
    token = token.replace("%", " percent ").replace('w/o',' without ')
    token = token.replace("$", " dollar ").replace('@',' at ')
    token = re.sub(r'\s+',' ',token).replace('w/','with')

    # Missing value
    if token in {"-", "--", ""}:
        return None
    if '/' in token and bool(re.fullmatch(r"[^a-z]+", token)):
      parts = token.split('/')
      parsed = [re.sub('[^0-9a-z. ]+',' ',p).strip() for p in parts if len(re.sub('[^0-9a-z. ]+',' ',p).strip())>1]
      return parsed
    return token
