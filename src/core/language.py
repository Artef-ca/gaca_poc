"""
Language detection and translation utilities.
Used by all merge scripts to populate 'Original Review Language' and 'Translated Review'.
"""

import logging
import time
import re
import pandas as pd
import os
from langdetect import detect, LangDetectException

from src.config import LANG_MAP
from src.core.llm import make_model

log = logging.getLogger(__name__)


def detect_language(text: str) -> tuple[str, str]:
    """Return (lang_code, lang_name) for a piece of text."""
    try:
        code = detect(str(text))
        return code, LANG_MAP.get(code, code)
    except LangDetectException:
        return 'unknown', 'Unknown'


def translate_batch(texts: list[str], model, batch_size: int = 20) -> list[str]:
    """
    Translate a list of texts to English via Gemini.
    English texts are returned as-is. Returns a list of the same length.
    """
    translations = [''] * len(texts)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        numbered = '\n'.join(f'{j + 1}. {t}' for j, t in enumerate(batch))
        prompt = (
            'Translate each of the following tweets to English. '
            'Return ONLY the translations numbered in the same order, one per line. '
            'If a tweet is already in English, return it as-is.\n\n' + numbered
        )
        try:
            resp = model.generate_content(prompt)
            lines = [l.strip() for l in resp.text.strip().split('\n') if l.strip()]
            cleaned = [re.sub(r'^\d+\.\s*', '', l) for l in lines]
            for j, t in enumerate(cleaned[:len(batch)]):
                translations[i + j] = t
        except Exception as e:
            log.warning('Translation batch %d failed: %s', i // batch_size + 1, e)
            time.sleep(2)
    return translations


def enrich_language_and_translation(df: pd.DataFrame, text_col: str = 'Original Review') -> pd.DataFrame:
    """
    Detect language and translate non-English rows.
    Adds 'Original Review Language' and 'Translated Review' columns.
    """
    print('Detecting languages...')
    lang_codes, lang_names = zip(*df[text_col].apply(detect_language))
    df = df.copy()
    df['Original Review Language'] = list(lang_names)
    df['_lang_code'] = list(lang_codes)

    non_english = df[df['_lang_code'] != 'en']
    print(f'  {len(df) - len(non_english)} English - copied as-is')
    print(f'  {len(non_english)} non-English - translating via Gemini...')

    df['Translated Review'] = df[text_col]

    if len(non_english) > 0:
        model = make_model(os.environ.get('GEMINI_API_KEY'))
        translated = translate_batch(non_english[text_col].tolist(), model)
        df.loc[non_english.index, 'Translated Review'] = translated

    df = df.drop(columns=['_lang_code'])
    return df
