"""
Prepare X / Twitter data for the enrichment pipeline.
Accepts one or more Excel or CSV scraper exports, combines, deduplicates,
applies keyword classification, and saves to X_COMBINED_PATH.

For incremental updates (new scraper files on top of existing data),
use src/x_data/enrich.py instead — it handles dedup against the existing file.

Usage:
    python -m src.data_prep.prepare_x path/to/file.xlsx ...

Example:
    python -m src.data_prep.prepare_x "X_data/TwitterData_*.xlsx"
"""

import sys
import os
import glob
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.config import (X_COMBINED_PATH, RAW_X_DIR,
                    AIRLINE_ENTITY_KEYWORDS, AIRPORT_ENTITY_KEYWORDS,
                    AIRPORT_GENERIC_KEYWORDS, AIRLINE_GENERIC_KEYWORDS)


def keyword_classify(text: str) -> tuple:
    if not isinstance(text, str):
        return None, None
    t = text.lower()
    for entity, kws in AIRPORT_ENTITY_KEYWORDS.items():
        if any(kw.lower() in t for kw in kws):
            return 'AIRPORT', entity
    for entity, kws in AIRLINE_ENTITY_KEYWORDS.items():
        if any(kw.lower() in t for kw in kws):
            return 'AIRLINE', entity
    if any(kw.lower() in t for kw in AIRPORT_GENERIC_KEYWORDS):
        return 'AIRPORT', 'Unknown'
    if any(kw.lower() in t for kw in AIRLINE_GENERIC_KEYWORDS):
        return 'AIRLINE', 'Unknown'
    return 'OTHER', 'Unknown'


def prepare(input_paths: list[str]) -> pd.DataFrame:
    dfs = []
    for pattern in input_paths:
        matched = glob.glob(pattern)
        if not matched:
            print(f'Warning: no files matched "{pattern}"')
            continue
        for path in matched:
            print(f'Loading: {path}')
            df = pd.read_excel(path, engine='openpyxl') if path.endswith('.xlsx') else pd.read_csv(path)
            dfs.append(df[['INTERNAL UNIQUE ID', 'CONTENT', 'PUBLISHED AT']].copy())
            print(f'  {len(dfs[-1])} rows')

    if not dfs:
        print('No input files loaded. Exiting.')
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined['INTERNAL UNIQUE ID'] = combined['INTERNAL UNIQUE ID'].astype(str)
    combined = combined.drop_duplicates(subset='INTERNAL UNIQUE ID')
    print(f'\nUnique tweets: {len(combined)}')

    combined['PUBLISHED AT'] = pd.to_datetime(combined['PUBLISHED AT'], utc=True, errors='coerce')
    print(f'Date range: {combined["PUBLISHED AT"].min()} -> {combined["PUBLISHED AT"].max()}')
    print(f'Null dates: {combined["PUBLISHED AT"].isna().sum()}')

    print('\nRunning keyword classification...')
    combined[['label', 'main_entity']] = combined['CONTENT'].apply(
        lambda x: pd.Series(keyword_classify(x))
    )

    print('\n── Label distribution ──')
    print(combined['label'].value_counts().to_string())
    print('\n── Entity distribution ──')
    print(combined['main_entity'].value_counts().to_string())

    os.makedirs(os.path.dirname(X_COMBINED_PATH), exist_ok=True)
    combined.to_csv(X_COMBINED_PATH, index=False, encoding='utf-8-sig')
    print(f'\nSaved: {X_COMBINED_PATH} ({len(combined)} rows)')
    print('Note: run src/x_data/enrich.py next for LLM fallback classification + language enrichment.')
    return combined


if __name__ == '__main__':
    paths = sys.argv[1:] if len(sys.argv) > 1 else [f'{RAW_X_DIR}/*.xlsx', f'{RAW_X_DIR}/*.csv']
    prepare(paths)
