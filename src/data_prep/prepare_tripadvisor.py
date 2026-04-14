"""
Prepare TripAdvisor reviews for the airline sentiment pipeline.
Accepts one or more source CSVs, combines, deduplicates, filters by MIN_DATE, and saves.

Usage:
    python -m src.data_prep.prepare_tripadvisor path/to/file1.csv path/to/file2.csv ...

Example:
    python -m src.data_prep.prepare_tripadvisor \\
        market_exp_extract/Fact_Reviews_v2.csv \\
        data/Fact_Reviews_new.csv
"""

import sys
import os
import glob
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.config import MIN_DATE, TRIPADVISOR_PATH, RAW_TRIPADVISOR_DIR


def prepare(input_paths: list[str]) -> pd.DataFrame:
    dfs = []
    for pattern in input_paths:
        matched = glob.glob(pattern)
        if not matched:
            print(f'Warning: no files matched "{pattern}"')
            continue
        for path in matched:
            print(f'Loading: {path}')
            df = pd.read_csv(path)
            dfs.append(df)
            print(f'  {len(df)} rows')

    if not dfs:
        print('No input files loaded. Exiting.')
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    print(f'\nCombined rows: {len(combined)}')

    # Parse dates first so we can sort — handles mixed formats across source files
    combined['date'] = pd.to_datetime(combined['date'], format='mixed', dayfirst=True, errors='coerce')
    # Sort newest-first so keep='first' always retains the most recent version of each review
    combined = combined.sort_values('date', ascending=False)
    combined = combined.drop_duplicates(subset='review_unique_id', keep='first')
    print(f'After dedup: {len(combined)}')

    combined = combined[combined['data_source'].str.lower() == 'tripadvisor']
    print(f'After TripAdvisor filter: {len(combined)}')

    combined = combined[combined['date'] >= pd.Timestamp(MIN_DATE)]
    combined['date'] = combined['date'].dt.strftime('%Y-%m-%d')
    print(f'After date filter (>= {MIN_DATE}): {len(combined)}')

    print('\n── By airline ──')
    print(combined.groupby('airline').size().sort_values(ascending=False).to_string())
    print('\n── Date range ──')
    print(combined['date'].min(), '->', combined['date'].max())

    os.makedirs(os.path.dirname(TRIPADVISOR_PATH), exist_ok=True)
    combined.to_csv(TRIPADVISOR_PATH, index=False, encoding='utf-8-sig')
    print(f'\nSaved: {TRIPADVISOR_PATH} ({len(combined)} rows)')
    return combined


if __name__ == '__main__':
    paths = sys.argv[1:] if len(sys.argv) > 1 else [f'{RAW_TRIPADVISOR_DIR}/*.csv']
    prepare(paths)
