"""
Prepare Google Maps reviews for the airport sentiment pipeline.
Accepts one or more scraper CSVs, combines, filters by MIN_DATE, maps airport codes, and saves.

Usage:
    python -m src.data_prep.prepare_google path/to/scraper_output.csv ...

Example:
    python -m src.data_prep.prepare_google "Google_review_data/Google Maps Reviews Scraper*.csv"
"""

import sys
import os
import glob
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.config import MIN_DATE, GOOGLE_PATH, RAW_GOOGLE_DIR

PLACE_TO_CODE = {
    'King Khalid International Airport'   : 'RUH',
    'King Fahd International Airport'     : 'DMM',
    'King Abdulaziz International Airport': 'JED',
}


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

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates()
    print(f'\nCombined rows: {len(df)}')

    df['published_at_datetime_x'] = pd.to_datetime(df['PUBLISHED AT DATETIME'], errors='coerce', utc=True)
    df = df[df['published_at_datetime_x'] >= MIN_DATE]
    print(f'After date filter (>= {MIN_DATE}): {len(df)}')

    df['Airport'] = df['PLACE NAME'].map(PLACE_TO_CODE)
    unmapped = df['Airport'].isna().sum()
    if unmapped > 0:
        print(f'Warning: {unmapped} rows with unmapped airport names:')
        print(df[df['Airport'].isna()]['PLACE NAME'].unique())
    df = df.dropna(subset=['Airport'])

    df = df.rename(columns={
        'SCORE'        : 'score_x',
        'TEXT'         : 'text',
        'ORIGINAL TEXT': 'original_text',
        'LANG'         : 'lang',
        'ORIGIN'       : 'origin',
    })
    df = df.reset_index(drop=True)
    df['Review_No'] = df.index + 1
    df = df[['Review_No', 'Airport', 'published_at_datetime_x',
             'text', 'score_x', 'lang', 'original_text', 'origin']]

    print('\n── By airport ──')
    print(df.groupby('Airport').size().sort_values(ascending=False).to_string())
    print('\n── Date range ──')
    print(df['published_at_datetime_x'].min(), '->', df['published_at_datetime_x'].max())

    os.makedirs(os.path.dirname(GOOGLE_PATH), exist_ok=True)
    df.to_csv(GOOGLE_PATH, index=False, encoding='utf-8-sig')
    print(f'\nSaved: {GOOGLE_PATH} ({len(df)} rows)')
    return df


if __name__ == '__main__':
    paths = sys.argv[1:] if len(sys.argv) > 1 else [f'{RAW_GOOGLE_DIR}/*.csv']
    prepare(paths)
