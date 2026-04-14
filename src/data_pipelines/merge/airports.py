"""
Final Merge - Airport.
Combines sentiment extraction output + subtopic mappings + Google metadata
into a dashboard-ready CSV. Appends X/Twitter data if available.

Run from project root:
    python -m src.data_pipelines.merge.airports
"""

import os
import sys
import pandas as pd
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from src.config import BATCH_DATE, GOOGLE_PATH, AIRPORT_OUTPUT, AIRPORT_X_OUTPUT, LANG_MAP, MIN_DATE
from src.utils.helpers import combine_csvs, explode_column, fix_subtopic_format

PHASE1_PATH = f'airport_sentiment/intermediate_sentiments/{BATCH_DATE}_batches/all_reviews_combined.csv'
PP_DIR      = f'airport_sentiment/intermediate_sentiments/{BATCH_DATE}_batches/painpoints'
MOD_DIR     = f'airport_sentiment/intermediate_sentiments/{BATCH_DATE}_batches/mods'

FINAL_COLS = ['Review_No', 'sentiment', 'topic', 'pp_mod', 'type',
              'Airport', 'date', 'rating', 'data_source',
              'Original Review Language', 'Original Review', 'Translated Review',
              'subtopic']


if __name__ == '__main__':
    print(f'Loading sentiment output: {PHASE1_PATH}')
    phase1 = pd.read_csv(PHASE1_PATH)
    phase1['review_id'] = phase1['review_id'].astype(int)

    pp  = explode_column(phase1, 'review_id', 'sentiment', 'topic', 'pain_points',        'Pain Point')
    mod = explode_column(phase1, 'review_id', 'sentiment', 'topic', 'moments_of_delight', 'Moment of Delight')

    print('Loading subtopic mappings...')
    pp_sub  = combine_csvs(PP_DIR)
    mod_sub = combine_csvs(MOD_DIR)
    pp  = pp.merge(pp_sub.rename(columns={'id': 'pp_mod_id'}),  on='pp_mod_id', how='left') if not pp_sub.empty  else pp.assign(subtopic=None)
    mod = mod.merge(mod_sub.rename(columns={'id': 'pp_mod_id'}), on='pp_mod_id', how='left') if not mod_sub.empty else mod.assign(subtopic=None)

    combined = pd.concat([pp, mod], ignore_index=True).drop(columns=['pp_mod_id'])

    print(f'Loading Google review data: {GOOGLE_PATH}')
    raw = pd.read_csv(GOOGLE_PATH)
    raw['date'] = pd.to_datetime(raw['published_at_datetime_x'], utc=True).dt.date
    raw['Original Review'] = raw.apply(
        lambda r: r['original_text'] if pd.notna(r['original_text']) and r['original_text'] != '' else r['text'],
        axis=1
    )
    raw['Translated Review'] = raw['text']
    raw = raw.rename(columns={'lang': 'Original Review Language', 'score_x': 'rating', 'origin': 'data_source'})
    raw = raw[['Review_No', 'Airport', 'date', 'rating', 'data_source',
               'Original Review Language', 'Original Review', 'Translated Review']]

    combined['Review_No'] = combined['Review_No'].astype(int)
    final = combined.merge(raw, on='Review_No', how='left')
    final = final[FINAL_COLS]

    final['topic'] = final['topic'].replace(
        {'Baggage': 'Baggage Services', 'Baggage Services - Airports': 'Baggage Services'}
    )
    final['date'] = pd.to_datetime(final['date'], errors='coerce')
    final = final[final['date'] >= MIN_DATE]
    final['date'] = final['date'].dt.strftime('%Y-%m-%d')

    final['Original Review Language'] = final['Original Review Language'].map(
        lambda x: LANG_MAP.get(str(x).lower().strip(), x) if pd.notna(x) else x
    )
    final['subtopic'] = final['subtopic'].apply(fix_subtopic_format)

    if os.path.exists(AIRPORT_X_OUTPUT):
        print(f'Appending X data: {AIRPORT_X_OUTPUT}')
        x_df = pd.read_csv(AIRPORT_X_OUTPUT, encoding='utf-8-sig')[FINAL_COLS]
        final = pd.concat([final, x_df], ignore_index=True)
        print(f'X rows added: {len(x_df)}')
    else:
        print('No X airport output found — skipping. Run x_data/merge_airports.py first.')

    final.to_csv(AIRPORT_OUTPUT, index=False, encoding='utf-8-sig')
    print(f'Saved: {AIRPORT_OUTPUT} ({len(final)} rows, {final["Review_No"].nunique()} unique reviews)')
    print('Airport merge complete.')
