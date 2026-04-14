"""
Final Merge - Airlines.
Combines sentiment extraction output + subtopic mappings + TripAdvisor metadata
into a dashboard-ready CSV. Appends X/Twitter data if available.

Run from project root:
    python -m src.data_pipelines.merge.airlines
"""

import os
import sys
import pandas as pd
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from src.config import BATCH_DATE, TRIPADVISOR_PATH, AIRLINE_OUTPUT, AIRLINE_X_OUTPUT, LANG_MAP, MIN_DATE
from src.utils.helpers import combine_csvs, explode_column, fix_subtopic_format

PHASE1_PATH   = f'airlines_sentiment/intermediate_sentiments/{BATCH_DATE}_batches/all_reviews_combined.csv'
PP_DIR        = f'airlines_sentiment/intermediate_sentiments/{BATCH_DATE}_batches/painpoints'
MOD_DIR       = f'airlines_sentiment/intermediate_sentiments/{BATCH_DATE}_batches/mods'

FINAL_COLS = ['Review_No', 'sentiment', 'topic', 'pp_mod', 'subtopic', 'type',
              'data_source', 'airline', 'date', 'rating',
              'Original Review Language', 'Original Review', 'Translated Review']


if __name__ == '__main__':
    print(f'Loading sentiment output: {PHASE1_PATH}')
    phase1 = pd.read_csv(PHASE1_PATH)

    pp  = explode_column(phase1, 'review_id', 'sentiment', 'topic', 'pain_points',        'Pain Point')
    mod = explode_column(phase1, 'review_id', 'sentiment', 'topic', 'moments_of_delight', 'Moment of Delight')

    print('Loading subtopic mappings...')
    pp_sub  = combine_csvs(PP_DIR)
    mod_sub = combine_csvs(MOD_DIR)
    pp  = pp.merge(pp_sub.rename(columns={'id': 'pp_mod_id'}),  on='pp_mod_id', how='left') if not pp_sub.empty  else pp.assign(subtopic=None)
    mod = mod.merge(mod_sub.rename(columns={'id': 'pp_mod_id'}), on='pp_mod_id', how='left') if not mod_sub.empty else mod.assign(subtopic=None)

    combined = pd.concat([pp, mod], ignore_index=True).drop(columns=['pp_mod_id'])

    print(f'Loading TripAdvisor data: {TRIPADVISOR_PATH}')
    raw = pd.read_csv(TRIPADVISOR_PATH)
    raw['date'] = pd.to_datetime(raw['date'], format='%Y-%m-%d', errors='coerce').dt.strftime('%Y-%m-%d')
    raw = raw.rename(columns={
        'review_unique_id' : 'Review_No',
        'review_language'  : 'Original Review Language',
        'raw_text'         : 'Original Review',
        'translated_text'  : 'Translated Review',
        'ratings'          : 'rating',
    })
    raw = raw[['Review_No', 'data_source', 'airline', 'date', 'rating',
               'Original Review Language', 'Original Review', 'Translated Review']]

    final = combined.merge(raw, on='Review_No', how='left')
    final = final[FINAL_COLS]
    final['date'] = pd.to_datetime(final['date'], errors='coerce')
    final = final[final['date'] >= MIN_DATE]
    final['date'] = final['date'].dt.strftime('%Y-%m-%d')

    final['Original Review Language'] = final['Original Review Language'].map(
        lambda x: LANG_MAP.get(str(x).lower().strip(), x) if pd.notna(x) else x
    )
    final['subtopic'] = final['subtopic'].apply(fix_subtopic_format)

    if os.path.exists(AIRLINE_X_OUTPUT):
        print(f'Appending X data: {AIRLINE_X_OUTPUT}')
        x_df = pd.read_csv(AIRLINE_X_OUTPUT, encoding='utf-8-sig')[FINAL_COLS]
        final = pd.concat([final, x_df], ignore_index=True)
        print(f'X rows added: {len(x_df)}')
    else:
        print('No X airline output found — skipping. Run x_data/merge_airlines.py first.')

    final.to_csv(AIRLINE_OUTPUT, index=False, encoding='utf-8-sig')
    print(f'Saved: {AIRLINE_OUTPUT} ({len(final)} rows, {final["Review_No"].nunique()} unique reviews)')
    print('Airline merge complete.')
