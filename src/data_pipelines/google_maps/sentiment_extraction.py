"""
Sentiment Extraction - Airport (Google Reviews).
Calls Gemini to extract sentiment / topics / pain points / moments of delight.
Saves batch CSVs to intermediate folders.

Run from project root:
    python -m src.data_pipelines.google_maps.sentiment_extraction
"""

import os
import sys
import pandas as pd
import logging
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from src.config import GOOGLE_PATH, TAXONOMY_PATH, BATCH_DATE
from src.core.llm import resolve_schema, call_and_parse, make_generation_config, make_model
from src.models.sentiment import ReviewsBatch
from src.prompts.loader import get_airport_review_prompt, get_valid_topics

OUTPUT_DIR = f'airport_sentiment/intermediate_sentiments/{BATCH_DATE}_batches'


def _process_batches(reviews_list, system_prompt, output_dir, model, batch_size=50, valid_topics=None):
    os.makedirs(output_dir, exist_ok=True)
    schema = resolve_schema(ReviewsBatch.model_json_schema())
    cfg = make_generation_config(response_schema=schema)

    df_list = []
    total = (len(reviews_list) + batch_size - 1) // batch_size

    for i in range(0, len(reviews_list), batch_size):
        batch_num = i // batch_size + 1
        batch_file = os.path.join(output_dir, f'review_batch_{batch_num}.csv')

        if os.path.exists(batch_file):
            print(f'Batch {batch_num}/{total} already done. Skipping.')
            df_list.append(pd.read_csv(batch_file))
            continue

        batch = reviews_list[i:i + batch_size]
        messages = [system_prompt] + [r['content'] for r in batch]

        try:
            result = call_and_parse(model, messages, cfg, ReviewsBatch)
        except Exception as e:
            log.error('Batch %s failed: %s', batch_num, e)
            continue

        rows = []
        for rev in result.reviews:
            for t in rev.topics:
                topic = t.topic if (valid_topics is None or t.topic in valid_topics) else 'Others'
                rows.append({
                    'review_id'          : rev.id,
                    'sentiment'          : rev.sentiment,
                    'topic'              : topic,
                    'pain_points'        : ', '.join(t.pain_points) or None,
                    'moments_of_delight' : ', '.join(t.moments_of_delight) or None,
                })

        df = pd.DataFrame(rows)
        df.to_csv(batch_file, index=False, encoding='utf-8-sig')
        df_list.append(df)
        print(f'Batch {batch_num}/{total} saved.')

    if df_list:
        combined = pd.concat(df_list, ignore_index=True)
        combined.to_csv(os.path.join(output_dir, 'all_reviews_combined.csv'), index=False, encoding='utf-8-sig')
        print(f'All batches combined: {len(combined)} rows.')
        return combined
    return pd.DataFrame()


if __name__ == '__main__':
    model = make_model(os.environ.get('GEMINI_API_KEY'))
    system_prompt = get_airport_review_prompt(TAXONOMY_PATH)

    print(f'Loading Google airport reviews: {GOOGLE_PATH}')
    reviews = pd.read_csv(GOOGLE_PATH)
    reviews_list = [
        {'content': f"Review_Number: {row['Review_No']}\nRating: {row['score_x']}\nReview: {row['text']}"}
        for _, row in reviews.iterrows()
    ]
    _process_batches(reviews_list, system_prompt, OUTPUT_DIR, model,
                     valid_topics=get_valid_topics(TAXONOMY_PATH, 'AIRPORT'))
    print('Airport sentiment extraction complete.')
