"""
Subtopic Mapping - Airport (Google Reviews).
Maps raw pain points / moments of delight to taxonomy subtopic labels via Gemini.

Run from project root:
    python -m src.data_pipelines.google_maps.subtopic_mapping
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

from src.config import TAXONOMY_PATH, BATCH_DATE
from src.core.llm import resolve_schema, call_and_parse, make_generation_config, make_model
from src.models.sentiment import SubtopicsBatch
from src.utils.helpers import build_subtopic_lists, explode_column
from src.prompts.loader import get_subtopic_mapping_prompt

PHASE1_PATH = f'airport_sentiment/intermediate_sentiments/{BATCH_DATE}_batches/all_reviews_combined.csv'
PP_OUTPUT   = f'airport_sentiment/intermediate_sentiments/{BATCH_DATE}_batches/painpoints'
MOD_OUTPUT  = f'airport_sentiment/intermediate_sentiments/{BATCH_DATE}_batches/mods'


def _categorise(df, allowed_topics, pains_df, delights_df, batch_size,
                model, output_folder, mode='pain_point'):
    os.makedirs(output_folder, exist_ok=True)
    schema = resolve_schema(SubtopicsBatch.model_json_schema())
    cfg = make_generation_config(response_schema=schema)

    label = 'pain point' if mode == 'pain_point' else 'moment of delight'
    subtopic_src = pains_df if mode == 'pain_point' else delights_df

    for topic in allowed_topics:
        topic_df = df[df['topic'] == topic]
        subtopics = subtopic_src[subtopic_src['topic'] == topic]['final_subtopic'].unique().tolist()
        if topic_df.empty or not subtopics:
            continue

        system_prompt = get_subtopic_mapping_prompt('airport', label, subtopics)
        data_list = [
            {'content': f"id: {row.pp_mod_id}\n{label.title()}: {row.pp_mod}"}
            for _, row in topic_df.iterrows()
        ]

        total = (len(data_list) + batch_size - 1) // batch_size
        for bn in range(total):
            out_file = os.path.join(output_folder, f'batch_{bn}_topic_{topic}.csv')
            if os.path.exists(out_file):
                print(f'Skipping: {out_file}')
                continue
            batch = data_list[bn * batch_size:(bn + 1) * batch_size]
            messages = [system_prompt] + [m['content'] for m in batch]
            try:
                result = call_and_parse(model, messages, cfg, SubtopicsBatch)
            except Exception as e:
                log.error('Batch %s topic %s failed: %s', bn, topic, e)
                continue
            pd.DataFrame([{'id': r.id, 'subtopic': r.subtopic} for r in result.reviews]).to_csv(
                out_file, index=False)
            print(f'Saved: {out_file}')


if __name__ == '__main__':
    print(f'Loading sentiment output: {PHASE1_PATH}')
    res = pd.read_csv(PHASE1_PATH)

    taxonomy_df = pd.read_csv(TAXONOMY_PATH)
    allowed = taxonomy_df[taxonomy_df['type'] == 'AIRPORT']['topic'].unique().tolist()
    pains, delights = build_subtopic_lists(taxonomy_df, 'AIRPORT')

    pp  = explode_column(res, 'review_id', 'sentiment', 'topic', 'pain_points',        'Pain Point')
    mod = explode_column(res, 'review_id', 'sentiment', 'topic', 'moments_of_delight', 'Moment of Delight')
    print(f'Pain points: {len(pp)} | Moments of delight: {len(mod)}')

    model = make_model(os.environ.get('GEMINI_API_KEY'))
    _categorise(pp,  allowed, pains, delights, 150, model, PP_OUTPUT,  'pain_point')
    _categorise(mod, allowed, pains, delights, 150, model, MOD_OUTPUT, 'moment_of_delight')
    print('Airport subtopic mapping complete.')
