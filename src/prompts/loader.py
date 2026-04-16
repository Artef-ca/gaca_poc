"""
Loads prompts from config/prompts.yaml and config/survey_prompts.yaml and injects runtime values.
All pipeline scripts import from here — never hardcode prompts in Python files.
"""

import os
import yaml
import pandas as pd

_BASE = os.path.join(os.path.dirname(__file__), '../../config')

with open(os.path.join(_BASE, 'prompts.yaml'), 'r', encoding='utf-8') as f:
    _PROMPTS = yaml.safe_load(f)

with open(os.path.join(_BASE, 'survey_prompts.yaml'), 'r', encoding='utf-8') as f:
    _SURVEY_PROMPTS = yaml.safe_load(f)


def _inject(template: str, **kwargs) -> str:
    """Safe placeholder replacement — avoids str.format() choking on JSON examples."""
    for key, value in kwargs.items():
        template = template.replace('{' + key + '}', value)
    return template


def _load_topics(taxonomy_path: str, entity_type: str) -> str:
    taxonomy = pd.read_csv(taxonomy_path)
    topics = taxonomy[taxonomy['type'] == entity_type]['topic'].unique().tolist() + ['Others']
    return ', '.join(topics)


def get_valid_topics(taxonomy_path: str, entity_type: str) -> frozenset:
    """Return the set of valid topic names for the given entity type, including 'Others'."""
    taxonomy = pd.read_csv(taxonomy_path)
    return frozenset(taxonomy[taxonomy['type'] == entity_type]['topic'].unique()) | {'Others'}


def get_airline_review_prompt(taxonomy_path: str) -> str:
    return _inject(_PROMPTS['airline']['review_extraction'], topics=_load_topics(taxonomy_path, 'AIRLINE'))


def get_airline_tweet_prompt(taxonomy_path: str) -> str:
    return _inject(_PROMPTS['airline']['tweet_extraction'], topics=_load_topics(taxonomy_path, 'AIRLINE'))


def get_airport_review_prompt(taxonomy_path: str) -> str:
    return _inject(_PROMPTS['airport']['review_extraction'], topics=_load_topics(taxonomy_path, 'AIRPORT'))


def get_airport_tweet_prompt(taxonomy_path: str) -> str:
    return _inject(_PROMPTS['airport']['tweet_extraction'], topics=_load_topics(taxonomy_path, 'AIRPORT'))


def get_x_classification_prompt() -> str:
    return _PROMPTS['tweet_classification']


def get_subtopic_mapping_prompt(entity_type: str, item_type: str, subtopics: list) -> str:
    """
    entity_type: 'airline' or 'airport'
    item_type: 'pain point' or 'moment of delight'
    subtopics: list of allowed subtopic strings
    """
    return _inject(_PROMPTS['subtopic_mapping'][entity_type.lower()],
                   type=item_type, subtopics=', '.join(subtopics))


# ── Survey prompts ────────────────────────────────────────────────────────────

def get_survey_first_pass_prompt() -> str:
    """Initial sentiment pass — model induces topics from the batch."""
    return _SURVEY_PROMPTS['sentiment']['first_pass']


def get_survey_second_pass_prompt(topics: list) -> str:
    """Subsequent sentiment passes — model reuses or extends discovered topics."""
    return _inject(_SURVEY_PROMPTS['sentiment']['second_pass'], topics=', '.join(topics))


def get_survey_subtopic_map_prompt(item_plural: str, item_singular: str) -> str:
    """
    Maps pain points / moments of delight to the fixed subtopics list from the yaml.
    item_plural:   e.g. 'pain points' or 'moments of delight'
    item_singular: e.g. 'pain point'  or 'moment of delight'
    """
    subtopics = ', '.join(_SURVEY_PROMPTS['subtopics_list'])
    return _inject(_SURVEY_PROMPTS['subtopic_mapping']['map_to_fixed'],
                   item_plural=item_plural, item_singular=item_singular, subtopics=subtopics)


def get_survey_subtopics_list() -> list:
    """Return the predefined survey subtopics list from the yaml."""
    return list(_SURVEY_PROMPTS['subtopics_list'])
