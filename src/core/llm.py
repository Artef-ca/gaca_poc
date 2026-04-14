"""
Shared Gemini LLM helpers: schema resolution, JSON cleaning, call-and-parse with retry.
Pydantic models live in src/models/ — import them from there.
"""

import re
import time
import logging
from pydantic import ValidationError
from google import genai
from google.genai import types

log = logging.getLogger(__name__)


class _GeminiModel:
    """Thin wrapper around genai.Client that preserves the old generate_content() interface."""
    def __init__(self, model_name: str, api_key: str):
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def generate_content(self, contents, generation_config=None):
        return self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=generation_config,
        )


def make_model(api_key: str) -> _GeminiModel:
    """Instantiate a Gemini model using the new google-genai SDK."""
    from src.config import MODEL_NAME
    return _GeminiModel(MODEL_NAME, api_key)


def resolve_schema(schema: dict) -> dict:
    """Inline $defs and strip unsupported fields so Gemini can parse the schema."""
    STRIP = {'$defs', 'title'}
    defs = schema.pop('$defs', {})

    def _resolve(obj):
        if isinstance(obj, dict):
            if '$ref' in obj:
                return _resolve(defs[obj['$ref'].split('/')[-1]].copy())
            return {k: _resolve(v) for k, v in obj.items() if k not in STRIP}
        if isinstance(obj, list):
            return [_resolve(i) for i in obj]
        return obj

    return _resolve(schema)


def clean_json(txt: str) -> str:
    txt = txt.strip()
    if txt.startswith('```'):
        txt = re.sub(r'^```[a-z]*\n|\n```$', '', txt, flags=re.I | re.S).strip()
    return txt


def call_and_parse(model, messages, cfg, Schema, retries: int = 3, backoff: float = 2.0):
    """Call Gemini, parse JSON response into Schema, retry on truncation or transient errors."""
    for attempt in range(1, retries + 1):
        try:
            resp = model.generate_content(messages, generation_config=cfg)
            return Schema.model_validate_json(clean_json(resp.text or ''))
        except ValidationError as ve:
            if 'EOF while parsing' in str(ve) and attempt < retries:
                log.warning('JSON truncated. Retrying (%s/%s).', attempt + 1, retries)
            else:
                raise
        except Exception as e:
            if attempt == retries:
                raise
            log.warning('Model call failed (%s). Retrying in %.1fs (%s/%s).', e, backoff, attempt + 1, retries)
        time.sleep(backoff)


def make_generation_config(response_schema=None, max_output_tokens: int = 60000):
    return types.GenerateContentConfig(
        response_mime_type='application/json',
        temperature=0,
        max_output_tokens=max_output_tokens,
        response_schema=response_schema,
    )
