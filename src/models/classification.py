"""
Pydantic models for X/Twitter tweet classification responses.
Used by the X data enrichment pipeline to label tweets as AIRLINE/AIRPORT/OTHER
and identify the specific entity (saudia, flynas, RUH, etc.).
"""

from typing import List, Literal
from pydantic import BaseModel

Label  = Literal['AIRLINE', 'AIRPORT', 'OTHER']
Entity = Literal['saudia', 'flynas', 'flyadeal', 'RUH', 'JED', 'DMM', 'Unknown']


class TweetClassification(BaseModel):
    id: str
    label: Label
    main_entity: Entity


class BatchClassification(BaseModel):
    tweets: List[TweetClassification]
