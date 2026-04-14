"""
Pydantic models for Gemini sentiment extraction responses.
Used by Phase 1 (sentiment extraction) across airlines, airports, and X pipelines.
"""

from typing import List, Literal
from pydantic import BaseModel, Field

Sentiment = Literal["positive", "neutral", "negative"]


class TopicDetails(BaseModel):
    topic: str
    pain_points: List[str] = Field(default_factory=list)
    moments_of_delight: List[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    id: str
    sentiment: Sentiment
    topics: List[TopicDetails]


class ReviewsBatch(BaseModel):
    reviews: List[ReviewResult]


class SubtopicResult(BaseModel):
    id: str
    subtopic: str


class SubtopicsBatch(BaseModel):
    reviews: List[SubtopicResult]
