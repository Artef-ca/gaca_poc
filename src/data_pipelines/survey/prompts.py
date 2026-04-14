"""
Pydantic models for the GACA survey pipeline.
Prompts are in config/survey_prompts.yaml — load them via src/prompts/loader.py.
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class TopicDetailsGem(BaseModel):
    topic: str
    pain_points: List[str] = Field(default_factory=list)
    moments_of_delight: List[str] = Field(default_factory=list)


class IndividualReviewGem(BaseModel):
    id: str
    sentiment: Literal["positive", "neutral", "negative"]
    topics: List[TopicDetailsGem]


class ReviewsAnalyzerGem(BaseModel):
    comments: List[IndividualReviewGem]


class PainPointCategorization(BaseModel):
    pain_point: str = Field(description="The pain point identified from the survey comments.")
    topic: str = Field(description="The current topic under which the pain point was categorized.")
    subtopic: str = Field(description="The most appropriate subtopic for this pain point.")


class PainPointAnalyzer(BaseModel):
    reviews: List[PainPointCategorization]


class MODCategorization(BaseModel):
    mod: str = Field(description="The moment of delight identified from the survey comments.")
    topic: str = Field(description="The current topic under which the moment of delight was categorized.")
    subtopic: str = Field(description="The most appropriate subtopic for this moment of delight.")


class MODAnalyzer(BaseModel):
    reviews: List[MODCategorization]
