from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class ValidationStatus(str, Enum):
    VALIDATE = "VALIDATE"
    NEEDS_WORK = "NEEDS_WORK"
    REJECT = "REJECT"

class StartupIdeaInput(BaseModel):
    idea: str = Field(..., description="The raw startup idea or pitch provided by the user.")

class Competitor(BaseModel):
    name: str
    pricing_model: str | None = None
    core_features: list[str] = Field(default_factory=list)

class MarketAssessment(BaseModel):
    competitors: list[Competitor]
    market_saturation_warning: bool
    summary: str

class TechFeasibility(BaseModel):
    github_repos_found: int
    average_stars: Optional[int] = None
    is_buildable: bool
    tech_stack_summary: str

class VCEvaluationOutput(BaseModel):
    status: ValidationStatus
    confidence_score: int = Field(ge=0, le=100, description="0-100 confidence in the assessment")
    market_assessment: MarketAssessment
    technical_feasibility: TechFeasibility
    developer_sentiment: str = Field(..., description="Summary of Hacker News sentiment")
    final_verdict: str = Field(..., description="The definitive reasoning from the VC Agent")