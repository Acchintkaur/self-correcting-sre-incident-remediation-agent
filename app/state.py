from typing import List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class GraphState(TypedDict):
    """
    Represents the state of our agentic RAG graph.
    """
    question: str              # User's initial or transformed question
    original_question: str     # Preserves original user intent across rewrites
    documents: List[str]       # Document chunks retrieved and filtered
    generation: str            # Final/intermediate generated answer
    retry_count: int           # Tracks retry loops to prevent infinite cycles
    max_retries: int           # Maximum allowable retries
    hallucination_feedback: Optional[str]  # Critique injected into optimizer if hallucinated
    web_fallback: bool         # Flag to trigger fallback web search if docs are insufficient


# -------------------------------------------------------------
# Structured Evaluator Schemas for LLM Function Calling
# -------------------------------------------------------------

class DocumentGrade(BaseModel):
    """Binary score for document chunk relevance evaluation."""
    binary_score: Literal["yes", "no"] = Field(
        description="Score 'yes' if the document contains keywords, facts, or instructions relevant to the question, else 'no'."
    )
    reasoning: str = Field(
        description="One-sentence rationale explaining why the chunk is relevant or irrelevant."
    )


class GroundednessGrade(BaseModel):
    """Binary score to evaluate if generation is grounded in retrieved facts."""
    binary_score: Literal["yes", "no"] = Field(
        description="Score 'yes' if every claim/command in the answer is grounded in the retrieved docs. Score 'no' if it contains hallucinations or ungrounded claims."
    )
    critique: str = Field(
        default="",
        description="If 'no', clearly identify the exact ungrounded commands, flags, or claims that must be corrected."
    )


class AnswerHelpfulnessGrade(BaseModel):
    """Binary score to evaluate if the answer completely resolves the incident query."""
    binary_score: Literal["yes", "no"] = Field(
        description="Score 'yes' if the answer directly addresses and solves the user's specific problem, else 'no'."
    )
    missing_aspects: str = Field(
        default="",
        description="If 'no', what crucial steps or explanations were omitted."
    )


class QueryRewrite(BaseModel):
    """Refined query optimized for search retrieval."""
    improved_query: str = Field(
        description="An optimized search query focusing on service names, error messages, and operational keywords."
    )

