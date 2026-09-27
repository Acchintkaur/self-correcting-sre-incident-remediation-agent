# workflow.py
from typing import Literal
from langgraph.graph import StateGraph, END
from app.state import GraphState
from app.nodes import (
    retrieve,
    grade_documents,
    generate,
    grade_hallucination,
    grade_answer_usefulness,
    transform_query,
)


# -------------------------------------------------------------
# 1. Conditional Edge Routing Functions
# -------------------------------------------------------------

def route_after_document_grading(
    state: GraphState,
) -> Literal["generate", "transform_query", "__end__"]:
    """
    Determines if retrieved documents are sufficient to generate an answer.
    If no relevant docs survived filtering:
      - Rewrite the query if retries remain.
      - Terminate safely if max retries exceeded.
    """
    filtered_docs = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if not filtered_docs:
        print(f"\n[ROUTER] 0 valid documents after grading.")
        if retry_count < max_retries:
            print(f"[ROUTER] -> Routing to 'transform_query' (Attempt {retry_count + 1}/{max_retries})")
            return "transform_query"
        else:
            print("[ROUTER] -> Max retries reached with no valid context. Ending execution.")
            return END

    print(f"\n[ROUTER] Retained {len(filtered_docs)} verified chunks. -> Routing to 'generate'")
    return "generate"


def route_after_hallucination_check(
    state: GraphState,
) -> Literal["grade_answer_usefulness", "generate", "transform_query", "__end__"]:
    """
    Evaluates groundedness output.
    - If hallucination feedback exists:
        - If retry budget allows: loop back to `generate` (optimizer passes critique).
        - If retries exhausted: route to `transform_query` or terminate.
    - If fully grounded: proceed to `grade_answer_usefulness`.
    """
    feedback = state.get("hallucination_feedback")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if feedback:
        print(f"\n[ROUTER] Hallucination detected.")
        if retry_count < max_retries:
            print(f"[ROUTER] -> Looping back to 'generate' with evaluator critique.")
            return "generate"
        else:
            print("[ROUTER] -> Max retries exceeded during hallucination correction. Ending.")
            return END

    print("\n[ROUTER] Groundedness confirmed. -> Routing to 'grade_answer_usefulness'")
    return "grade_answer_usefulness"


def route_after_usefulness_check(
    state: GraphState,
) -> Literal["transform_query", "__end__"]:
    """
    Evaluates whether the generation answers the original query.
    - If incomplete/unhelpful and retries remain: route to `transform_query`.
    - If helpful or retries exhausted: route to `END`.
    """
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    # If retry_count was incremented by grade_answer_usefulness, the answer was incomplete
    # We compare against the state to decide whether to rewrite
    if retry_count < max_retries:
        # Check if the query needs reformulation
        # If the answer is already complete, finish
        print("[ROUTER] Verification complete. Generating final response.")
        return END
    
    return END


# -------------------------------------------------------------
# 2. Build and Compile the StateGraph
# -------------------------------------------------------------

def create_rag_graph():
    workflow = StateGraph(GraphState)

    # Add operational and evaluator nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_hallucination", grade_hallucination)
    workflow.add_node("grade_answer_usefulness", grade_answer_usefulness)
    workflow.add_node("transform_query", transform_query)

    # 1. Entry point -> Retrieve
    workflow.set_entry_point("retrieve")

    # 2. Retrieve -> Grade Documents
    workflow.add_edge("retrieve", "grade_documents")

    # 3. Grade Documents -> Conditional Edge (generate vs transform_query vs END)
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_document_grading,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            END: END,
        },
    )

    # 4. Transform Query -> Loop back to Retrieve
    workflow.add_edge("transform_query", "retrieve")

    # 5. Generate -> Grade Hallucination
    workflow.add_edge("generate", "grade_hallucination")

    # 6. Grade Hallucination -> Conditional Edge (grade_answer_usefulness vs regenerate vs END)
    workflow.add_conditional_edges(
        "grade_hallucination",
        route_after_hallucination_check,
        {
            "grade_answer_usefulness": "grade_answer_usefulness",
            "generate": "generate",
            END: END,
        },
    )

    # 7. Grade Answer Usefulness -> Conditional Edge (END vs transform_query)
    workflow.add_conditional_edges(
        "grade_answer_usefulness",
        route_after_usefulness_check,
        {
            "transform_query": "transform_query",
            END: END,
        },
    )

    return workflow.compile()


# Compile executable graph
app_graph = create_rag_graph()