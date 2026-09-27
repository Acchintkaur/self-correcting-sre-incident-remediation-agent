# evaluate.py
import os
import json
import csv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from app.workflow import app_graph
import app.nodes as nodes

load_dotenv()

# -------------------------------------------------------------
# 1. Pydantic Models for LLM-as-a-Judge Scoring
# -------------------------------------------------------------
class FaithfulnessJudgement(BaseModel):
    reasoning: str = Field(description="Explanation of whether every claim in the answer is found in the context.")
    score: float = Field(description="Float between 0.0 and 1.0 representing proportion of grounded claims.")

class AnswerRelevanceJudgement(BaseModel):
    reasoning: str = Field(description="Explanation of whether the answer directly and completely solves the prompt.")
    score: float = Field(description="Float between 0.0 and 1.0 representing semantic relevance to query.")

class ContextPrecisionJudgement(BaseModel):
    reasoning: str = Field(description="Evaluation of whether retrieved chunks are relevant to the question.")
    score: float = Field(description="Float between 0.0 and 1.0 representing precision of retrieved documents.")


# -------------------------------------------------------------
# 2. Main Evaluation Pipeline
# -------------------------------------------------------------
def run_evaluation(dataset_path: str = "benchmark_testset.json"):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing {dataset_path}. Run generate_testset.py first!")

    with open(dataset_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

    faithfulness_grader = judge_llm.with_structured_output(FaithfulnessJudgement)
    relevance_grader = judge_llm.with_structured_output(AnswerRelevanceJudgement)
    precision_grader = judge_llm.with_structured_output(ContextPrecisionJudgement)

    results = []
    total = len(benchmark_data)

    print(f"\n{'='*70}")
    print(f"  STARTING BENCHMARK EVALUATION ({total} INCIDENT SAMPLES)")
    print(f"{'='*70}")

    for idx, item in enumerate(benchmark_data, start=1):
        q = item["question"]
        q_type = item.get("question_type", "general")
        print(f"\n[{idx}/{total}] ({q_type.upper()}) Query: {q}")

        initial_state = {
            "question": q,
            "original_question": q,
            "documents": [],
            "generation": "",
            "retry_count": 0,
            "max_retries": 2,
            "hallucination_feedback": None,
            "web_fallback": False,
        }

        # 1. Execute through LangGraph State Machine
        final_state = app_graph.invoke(initial_state)

        answer = final_state.get("generation", "")
        contexts = final_state.get("documents", [])
        context_str = "\n\n".join(contexts) if contexts else "None"
        retries_used = final_state.get("retry_count", 0)

        # 2. Faithfulness Metric
        f_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an automated evaluation judge. Calculate Faithfulness (0.0 to 1.0): what fraction of claims in the generated answer are directly supported by the context without hallucination?"),
            ("human", "Context:\n{context}\n\nGenerated Answer:\n{answer}")
        ])
        f_eval: FaithfulnessJudgement = (f_prompt | faithfulness_grader).invoke({
            "context": context_str,
            "answer": answer
        })

        # 3. Answer Relevance Metric
        r_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an automated evaluation judge. Calculate Answer Relevance (0.0 to 1.0): how directly, accurately, and completely does the generated answer resolve the user prompt?"),
            ("human", "User Question:\n{question}\n\nGenerated Answer:\n{answer}")
        ])
        r_eval: AnswerRelevanceJudgement = (r_prompt | relevance_grader).invoke({
            "question": q,
            "answer": answer
        })

        # 4. Context Precision Metric
        p_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an automated evaluation judge. Calculate Context Precision (0.0 to 1.0): were the retrieved chunks actually relevant to the operational question?"),
            ("human", "Question:\n{question}\n\nRetrieved Chunks:\n{context}")
        ])
        p_eval: ContextPrecisionJudgement = (p_prompt | precision_grader).invoke({
            "question": q,
            "context": context_str
        })

        row = {
            "id": idx,
            "question_type": q_type,
            "question": q,
            "retries_used": retries_used,
            "faithfulness": round(f_eval.score, 3),
            "answer_relevance": round(r_eval.score, 3),
            "context_precision": round(p_eval.score, 3),
            "f_reason": f_eval.reasoning,
            "r_reason": r_eval.reasoning
        }
        results.append(row)
        print(f"  -> Scores: Faithfulness: {row['faithfulness']} | Relevance: {row['answer_relevance']} | Precision: {row['context_precision']} | Retries: {retries_used}")

    nodes.close_client()

    # -------------------------------------------------------------
    # Compute Aggregate Metrics
    # -------------------------------------------------------------
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["answer_relevance"] for r in results) / len(results)
    avg_prec = sum(r["context_precision"] for r in results) / len(results)
    total_retries = sum(r["retries_used"] for r in results)

    print("\n" + "=" * 70)
    print("                      AGGREGATE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Evaluated Samples:   {len(results)}")
    print(f"Average Faithfulness:      {avg_faith:.3f}  (Target: > 0.90)")
    print(f"Average Answer Relevance:  {avg_rel:.3f}  (Target: > 0.85)")
    print(f"Average Context Precision: {avg_prec:.3f}  (Target: > 0.85)")
    print(f"Self-Correction Retries:   {total_retries} loop(s) triggered and resolved")
    print("=" * 70)

    # Export to CSV
    csv_file = "benchmark_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nItemized CSV report exported to: {os.path.abspath(csv_file)}\n")


if __name__ == "__main__":
    run_evaluation()