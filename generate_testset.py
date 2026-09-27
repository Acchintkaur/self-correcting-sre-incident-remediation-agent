# generate_testset.py
import json
import glob
import random
import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


class SyntheticIncidentQA(BaseModel):
    question: str = Field(description="A realistic incident or operational question an SRE/DevOps engineer would ask.")
    question_type: str = Field(description="'direct_command', 'symptom_based', or 'troubleshooting'")
    ground_truth: str = Field(description="The factual remediation steps strictly derived from the context excerpt.")

def generate_benchmark_dataset(num_samples: int = 15):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    generator = llm.with_structured_output(SyntheticIncidentQA)

    files = glob.glob("./data/runbooks/**/*.md", recursive=True)
    if not files:
        raise FileNotFoundError("No markdown files found in ./data/runbooks. Please verify ingestion directory.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Principal SRE creating an evaluation benchmark for an AI Incident Copilot.\n"
            "Given the following runbook excerpt, generate one realistic operational question "
            "and a factual ground-truth answer based ONLY on the text.\n"
            "Vary the question style: some should be direct CLI command queries, others symptom descriptions."
        )),
        ("human", "Runbook Context:\n{context}")
    ])

    testset = []
    print(f"\n[GENERATOR] Sampling {num_samples} documents from runbooks to generate benchmark...")

    random.seed(42)
    # Filter files that have sufficient technical substance
    valid_files = [f for f in files if os.path.getsize(f) > 500]
    sampled_files = random.sample(valid_files, min(num_samples, len(valid_files)))

    for idx, fpath in enumerate(sampled_files, 1):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()

            # Take a 1200-character representative snippet
            excerpt = content[:1500]
            chain = prompt | generator
            qa_pair: SyntheticIncidentQA = chain.invoke({"context": excerpt})

            testset.append({
                "id": idx,
                "question": qa_pair.question,
                "question_type": qa_pair.question_type,
                "ground_truth": qa_pair.ground_truth,
                "source_file": os.path.relpath(fpath, "./data/runbooks")
            })
            print(f"  [{idx}/{num_samples}] Generated QA: {qa_pair.question[:65]}...")
        except Exception as e:
            print(f"Skipping {fpath}: {e}")

    output_path = "benchmark_testset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2)

    print(f"\n[SUCCESS] Generated {len(testset)} benchmark samples -> {os.path.abspath(output_path)}\n")

if __name__ == "__main__":
    generate_benchmark_dataset(num_samples=15)