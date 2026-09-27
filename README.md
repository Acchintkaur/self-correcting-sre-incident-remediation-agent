# Autonomous SRE Incident Remediation Copilot

An enterprise-grade, self-correcting Retrieval-Augmented Generation (RAG) engine designed for infrastructure incident response and Site Reliability Engineering (SRE) operational playbooks.

Unlike traditional naive RAG architectures that suffer from context hallucinations and silent failures, this system implements an Evaluator-Optimizer State Machine with closed-loop verification, ensuring that mission-critical CLI commands and recovery steps are strictly grounded in verified runbooks before reaching the operator.

---
## Architecture and State Machine
![alt text](image.png)

## Core Capabilities
- Two-Tier Model Routing: High-throughput lightweight models (gpt-4o-mini) for pre-generation chunk filtering and intent checks, with deep reasoning models (gpt-4o) reserved for synthesis and hallucination audits.
- Deterministic Pre-Generation Filtering: Discards noisy or adjacent service chunks using strict Pydantic scoring before context enters the generation prompt.
- Autonomous Query Reformulation: Rewrites symptom-based or colloquial queries into technical domain keywords when initial vector search recall is low.
- Evaluator-Optimizer Feedback Loop: Programmatically extracts hallucination critiques and injects them back into the generator node for automated self-correction.
- Real-Time Telemetry via SSE: Asynchronous FastAPI backend streaming intermediate state transitions and payload events over Server-Sent Events.


## Benchmark and Evaluation

The engine was evaluated using an automated LLM-as-a-Judge scoring harness across 15 synthetic real-world infrastructure scenarios (PostgreSQL failovers, HAProxy connection saturation, Redis memory pressure, and connection pool exhaustion):

| Metric | Score | Target | Description |
| :---- | :---: | :---: | :---- |
| Answer Relevance | 0.840 | \> 0.80 | Semantic alignment, completeness, and directness in resolving the incident query (+10% post prompt tuning). |
| Faithfulness | 0.800 | \> 0.80 | Fraction of claims and CLI commands strictly supported by the source runbooks. |
| Context Precision | 0.800 | \> 0.80 | Proportion of relevant runbook passages ranked at the top of retrieved context. |
| Self-Correction Loops | 9 Resolved | N/A | Total autonomous retry loops triggered and resolved without human intervention. |

---

## Repository Layout 

```
self_correcting_rag/
|-- app/
|   |-- __init__.py
|   |-- state.py
|   |-- nodes.py
|   |-- workflow.py
|   |-- main.py
|-- data/
|   |-- runbooks/
|-- tests/
|   |-- __init__.py
|   |-- test_client.py
|   |-- test_graph.py
|   |-- test_search.py
|-- ingest.py
|-- generate_testset.py
|-- evaluate.py
|-- benchmark_results.csv
|-- benchmark_testset.json
|-- .env.example
|-- .gitignore
|-- requirements.txt
|-- README.md
```

## Getting Started

### 1. Clone and Set Up Virtual Environment
```
git clone <your-repository-url>

cd self-correcting-incident-copilot

python -m venv .venv
```

#### Windows (PowerShell)
```
.\.venv\Scripts\activate
```

#### Linux / macOS
```
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Variables

Copy `.env.example` to `.env` and set your API key:
```
cp .env.example .env
```
Inside `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Ingest Runbooks

Parse and index the runbooks into the local persistent Qdrant vector store:
```
python ingest.py
```

### 4. Run the API Server

Start the asynchronous FastAPI server:
```
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Interactive OpenAPI documentation will be accessible at `/docs` on the host.

---

## API Usage and Streaming

### Stream Incident Remediation

`POST /api/v1/incident/stream`

#### Request Payload
```
{
    "query": "What is the procedure to failover a Patroni PostgreSQL leader node safely?",
    "max_retries": 2
}
```

#### Test via Client Script

Run the test client in another terminal:
```
python tests/test_client.py
```

#### Sample SSE Stream Output
```
data: {"event": "started", "query": "What is the procedure to failover a Patroni PostgreSQL leader node safely?"}

data: {"event": "node_complete", "node": "retrieve", "retries": 0, "message": "Retrieved 5 candidate chunks from Qdrant."}

data: {"event": "node_complete", "node": "grade_documents", "retries": 0, "message": "Evaluation complete: 1 verified relevant chunks retained."}

data: {"event": "node_complete", "node": "generate", "retries": 0, "message": "Remediation instructions generated from verified context."}

data: {"event": "answer_payload", "answer": "To safely failover a Patroni PostgreSQL leader node:\n\n1. Connect via SSH to a replica node...\n2. Run `sudo gitlab-patronictl list`...\n3. Execute \`sudo gitlab-patronictl failover`..."}

data: {"event": "node_complete", "node": "grade_hallucination", "retries": 0, "message": "Zero hallucinations detected. Output verified strictly against source context."}

data: {"event": "node_complete", "node": "grade_answer_usefulness", "retries": 0, "message": "Answer intent verified against original query."}

data: {"event": "done"}
```
---

## Evaluation Suite

To reproduce the benchmark or evaluate against custom scenarios:

### 1. Generate synthetic incident QA benchmark triplets
```
python generate_testset.py
```

### 2. Run the quantitative LLM-as-a-judge evaluation harness
```
python evaluate.py
```
Scoring outputs and full reasoning breakdowns will be exported to `benchmark_results.csv`.
