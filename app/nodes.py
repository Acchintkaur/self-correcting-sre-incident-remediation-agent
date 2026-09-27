import os
import atexit
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from app.state import (
    GraphState,
    DocumentGrade,
    GroundednessGrade,
    AnswerHelpfulnessGrade,
    QueryRewrite,
)
from dotenv import load_dotenv

load_dotenv()


llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_eval = ChatOpenAI(model="gpt-4o", temperature=0)

PERSIST_DIR = os.path.join(os.getcwd(), "qdrant_storage")
COLLECTION_NAME = "infra_runbooks"

client = QdrantClient(path=PERSIST_DIR, force_disable_check_same_thread=True)

atexit.register(client.close)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

_client = None
_vectorstore = None
_retriever = None 

def close_client():
    global _client, _vectorstore, _retriever
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None
        _vectorstore = None
        _retriever = None


def get_retriever():
    global _client, _vectorstore, _retriever
    if _retriever is None:
        PERSIST_DIR = os.path.join(os.getcwd(), "qdrant_storage")
        COLLECTION_NAME = "infra_runbooks"
        _client = QdrantClient(path=PERSIST_DIR)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _vectorstore = QdrantVectorStore(client=_client, collection_name=COLLECTION_NAME, embedding=embeddings)
        _retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})
    return _retriever


atexit.register(close_client)


def retrieve(state: GraphState) -> dict:
    """
    Retrieves candidate runbook chunks from Qdrant.
    """
    query = state["question"]
    print(f"\n[NODE: retrieve] Fetching candidate chunks for query: '{query}'")
    
    # retriever_instance = get_retriever()
    docs = retriever.invoke(query)
    doc_texts = [d.page_content for d in docs]
    
    return {"documents": doc_texts}


def grade_documents(state: GraphState) -> dict:
    """
    Evaluator Node (Pre-generation):
    Filters out noisy or irrelevant chunks. Only relevant runbooks proceed.
    """
    question = state["question"]
    documents = state["documents"]
    print(f"[NODE: grade_documents] Grading {len(documents)} retrieved chunks...")

    grader = llm_fast.with_structured_output(DocumentGrade)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict SRE Document Reviewer. Evaluate whether the provided runbook excerpt "
            "contains relevant facts, procedures, or CLI commands to answer the technical query.\n"
            "Return binary_score as 'yes' or 'no' along with your reasoning."
        ),
        ("human", "User Question: {question}\n\nDocument Chunk:\n{document}")
    ])

    valid_documents = []
    for idx, doc in enumerate(documents, 1):
        chain = prompt | grader
        result: DocumentGrade = chain.invoke({"question": question, "document": doc})
        if result.binary_score.lower() == "yes":
            print(f"  -> Chunk {idx}: ACCEPTED ({result.reasoning})")
            valid_documents.append(doc)
        else:
            print(f"  -> Chunk {idx}: DISCARDED ({result.reasoning})")

    return {"documents": valid_documents}


def generate(state: GraphState) -> dict:
    """
    Optimizer/Generator Node:
    Synthesizes the technical response strictly using verified runbook context.
    If prior attempts hallucinated, incorporates evaluator critique for self-correction.
    """
    question = state["original_question"]
    documents = state["documents"]
    feedback = state.get("hallucination_feedback")

    print("[NODE: generate] Generating remediation response...")

    context_str = "\n\n---\n\n".join(documents)

    system_prompt = (
        "You are an expert Site Reliability Engineer (SRE). Answer the incident question "
        "thoroughly, explaining the root cause and providing actionable, step-by-step troubleshooting "
        "commands and explanations based ONLY on the provided context.\n"
        "- Directly and completely answer all aspects of the user's question.\n"
        "- Do NOT invent flags, parameters, or configurations.\n"
        "- If a command or procedure is not explicitly mentioned in the context, do NOT assume it.\n"
        "- Format bash/CLI commands inside standard markdown code blocks."
    )

    # Inject evaluator feedback into the optimizer prompt if this is a correction loop
    if feedback:
        print(f"  [Optimizer Active] Injecting critique into prompt: {feedback}")
        system_prompt += (
            f"\n\nCRITICAL FIX REQUIRED FROM PREVIOUS DRAFT:\n"
            f"The previous generation failed verification with the following critique:\n'{feedback}'\n"
            f"Correct this error strictly using the provided context."
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Runbook Context:\n{context}\n\nIncident Question: {question}")
    ])

    chain = prompt | llm_eval
    response = chain.invoke({"context": context_str, "question": question})

    return {
        "generation": response.content,
        "hallucination_feedback": None  # Reset feedback once consumed
    }


def grade_hallucination(state: GraphState) -> dict:
    """
    Evaluator Node (Post-generation):
    Validates whether every claim/command in the answer is grounded in the retrieved documents.
    """
    documents = state["documents"]
    generation = state["generation"]
    print("[NODE: grade_hallucination] Verifying answer groundedness...")

    grader = llm_eval.with_structured_output(GroundednessGrade)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict Hallucination Auditor for production infrastructure commands.\n"
            "Assess whether the generated response is strictly grounded in the context facts.\n"
            "- If the response includes flags, tools, or procedures NOT in the context, score 'no'.\n"
            "- If all facts and commands are supported, score 'yes'."
        ),
        ("human", "Context Documents:\n{context}\n\nGenerated Response:\n{generation}")
    ])

    chain = prompt | grader
    result: GroundednessGrade = chain.invoke({
        "context": "\n\n".join(documents),
        "generation": generation
    })

    if result.binary_score.lower() == "yes":
        print("  -> Groundedness Check: PASSED (Zero hallucination)")
        return {"hallucination_feedback": None}
    else:
        print(f"  -> Groundedness Check: FAILED! Critique: {result.critique}")
        return {
            "hallucination_feedback": result.critique,
            "retry_count": state.get("retry_count", 0) + 1
        }


def grade_answer_usefulness(state: GraphState) -> dict:
    """
    Evaluator Node (Final Quality):
    Verifies that the generated answer fully addresses the user's specific problem.
    """
    question = state["original_question"]
    generation = state["generation"]
    print("[NODE: grade_answer_usefulness] Checking if answer resolves user question...")

    grader = llm_fast.with_structured_output(AnswerHelpfulnessGrade)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an SRE Incident Manager. Check if the generated solution directly answers "
            "and solves the user's specific incident question. Return 'yes' or 'no'."
        ),
        ("human", "Original Question: {question}\n\nGenerated Solution:\n{generation}")
    ])

    chain = prompt | grader
    result: AnswerHelpfulnessGrade = chain.invoke({
        "question": question,
        "generation": generation
    })

    if result.binary_score.lower() == "yes":
        print("  -> Answer Usefulness Check: PASSED")
        return {"retry_count": state.get("retry_count", 0)}
    else:
        print(f"  -> Answer Usefulness Check: INCOMPLETE. Missing: {result.missing_aspects}")
        return {"retry_count": state.get("retry_count", 0) + 1}


def transform_query(state: GraphState) -> dict:
    """
    Optimizer Node (Search Query Reformulation):
    Rephrases the query using technical domain terms to improve vector search recall.
    """
    current_query = state["question"]
    print(f"\n[NODE: transform_query] Rewriting query '{current_query}' for better recall...")

    rewriter = llm_fast.with_structured_output(QueryRewrite)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an SRE Query Reformulation Expert. The current query failed to retrieve "
            "sufficient relevant runbooks. Rewrite the query to use specific technical terminology, "
            "service names, error patterns, or standard CLI command keywords."
        ),
        ("human", "Failed Query: {query}")
    ])

    chain = prompt | rewriter
    result: QueryRewrite = chain.invoke({"query": current_query})
    print(f"  -> Improved Query: '{result.improved_query}'")

    return {
        "question": result.improved_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "web_fallback": True
    }

