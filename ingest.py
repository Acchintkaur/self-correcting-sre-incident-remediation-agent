import os
import glob 
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = os.path.join(os.getcwd(), "data", "runbooks")
PERSIST_DIR = os.path.join(os.getcwd(), "qdrant_storage")
COLLECTION_NAME = "infra_runbooks"

def load_and_chunk_runbooks():
    md_files = glob.glob(os.path.join(DOCS_DIR, "**", "*.md"), recursive=True)
    md_files += glob.glob(os.path.join(DOCS_DIR, "**", "*.markdown"), recursive=True)

    print(f"Discovered {len(md_files)} markdown files across runbook folders.")

    headers_to_split = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n```", "\n\n", "\n", " ", ""]
    )

    all_processed_docs = []

    for file_path in md_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            
            if not content:
                continue

            # Extract relative folder hierarchy as service metadata
            rel_path = os.path.relpath(file_path, DOCS_DIR)
            path_parts = rel_path.split(os.sep)
            service_name = path_parts[0] if len(path_parts) > 1 else "general"

            # 1. Structural Markdown Header Splitting
            header_splits = markdown_splitter.split_text(content)
            
            # 2. Token-bounded sub-splitting
            chunks = text_splitter.split_documents(header_splits)

            for chunk in chunks:
                chunk.metadata["service"] = service_name
                chunk.metadata["filename"] = os.path.basename(file_path)
                chunk.metadata["relative_path"] = rel_path
                all_processed_docs.append(chunk)

        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

    print(f"Generated {len(all_processed_docs)} granular semantic chunks.")
    return all_processed_docs

def build_qdrant_index():
    docs = load_and_chunk_runbooks()
    if not docs:
        print("No documents found to index. Check your data/runbooks folder.")
        return 

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Embedding and persisting to local Qdrant collection...")
    qdrant = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        path=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"Ingestion complete. Database saved to: {PERSIST_DIR}")

if __name__ == "__main__":
    build_qdrant_index()