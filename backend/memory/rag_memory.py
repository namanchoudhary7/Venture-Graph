"""
backend/memory/rag_memory.py
 
Stores every completed VC evaluation as a vector embedding in ChromaDB.
Before the research node runs, similar past evaluations are retrieved and
injected into the system prompt as additional context — giving the agent
pattern recognition across hundreds of past ideas.
 
Install:
    pip install chromadb sentence-transformers
"""

import json
import uuid
from datetime import datetime
from typing import Optional
 
import chromadb
from chromadb.utils import embedding_functions

_client = chromadb.PersistentClient(path="./chroma_store")

_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2" 
)

_collection = _client.get_or_create_collection(
    name="vc_evaluations",
    embedding_function=_ef,
    metadata={"hnsw:space": "cosine"},  
)

def store_evaluation(idea: str, report: dict) -> str:
    """
    Embed and store a completed evaluation.
 
    Args:
        idea:   The raw startup idea string.
        report: The VCEvaluationOutput dict (final_report from AgentState).
 
    Returns:
        The unique document ID assigned to this entry.
    """
    doc_id = str(uuid.uuid4())

    _collection.add(
        ids=[doc_id],
        documents=[idea],                       
        metadatas=[{
            "idea":             idea,
            "status":           report.get("status", "UNKNOWN"),
            "confidence_score": str(report.get("confidence_score", 0)),
            "final_verdict":    report.get("final_verdict", "")[:500],  
            "report_json":      json.dumps(report)[:2000],              
            "evaluated_at":     datetime.utcnow().isoformat(),
        }],
    )
    print(f"[RAG] Stored evaluation '{idea[:60]}' → id={doc_id}")
    return doc_id

def retrieve_similar(idea: str, n_results: int = 3) -> list[dict]:
    """
    Retrieve the N most semantically similar past evaluations for a given idea.
 
    Args:
        idea:      The startup idea to search against.
        n_results: How many similar evaluations to return (default 3).
 
    Returns:
        List of dicts with keys: idea, status, confidence_score,
        final_verdict, similarity_score.
    """
    total = _collection.count()
    if total == 0:
        return []   
 
    results = _collection.query(
        query_texts=[idea],
        n_results=min(n_results, total),  
        include=["metadatas", "distances"],
    )
 
    similar = []
    for meta, distance in zip(
        results["metadatas"][0],
        results["distances"][0],
    ):
        similar.append({
            "idea":             meta.get("idea", ""),
            "status":           meta.get("status", ""),
            "confidence_score": int(meta.get("confidence_score", 0)),
            "final_verdict":    meta.get("final_verdict", ""),
            "similarity_score": round(1 - distance, 3), 
        })
 
    return similar

def format_context_for_prompt(similar_evals: list[dict]) -> str:
    """
    Formats retrieved evaluations into a concise block ready to be injected
    into the research node's system prompt.
    """
    if not similar_evals:
        return ""
 
    lines = ["=== SIMILAR PAST EVALUATIONS (use as reference patterns) ==="]
    for i, ev in enumerate(similar_evals, 1):
        lines.append(
            f"\n[{i}] Idea: {ev['idea']}\n"
            f"    Decision: {ev['status']} | Score: {ev['confidence_score']}/100 "
            f"| Similarity: {ev['similarity_score']}\n"
            f"    Verdict: {ev['final_verdict']}"
        )
    lines.append("\n=== END OF PAST EVALUATIONS ===")
    return "\n".join(lines)

def get_collection_stats() -> dict:
    """Returns basic stats about the memory store — useful for the health endpoint."""
    return {
        "total_evaluations": _collection.count(),
        "collection_name":   _collection.name,
        "embedding_model":   "all-MiniLM-L6-v2",
    }