"""Legal Graph-RAG with the Finetuned model (Project 6).

A self-contained project fusing three components:
  1. the legal system prompt + safety gate (from Prompt Design, Project 1),
  2. Neo4j graph retrieval over the Egyptian Civil Code (Project 3),
  3. answer generation by the finetuned QLoRA knowledge adapter (Project 2).
"""
from .config import get_settings
from .pipeline import LegalGraphRagFinetuned

__all__ = ["get_settings", "LegalGraphRagFinetuned"]
