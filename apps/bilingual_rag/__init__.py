"""Bilingual RAG over the Egyptian Civil Code (Project 4)."""
from .config import BRagSettings, get_settings
from .pipeline import BilingualRAGPipeline

__all__ = ["BRagSettings", "get_settings", "BilingualRAGPipeline"]
