from __future__ import annotations

from typing import List

from .abstracts import LLMProvider
from .models import Citation, Chunk
from .query_planner import QueryPlanner
from .retrieval import Retriever
from .storage import KnowledgeLibrary


class GroundedQA:
    def __init__(
        self,
        retriever: Retriever,
        query_planner: QueryPlanner,
        llm_provider: LLMProvider,
        debug: bool = False,
    ):
        self.retriever = retriever
        self.query_planner = query_planner
        self.llm_provider = llm_provider
        self.debug = debug

    def answer_query(self, query: str, library: KnowledgeLibrary, top_k: int = 5) -> dict:
        plan = self.query_planner.plan(query)
        chunks = self.retriever.semantic_search(query, top_k=top_k)
        citations = [library.get_citation(chunk) for chunk in chunks]
        prompt = self.build_prompt(query, plan, chunks, citations)
        if self.debug:
            print("[GroundedQA] query plan:", plan)
            print("[GroundedQA] prompt assembled")
        answer = self.llm_provider.generate(prompt)
        return {
            "query": query,
            "plan": plan,
            "answer": answer,
            "citations": [citation.to_dict() for citation in citations],
            "chunks": [chunk.to_dict() for chunk in chunks],
        }

    def build_prompt(self, query: str, plan: str, chunks: List[Chunk], citations: List[Citation]) -> str:
        retrieved_sections = []
        for chunk, citation in zip(chunks, citations):
            citation_marker = f"[CITATION:{citation.citation_id}]"
            retrieved_sections.append(f"{citation_marker} {chunk.text.strip()}\n")
        retrieved_text = "\n---\n".join(retrieved_sections)
        citation_legend = "\n".join(
            f"{citation.citation_id}: {citation.filename or citation.document_id} | page={citation.page_number} | section={citation.section}"
            for citation in citations
        )
        prompt = (
            "You are a grounded research assistant. Answer the user query using only the provided retrieved knowledge chunks. "
            "Do not invent citation metadata. Preserve the distinction between user context, retrieved knowledge, and generated response.\n\n"
            f"Query plan: {plan}\n"
            f"User query: {query}\n\n"
            "Retrieved knowledge:\n"
            f"{retrieved_text}\n\n"
            "Instructions:\n"
            "- Use the retrieved chunks as authoritative evidence.\n"
            "- Include citation markers in your answer where appropriate.\n"
            "- Do not fabricate page numbers, authors, or document metadata.\n"
            "- If the answer is not contained in the provided material, say so clearly.\n\n"
            "Citation metadata:\n"
            f"{citation_legend}\n"
        )
        return prompt
