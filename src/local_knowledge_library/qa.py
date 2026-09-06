from __future__ import annotations

from typing import List

from .abstracts import LLMProvider
from .models import Citation, Chunk, RecipeFact, make_recipe_fact_citation_id
from .providers.ollama_providers import DummyLLMProvider
from .query_planner import QueryPlanner
from .recipe_extraction import AggregateQueryPlan, interpret_aggregate_query
from .retrieval import Retriever
from .storage import KnowledgeLibrary

# A real full-book run (211 recipes, qwen2:1.5b) produced 8-way and 16-way
# ties at the minimum ingredient/step count - real recipes tying that widely
# is implausible; it's a sign of the small local model under-extracting on
# some of them, not that they're genuinely identical. Above this many
# winners, say so rather than presenting the list as precise.
_WIDE_TIE_CAVEAT_THRESHOLD = 3


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

        # Superlative/aggregate questions ("fewest ingredients", "cheapest")
        # ask for a MIN/MAX over a fact computed across the whole library -
        # no amount of top-k similarity search answers that, so this branches
        # to a real computation over RecipeFact data instead. Checked
        # independently of `plan` (which is just a cosmetic label for the
        # prompt) so this can't be silently skipped by a query that doesn't
        # happen to match the planner's own separate keyword list.
        aggregate_plan = interpret_aggregate_query(query)
        if aggregate_plan is not None:
            return self._answer_aggregate_query(query, plan, aggregate_plan, library)

        # hybrid_search falls back to pure semantic results when no
        # keyword_searcher is configured (e.g. DummyEmbedder-only test
        # setups), so this is safe even where the retriever wasn't built
        # with keyword search wired in.
        chunks = self.retriever.hybrid_search(query, top_k=top_k)
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
            # So the GUI never mistakes a DummyLLMProvider fallback (no real
            # Ollama reachable) for a real answer - see docs/how_to_use.md's
            # long-standing "no field distinguishing real from dummy" gap.
            "answer_source": "dummy" if isinstance(self.llm_provider, DummyLLMProvider) else "llm",
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

    def _answer_aggregate_query(
        self, query: str, plan: str, aggregate_plan: AggregateQueryPlan, library: KnowledgeLibrary
    ) -> dict:
        # "computed" (not "llm"/"dummy") for every path below: these answers
        # are either a deterministic Python MIN/MAX or a fixed refusal
        # string, never LLM-generated - a real third answer_source, not a
        # stand-in for one of the other two.
        if not aggregate_plan.answerable:
            answer = (
                "I can't answer that from the ingested material: cost/price isn't tracked in the "
                "source text, so any number I gave would be a guess, not a grounded citation."
            )
            return {
                "query": query, "plan": plan, "answer": answer,
                "citations": [], "chunks": [], "answer_source": "computed",
            }

        facts = library.list_recipe_facts()
        if not facts:
            answer = (
                "This library doesn't have recipe data extracted yet. Enable recipe extraction in "
                "this library's settings and re-ingest, then ask again."
            )
            return {
                "query": query, "plan": plan, "answer": answer,
                "citations": [], "chunks": [], "answer_source": "computed",
            }

        def metric_value(fact: RecipeFact) -> int:
            return fact.ingredient_count if aggregate_plan.metric == "ingredient_count" else fact.step_count

        target_value = (
            min(metric_value(fact) for fact in facts)
            if aggregate_plan.direction == "min"
            else max(metric_value(fact) for fact in facts)
        )
        winners = [fact for fact in facts if metric_value(fact) == target_value]

        metric_label = "ingredients" if aggregate_plan.metric == "ingredient_count" else "steps"
        superlative = "fewest" if aggregate_plan.direction == "min" else "most"
        descriptions = [f"{fact.recipe_name} ({metric_value(fact)} {metric_label})" for fact in winners]
        if len(winners) == 1:
            answer = f"The recipe with the {superlative} {metric_label} is {descriptions[0]}."
        else:
            answer = f"{len(winners)} recipes are tied for the {superlative} {metric_label}: " + "; ".join(
                descriptions
            ) + "."
            if len(winners) > _WIDE_TIE_CAVEAT_THRESHOLD:
                answer += (
                    f" A tie this wide is often a sign the extraction under-counted {metric_label} "
                    "on some of these recipes rather than them being genuinely identical - treat this "
                    "as a starting point to check by hand, not a precise ranking."
                )

        citations = [self._citation_for_fact(fact, library) for fact in winners]
        chunks = [self._pseudo_chunk_for_fact(fact) for fact in winners]
        return {
            "query": query,
            "plan": plan,
            "answer": answer,
            "citations": [citation.to_dict() for citation in citations],
            "chunks": [chunk.to_dict() for chunk in chunks],
            "answer_source": "computed",
        }

    def _citation_for_fact(self, fact: RecipeFact, library: KnowledgeLibrary) -> Citation:
        source = library.sources.get(fact.source_id)
        return Citation(
            citation_id=make_recipe_fact_citation_id(fact),
            library_id=fact.library_id,
            source_id=fact.source_id,
            document_id=fact.document_id,
            chunk_id="",
            page_number=fact.page_number,
            section=fact.recipe_name,
            filename=source.filename if source else None,
            file_type=source.file_type if source else None,
        )

    def _pseudo_chunk_for_fact(self, fact: RecipeFact) -> Chunk:
        """The GUI's "raw retrieved chunks" view expects Chunk-shaped data;
        reusing it here surfaces the recipe's real source excerpt as evidence
        with no GUI changes needed."""
        citation_id = make_recipe_fact_citation_id(fact)
        return Chunk(
            chunk_id=citation_id,
            library_id=fact.library_id,
            source_id=fact.source_id,
            document_id=fact.document_id,
            text=fact.source_excerpt,
            metadata={"page_number": str(fact.page_number or ""), "section": fact.recipe_name},
            citation_id=citation_id,
        )
