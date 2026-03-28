import os
import re

from db.queries import search_chunks_by_embedding, search_chunks_by_text, get_neighbor_chunks


async def expand_query(question: str) -> list:
    """Generate multiple search queries from the user's question using Groq."""
    if not os.environ.get("GROQ_API_KEY"):
        return [question]

    try:
        from services.llm import generate
        response_text = generate(
            f"""You are an ATPL aviation expert. Given this question, generate 3 different search queries
that would help find relevant information across ATPL study material.
Include different angles, synonyms, and related aviation concepts.

Question: {question}

Reply with exactly 3 queries, one per line. No numbering, no explanations."""
        )
        queries = [q.strip() for q in response_text.strip().split("\n") if q.strip()]
        return queries[:3]
    except Exception:
        return [question]


async def retrieve_chunks(question: str) -> list:
    """Hybrid retrieval: vector similarity + keyword search."""

    queries = await expand_query(question)
    queries.append(question)

    seen_ids = set()
    all_chunks = []

    # Vector search for each query — more results per query
    for query in queries:
        results = search_chunks_by_embedding(query, top_k=8)
        for chunk in results:
            if chunk["id"] not in seen_ids and chunk.get("similarity", 0) > 0.3:
                seen_ids.add(chunk["id"])
                all_chunks.append(chunk)

    # Keyword search — more results
    keyword_results = search_chunks_by_text(question, top_k=10)
    for chunk in keyword_results:
        if chunk["id"] not in seen_ids:
            seen_ids.add(chunk["id"])
            all_chunks.append(chunk)

    # Sort by similarity (best first), take top 20
    all_chunks.sort(key=lambda c: c.get("similarity", 0), reverse=True)

    # Fetch neighbor chunks for context continuity
    enriched = []
    for chunk in all_chunks[:20]:
        neighbors = get_neighbor_chunks(chunk["document_id"], chunk["chunk_index"])
        neighbor_text = "\n".join(n["content"] for n in neighbors if n["id"] != chunk["id"])
        chunk["context"] = neighbor_text[:1500]
        enriched.append(chunk)

    return enriched


def build_context(chunks: list) -> str:
    """Assemble retrieved chunks into structured context."""
    if not chunks:
        return "No relevant material found."

    by_subject = {}
    for chunk in chunks:
        subj = chunk.get("subject", "Unknown")
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(chunk)

    context = ""
    for subject, subj_chunks in by_subject.items():
        context += f"\n\n=== {subject} ===\n"
        for chunk in subj_chunks:
            chapter = chunk.get('chapter', '').strip()
            section = chunk.get('section', '').strip()
            ref_parts = [p for p in [chapter, section] if p]
            ref_label = " > ".join(ref_parts) if ref_parts else "General"
            ref = f"[{ref_label}, pages {chunk.get('page_start', '?')}-{chunk.get('page_end', '?')}]"
            context += f"\n{ref}\n{chunk['content']}\n"
            if chunk.get("context"):
                context += f"\n[Surrounding material:]\n{chunk['context']}\n"

    return context


SYSTEM_PROMPT = """You are an expert ATPL (Airline Transport Pilot License) instructor helping a pilot study.

YOUR JOB: Answer the question using ONLY the study material provided below. Synthesize information from multiple sections when needed to give a complete answer.

ANSWER GUIDELINES:
- Be thorough — combine relevant information from different parts of the material
- Use precise aviation terminology
- Cite sources: mention which section/chapter and page numbers your answer draws from
- For calculations or formulas, show the steps
- Structure your answer with clear formatting (use bullet points, headers if needed)

CONFIDENCE RATING — at the very end, on its own line, write exactly one of:
CONFIDENCE: high
CONFIDENCE: medium
CONFIDENCE: low

Use these criteria:
- HIGH: The study material directly addresses this question. You found clear, specific information.
- MEDIUM: The material contains relevant information that answers the question, even if you had to combine multiple sections or interpret slightly. This is the right rating for most well-supported answers.
- LOW: The material does not meaningfully address this topic. Only use LOW when the material genuinely lacks relevant content — not just because the answer requires synthesis.

Default to MEDIUM when in doubt. Reserve LOW for genuine gaps in the material."""


async def answer_question(question: str) -> dict:
    """Full RAG pipeline: retrieve -> synthesize -> return with sources."""

    chunks = await retrieve_chunks(question)
    context = build_context(chunks)

    if not os.environ.get("GROQ_API_KEY"):
        return {
            "answer": "**Groq API key not configured.**\n\nPlease set `GROQ_API_KEY` in your `.env` file.\n\nGet a free key at: [console.groq.com](https://console.groq.com)",
            "confidence": "low",
            "sources": [],
        }

    try:
        from services.llm import generate
        answer_text = generate(
            f"""Question: {question}

Study Material:
{context}""",
            system=SYSTEM_PROMPT,
        )
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "confidence": "low",
            "sources": [],
        }

    # Extract confidence
    confidence = "medium"
    lower = answer_text.lower()
    if "confidence: high" in lower:
        confidence = "high"
    elif "confidence: low" in lower:
        confidence = "low"

    answer_clean = re.sub(r'\n*CONFIDENCE:\s*(high|medium|low)\s*$', '', answer_text, flags=re.IGNORECASE).strip()

    # Build source references
    sources = []
    seen_refs = set()
    for chunk in chunks[:20]:
        ref_key = f"{chunk.get('document_id')}-{chunk.get('page_start')}"
        if ref_key not in seen_refs:
            seen_refs.add(ref_key)
            sources.append({
                "chunk_id": chunk["id"],
                "document": chunk.get("subject", "Unknown"),
                "chapter": chunk.get("chapter", ""),
                "section": chunk.get("section", ""),
                "page_start": chunk.get("page_start", "?"),
                "page_end": chunk.get("page_end", "?"),
            })

    return {
        "answer": answer_clean,
        "confidence": confidence,
        "sources": sources,
    }
