import os
import re

from services.embeddings import get_embedding
from db.queries import (
    search_chunks_vector,
    search_chunks_fulltext,
    get_neighbor_chunks,
)


def _extract_search_terms(question: str) -> str:
    """Extract key terms from a natural language question for full-text search."""
    # Remove common question words that add noise to full-text search
    stopwords = {
        "what", "which", "when", "where", "who", "whom", "why", "how",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having",
        "do", "does", "did", "doing",
        "a", "an", "the", "and", "but", "or", "nor", "not",
        "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "about", "between", "through", "during",
        "can", "could", "would", "should", "will", "shall", "may", "might",
        "this", "that", "these", "those", "it", "its",
        "me", "my", "we", "our", "you", "your", "they", "their",
        "tell", "explain", "describe", "define", "much", "many",
    }
    words = re.findall(r'\b[a-zA-Z0-9]+\b', question.lower())
    key_terms = [w for w in words if w not in stopwords and len(w) > 1]
    return " ".join(key_terms)


async def retrieve_chunks(question: str) -> list:
    """Hybrid retrieval: vector similarity + full-text search, merged and deduplicated."""

    # 1. Vector search (semantic — handles natural language)
    vector_chunks = []
    try:
        query_embedding = get_embedding(question)
        vector_chunks = search_chunks_vector(query_embedding, top_k=15)
    except Exception as e:
        print(f"Vector search failed: {e}")

    # 2. Full-text search (keyword — handles exact terms, abbreviations)
    fulltext_chunks = []
    try:
        search_terms = _extract_search_terms(question)
        if search_terms:
            fulltext_chunks = search_chunks_fulltext(search_terms, top_k=10)
    except Exception as e:
        print(f"Full-text search failed: {e}")

    # 3. Merge and deduplicate, preserving rank order
    seen_ids = set()
    merged = []

    # Vector results first (usually more relevant for natural language)
    for chunk in vector_chunks:
        cid = chunk["id"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            chunk["_source"] = "vector"
            chunk["_rank"] = len(merged)
            merged.append(chunk)

    # Then full-text results (may catch exact term matches vector missed)
    for chunk in fulltext_chunks:
        cid = chunk["id"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            chunk["_source"] = "fulltext"
            chunk["_rank"] = len(merged)
            merged.append(chunk)
        else:
            # Boost chunks found by both methods
            for m in merged:
                if m["id"] == cid:
                    m["_source"] = "both"
                    break

    # 4. Sort: "both" first, then by original rank
    merged.sort(key=lambda c: (0 if c.get("_source") == "both" else 1, c.get("_rank", 99)))

    # 5. Fetch neighbor chunks for top results to provide surrounding context
    enriched = []
    for chunk in merged[:12]:
        neighbors = get_neighbor_chunks(chunk["document_id"], chunk["chunk_index"])
        neighbor_text = "\n".join(
            n["content"] for n in neighbors if n["id"] != chunk["id"]
        )
        chunk["context"] = neighbor_text[:2000]
        enriched.append(chunk)

    return enriched


def build_context(chunks: list) -> str:
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

            similarity = chunk.get("similarity", "")
            source = chunk.get("_source", "")
            meta = f" (match: {source}" + (f", sim: {similarity:.3f}" if similarity else "") + ")"

            context += f"\n{ref}{meta}\n{chunk['content']}\n"
            if chunk.get("context"):
                context += f"\n[Surrounding material:]\n{chunk['context']}\n"

    return context


SYSTEM_PROMPT = """You are an expert ATPL (Airline Transport Pilot License) instructor helping a pilot study for their exams.

YOUR JOB: Answer the question using the study material provided below. The material was retrieved via semantic search — it IS relevant even if the exact words don't match the question.

CRITICAL RULES:
1. ALWAYS attempt to answer from the material. The retrieval system found these chunks because they are semantically related to the question.
2. Synthesize information across multiple chunks/sections when needed.
3. If the material discusses the topic but doesn't give a direct answer, explain what the material DOES say about it and extrapolate where reasonable.
4. Only say "not found in the material" if the provided chunks are truly about a completely different topic.

ANSWER FORMAT:
- Use precise aviation terminology
- For calculations or formulas, show the steps
- Cite sources: mention which section/chapter and page numbers
- Structure with clear headings when the answer is complex
- Keep it thorough but focused

CONFIDENCE RATING — at the very end, on its own line, write exactly one of:
CONFIDENCE: high — Material directly answers the question with clear, specific information.
CONFIDENCE: medium — Material contains relevant information; answer required some synthesis or interpretation.
CONFIDENCE: low — Material touches the topic tangentially; answer includes significant extrapolation."""


async def answer_question(question: str) -> dict:
    chunks = await retrieve_chunks(question)
    context = build_context(chunks)

    if not os.environ.get("GROQ_API_KEY"):
        return {
            "answer": "**Groq API key not configured.**",
            "confidence": "low",
            "sources": [],
        }

    try:
        from services.llm import generate
        answer_text = generate(
            f"""Question: {question}

Study Material (retrieved via semantic search — these chunks are relevant):
{context}""",
            system=SYSTEM_PROMPT,
        )
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "confidence": "low",
            "sources": [],
        }

    confidence = "medium"
    lower = answer_text.lower()
    if "confidence: high" in lower:
        confidence = "high"
    elif "confidence: low" in lower:
        confidence = "low"

    answer_clean = re.sub(r'\n*CONFIDENCE:\s*(high|medium|low)\s*$', '', answer_text, flags=re.IGNORECASE).strip()

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
