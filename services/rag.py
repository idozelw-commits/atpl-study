import os
import re

from db.queries import search_chunks_fulltext, get_neighbor_chunks


async def retrieve_chunks(question: str) -> list:
    """Full-text search via Postgres."""
    all_chunks = search_chunks_fulltext(question, top_k=20)

    # Fetch neighbor chunks for context
    enriched = []
    for chunk in all_chunks[:15]:
        neighbors = get_neighbor_chunks(chunk["document_id"], chunk["chunk_index"])
        neighbor_text = "\n".join(n["content"] for n in neighbors if n["id"] != chunk["id"])
        chunk["context"] = neighbor_text[:1500]
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
- Structure your answer with clear formatting

CONFIDENCE RATING — at the very end, on its own line, write exactly one of:
CONFIDENCE: high
CONFIDENCE: medium
CONFIDENCE: low

Use these criteria:
- HIGH: The study material directly addresses this question. You found clear, specific information.
- MEDIUM: The material contains relevant information that answers the question, even if you had to combine multiple sections or interpret slightly.
- LOW: The material does not meaningfully address this topic. Only use LOW when the material genuinely lacks relevant content.

Default to MEDIUM when in doubt. Reserve LOW for genuine gaps in the material."""


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
