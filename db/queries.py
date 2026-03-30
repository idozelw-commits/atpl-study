from db.connection import get_supabase


# ── Documents ──────────────────────────────────────────────

def insert_document(filename: str, page_count: int) -> dict:
    sb = get_supabase()
    name = filename.upper()
    subject = filename
    for keyword in ["QRH", "FCTM", "FCOM", "OMA", "MEL", "SOP", "AOM", "LIDO"]:
        if keyword in name:
            subject = keyword
            break
    result = sb.table("documents").insert({
        "filename": filename,
        "subject": subject,
        "page_count": page_count,
        "processing_status": "pending",
    }).execute()
    return result.data[0]


def update_document_status(doc_id: str, status: str, progress: float = None):
    sb = get_supabase()
    data = {"processing_status": status}
    if progress is not None:
        data["processing_progress"] = progress
    sb.table("documents").update(data).eq("id", doc_id).execute()


def get_all_documents() -> list:
    sb = get_supabase()
    result = sb.table("documents").select("*").order("upload_date", desc=True).execute()
    return result.data


def get_document(doc_id: str) -> dict:
    sb = get_supabase()
    result = sb.table("documents").select("*").eq("id", doc_id).execute()
    return result.data[0] if result.data else None


def delete_document(doc_id: str):
    sb = get_supabase()
    sb.table("chunks").delete().eq("document_id", doc_id).execute()
    sb.table("documents").delete().eq("id", doc_id).execute()


def update_document_meta(doc_id: str, labels: str = None, notes: str = None):
    sb = get_supabase()
    data = {}
    if labels is not None:
        data["labels"] = labels
    if notes is not None:
        data["notes"] = notes
    if data:
        sb.table("documents").update(data).eq("id", doc_id).execute()


# ── Chunks ─────────────────────────────────────────────────

def insert_chunks(chunks: list):
    """Insert chunks into Supabase."""
    sb = get_supabase()
    for i in range(0, len(chunks), 50):
        batch = chunks[i:i + 50]
        sb.table("chunks").insert(batch).execute()


def search_chunks_vector(query_embedding: list, top_k: int = 15) -> list:
    """Semantic search via pgvector cosine similarity."""
    sb = get_supabase()
    result = sb.rpc("match_chunks", {
        "query_embedding": query_embedding,
        "match_count": top_k,
    }).execute()
    return result.data


def search_chunks_fulltext(query: str, top_k: int = 15) -> list:
    """Full-text search via Postgres tsvector."""
    sb = get_supabase()
    result = sb.rpc("search_chunks", {
        "search_query": query,
        "match_count": top_k,
    }).execute()
    return result.data


def get_neighbor_chunks(document_id: str, chunk_index: int) -> list:
    sb = get_supabase()
    result = sb.table("chunks").select("*").eq(
        "document_id", document_id
    ).gte("chunk_index", chunk_index - 1).lte(
        "chunk_index", chunk_index + 1
    ).order("chunk_index").execute()
    return result.data


def get_chunks_without_embeddings(limit: int = 100) -> list:
    """Get chunks that don't have embeddings yet."""
    sb = get_supabase()
    result = sb.table("chunks").select(
        "id, content, subject, chapter, section"
    ).is_("embedding", "null").limit(limit).execute()
    return result.data


def update_chunk_embedding(chunk_id: str, embedding: list):
    """Update a single chunk's embedding."""
    sb = get_supabase()
    sb.table("chunks").update({
        "embedding": embedding,
    }).eq("id", chunk_id).execute()


def update_chunks_embeddings_batch(updates: list):
    """Update embeddings for multiple chunks. Each item: {id, embedding}."""
    sb = get_supabase()
    for item in updates:
        sb.table("chunks").update({
            "embedding": item["embedding"],
        }).eq("id", item["id"]).execute()


def count_chunks_without_embeddings() -> int:
    """Count chunks missing embeddings."""
    sb = get_supabase()
    result = sb.table("chunks").select(
        "id", count="exact"
    ).is_("embedding", "null").execute()
    return result.count or 0


def count_total_chunks() -> int:
    """Count total chunks."""
    sb = get_supabase()
    result = sb.table("chunks").select(
        "id", count="exact"
    ).execute()
    return result.count or 0


# ── Q&A History ────────────────────────────────────────────

def insert_qa(question: str, answer: str, source_chunk_ids: list, confidence: str) -> dict:
    sb = get_supabase()
    import json
    result = sb.table("qa_conversations").insert({
        "question": question,
        "answer": answer,
        "source_chunks": json.dumps(source_chunk_ids),
        "confidence": confidence,
    }).execute()
    return result.data[0]
