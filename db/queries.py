import json
import uuid
from db.connection import get_db, get_chroma


# ── Documents ──────────────────────────────────────────────

def insert_document(filename: str, page_count: int) -> dict:
    conn = get_db()
    doc_id = str(uuid.uuid4())
    # Auto-detect subject from filename
    name = filename.upper()
    subject = filename  # default: use filename as subject
    for keyword in ["QRH", "FCTM", "FCOM", "OMA", "MEL", "SOP", "AOM", "LIDO"]:
        if keyword in name:
            subject = keyword
            break
    conn.execute(
        "INSERT INTO documents (id, filename, subject, page_count, processing_status) VALUES (?, ?, ?, ?, 'pending')",
        (doc_id, filename, subject, page_count),
    )
    conn.commit()
    doc = dict(conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone())
    conn.close()
    return doc


def update_document_status(doc_id: str, status: str, progress: float = None):
    conn = get_db()
    if progress is not None:
        conn.execute("UPDATE documents SET processing_status = ?, processing_progress = ? WHERE id = ?", (status, progress, doc_id))
    else:
        conn.execute("UPDATE documents SET processing_status = ? WHERE id = ?", (status, doc_id))
    conn.commit()
    conn.close()


def get_all_documents() -> list:
    conn = get_db()
    rows = conn.execute("SELECT * FROM documents ORDER BY upload_date DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_document(doc_id: str) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_document(doc_id: str):
    """Delete a document and all its chunks from SQLite and ChromaDB."""
    conn = get_db()
    # Get chunk IDs for ChromaDB cleanup
    chunk_ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,)).fetchall()]
    # Delete from SQLite
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    # Delete from ChromaDB
    if chunk_ids:
        collection = get_chroma()
        # ChromaDB delete in batches
        for i in range(0, len(chunk_ids), 100):
            batch = chunk_ids[i:i + 100]
            try:
                collection.delete(ids=batch)
            except Exception:
                pass


def update_document_meta(doc_id: str, labels: str = None, notes: str = None):
    """Update labels and/or notes on a document."""
    conn = get_db()
    if labels is not None:
        conn.execute("UPDATE documents SET labels = ? WHERE id = ?", (labels, doc_id))
    if notes is not None:
        conn.execute("UPDATE documents SET notes = ? WHERE id = ?", (notes, doc_id))
    conn.commit()
    conn.close()


# ── Chunks ─────────────────────────────────────────────────

def insert_chunks(chunks: list):
    """Insert chunks into SQLite and ChromaDB (ChromaDB handles embeddings)."""
    conn = get_db()
    collection = get_chroma()

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        ids.append(chunk_id)

        # SQLite
        conn.execute(
            """INSERT INTO chunks (id, document_id, chunk_index, content, summary, subject, chapter, section, page_start, page_end, token_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (chunk_id, chunk["document_id"], chunk["chunk_index"], chunk["content"],
             chunk.get("summary", ""), chunk["subject"], chunk.get("chapter", ""),
             chunk.get("section", ""), chunk.get("page_start"), chunk.get("page_end"),
             chunk.get("token_count", 0)),
        )

        # ChromaDB — use summary + content for better embedding
        doc_text = f"{chunk.get('summary', '')} {chunk['content']}"
        documents.append(doc_text[:8000])
        metadatas.append({
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
            "subject": chunk["subject"],
            "chapter": chunk.get("chapter", ""),
            "section": chunk.get("section", ""),
            "page_start": chunk.get("page_start", 0),
            "page_end": chunk.get("page_end", 0),
        })

    conn.commit()
    conn.close()

    # Batch insert into ChromaDB (it generates embeddings automatically)
    if ids:
        # ChromaDB has a batch limit, insert in batches of 40
        for i in range(0, len(ids), 40):
            collection.add(
                ids=ids[i:i+40],
                documents=documents[i:i+40],
                metadatas=metadatas[i:i+40],
            )


def search_chunks_by_embedding(query: str, top_k: int = 10) -> list:
    """Vector similarity search via ChromaDB (it embeds the query automatically)."""
    collection = get_chroma()
    results = collection.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])

    chunks = []
    if results["ids"] and results["ids"][0]:
        conn = get_db()
        for i, chunk_id in enumerate(results["ids"][0]):
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
            if row:
                chunk = dict(row)
                chunk["similarity"] = 1 - (results["distances"][0][i] if results["distances"] else 0)
                chunks.append(chunk)
        conn.close()

    return chunks


def search_chunks_by_text(query: str, top_k: int = 5) -> list:
    """Simple text search using SQLite LIKE."""
    conn = get_db()
    words = query.split()
    if not words:
        conn.close()
        return []
    conditions = " OR ".join(["content LIKE ?" for _ in words])
    params = [f"%{w}%" for w in words]

    rows = conn.execute(
        f"SELECT * FROM chunks WHERE {conditions} LIMIT ?",
        params + [top_k],
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_neighbor_chunks(document_id: str, chunk_index: int) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM chunks WHERE document_id = ? AND chunk_index BETWEEN ? AND ? ORDER BY chunk_index",
        (document_id, chunk_index - 1, chunk_index + 1),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Q&A History ────────────────────────────────────────────

def insert_qa(question: str, answer: str, source_chunk_ids: list, confidence: str) -> dict:
    conn = get_db()
    qa_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO qa_conversations (id, question, answer, source_chunks, confidence) VALUES (?, ?, ?, ?, ?)",
        (qa_id, question, answer, json.dumps(source_chunk_ids), confidence),
    )
    conn.commit()
    conn.close()
    return {"id": qa_id}
