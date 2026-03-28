import os
import sqlite3
import chromadb

_chroma_client = None
_chroma_collection = None

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "atpl.db")
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma")


def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_chroma():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="atpl_chunks",
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            subject TEXT NOT NULL DEFAULT '',
            page_count INTEGER,
            processing_status TEXT DEFAULT 'pending',
            processing_progress REAL DEFAULT 0,
            upload_date TEXT DEFAULT (datetime('now')),
            labels TEXT DEFAULT '',
            notes TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            summary TEXT DEFAULT '',
            subject TEXT NOT NULL,
            chapter TEXT DEFAULT '',
            section TEXT DEFAULT '',
            page_start INTEGER,
            page_end INTEGER,
            token_count INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );

        CREATE TABLE IF NOT EXISTS qa_conversations (
            id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            source_chunks TEXT DEFAULT '[]',
            confidence TEXT DEFAULT 'medium',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_idx ON chunks(document_id, chunk_index);
    """)
    # Migrate: add labels and notes columns if missing
    cols = [r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()]
    if "labels" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN labels TEXT DEFAULT ''")
    if "notes" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN notes TEXT DEFAULT ''")

    conn.commit()
    conn.close()
