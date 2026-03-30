-- Run this in Supabase SQL Editor to set up the database

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    subject TEXT NOT NULL,
    page_count INT,
    processing_status TEXT DEFAULT 'pending',
    processing_progress FLOAT DEFAULT 0,
    file_storage_path TEXT,
    upload_date TIMESTAMPTZ DEFAULT now()
);

-- Chunks table with embeddings
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT DEFAULT '',
    subject TEXT NOT NULL,
    chapter TEXT DEFAULT '',
    section TEXT DEFAULT '',
    page_start INT,
    page_end INT,
    token_count INT,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Q&A conversation history
CREATE TABLE qa_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    source_chunks TEXT[] DEFAULT '{}',
    confidence TEXT DEFAULT 'medium',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX chunks_document_id_idx ON chunks (document_id);
CREATE INDEX chunks_chunk_index_idx ON chunks (document_id, chunk_index);
CREATE INDEX chunks_fts_idx ON chunks USING gin (to_tsvector('english', content));

-- RPC: Vector similarity search
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1536),
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_index INT,
    content TEXT,
    summary TEXT,
    subject TEXT,
    chapter TEXT,
    section TEXT,
    page_start INT,
    page_end INT,
    token_count INT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.chunk_index,
        c.content,
        c.summary,
        c.subject,
        c.chapter,
        c.section,
        c.page_start,
        c.page_end,
        c.token_count,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM chunks c
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- RPC: Full-text keyword search (OR-based for better recall with natural language)
CREATE OR REPLACE FUNCTION search_chunks_text(
    search_query TEXT,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_index INT,
    content TEXT,
    summary TEXT,
    subject TEXT,
    chapter TEXT,
    section TEXT,
    page_start INT,
    page_end INT,
    token_count INT,
    rank FLOAT
)
LANGUAGE plpgsql
AS $$
DECLARE
    or_query tsquery;
BEGIN
    -- Build OR query: split words and join with |
    or_query := array_to_string(
        regexp_split_to_array(trim(search_query), '\s+'),
        ' | '
    )::tsquery;

    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.chunk_index,
        c.content,
        c.summary,
        c.subject,
        c.chapter,
        c.section,
        c.page_start,
        c.page_end,
        c.token_count,
        ts_rank(to_tsvector('english', c.content), or_query) AS rank
    FROM chunks c
    WHERE to_tsvector('english', c.content) @@ or_query
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- RPC: Update chunk embedding (PostgREST can't directly write vector columns)
CREATE OR REPLACE FUNCTION update_chunk_embedding(
    chunk_id UUID,
    new_embedding vector(1536)
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE chunks SET embedding = new_embedding WHERE id = chunk_id;
END;
$$;

-- Storage bucket (run this separately or create via Supabase dashboard)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('pdfs', 'pdfs', false);
