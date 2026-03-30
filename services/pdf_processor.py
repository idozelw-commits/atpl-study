import gc
import tempfile
import os
from collections import Counter
import fitz  # PyMuPDF

from db.queries import insert_chunks, update_document_status, update_chunks_embeddings_batch


def process_pdf_sync(doc_id: str, pdf_bytes: bytes, subject: str, filename: str):
    """Memory-safe PDF processing: write to temp file, process page-by-page."""

    # Write PDF to temp file instead of keeping bytes in memory
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # Release the bytes from memory immediately
    del pdf_bytes
    gc.collect()

    try:
        _process_from_file(doc_id, tmp_path, subject, filename)
    finally:
        os.unlink(tmp_path)


def _process_from_file(doc_id: str, pdf_path: str, subject: str, filename: str):
    """Process PDF from file path, page by page, minimal memory."""

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    update_document_status(doc_id, "processing", 0.1)

    # Step 1: Sample pages for font detection (only need ~50 pages)
    body_size = _detect_body_font_size(pdf_path, total_pages)
    update_document_status(doc_id, "processing", 0.15)

    # Step 2: Extract text and chunk, processing 20 pages at a time
    all_chunks = []
    current_chunk = {"text": "", "chapter": "", "section": "", "page_start": 1, "page_end": 1}
    prev_tail = ""
    batch_size = 20

    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)

        doc = fitz.open(pdf_path)
        for page_num in range(batch_start, batch_end):
            page = doc[page_num]
            # Use lightweight text extraction with position info
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    text = "".join(span["text"] for span in line["spans"])
                    if not text.strip():
                        continue
                    max_size = max(span["size"] for span in line["spans"])
                    is_bold = any("bold" in span["font"].lower() for span in line["spans"])

                    is_heading = False
                    heading_level = 3
                    size = round(max_size, 1)
                    if size > body_size + 1.5 or (size > body_size + 0.5 and is_bold):
                        is_heading = True
                        heading_level = 1 if size > body_size + 3 else 2
                    elif is_bold and len(text.strip()) < 100:
                        is_heading = True
                        heading_level = 3

                    # Chunking inline
                    if is_heading and heading_level <= 2:
                        # Flush current chunk
                        chunk_text = current_chunk["text"].strip()
                        if chunk_text and len(chunk_text) > 100:
                            content = (prev_tail + " " + chunk_text).strip() if prev_tail else chunk_text
                            all_chunks.append({
                                "content": content,
                                "chapter": current_chunk["chapter"],
                                "section": current_chunk["section"],
                                "page_start": current_chunk["page_start"],
                                "page_end": current_chunk["page_end"],
                            })
                            prev_tail = chunk_text[-200:] if len(chunk_text) > 200 else chunk_text

                        if heading_level <= 1:
                            current_chunk = {"text": text.strip() + "\n", "chapter": text.strip(), "section": "", "page_start": page_num + 1, "page_end": page_num + 1}
                        else:
                            current_chunk = {"text": text.strip() + "\n", "chapter": current_chunk.get("chapter", ""), "section": text.strip(), "page_start": page_num + 1, "page_end": page_num + 1}
                    else:
                        if is_heading:
                            current_chunk["text"] += "\n" + text.strip() + "\n"
                        else:
                            current_chunk["text"] += text.strip() + "\n"
                        current_chunk["page_end"] = page_num + 1

                        if len(current_chunk["text"]) > 2500:
                            chunk_text = current_chunk["text"].strip()
                            if chunk_text and len(chunk_text) > 100:
                                content = (prev_tail + " " + chunk_text).strip() if prev_tail else chunk_text
                                all_chunks.append({
                                    "content": content,
                                    "chapter": current_chunk["chapter"],
                                    "section": current_chunk["section"],
                                    "page_start": current_chunk["page_start"],
                                    "page_end": current_chunk["page_end"],
                                })
                                prev_tail = chunk_text[-200:] if len(chunk_text) > 200 else chunk_text
                            current_chunk = {"text": "", "chapter": current_chunk["chapter"], "section": current_chunk["section"], "page_start": page_num + 1, "page_end": page_num + 1}

        doc.close()
        gc.collect()

        progress = 0.15 + 0.25 * min(batch_end, total_pages) / total_pages
        update_document_status(doc_id, "processing", progress)
        print(f"  [{filename}] Extracted {batch_end}/{total_pages} pages, {len(all_chunks)} chunks so far")

    # Flush last chunk
    chunk_text = current_chunk["text"].strip()
    if chunk_text and len(chunk_text) > 100:
        content = (prev_tail + " " + chunk_text).strip() if prev_tail else chunk_text
        all_chunks.append({
            "content": content,
            "chapter": current_chunk["chapter"],
            "section": current_chunk["section"],
            "page_start": current_chunk["page_start"],
            "page_end": current_chunk["page_end"],
        })

    update_document_status(doc_id, "processing", 0.4)

    if not all_chunks:
        update_document_status(doc_id, "error")
        return

    # Step 3: Insert into Supabase
    chunk_records = []
    for idx, chunk in enumerate(all_chunks):
        chunk_records.append({
            "document_id": doc_id,
            "chunk_index": idx,
            "content": chunk["content"],
            "subject": subject,
            "chapter": chunk.get("chapter", ""),
            "section": chunk.get("section", ""),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "token_count": len(chunk["content"]) // 4,
        })

    total = len(chunk_records)
    batch_size = 50
    for i in range(0, total, batch_size):
        batch = chunk_records[i:i + batch_size]
        insert_chunks(batch)
        progress = 0.4 + 0.45 * min(i + batch_size, total) / total
        update_document_status(doc_id, "processing", progress)

    print(f"  [{filename}] Inserted {total} chunks from {total_pages} pages")

    # Step 4: Generate embeddings
    try:
        from services.embeddings import get_embeddings_batch
        from db.connection import get_supabase
        sb = get_supabase()

        # Fetch the chunks we just inserted (they have IDs now)
        result = sb.table("chunks").select("id, content").eq(
            "document_id", doc_id
        ).order("chunk_index").execute()
        db_chunks = result.data

        texts = [c["content"][:8000] for c in db_chunks]
        embed_batch_size = 50
        for i in range(0, len(texts), embed_batch_size):
            batch_texts = texts[i:i + embed_batch_size]
            batch_chunks = db_chunks[i:i + embed_batch_size]
            embeddings = get_embeddings_batch(batch_texts)
            updates = [
                {"id": batch_chunks[j]["id"], "embedding": embeddings[j]}
                for j in range(len(batch_chunks))
            ]
            update_chunks_embeddings_batch(updates)
            progress = 0.85 + 0.15 * min(i + embed_batch_size, len(texts)) / len(texts)
            update_document_status(doc_id, "processing", progress)

        print(f"  [{filename}] Embeddings generated for {len(db_chunks)} chunks")
    except Exception as e:
        print(f"  [{filename}] Embedding generation failed (search will use full-text only): {e}")

    print(f"  [{filename}] Done: {total} chunks from {total_pages} pages")


def _detect_body_font_size(pdf_path: str, total_pages: int) -> float:
    """Sample pages to detect the most common (body) font size."""
    doc = fitz.open(pdf_path)
    all_sizes = []

    # Sample up to 30 pages evenly distributed
    sample_pages = min(30, total_pages)
    step = max(1, total_pages // sample_pages)

    for i in range(0, total_pages, step):
        if len(all_sizes) > 2000:
            break
        page = doc[i]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        all_sizes.append(round(span["size"], 1))

    doc.close()

    if not all_sizes:
        return 10.0

    size_counts = Counter(all_sizes)
    return size_counts.most_common(1)[0][0]
