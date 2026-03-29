from collections import Counter
import fitz  # PyMuPDF

from db.queries import insert_chunks, update_document_status


def extract_text_with_structure(pdf_bytes: bytes) -> list:
    """Extract text page by page, keeping memory low."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    pages = []
    # Process in batches of 30 pages to keep memory under control
    batch_size = 30
    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        for page_num in range(batch_start, batch_end):
            page = doc[page_num]
            # Use "text" mode (lighter than "dict") then parse separately for headings
            blocks = page.get_text("dict")["blocks"]
            page_texts = []
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    text = "".join(span["text"] for span in line["spans"])
                    if not text.strip():
                        continue
                    max_size = max(span["size"] for span in line["spans"])
                    is_bold = any("bold" in span["font"].lower() for span in line["spans"])
                    page_texts.append({
                        "text": text.strip(),
                        "font_size": max_size,
                        "is_bold": is_bold,
                        "page": page_num + 1,
                    })
            pages.append({"page_num": page_num + 1, "lines": page_texts})
        # Release page resources after each batch
        import gc
        gc.collect()
    doc.close()
    return pages


def detect_headings(pages: list) -> list:
    all_sizes = [line["font_size"] for page in pages for line in page["lines"]]
    if not all_sizes:
        return pages
    size_counts = Counter(round(s, 1) for s in all_sizes)
    body_size = size_counts.most_common(1)[0][0]
    for page in pages:
        for line in page["lines"]:
            size = round(line["font_size"], 1)
            if size > body_size + 1.5 or (size > body_size + 0.5 and line["is_bold"]):
                line["is_heading"] = True
                line["heading_level"] = 1 if size > body_size + 3 else 2
            elif line["is_bold"] and len(line["text"]) < 100:
                line["is_heading"] = True
                line["heading_level"] = 3
            else:
                line["is_heading"] = False
    return pages


def semantic_chunk(pages: list, max_chars: int = 8000, overlap_chars: int = 200) -> list:
    chunks = []
    current_chunk = {"text": "", "chapter": "", "section": "", "page_start": 1, "page_end": 1}
    prev_tail = ""

    def flush_chunk():
        nonlocal prev_tail
        text = current_chunk["text"].strip()
        if text and len(text) > 100:
            content = (prev_tail + " " + text).strip() if prev_tail else text
            chunks.append({
                "content": content,
                "chapter": current_chunk["chapter"],
                "section": current_chunk["section"],
                "page_start": current_chunk["page_start"],
                "page_end": current_chunk["page_end"],
            })
            prev_tail = text[-overlap_chars:] if len(text) > overlap_chars else text

    for page in pages:
        for line in page["lines"]:
            if line.get("is_heading") and line.get("heading_level", 3) <= 2:
                flush_chunk()
                if line.get("heading_level", 3) <= 1:
                    current_chunk = {"text": line["text"] + "\n", "chapter": line["text"], "section": "", "page_start": line["page"], "page_end": line["page"]}
                else:
                    current_chunk = {"text": line["text"] + "\n", "chapter": current_chunk.get("chapter", ""), "section": line["text"], "page_start": line["page"], "page_end": line["page"]}
            else:
                if line.get("is_heading"):
                    current_chunk["text"] += "\n" + line["text"] + "\n"
                else:
                    current_chunk["text"] += line["text"] + "\n"
                current_chunk["page_end"] = line["page"]
                if len(current_chunk["text"]) > max_chars:
                    flush_chunk()
                    current_chunk = {"text": "", "chapter": current_chunk["chapter"], "section": current_chunk["section"], "page_start": line["page"], "page_end": line["page"]}

    flush_chunk()
    return chunks


def process_pdf_sync(doc_id: str, pdf_bytes: bytes, subject: str, filename: str):
    """Extract, chunk, store in Supabase. No embeddings needed — Postgres handles search."""
    pages = extract_text_with_structure(pdf_bytes)
    update_document_status(doc_id, "processing", 0.2)

    pages = detect_headings(pages)
    update_document_status(doc_id, "processing", 0.3)

    chunks = semantic_chunk(pages)
    update_document_status(doc_id, "processing", 0.5)

    if not chunks:
        update_document_status(doc_id, "error")
        return

    # Prepare records — Postgres trigger auto-generates search_vector
    chunk_records = []
    for idx, chunk in enumerate(chunks):
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

    # Insert in batches
    total = len(chunk_records)
    batch_size = 50
    for i in range(0, total, batch_size):
        batch = chunk_records[i:i + batch_size]
        insert_chunks(batch)
        progress = 0.5 + 0.5 * min(i + batch_size, total) / total
        update_document_status(doc_id, "processing", progress)
        print(f"  [{filename}] {min(i + batch_size, total)}/{total} chunks")
