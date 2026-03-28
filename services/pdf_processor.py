import os
import re
from collections import Counter

import fitz  # PyMuPDF

from db.queries import insert_chunks, update_document_status


def extract_text_with_structure(pdf_bytes: bytes) -> list:
    """Extract text from PDF with page numbers and basic structure detection."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
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

    doc.close()
    return pages


def detect_headings(pages: list) -> list:
    """Identify heading lines based on font size and boldness."""
    all_sizes = []
    for page in pages:
        for line in page["lines"]:
            all_sizes.append(line["font_size"])

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
    """Split pages into semantic chunks based on headings. Larger chunks = better context."""
    chunks = []
    current_chunk = {
        "text": "",
        "chapter": "",
        "section": "",
        "page_start": 1,
        "page_end": 1,
    }
    prev_tail = ""  # For overlap

    def flush_chunk():
        nonlocal prev_tail
        text = current_chunk["text"].strip()
        if text and len(text) > 100:  # Min 100 chars to filter junk
            # Prepend overlap from previous chunk
            content = (prev_tail + " " + text).strip() if prev_tail else text
            chunks.append({
                "content": content,
                "chapter": current_chunk["chapter"],
                "section": current_chunk["section"],
                "page_start": current_chunk["page_start"],
                "page_end": current_chunk["page_end"],
            })
            # Save tail for next chunk's overlap
            prev_tail = text[-overlap_chars:] if len(text) > overlap_chars else text

    for page in pages:
        for line in page["lines"]:
            if line.get("is_heading"):
                # Only flush on major headings (level 1-2), accumulate level 3
                if line.get("heading_level", 3) <= 2:
                    flush_chunk()
                    if line.get("heading_level", 3) <= 1:
                        current_chunk = {
                            "text": line["text"] + "\n",
                            "chapter": line["text"],
                            "section": "",
                            "page_start": line["page"],
                            "page_end": line["page"],
                        }
                    else:
                        current_chunk = {
                            "text": line["text"] + "\n",
                            "chapter": current_chunk.get("chapter", ""),
                            "section": line["text"],
                            "page_start": line["page"],
                            "page_end": line["page"],
                        }
                else:
                    # Level 3 headings: keep in current chunk
                    current_chunk["text"] += "\n" + line["text"] + "\n"
                    current_chunk["page_end"] = line["page"]
            else:
                current_chunk["text"] += line["text"] + "\n"
                current_chunk["page_end"] = line["page"]

                if len(current_chunk["text"]) > max_chars:
                    flush_chunk()
                    current_chunk = {
                        "text": "",
                        "chapter": current_chunk["chapter"],
                        "section": current_chunk["section"],
                        "page_start": line["page"],
                        "page_end": line["page"],
                    }

    flush_chunk()
    return chunks


def enrich_chunks_with_summaries(chunks: list, subject: str) -> list:
    """Add topic summaries using Groq LLM (free tier)."""
    if not os.environ.get("GROQ_API_KEY"):
        return chunks

    try:
        from services.llm import generate
    except Exception:
        return chunks

    for i in range(0, len(chunks), 10):
        batch = chunks[i:i + 10]
        batch_text = ""
        for j, chunk in enumerate(batch):
            batch_text += f"\n---CHUNK {j}---\n{chunk['content'][:500]}\n"

        try:
            response_text = generate(
                f"""You are summarizing aviation study material for ATPL exams.
Subject: {subject}

For each chunk below, write a ONE-LINE topic summary.
Reply with one summary per line, prefixed with the chunk number.

{batch_text}

Format:
0: [summary]
1: [summary]
..."""
            )

            for line in response_text.strip().split("\n"):
                match = re.match(r"(\d+):\s*(.+)", line)
                if match:
                    idx = int(match.group(1))
                    summary = match.group(2).strip()
                    if i + idx < len(chunks):
                        chunks[i + idx]["summary"] = summary
        except Exception as e:
            print(f"Enrichment batch {i} failed: {e}")
            continue

    return chunks


def process_pdf_sync(doc_id: str, pdf_bytes: bytes, subject: str, filename: str):
    """Full pipeline (sync, runs in thread): extract -> chunk -> enrich -> embed -> store."""

    # Step 1: Extract text with structure
    pages = extract_text_with_structure(pdf_bytes)
    update_document_status(doc_id, "processing", 0.1)

    # Step 2: Detect headings
    pages = detect_headings(pages)
    update_document_status(doc_id, "processing", 0.2)

    # Step 3: Semantic chunking
    chunks = semantic_chunk(pages)
    update_document_status(doc_id, "processing", 0.3)

    if not chunks:
        update_document_status(doc_id, "error")
        return

    # Step 4: Skip enrichment during upload (saves rate limits for Q&A)
    update_document_status(doc_id, "processing", 0.6)

    # Step 5: Prepare and insert (ChromaDB handles embeddings automatically)
    chunk_records = []
    for idx, chunk in enumerate(chunks):
        chunk_records.append({
            "document_id": doc_id,
            "chunk_index": idx,
            "content": chunk["content"],
            "summary": chunk.get("summary", ""),
            "subject": subject,
            "chapter": chunk.get("chapter", ""),
            "section": chunk.get("section", ""),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "token_count": len(chunk["content"]) // 4,
        })

    # Insert in small batches with progress updates
    total = len(chunk_records)
    batch_size = 20
    for i in range(0, total, batch_size):
        batch = chunk_records[i:i + batch_size]
        insert_chunks(batch)
        progress = 0.6 + 0.4 * min(i + batch_size, total) / total
        update_document_status(doc_id, "processing", progress)
        print(f"  [{filename}] Inserted {min(i + batch_size, total)}/{total} chunks")
