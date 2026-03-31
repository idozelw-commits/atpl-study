import os
import threading
from typing import List

import fitz
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from db.queries import insert_document, get_all_documents, get_document, delete_document, update_document_meta

router = APIRouter(prefix="/upload")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("")
async def upload_page(request: Request):
    documents = get_all_documents()
    return templates.TemplateResponse(name="upload.html", request=request, context={"documents": documents})


@router.post("/pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    uploaded = []

    for file in files[:5]:
        content = await file.read()
        filename = file.filename

        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(content)

        pdf = fitz.open(stream=content, filetype="pdf")
        page_count = len(pdf)
        pdf.close()

        doc = insert_document(filename, page_count)
        uploaded.append({"id": doc["id"], "filename": filename, "page_count": page_count})

        t = threading.Thread(target=run_processing_sync, args=(doc["id"], content, doc["subject"], filename), daemon=True)
        t.start()

    return JSONResponse({"uploads": uploaded})


def run_processing_sync(doc_id: str, pdf_bytes: bytes, subject: str, filename: str):
    from db.queries import update_document_status
    from services.pdf_processor import process_pdf_sync
    try:
        update_document_status(doc_id, "processing", 0.0)
        process_pdf_sync(doc_id, pdf_bytes, subject, filename)
        update_document_status(doc_id, "done", 1.0)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        update_document_status(doc_id, "error")


@router.get("/status/{doc_id}")
async def get_status(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        return JSONResponse({"status": "error", "filename": "Unknown", "progress": 0})
    return JSONResponse({
        "status": doc["processing_status"],
        "filename": doc["filename"],
        "progress": doc.get("processing_progress", 0),
    })


@router.delete("/document/{doc_id}")
async def remove_document(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        return JSONResponse({"error": "Not found"}, status_code=404)
    # Remove PDF file
    filepath = os.path.join(UPLOAD_DIR, doc["filename"])
    if os.path.exists(filepath):
        os.remove(filepath)
    # Remove from DB + ChromaDB
    delete_document(doc_id)
    return JSONResponse({"ok": True})


@router.post("/document/{doc_id}/meta")
async def update_meta(doc_id: str, labels: str = Form(""), notes: str = Form("")):
    doc = get_document(doc_id)
    if not doc:
        return JSONResponse({"error": "Not found"}, status_code=404)
    update_document_meta(doc_id, labels=labels, notes=notes)
    return JSONResponse({"ok": True})


@router.post("/document/{doc_id}/retry")
async def retry_document(doc_id: str, file: UploadFile = File(...)):
    """Re-upload and reprocess a stuck/failed document."""
    from db.queries import update_document_status
    doc = get_document(doc_id)
    if not doc:
        return JSONResponse({"error": "Not found"}, status_code=404)

    content = await file.read()

    # Delete old chunks, reset status
    from db.connection import get_supabase
    sb = get_supabase()
    sb.table("chunks").delete().eq("document_id", doc_id).execute()
    update_document_status(doc_id, "pending", 0.0)

    t = threading.Thread(target=run_processing_sync, args=(doc_id, content, doc.get("subject", ""), doc["filename"]), daemon=True)
    t.start()

    return JSONResponse({"ok": True, "id": doc_id})


@router.get("/documents")
async def get_documents_list(request: Request):
    documents = get_all_documents()
    return templates.TemplateResponse(name="components/documents_list.html", request=request, context={"documents": documents})


_embed_status = {"running": False, "processed": 0, "total": 0, "error": None}


def _run_embed_all():
    """Background embedding job. Only processes chunks without embeddings (has_embedding=false)."""
    import traceback
    import time
    global _embed_status
    try:
        from services.embeddings import get_embeddings_batch
        from db.connection import get_supabase

        sb = get_supabase()

        # Count remaining
        remaining = sb.table("chunks").select("id", count="exact").eq("has_embedding", False).execute()
        total_remaining = remaining.count or 0
        total_all = sb.table("chunks").select("id", count="exact").execute().count or 0

        _embed_status["total"] = total_all
        _embed_status["processed"] = total_all - total_remaining
        _embed_status["error"] = None

        if total_remaining == 0:
            print(f"  [embed-all] All {total_all} chunks already embedded")
            return

        print(f"  [embed-all] {total_remaining} chunks remaining to embed (of {total_all} total)")

        batch_size = 20
        while True:
            # Fetch chunks that need embedding
            chunks = sb.table("chunks").select("id, content").eq(
                "has_embedding", False
            ).limit(batch_size).execute().data

            if not chunks:
                break

            # Get embeddings from OpenAI
            texts = [c["content"][:8000] for c in chunks]
            embeddings = get_embeddings_batch(texts)

            # Write embeddings + set flag
            for i, chunk in enumerate(chunks):
                try:
                    sb.rpc("update_chunk_embedding", {
                        "chunk_id": chunk["id"],
                        "new_embedding": embeddings[i],
                    }).execute()
                    sb.table("chunks").update({"has_embedding": True}).eq("id", chunk["id"]).execute()
                except Exception as e:
                    print(f"  [embed-all] Failed chunk {chunk['id']}: {e}")
                    time.sleep(1)
                    try:
                        sb.rpc("update_chunk_embedding", {
                            "chunk_id": chunk["id"],
                            "new_embedding": embeddings[i],
                        }).execute()
                        sb.table("chunks").update({"has_embedding": True}).eq("id", chunk["id"]).execute()
                    except Exception as e2:
                        print(f"  [embed-all] Retry failed: {e2}")

            _embed_status["processed"] += len(chunks)
            print(f"  [embed-all] {_embed_status['processed']}/{total_all} chunks embedded")

    except Exception as e:
        traceback.print_exc()
        _embed_status["error"] = f"{type(e).__name__}: {str(e)}"
    finally:
        _embed_status["running"] = False


@router.post("/embed-all")
async def embed_all_chunks():
    """Start background embedding job for all chunks."""
    global _embed_status
    if _embed_status["running"]:
        return JSONResponse({
            "status": "running",
            "processed": _embed_status["processed"],
            "total": _embed_status["total"],
        })

    _embed_status["running"] = True
    t = threading.Thread(target=_run_embed_all, daemon=True)
    t.start()

    return JSONResponse({"status": "started", "message": "Embedding job started in background. Check /upload/embed-status for progress."})


@router.get("/embed-status")
async def embed_status():
    """Check progress of background embedding job."""
    return JSONResponse({
        "running": _embed_status["running"],
        "processed": _embed_status["processed"],
        "total": _embed_status["total"],
        "error": _embed_status["error"],
    })
