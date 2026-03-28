import asyncio
import markdown
from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates

from services.rag import answer_question
from db.queries import insert_qa

router = APIRouter(prefix="/qa")
templates = Jinja2Templates(directory="templates")


@router.post("/ask")
async def ask(request: Request, question: str = Form(...)):
    # Run RAG in thread pool so ChromaDB doesn't block the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, answer_question_sync, question)

    answer_html = markdown.markdown(result["answer"], extensions=["tables", "fenced_code"])

    source_ids = [s.get("chunk_id", "") for s in result["sources"]]
    insert_qa(question, result["answer"], source_ids, result["confidence"])

    return templates.TemplateResponse(name="components/answer.html", request=request, context={
        "answer_html": answer_html,
        "confidence": result["confidence"],
        "sources": result["sources"],
    })


def answer_question_sync(question: str) -> dict:
    """Sync wrapper for the RAG pipeline."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(answer_question(question))
    finally:
        loop.close()
