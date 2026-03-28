from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import traceback

load_dotenv()

app = FastAPI(title="ATPL Study Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize database on startup
try:
    from db.connection import init_db
    init_db()
    print("Database initialized OK")
except Exception as e:
    print(f"Database init failed: {e}")
    traceback.print_exc()

from routers import upload, qa

app.include_router(upload.router)
app.include_router(qa.router)


@app.get("/")
async def home(request: Request):
    try:
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        return JSONResponse({"error": str(e), "tb": traceback.format_exc()}, status_code=500)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})
