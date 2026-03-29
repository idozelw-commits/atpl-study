import os
from supabase import create_client, Client

_client = None

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://yogtwgdglftbplvikxud.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")


def get_supabase() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def init_db():
    """Verify Supabase connection."""
    sb = get_supabase()
    sb.table("documents").select("id").limit(1).execute()
    print("Supabase connected OK")
