
import os, time, aiosqlite
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

DB_PATH = os.getenv("DB_PATH", "/data/leaderboard.db")

app = FastAPI(title="Web Tetris Server", version="1.0")

# Mount static
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

INIT_SQL = '''
CREATE TABLE IF NOT EXISTS scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  name TEXT NOT NULL,
  ip   TEXT NOT NULL,
  score INTEGER NOT NULL
);
'''

@app.on_event("startup")
async def startup():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(INIT_SQL)
        await db.commit()

class ScoreIn(BaseModel):
    name: str
    score: int

async def insert_score(name:str, ip:str, score:int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO scores (ts, name, ip, score) VALUES (?, ?, ?, ?)",
                         (int(time.time()), name[:40], ip[:64], int(score)))
        await db.commit()

async def fetch_top(limit:int=50)->List[Dict[str,Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT ts, name, ip, score FROM scores ORDER BY score DESC, ts ASC LIMIT ?", (limit,)) as cur:
            rows = await cur.fetchall()
            return [ {"ts":r[0], "name":r[1], "ip":r[2], "score":r[3]} for r in rows ]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/leaderboard", response_class=HTMLResponse)
async def leaderboard_page(request: Request):
    return templates.TemplateResponse("leaderboard.html", {"request": request})

@app.get("/api/leaderboard")
async def get_leaderboard(limit:int=50):
    data = await fetch_top(limit=limit)
    return {"items": data}

@app.post("/api/score")
async def post_score(score: ScoreIn, request: Request):
    if score.score < 0 or score.score > 10_000_000:
        raise HTTPException(400, "invalid score")
    client_host = request.client.host if request.client else "unknown"
    await insert_score(score.name.strip() or "Anonymous", client_host, score.score)
    return {"ok": True}

@app.get("/healthz")
async def healthz():
    return {"ok": True}
