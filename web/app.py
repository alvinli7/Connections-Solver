from __future__ import annotations
from typing import List, Optional, Dict, Any
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from solver.app import solve_connections, SolveParams
from solver.service import SCORER

app = FastAPI(title="Connections Solver API", version="1.0")

# CORS: allow browser frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="web/static", html=True), name="static")

def confidence_label(gap: float | None) -> str:
    if gap is None:
        return "n/a"
    if gap > 0.8:
        return "high confidence"
    if gap > 0.3:
        return "medium confidence"
    return "ambiguous"

class SolveRequest(BaseModel):
    words: List[str] = Field(..., description="Exactly 16 entries. Phrases allowed, e.g. 'ICE CREAM'.")
    top_n: int = 5
    k: int = 400
    cap: int = 80
    min_score: float = -0.25
    explain: bool = False
    include_details: bool = False   
    include_more_solutions: bool = False

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

@app.on_event("startup")
def startup_message():
    print("\nOpen the web app: http://127.0.0.1:8000/static/index.html\n")

@app.post("/solve")
def solve(req: SolveRequest) -> Dict[str, Any]:
    params = SolveParams(
        top_n=req.top_n,
        k=req.k,
        cap=req.cap,
        min_score=req.min_score,
        explain=req.explain,
    )

    result = solve_connections(req.words, params, SCORER)

    # If solver failed, return as is (still clean JSON)
    if not result.get("ok", False):
        return result

    # Build clean response: only best solution + confidence by default
    solutions = result.get("solutions", [])
    best = solutions[0] if solutions else None

    gap = result.get("confidence_gap", None)
    label = confidence_label(gap)

    clean: Dict[str, Any] = {
        "ok": True,
        "best_solution": best,   
        "confidence_gap": gap,
        "confidence_label": label,
    }

    if req.include_more_solutions:
        clean["other_solutions"] = solutions[1:]  

    if req.include_details:
        clean["words"] = result.get("words", [])
        clean["params"] = result.get("params", {})
        clean["candidate_stats"] = result.get("candidate_stats", {})
    return clean

