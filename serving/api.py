"""FastAPI application — assembles all route modules."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from serving.routes import ab_test, batch, recommend, trigger, viz

app = FastAPI(
    title="MovieLens RecSys API",
    version="0.1.0",
    description="Two-Tower + InfoNCE recommendation system with SDFT self-distillation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router)
app.include_router(ab_test.router)
app.include_router(batch.router)
app.include_router(trigger.router)
app.include_router(viz.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
