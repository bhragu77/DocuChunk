"""
PipelineRun — one recorded execution of the RAG/agent pipeline, for the Pipeline
dashboard's record-then-replay theater.

Every dashboard query runs the REAL pipeline once and serializes exactly what
happened — spans, agent steps, timings, token/₹ cost, retrieved chunks + scores,
groundedness verdict, provider provenance — into `artifact` (JSON). The dashboard
renders that artifact, so "faithful per query" is guaranteed by construction: we
store what ran, we never synthesize it.

A handful of columns are DENORMALIZED out of the artifact (status, total_ms,
cost_inr, step_count, confidence, provider) so the run gallery lists without
parsing every blob. `doc_id` is intentionally NOT a foreign key: a run is a
historical record and must survive the document being deleted later.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    query: Mapped[str] = mapped_column(Text, nullable=False, default="")
    doc_id: Mapped[str | None] = mapped_column(String, nullable=True)  # denormalized, not FK

    # Provenance — the model tier that actually ran ("gemini" | "offline" | "default")
    # and its concrete model name. Stamped on every artifact so the graph never
    # implies a capability the recorded run didn't exercise.
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    model_name: Mapped[str] = mapped_column(String(80), nullable=False, default="")

    # ── denormalized gallery summary (mirrors artifact["totals"]/["verdict"]) ──
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="ok")  # ok|error|abstained
    total_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cost_inr: Mapped[float | None] = mapped_column(Float, nullable=True)
    step_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    multi_hop: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # The full trace artifact the dashboard replays.
    artifact: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
