"""
Admin-gated evaluation endpoint.

GET /eval/run?k=5 → runs the dense-only retrieval harness in an isolated temp store
and returns the metrics report as JSON, so the baseline is runnable without a shell.

Gating: enabled only for emails listed in EVAL_ADMIN_EMAILS (config). Empty (the
default) means the endpoint is disabled for everyone → 403. The harness never touches
the production Chroma/DB (it builds its own temp store per run), but running the full
ingest on request is still an admin-only operation.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.config import get_settings
from app.core.dependencies import get_current_user
from app.eval.gen_harness import check_gate, run_gen_eval
from app.eval.harness import run_eval
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["eval"])


def require_eval_admin(current_user: User = Depends(get_current_user)) -> User:
    allow = get_settings().eval_admin_emails_list
    if not allow or (current_user.email or "").lower() not in allow:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Evaluation endpoint is admin-only (configure EVAL_ADMIN_EMAILS).",
        )
    return current_user


@router.get("/run")
def run_evaluation(
    k: int = Query(default=5, ge=1, le=50, description="top-k for retrieval metrics"),
    _admin: User = Depends(require_eval_admin),
):
    """Run the dense-only baseline harness and return the full metrics report."""
    logger.info("eval: running dense-only harness (k=%d) for admin=%s", k, _admin.email)
    report = run_eval(k=k)
    return report


@router.get("/gen")
def run_generation_evaluation(
    request: Request,
    k: int = Query(default=5, ge=1, le=50, description="top-k for retrieval"),
    _admin: User = Depends(require_eval_admin),
):
    """Run the generation-quality harness (Phase 11) and return the metrics report
    plus the regression-gate verdict.

    Uses the CONFIGURED generation backend when one is wired onto app.state
    (app.state.llm_fn / verify_fn) — so on a Gemini deployment this returns the real
    neural faithfulness/relevancy numbers. When generation is not configured
    (llm_fn is None) it falls back to the deterministic offline surrogate, exactly
    like the CLI's default, so the endpoint always returns a usable report.
    """
    llm_fn = getattr(request.app.state, "llm_fn", None)
    verify_fn = getattr(request.app.state, "verify_fn", None) or llm_fn
    gen_model = getattr(request.app.state, "gen_model_name", None) or "llm"
    profile = "neural (configured provider)" if llm_fn is not None else "offline surrogate"
    logger.info("eval: running generation harness (k=%d, %s) for admin=%s", k, profile, _admin.email)

    report = run_gen_eval(k=k, llm_fn=llm_fn, judge_fn=verify_fn, gen_model=gen_model)
    passed, failures = check_gate(report)
    report["gate"] = {"passed": passed, "failures": failures}
    return report
