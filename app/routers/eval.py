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

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import get_settings
from app.core.dependencies import get_current_user
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
