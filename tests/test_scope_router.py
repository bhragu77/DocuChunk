"""
Scope-Aware Retrieval Routing — the pure heuristic classifiers.

classify_task decides the OUTPUT SHAPE; classify_scope decides LOCAL (top-k) vs
GLOBAL (whole document). Both are pure functions — no LLM, no I/O — so these
tests are fast and deterministic. The canonical fixtures are the exact Keebler
questions that motivated the feature: fact questions must stay LOCAL/ANSWER;
enumeration / table / classify / summarize / critique / infer questions must
route GLOBAL with the right task.
"""
import pytest

from app.generation.query_classifier import (
    AnswerTask,
    RetrievalScope,
    classify_scope,
    classify_task,
    task_instructions_for,
)


# ── The six failing Keebler questions → (task, scope) ─────────────────────────

@pytest.mark.parametrize("query,task", [
    ("List all elves along with their roles in one sentence.", AnswerTask.ENUMERATE),
    ("Create a structured table mapping each elf to their role.", AnswerTask.TABLE),
    ("Classify elves into categories: marketing, production, support roles.", AnswerTask.CLASSIFY),
    ("If a new elf were added for logistics, what responsibilities might they have "
     "based on existing roles?", AnswerTask.INFER),
    ("Summarize the paragraph in 2-3 sentences.", AnswerTask.SUMMARIZE),
    ("Identify any inconsistencies or unusual descriptions in the paragraph.", AnswerTask.CRITIQUE),
])
def test_keebler_global_questions_get_the_right_task(query, task):
    assert classify_task(query) is task
    # Every non-ANSWER task is inherently GLOBAL.
    assert classify_scope(query) is RetrievalScope.GLOBAL


# ── Fact questions must stay LOCAL / ANSWER (don't regress the fast path) ──────

@pytest.mark.parametrize("query", [
    "Who is the accountant?",
    "Which elf is described as young?",
    "Who promoted Munch-ems?",
    "What is the relationship between Ernie and Ma Keebler?",
    "Name any three Keebler elves mentioned in the paragraph.",  # "any three", not "all"
    "Who is the peanut butter baker?",
])
def test_fact_questions_stay_local(query):
    assert classify_task(query) is AnswerTask.ANSWER
    assert classify_scope(query) is RetrievalScope.LOCAL


# ── Precedence: ENUMERATE beats SUMMARIZE when "list/all" + "one sentence" collide

def test_list_all_in_one_sentence_is_enumerate_not_summarize():
    q = "List all elves along with their roles in one sentence."
    # "in one sentence" is a SUMMARIZE cue, but "list"/"all" (ENUMERATE) is checked
    # first — we want the exhaustive list, not a compressed summary.
    assert classify_task(q) is AnswerTask.ENUMERATE


def test_plain_summary_without_enumeration_is_summarize():
    assert classify_task("Summarize focusing only on roles") is AnswerTask.SUMMARIZE


# ── Triggers don't fire inside unrelated words (space-padded matching) ─────────

@pytest.mark.parametrize("query", [
    "What is the account balance?",   # "account" must not fire "count"
    "Who lives in the small house?",  # "small" must not fire "all"
    "Which teacher is mentioned?",    # "teacher" must not fire "each"
])
def test_no_false_global_from_substrings(query):
    assert classify_scope(query) is RetrievalScope.LOCAL


# ── Standalone global cues (structure questions) route GLOBAL even as ANSWER ───

def test_reconstruct_hierarchy_is_global():
    q = "Reconstruct the hierarchy of cookie production using the given characters."
    assert classify_scope(q) is RetrievalScope.GLOBAL


# ── Empty / whitespace query is a safe default ────────────────────────────────

def test_empty_query_defaults_local_answer():
    assert classify_task("") is AnswerTask.ANSWER
    assert classify_scope("") is RetrievalScope.LOCAL
    assert classify_scope("   ", AnswerTask.ANSWER) is RetrievalScope.LOCAL


# ── Task instructions block ───────────────────────────────────────────────────

def test_task_instructions_present_for_tasks_absent_for_answer():
    assert task_instructions_for(AnswerTask.ANSWER) == ""
    assert task_instructions_for(None) == ""
    for task in (AnswerTask.ENUMERATE, AnswerTask.TABLE, AnswerTask.CLASSIFY,
                 AnswerTask.SUMMARIZE, AnswerTask.CRITIQUE, AnswerTask.INFER):
        assert task_instructions_for(task), f"{task} should have a TASK block"


def test_scope_passing_task_avoids_reclassification():
    # classify_scope trusts an explicitly-passed task.
    assert classify_scope("anything at all", AnswerTask.TABLE) is RetrievalScope.GLOBAL
    assert classify_scope("anything at all", AnswerTask.ANSWER) is RetrievalScope.LOCAL
