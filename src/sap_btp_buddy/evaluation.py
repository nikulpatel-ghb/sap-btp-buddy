"""Evaluation orchestration; metric definitions stay out of notebooks and UI."""

from .models import AnswerFn, EvaluationReport, QuestionSet


def evaluate_questions(
    questions: QuestionSet,
    answer_fn: AnswerFn,
) -> EvaluationReport:
    """Run a frozen question set and return per-question evidence for reporting."""

    raise NotImplementedError(
        "Evaluation implementation is the next build milestone; keep this public API stable."
    )
