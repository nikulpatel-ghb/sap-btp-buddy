"""Answering and evidence-policy boundary."""

from .config import RunConfig
from .models import Answer


def answer_question(
    question: str,
    index: object,
    *,
    config: RunConfig | None = None,
) -> Answer:
    """Retrieve evidence, route answerability, and generate a cited answer."""

    raise NotImplementedError(
        "Answering implementation is the next build milestone; keep this public API stable."
    )
