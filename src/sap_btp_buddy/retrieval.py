"""Dense, lexical, and hybrid retrieval boundary."""

from typing import Iterable

from .models import Chunk, SearchResult


def retrieve(
    query: str,
    chunks: Iterable[Chunk],
    *,
    mode: str = "hybrid",
    top_k: int = 5,
) -> list[SearchResult]:
    """Retrieve bounded candidates from approved indexed chunks."""

    if hasattr(chunks, "retrieve"):
        return chunks.retrieve(query, mode=mode, top_k=top_k)  # type: ignore[no-any-return]

    raise NotImplementedError("Retrieval is PENDING; no candidates were produced.")
