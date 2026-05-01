from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: str
    source_domain: str
    filename: str
    text: str
    char_start: int


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float


@dataclass
class RetrievalResult:
    hits: list[RetrievalHit]
    confidence: str

    @property
    def chunks(self) -> list[Chunk]:
        return [hit.chunk for hit in self.hits]


@dataclass
class EscalationSignal:
    hard_flags: list[str] = field(default_factory=list)
    soft_flags: list[str] = field(default_factory=list)
