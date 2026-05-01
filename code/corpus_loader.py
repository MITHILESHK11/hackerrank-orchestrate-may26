from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from pathlib import Path

from models import Chunk


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)

    def get_text(self) -> str:
        return "".join(self.parts)


def _read_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        parser = _HTMLTextExtractor()
        parser.feed(raw)
        raw = parser.get_text()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    return raw.strip()


def _domain_from_path(relative_path: Path) -> str:
    if not relative_path.parts:
        return "generic"
    root = relative_path.parts[0].lower()
    if root in {"hackerrank", "claude", "visa"}:
        return root
    return "generic"


def _paragraph_segments(text: str) -> list[tuple[str, int]]:
    segments: list[tuple[str, int]] = []
    for match in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|$)", text):
        value = match.group(0).strip()
        if value:
            segments.append((value, match.start()))
    return segments


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _chunk_from_segments(
    segments: list[tuple[str, int]],
    chunk_size_words: int,
    overlap_words: int,
) -> list[tuple[str, int]]:
    if not segments:
        return []

    chunks: list[tuple[str, int]] = []
    i = 0
    while i < len(segments):
        word_total = 0
        j = i
        while j < len(segments):
            segment_words = _word_count(segments[j][0])
            if word_total > 0 and word_total + segment_words > chunk_size_words:
                break
            word_total += segment_words
            j += 1
            if word_total >= chunk_size_words:
                break

        if j == i:
            j += 1

        joined = "\n\n".join(segment for segment, _ in segments[i:j]).strip()
        start = segments[i][1]
        if joined:
            chunks.append((joined, start))

        if j >= len(segments):
            break

        overlap_target = overlap_words
        k = j - 1
        back_words = 0
        while k >= i:
            back_words += _word_count(segments[k][0])
            if back_words >= overlap_target:
                break
            k -= 1
        i = max(k, i + 1)

    return chunks


def _fallback_sliding_windows(
    text: str,
    chunk_size_words: int,
    overlap_words: int,
) -> list[tuple[str, int]]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    step = max(1, chunk_size_words - overlap_words)
    chunks: list[tuple[str, int]] = []
    start_idx = 0
    while start_idx < len(words):
        end_idx = min(len(words), start_idx + chunk_size_words)
        fragment = " ".join(words[start_idx:end_idx]).strip()
        if fragment:
            anchor = fragment[:50]
            char_start = text.find(anchor)
            if char_start < 0:
                char_start = 0
            chunks.append((fragment, char_start))
        if end_idx == len(words):
            break
        start_idx += step
    return chunks


def load_corpus(data_dir: str) -> list[Chunk]:
    base = Path(data_dir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {base}")

    allowed_ext = {".txt", ".md", ".html", ".htm"}
    file_paths: list[Path] = []
    for root, _, files in os.walk(base):
        for filename in files:
            candidate = Path(root) / filename
            if candidate.suffix.lower() in allowed_ext:
                file_paths.append(candidate)
    file_paths.sort(key=lambda p: str(p).lower())

    chunks: list[Chunk] = []
    for path in file_paths:
        rel = path.relative_to(base)
        domain = _domain_from_path(rel)
        text = _read_text(path)
        if not text:
            continue

        segments = _paragraph_segments(text)
        built = _chunk_from_segments(
            segments=segments,
            chunk_size_words=300,
            overlap_words=60,
        )
        if not built:
            built = _fallback_sliding_windows(
                text=text,
                chunk_size_words=300,
                overlap_words=60,
            )

        for idx, (chunk_text, char_start) in enumerate(built):
            chunk_id = f"{domain}:{rel.as_posix()}:{idx}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    source_domain=domain,
                    filename=rel.as_posix(),
                    text=chunk_text,
                    char_start=max(0, int(char_start)),
                )
            )

    chunks.sort(key=lambda c: (c.source_domain, c.filename, c.char_start))
    return chunks


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1] / "data"
    loaded = load_corpus(str(base_dir))
    assert loaded
    domains = sorted({c.source_domain for c in loaded})
    print(f"Loaded {len(loaded)} chunks from {len(domains)} domains")
