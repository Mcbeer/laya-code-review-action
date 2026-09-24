"""Split a unified diff into deterministic, model-sized chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterator

TokenCounter = Callable[[str], int]

_FILE_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+)$")


@dataclass(frozen=True)
class Chunk:
    path: str
    hunk: str
    body: str

    def as_state(self) -> str:
        return render_state(self.path, self.hunk, self.body)


def render_state(path: str, hunk: str, body: str) -> str:
    """The exact text the model sees for one chunk. Evaluation uses it too."""
    return f"File: {path}\n{hunk}\n{body}"


@dataclass
class _FileDiff:
    path: str
    deleted: bool = False
    binary: bool = False
    hunks: list[list[str]] = field(default_factory=list)


def chunk_diff(diff_text: str, count_tokens: TokenCounter, budget: int) -> list[Chunk]:
    """Chunk every hunk that adds lines, so no chunk exceeds `budget` tokens.

    Deleted files, binary files and hunks without added lines are skipped: the
    rules judge code the PR introduces. A single line longer than the budget
    still becomes its own chunk and is truncated by the model.
    """
    chunks: list[Chunk] = []
    for file in _parse_files(diff_text):
        if file.deleted or file.binary:
            continue
        for hunk in file.hunks:
            chunks.extend(_split_hunk(file.path, hunk, count_tokens, budget))
    return chunks


def _parse_files(diff_text: str) -> list[_FileDiff]:
    files: list[_FileDiff] = []
    current: _FileDiff | None = None
    for line in diff_text.splitlines():
        header = _FILE_HEADER.match(line)
        if header:
            current = _FileDiff(path=header.group(2))
            files.append(current)
        elif current is None:
            continue
        elif line.startswith("@@"):
            current.hunks.append([line])
        elif current.hunks:
            current.hunks[-1].append(line)
        elif line == "+++ /dev/null":
            current.deleted = True
        elif line.startswith("Binary files ") or line == "GIT binary patch":
            current.binary = True
    return files


def _split_hunk(
    path: str, lines: list[str], count_tokens: TokenCounter, budget: int
) -> Iterator[Chunk]:
    header, body = lines[0], lines[1:]
    # Token counts are summed per line (+1 for the newline) so chunking stays
    # linear in hunk size. The sum can differ slightly from tokenizing the
    # joined text; callers keep a safety margin in `budget`.
    base = count_tokens(render_state(path, header, ""))
    current: list[str] = []
    used = base
    for line in body:
        cost = count_tokens(line) + 1
        if current and used + cost > budget:
            yield from _emit(path, header, current)
            current, used = [], base
        current.append(line)
        used += cost
    yield from _emit(path, header, current)


def _emit(path: str, header: str, lines: list[str]) -> Iterator[Chunk]:
    if any(line.startswith("+") for line in lines):
        yield Chunk(path=path, hunk=header, body="\n".join(lines))
