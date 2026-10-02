"""Audit built chapters for the book's editorial rules.

Checks each generated chapter (`book/**/ch*.md`, not `.src.md`) for:
  - unexpanded @@FILE@@ / @@RUN@@ markers
  - mathematical symbols or LaTeX in prose (code blocks and inline code are skipped)
  - filler words the style guide forbids ("simply", "obviously", "as everyone knows")
  - relative links that point to files that do not exist

Run from the repository root:
    python3 tools/audit_chapters.py          # all chapters
    python3 tools/audit_chapters.py ch03     # chapters whose path contains "ch03"
Exit status is 1 if any problem is found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATH = re.compile(r"[=×÷√∑∏≈≠≤≥±∞^]|\$|\\\(|\\\[")
FILLER = re.compile(r"\b(simply|obviously|as everyone knows)\b", re.IGNORECASE)
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def prose_lines(text: str) -> list[tuple[int, str]]:
    """Lines outside fenced code blocks, with inline code removed."""
    lines, in_fence = [], False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append((number, re.sub(r"`[^`]*`", "", line)))
    return lines


def audit(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    problems = []
    if "@@FILE" in text or "@@RUN" in text:
        problems.append("unexpanded marker")
    for number, line in prose_lines(text):
        prose = LINK.sub("]", line)  # link targets may contain symbols
        if MATH.search(prose):
            problems.append(f"line {number}: math symbol: {line.strip()[:100]}")
        if FILLER.search(prose):
            problems.append(f"line {number}: filler word: {line.strip()[:100]}")
    for target in LINK.findall(text):
        if not target.startswith(("http://", "https://", "mailto:")) and not (path.parent / target).exists():
            problems.append(f"broken link: {target}")
    return problems


def main() -> int:
    selector = sys.argv[1] if len(sys.argv) > 1 else ""
    chapters = sorted(
        p for p in (ROOT / "book").rglob("ch*.md") if not p.name.endswith(".src.md") and selector in str(p)
    )
    failed = False
    for chapter in chapters:
        problems = audit(chapter)
        print(f"{chapter.relative_to(ROOT)}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
        for problem in problems:
            print(f"    {problem}")
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
