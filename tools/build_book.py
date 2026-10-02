"""Expand include markers in chapter sources so listings and outputs match the real code.

Chapter sources are `book/**/*.src.md`. Each is expanded into the `.md` file next
to it (the file readers open). Two markers are supported, each on a line by itself:

    @@FILE <path relative to repo root>@@   -> replaced by that file's contents
    @@RUN <shell command>@@                 -> replaced by the command's stdout,
                                               run from code/ with python3

Run from the repository root:
    python3 tools/build_book.py            # build every chapter
    python3 tools/build_book.py ch01       # build chapters whose path contains "ch01"
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
MARKER = re.compile(r"^@@(FILE|RUN) (.+)@@$", re.MULTILINE)


def expand(match: re.Match[str]) -> str:
    kind, argument = match.group(1), match.group(2).strip()
    if kind == "FILE":
        return (ROOT / argument).read_text(encoding="utf-8").rstrip("\n")
    result = subprocess.run(
        argument, shell=True, cwd=CODE_DIR, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {argument}\n{result.stderr}")
    return result.stdout.rstrip("\n")


def main() -> None:
    selector = sys.argv[1] if len(sys.argv) > 1 else ""
    sources = sorted(p for p in (ROOT / "book").rglob("*.src.md") if selector in str(p))
    if not sources:
        raise SystemExit(f"No chapter sources match {selector!r}")
    for source in sources:
        target = source.with_name(source.name.replace(".src.md", ".md"))
        text = MARKER.sub(expand, source.read_text(encoding="utf-8"))
        target.write_text(text, encoding="utf-8")
        print(f"built {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
