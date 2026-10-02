"""Chapter 2.8: paths, text encodings, JSON, and TOML.

Writes only inside a temporary directory, which is deleted at the end.
Run from `code/`:  python examples/ch02/files_tour.py
"""

import json
import locale
import tempfile
import tomllib
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    # pathlib: build paths with "/", which works on every operating system.
    path = root / "notes" / "harbor.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    print("name:", path.name, "| suffix:", path.suffix, "| stem:", path.stem, "| parent:", path.parent.name)

    # Text vs bytes. Text is characters; files store bytes. An *encoding* is the
    # rule that converts between them. Always say which encoding you mean.
    text = "Café at the pier ☕"
    path.write_text(text, encoding="utf-8")
    raw = path.read_bytes()
    print("characters:", len(text), "| bytes on disk (UTF-8):", len(raw))
    print("first bytes:", raw[:6])
    print("default encoding on this machine:", locale.getpreferredencoding(False))

    # Reading with the wrong encoding garbles text or fails outright.
    print("read as latin-1:", repr(path.read_text(encoding="latin-1")))
    try:
        path.read_text(encoding="ascii")
    except UnicodeDecodeError as error:
        print("read as ascii: UnicodeDecodeError -", error.reason)

    # JSON: the format for data that programs write and read back.
    record = {"run": "demo", "context_size": 2, "tokens": ["the", "lamp"], "loss": None}
    json_path = root / "record.json"
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    print("JSON round trip equal:", loaded == record, "| None became:", json.dumps(None))
    print("JSON turns tuples into lists:", json.loads(json.dumps({"context": ("the", "keeper")})))

    # TOML: the format for configuration that people write by hand. It allows
    # comments, which JSON does not. Python can read TOML, but not write it.
    toml_path = root / "config.toml"
    toml_path.write_text(
        '# a comment\nseed = 0\n\n[model]\ncontext_size = 2\nlowercase = true\n', encoding="utf-8"
    )
    with open(toml_path, "rb") as file:  # tomllib needs binary mode
        config = tomllib.load(file)
    print("TOML:", config)

    # Listing files.
    print("files:", sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()))

print("temporary directory removed:", not root.exists())
