"""Extract paragraphs from a .docx with structure preserved (Teams transcripts have
speaker label + timestamp + utterance per paragraph)."""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def extract(path: Path) -> list[str]:
    paragraphs: list[str] = []
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8")
    root = ET.fromstring(xml)
    body = root.find("w:body", NS)
    if body is None:
        return paragraphs
    for p in body.findall(".//w:p", NS):
        # Each paragraph: collect all text runs in order
        texts = [t.text or "" for t in p.findall(".//w:t", NS)]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return paragraphs


def main(args):
    if not args:
        print("usage: extract_docx.py <file.docx>", file=sys.stderr)
        return 2
    path = Path(args[0])
    paragraphs = extract(path)
    for p in paragraphs:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
