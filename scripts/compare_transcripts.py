"""Cross-check named entities and key terms between v2 output and Teams docx."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def count(text: str, needle: str) -> int:
    return len(re.findall(r"\b" + re.escape(needle) + r"\b", text))


def main() -> int:
    v2_path = Path("output/v2-smoke-teams-diarised.txt")
    docx_text_path = Path("output/teams-docx-extracted.txt")

    v2 = v2_path.read_text(encoding="utf-8")
    teams = docx_text_path.read_text(encoding="utf-8") if docx_text_path.exists() else ""

    names = [
        "Pradeep", "Jacob", "James", "David", "Ejaz", "Scott", "Claire",
        "Harry", "Kenneth", "Castillo", "Nugent", "Gould", "Georgia", "Peter",
    ]
    terms = [
        "RITM", "macro block", "website block", "MVP", "SC task", "SE task",
        "exemption", "ServiceNow", "cyber assurance", "cyber defence",
        "cybersecurity", "draft", "draught",
    ]

    print(f"{'Term':<22} {'v2':>4} {'Teams':>6}")
    print("-" * 36)
    for n in names + terms:
        v = count(v2, n) if " " not in n else v2.lower().count(n.lower())
        t = count(teams, n) if " " not in n else teams.lower().count(n.lower())
        print(f"{n:<22} {v:>4} {t:>6}")

    print()
    print(f"v2 chars : {len(v2):>7}  ({v2_path})")
    print(f"teams chars: {len(teams):>5}  ({docx_text_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
