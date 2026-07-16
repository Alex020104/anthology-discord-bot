from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def masked(value: str) -> str:
    if not value:
        return "MISSING"
    if len(value) <= 10:
        return "SET"
    return value[:4] + "..." + value[-4:]


def is_placeholder(value: str) -> bool:
    lowered = (value or "").strip().lower()
    return (
        not lowered
        or "paste_" in lowered
        or "put_" in lowered
        or "change-me" in lowered
        or "твой_" in lowered
        or "новый_" in lowered
        or "любая_" in lowered
    )


def main() -> int:
    print("Anthology Discord Bot setup check")
    print("root:", ROOT)
    print(".env:", "exists" if (ROOT / ".env").exists() else "missing")

    discord_token = os.getenv("DISCORD_TOKEN", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
    bridge_token = os.getenv("BRIDGE_TOKEN", "").strip()
    port = os.getenv("PORT", "8787").strip()

    print("DISCORD_TOKEN:", masked(discord_token))
    print("OPENAI_API_KEY:", masked(openai_key))
    print("OPENAI_MODEL:", model or "MISSING")
    print("BRIDGE_TOKEN:", masked(bridge_token))
    print("PORT:", port)

    knowledge = sorted((ROOT / "knowledge").glob("*.md"))
    print("knowledge files:", len(knowledge))

    missing = []
    if is_placeholder(discord_token):
        missing.append("DISCORD_TOKEN")
    if is_placeholder(openai_key):
        missing.append("OPENAI_API_KEY")
    if is_placeholder(model):
        missing.append("OPENAI_MODEL")

    if missing:
        print("ERROR: missing required env:", ", ".join(missing))
        return 1

    try:
        import discord  # noqa: F401
        import fastapi  # noqa: F401
        import openai  # noqa: F401
        import uvicorn  # noqa: F401
    except Exception as exc:
        print("ERROR: dependency import failed:", type(exc).__name__, exc)
        return 2

    print("dependencies: ok")
    print("basic setup: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
