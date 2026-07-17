from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


STOPWORDS = {
    "юра", "что", "как", "где", "куда", "когда", "почему", "если", "или", "это",
    "мне", "меня", "надо", "нужно", "можно", "делать", "сделать", "помоги",
    "вопрос", "ответ", "проблема", "ошибка", "баг", "игра", "игре", "антология",
    "anthology", "anomaly", "stalker", "the", "and", "for", "with", "what", "how",
    "where", "when", "why", "can", "should",
}

IMPORTANT_PREFIXES = (
    "ошиб", "баг", "вылет", "краш", "завис", "фриз", "микрофриз", "статтер",
    "установ", "скач", "архив", "7zip", "7-zip", "crdownload", "лаунчер",
    "обнов", "github", "permission", "denied", "доступ", "модпак", "оригинал",
    "профил", "mo2", "hard", "стандарт", "правил", "бан", "модер", "админ",
)


def normalize(text: str) -> str:
    return (text or "").casefold().replace("ё", "е")


def tokenize(text: str) -> list[str]:
    q = normalize(text)
    out: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-zа-я0-9][a-zа-я0-9_+\\-]{2,}", q):
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def token_variants(token: str) -> set[str]:
    variants = {token}
    endings = (
        "ами", "ями", "ого", "его", "ому", "ему", "иях", "ией", "иям",
        "ия", "ию", "ии", "ом", "ем", "ой", "ый", "ий", "ая", "ое", "ые",
        "ам", "ям", "ах", "ях", "ов", "ев", "ей", "ых", "их",
        "ы", "и", "а", "я", "у", "ю", "е", "о",
    )
    for ending in endings:
        if token.endswith(ending) and len(token) - len(ending) >= 4:
            variants.add(token[: -len(ending)])
    if len(token) >= 7:
        variants.add(token[:7])
    return variants


@lru_cache(maxsize=1)
def load_entries(root: str) -> list[dict]:
    path = Path(root) / "knowledge" / "support_knowledge_index.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = payload.get("entries", [])
    for entry in entries:
        haystack = " ".join(
            [
                entry.get("source", ""),
                entry.get("title", ""),
                entry.get("text", ""),
            ]
        )
        entry["_search"] = normalize(haystack)
        entry["_title"] = normalize(entry.get("title", ""))
    return entries


def score_entry(entry: dict, tokens: list[str], question: str) -> int:
    haystack = entry.get("_search", "")
    title = entry.get("_title", "")
    q = normalize(question)
    score = 0
    for token in tokens:
        variants = token_variants(token)
        if any(v in title for v in variants):
            score += 14
        if any(v in haystack for v in variants):
            score += 7
    for prefix in IMPORTANT_PREFIXES:
        if prefix in q:
            score += 18 if prefix in haystack else -4
    if "github" in q or "гитхаб" in q:
        score += 15 if ("github" in haystack or "гитхаб" in haystack) else 0
    if "7zip" in q or "7-zip" in q or "архив" in q:
        score += 20 if ("7zip" in haystack or "7-zip" in haystack or "архив" in haystack) else 0
    if "профил" in q or "hard" in q or "хард" in q:
        score += 20 if ("профил" in haystack or "hard" in haystack or "хард" in haystack) else 0
    return score


def trim(text: str, max_chars: int = 1800) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(".", 1)[0].strip()
    if len(cut) < max_chars * 0.55:
        cut = text[:max_chars].rstrip(" ,;:")
    return cut + "..."


def find_answer(question: str, root: str, min_score: int = 22, max_chars: int = 1800) -> str | None:
    entries = load_entries(root)
    if not entries:
        return None
    tokens = tokenize(question)
    if not tokens:
        return None
    best: dict | None = None
    best_score = -999
    second_score = -999
    for entry in entries:
        score = score_entry(entry, tokens, question)
        if score > best_score:
            second_score = best_score
            best = entry
            best_score = score
        elif score > second_score:
            second_score = score
    if not best or best_score < max(min_score, len(tokens) * 4):
        return None
    if best_score < 45 and second_score > 0 and best_score - second_score < 3:
        # Avoid confident answers when several unrelated docs score almost the same.
        return None
    title = best.get("title") or best.get("source") or "Anthology knowledge"
    text = best.get("text", "")
    if normalize(text).startswith(normalize(title)):
        return trim(text, max_chars=max_chars)
    return trim(f"{title}: {text}", max_chars=max_chars)
