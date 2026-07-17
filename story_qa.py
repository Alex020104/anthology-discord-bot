from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


STOPWORDS = {
    "юра", "что", "как", "где", "куда", "когда", "почему", "если", "или", "это", "там", "тут",
    "мне", "меня", "надо", "нужно", "можно", "сюжет", "сюжета", "сюжете", "квест", "квесте",
    "задание", "задании", "объясни", "расскажи", "помоги", "застрял", "застряла", "проблема",
    "нахожусь", "находится", "делать", "идти", "игрок", "игрока", "по", "на", "в", "из", "от",
    "the", "and", "for", "with", "what", "how", "where", "when", "why",
}

GAME_ALIASES = {
    "Тень Чернобыля": ("тень", "чернобыл", "shadow", "soc", "тч"),
    "Зов Припяти": ("зов", "припят", "call", "cop", "зп"),
    "Чистое Небо": ("чист", "небо", "clear", "sky", "cs", "чн"),
}

IMPORTANT_PREFIXES = (
    "пулем", "военн", "кордон", "убежать", "убива", "стреля", "миновать", "блокпост",
    "химэр", "химер", "азот", "цемент", "инструмент", "соколов", "тополь", "ноутбук",
    "наемник", "наёмник", "ренегат", "волк", "бандит", "круглов", "сахаров", "хабар",
    "пуля", "свобод", "долгов", "штурм", "x-8", "х-8", "x8", "х8",
)


def normalize(text: str) -> str:
    text = (text or "").casefold().replace("ё", "е")
    text = text.replace("x-", "х-").replace("x8", "х8").replace("x-8", "х-8")
    return text


def tokenize(text: str) -> list[str]:
    text = normalize(text)
    out: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-zа-я0-9][a-zа-я0-9\-]{2,}", text):
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def token_variants(token: str) -> set[str]:
    variants = {token}
    endings = (
        "ами", "ями", "ого", "его", "ому", "ему", "иях", "ией", "иям", "ями",
        "ия", "ию", "ии", "ом", "ем", "ой", "ый", "ий", "ая", "ое", "ые",
        "ам", "ям", "ах", "ях", "ов", "ев", "ей", "ых", "их",
        "ы", "и", "а", "я", "у", "ю", "е", "о",
    )
    for ending in endings:
        if token.endswith(ending) and len(token) - len(ending) >= 4:
            variants.add(token[: -len(ending)])
    if len(token) >= 6:
        variants.add(token[:6])
    if len(token) >= 8:
        variants.add(token[:8])
    return variants


def wanted_game(question: str) -> str:
    q = normalize(question)
    for game, aliases in GAME_ALIASES.items():
        if any(alias in q for alias in aliases):
            return game
    return ""


@lru_cache(maxsize=1)
def load_entries(root: str) -> list[dict]:
    path = Path(root) / "knowledge" / "story_qa_index.json"
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
                entry.get("game", ""),
                entry.get("section", ""),
                entry.get("title", ""),
                entry.get("parent", ""),
                " ".join(entry.get("questions", [])),
                " ".join(entry.get("keywords", [])),
                entry.get("raw_paragraph", ""),
                entry.get("answer", ""),
            ]
        )
        entry["_search"] = normalize(haystack)
        entry["_title"] = normalize(entry.get("section") or entry.get("title") or "")
        entry["_raw"] = normalize(entry.get("raw_paragraph", ""))
    return entries


def _contains_any(haystack: str, prefixes: tuple[str, ...] | list[str]) -> bool:
    return any(prefix in haystack for prefix in prefixes)


def score_entry(entry: dict, q_tokens: list[str], game: str, question: str) -> int:
    haystack = entry.get("_search", "")
    title = entry.get("_title", "")
    raw = entry.get("_raw", "")
    score = 0
    if game:
        score += 40 if entry.get("game") == game else -80

    for token in q_tokens:
        variants = token_variants(token)
        if any(v in title for v in variants):
            score += 18
        if any(v in raw for v in variants):
            score += 14
        elif any(v in haystack for v in variants):
            score += 6

    q = normalize(question)
    q_important = [p for p in IMPORTANT_PREFIXES if p in q]
    for prefix in q_important:
        if prefix in raw:
            score += 30
        elif prefix in haystack:
            score += 12
        else:
            score -= 10

    # Common stuck-player situations: prefer the concrete paragraph, not the broad chapter intro.
    if "кордон" in q and "кордон" in title:
        score += 35
    if _contains_any(q, ("пулем", "пулемет", "пулеметчик")):
        score += 80 if "пулем" in raw else -25
    if _contains_any(q, ("убежать", "убива", "стреля")) and _contains_any(raw, ("огнем", "миновать", "блокпост", "пулем")):
        score += 35
    if "база чистое небо" in title and "кордон" in q:
        score -= 45

    if {"команд", "припят"} & set(q_tokens) and "припять-1" in title:
        score += 25
    if "химер" in q or "химэр" in q:
        score += 35 if ("химер" in haystack or "химэр" in haystack) else -20
    if "цемент" in q:
        score += 35 if "цемент" in haystack else -20
    if any(t.startswith("азот") for t in q_tokens):
        score += 35 if "азот" in haystack else -20
    if _contains_any(q, ("инструмент", "радио", "запчаст")):
        score += 30 if ("радиотехника" in title or "инструмент" in haystack or "запчаст" in haystack) else 0
    return score


def clean_answer(text: str) -> str:
    service_phrases = (
        "Если у игрока пропал маркер, ориентируй его по названию текущей локации/NPC из этого ответа и не подменяй другим квестом.",
    )
    text = text or ""
    for phrase in service_phrases:
        text = text.replace(phrase, "")
    text = re.sub(r"\s{2,}", " ", text).strip()
    # Do not end the answer with obvious chopped connective words.
    text = re.sub(r"[\s,;:—-]+(чтобы|который|которая|которые|где|если|так как|и|в|на|по)$", ".", text, flags=re.I)
    return text.strip()


def find_answer(question: str, root: str, min_score: int = 24) -> str | None:
    entries = load_entries(root)
    if not entries:
        return None
    q_tokens = tokenize(question)
    if not q_tokens:
        return None
    game = wanted_game(question)
    best: dict | None = None
    best_score = -999
    for entry in entries:
        score = score_entry(entry, q_tokens, game, question)
        if score > best_score:
            best = entry
            best_score = score
    if not best or best_score < min_score:
        return None
    return clean_answer(best.get("answer") or "")
