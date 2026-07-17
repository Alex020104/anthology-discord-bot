from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


SOURCE_DIR_NAME = "full_sources"
SUPPORTED_EXTENSIONS = {".md", ".txt"}

STOPWORDS = {
    "юра", "что", "как", "где", "куда", "когда", "почему", "если", "или",
    "это", "там", "тут", "мне", "меня", "надо", "нужно", "можно",
    "сюжет", "сюжета", "сюжете", "квест", "квесте", "задание", "задании",
    "объясни", "расскажи", "помоги", "застрял", "застряла", "проблема",
    "находится", "делать", "идти", "игрок", "игрока", "мой", "моя",
    "the", "and", "for", "with", "what", "how", "where", "when", "why",
}

GAME_HINTS = {
    "Тень Чернобыля": ("тень", "чернобыл", "shadow", "soc"),
    "Зов Припяти": ("зов", "припят", "call", "cop"),
    "Чистое Небо": ("чист", "небо", "clear", "sky"),
}

GAME_ALIAS_TOKENS = {
    alias
    for aliases in GAME_HINTS.values()
    for alias in aliases
    if len(alias) >= 3 and alias not in {"soc", "cop"}
}

STORY_HINTS = (
    "сюжет", "квест", "задание", "маркер", "куда идти", "что должно произойти",
    "тайник", "лаборатор", "подземель", "локац", "npc", "нпс", "сталкер",
    "стрелок", "круглов", "сахаров", "волк", "сидорович", "глухар", "тремор",
    "кардан", "азот", "соколов", "тополь", "зверобой", "ной", "лоцман",
    "кордон", "затон", "юпитер", "припять", "агропром", "свалка", "бар",
    "янтар", "рыж", "чаэс", "х-8", "х8", "x-8", "x8", "скат-", "б2", "б28",
)


def normalize(text: str) -> str:
    text = (text or "").casefold().replace("ё", "е")
    text = text.replace("x-", "х-").replace("x8", "х8").replace("x-8", "х-8")
    return text


def tokenize(text: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-zа-я0-9][a-zа-я0-9_+\\-]{2,}", normalize(text)):
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


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


def wanted_game(question: str) -> str:
    q = normalize(question)
    for game, aliases in GAME_HINTS.items():
        if any(alias in q for alias in aliases):
            return game
    return ""


def looks_like_story_question(question: str) -> bool:
    q = normalize(question)
    if wanted_game(q):
        return True
    if re.search(r"\bскат\s*-?\s*\d+\b", q):
        return True
    return any(hint in q for hint in STORY_HINTS)


def split_sections(text: str, source_name: str) -> list[dict]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[dict] = []
    current_title = source_name
    current: list[str] = []

    def flush() -> None:
        body = "\n".join(current).strip()
        if body:
            sections.append({"title": current_title.strip() or source_name, "text": body})

    for line in lines:
        stripped = line.strip()
        is_heading = (
            stripped.startswith("#")
            or (stripped and len(stripped) <= 120 and not stripped.endswith(".") and not stripped.startswith(("-", "•", "1.", "2.", "3.")))
        )
        if is_heading and len(current) >= 2:
            flush()
            current = []
            current_title = stripped.lstrip("#").strip() or source_name
        else:
            current.append(line)
    flush()
    return sections


def chunk_text(section: dict, source_name: str, target_chars: int = 2600, overlap_chars: int = 500) -> list[dict]:
    text = section["text"].strip()
    if len(text) <= target_chars:
        return [{"source": source_name, "title": section["title"], "text": text}]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[dict] = []
    buf = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if buf and len(buf) + len(paragraph) + 2 > target_chars:
            chunks.append({"source": source_name, "title": section["title"], "text": buf.strip()})
            buf = buf[-overlap_chars:].strip()
        buf = (buf + "\n\n" + paragraph).strip()
    if buf:
        chunks.append({"source": source_name, "title": section["title"], "text": buf.strip()})
    return chunks


@lru_cache(maxsize=1)
def load_chunks(root: str) -> list[dict]:
    source_dir = Path(root) / "knowledge" / SOURCE_DIR_NAME
    if not source_dir.exists():
        return []
    chunks: list[dict] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
            continue
        if path.name.casefold() == "readme.md":
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace").strip()
        except Exception:
            continue
        if not text:
            continue
        stem = path.stem.casefold()
        if "soc" in stem:
            source_name = "Тень Чернобыля"
        elif "cs" in stem:
            source_name = "Чистое Небо"
        elif "cop" in stem:
            source_name = "Зов Припяти"
        else:
            source_name = path.stem.replace("_", " ")
        for section in split_sections(text, source_name):
            chunks.extend(chunk_text(section, source_name))
    for chunk in chunks:
        haystack = " ".join([chunk.get("source", ""), chunk.get("title", ""), chunk.get("text", "")])
        chunk["_search"] = normalize(haystack)
        chunk["_title"] = normalize(chunk.get("title", ""))
    return chunks


def score_chunk(chunk: dict, tokens: list[str], question: str, game: str) -> int:
    haystack = chunk.get("_search", "")
    title = chunk.get("_title", "")
    score = 0
    if game:
        score += 70 if game.casefold() in haystack else -90
    q = normalize(question)
    if title and title in q and title != normalize(chunk.get("source", "")):
        score += 100
    for token in tokens:
        if token in GAME_ALIAS_TOKENS:
            continue
        variants = token_variants(token)
        if any(v in title for v in variants):
            score += 45
        if any(v in haystack for v in variants):
            score += 8
    if any(word in q for word in ("убить", "убью", "перебить", "застрелить", "атаковать")):
        if any(word in haystack for word in ("перебить", "рейд", "медвед", "атак", "расправ")):
            score += 65
        if any(word in haystack for word in ("выкуп", "заплат", "артефакт", "обмен")) and not any(word in haystack for word in ("перебить", "рейд", "атак")):
            score -= 25
    for exact in re.findall(r"[a-zа-я]+-?\d+", q):
        if exact and exact in title:
            score += 90
        elif exact and exact in haystack:
            score += 35
    for important in ("глухар", "тремор", "кровосос", "кардан", "азот", "химера", "ноутбук", "наемник", "соколов", "тополь", "пулемет", "кордон", "припят", "х-8", "скат"):
        if important in q:
            score += 35 if important in haystack else -12
    return score


def trim_answer(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(".", 1)[0].strip()
    if len(cut) < max_chars * 0.55:
        cut = text[:max_chars].rstrip(" ,;:")
    return cut + "..."


def find_context(question: str, root: str, min_score: int = 34) -> dict | None:
    chunks = load_chunks(root)
    if not chunks:
        return None
    if not looks_like_story_question(question):
        return None
    tokens = tokenize(question)
    if not tokens:
        return None
    game = wanted_game(question)
    candidate_chunks = [chunk for chunk in chunks if not game or chunk.get("source") == game]
    if not candidate_chunks:
        candidate_chunks = chunks
    scored = sorted(
        ((score_chunk(chunk, tokens, question, game), chunk) for chunk in candidate_chunks),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -999
    if best_score < max(min_score, len(tokens) * 5):
        return None
    if best_score < 70 and second_score > 0 and best_score - second_score < 5:
        return None
    return {
        "source": best.get("source") or "full_sources",
        "title": best.get("title") or best.get("source") or "Источник",
        "text": best.get("text", ""),
        "score": best_score,
    }


def find_answer(question: str, root: str, min_score: int = 34, max_chars: int = 1800) -> str | None:
    best = find_context(question, root, min_score=min_score)
    if not best:
        return None
    title = best.get("title") or best.get("source") or "Источник"
    source = best.get("source") or "full_sources"
    answer = trim_answer(best.get("text", ""), max_chars=max_chars)
    return f"{title} ({source}): {answer}"


def clear_cache() -> None:
    load_chunks.cache_clear()
