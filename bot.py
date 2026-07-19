from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any
from pathlib import Path

import discord
import uvicorn
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from openai import AsyncOpenAI

import story_qa
import general_knowledge
import full_sources


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO)

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
BOT_DISPLAY_NAME = os.getenv("BOT_DISPLAY_NAME", "\u042e\u0440\u0430 \u0421\u0435\u043c\u0435\u0446\u043a\u0438\u0439").strip()
DEFAULT_TRIGGER_NAMES = "anthology_bot,\u044e\u0440\u0430,\u044e\u0440\u0430 \u0441\u0435\u043c\u0435\u0446\u043a\u0438\u0439,\u0430\u043d\u0442\u043e\u043b\u043e\u0433\u0438\u044f \u0431\u043e\u0442,yura,yura semetsky,anthology bot"
BOT_TRIGGER_NAMES = [
    item.strip().casefold()
    for item in os.getenv("BOT_TRIGGER_NAMES", DEFAULT_TRIGGER_NAMES).split(",")
    if item.strip()
]
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID", "").strip()
BRIDGE_TOKEN = os.getenv("BRIDGE_TOKEN", "").strip()
PORT = int(os.getenv("PORT", "8787"))
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "3600"))
DISCORD_CHUNK_CHARS = 1850
BRIDGE_RATE_SECONDS = float(os.getenv("BRIDGE_RATE_SECONDS", "2.0"))
CONVERSATION_TTL_SECONDS = float(os.getenv("CONVERSATION_TTL_SECONDS", "900"))
OPENAI_ENABLED = os.getenv("OPENAI_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
AUTO_REPLY_QUESTION_CHANNEL_IDS = {
    item.strip()
    for item in os.getenv("AUTO_REPLY_QUESTION_CHANNEL_IDS", "").split(",")
    if item.strip()
}

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is not set.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, max_retries=0, timeout=12.0)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
app = FastAPI(title="Anthology Discord Bot Bridge")
CONVERSATION_CONTEXT: dict[str, dict[str, str]] = {}
BRIDGE_LAST_REQUEST_BY_IP: dict[str, float] = {}


def iter_knowledge_paths() -> list[Path]:
    paths = sorted(KNOWLEDGE_DIR.rglob("*.md"), key=lambda item: item.name.casefold())
    story_prefixes = ("quest_", "stalker_")
    low_priority_prefixes = ("downloads_",)
    story = [path for path in paths if path.name.casefold().startswith(story_prefixes)]
    normal = [
        path
        for path in paths
        if path not in story and not path.name.casefold().startswith(low_priority_prefixes)
    ]
    low_priority = [path for path in paths if path.name.casefold().startswith(low_priority_prefixes)]
    return story + normal + low_priority


def load_knowledge() -> str:
    parts: list[str] = []
    for path in iter_knowledge_paths():
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if text:
            chunk = f"## {path.relative_to(KNOWLEDGE_DIR).as_posix()}\n{text}"
            parts.append(chunk)
    return "\n\n".join(parts)


KNOWLEDGE = load_knowledge()


def trim_answer(text: str) -> str:
    text = " ".join((text or "").replace("\r", "\n").split())
    if len(text) > MAX_ANSWER_CHARS:
        return text[: MAX_ANSWER_CHARS - 1].rstrip() + "..."
    return text


def compact_text(text: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def latest_player_question(text: str) -> str:
    value = text or ""
    marker = "Уточнение игрока:"
    if marker in value:
        value = value.rsplit(marker, 1)[-1]
        value = value.split("\n\nВажно:", 1)[0]
    return value.strip()


def token_variants(token: str) -> set[str]:
    token = (token or "").casefold().strip()
    variants = {token} if token else set()
    if len(token) < 5:
        return variants
    endings = (
        "\u0430\u043c\u0438", "\u044f\u043c\u0438", "\u043e\u0433\u043e", "\u0435\u0433\u043e", "\u043e\u043c\u0443", "\u0435\u043c\u0443",
        "\u0438\u0438", "\u0438\u044e", "\u0438\u044f", "\u0435\u043c", "\u043e\u043c", "\u044b\u0439", "\u0438\u0439", "\u043e\u0439",
        "\u0430\u043c", "\u044f\u043c", "\u0430\u0445", "\u044f\u0445", "\u043e\u0432", "\u0435\u0432", "\u0435\u0439",
        "\u0430\u043c", "\u044f\u043c", "\u0430\u043c\u0438", "\u044f\u043c\u0438", "\u044b", "\u0438", "\u0430", "\u044f",
        "\u0443", "\u044e", "\u0435", "\u043e", "\u0439",
    )
    for ending in endings:
        if token.endswith(ending) and len(token) - len(ending) >= 4:
            variants.add(token[: -len(ending)])
    special = {
        "\u0432\u043e\u043b\u043a\u043e\u043c": "\u0432\u043e\u043b\u043a",
        "\u0432\u043e\u043b\u043a\u0430": "\u0432\u043e\u043b\u043a",
        "\u043a\u0440\u043e\u0442\u0430": "\u043a\u0440\u043e\u0442",
        "\u043a\u0440\u043e\u0442\u0443": "\u043a\u0440\u043e\u0442",
        "\u0431\u0430\u043d\u0434\u0438\u0442\u0430\u043c\u0438": "\u0431\u0430\u043d\u0434\u0438\u0442",
        "\u0431\u0430\u043d\u0434\u0438\u0442\u043e\u0432": "\u0431\u0430\u043d\u0434\u0438\u0442",
        "\u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u0438\u0438": "\u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440",
        "\u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u0438\u044e": "\u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440",
        "\u043f\u0440\u0438\u043f\u044f\u0442\u0438": "\u043f\u0440\u0438\u043f\u044f\u0442",
        "\u0430\u0433\u0440\u043e\u043f\u0440\u043e\u043c\u0435": "\u0430\u0433\u0440\u043e\u043f\u0440\u043e\u043c",
    }
    if token in special:
        variants.add(special[token])
    return {v for v in variants if len(v) >= 4}


def match_score(text: str, tokens: list[str], *, title: bool = False) -> int:
    text = (text or "").casefold()
    score = 0
    for token in tokens:
        variants = token_variants(token)
        if not variants:
            continue
        exact = any(v in text for v in variants)
        if exact:
            score += 10 if title else 6
            continue
        prefix = next((v for v in variants if len(v) >= 5 and v[:5] in text), None)
        if prefix:
            score += 4 if title else 2
    return score


def wanted_story_game(text: str) -> str | None:
    text = (text or "").casefold()
    if "\u0437\u043e\u0432" in text and "\u043f\u0440\u0438\u043f\u044f\u0442" in text:
        return "cop"
    if ("\u0442\u0435\u043d\u044c" in text and "\u0447\u0435\u0440\u043d\u043e\u0431" in text) or "\u0442\u0447" in text:
        return "soc"
    if "\u0447\u0438\u0441\u0442\u043e\u0435" in text and "\u043d\u0435\u0431\u043e" in text:
        return "cs"
    return None


def story_game_score(text: str, wanted: str | None) -> int:
    if not wanted:
        return 0
    text = (text or "").casefold()
    markers = {
        "cop": ("\u0437\u043e\u0432 \u043f\u0440\u0438\u043f\u044f\u0442", "\u0437\u043f:", "\u0437\u043f "),
        "soc": ("\u0442\u0435\u043d\u044c \u0447\u0435\u0440\u043d\u043e\u0431", "\u0442\u0447:", "\u0442\u0447 "),
        "cs": ("\u0447\u0438\u0441\u0442\u043e\u0435 \u043d\u0435\u0431\u043e", "\u0447\u043d:", "\u0447\u043d "),
    }
    own = any(marker in text for marker in markers[wanted])
    other = any(
        marker in text
        for game, game_markers in markers.items()
        if game != wanted
        for marker in game_markers
    )
    if other and not own:
        return -80
    if own:
        return 16
    return 0


def story_route_answer(lowered: str, language: str) -> str:
    if language == "English":
        return (
            "If a story marker is missing, navigate by the main story chain and named transitions. "
            "Shadow of Chernobyl: Cordon -> Garbage -> Agroprom -> Bar/Rostok -> Dark Valley -> X-18 -> Yantar/X-16 -> Radar/X-10 -> Pripyat -> CNPP. "
            "Clear Sky: Swamps -> Cordon -> Garbage -> Dark Valley -> Agroprom -> Yantar -> Red Forest -> Limansk -> Hospital -> CNPP. "
            "Call of Pripyat: Zaton -> Jupiter -> Pripyat -> finale/evacuation. "
            "For exact left/right directions, mention your current location, entry point, and the nearest landmark."
        )
    if "\u0442\u0435\u043d\u044c" in lowered or "\u0447\u0435\u0440\u043d\u043e\u0431" in lowered:
        return (
            "Если в Тени Чернобыля пропал маркер, иди по основной цепочке: "
            "Кордон -> Свалка -> Агропром -> Бар/Росток -> Тёмная Долина -> X-18 -> Янтарь/X-16 -> Радар/X-10 -> Припять -> ЧАЭС. "
            "Ориентиры: Кордон — Сидорович/Волк/Шустрый; Свалка — Бес/Серый; Агропром — Крот, подземелья и база военных; Бар — Бармен; "
            "Тёмная Долина — база бандитов и X-18; Янтарь — учёные и X-16; Радар — X-10 и проход на Припять. "
            "Точное «лево/право» можно дать только от конкретного входа или ориентира."
        )
    if "\u0437\u043e\u0432" in lowered or "\u043f\u0440\u0438\u043f\u044f\u0442" in lowered:
        return (
            "Если в Зове Припяти пропал маркер, держись цепочки: Затон -> Юпитер -> Припять -> финал/эвакуация. "
            "На Затоне ориентиры — Скадовск, вертолёты Скат и Ной для прохода на плато. "
            "На Юпитере — станция Янов, завод Юпитер, документы и подготовка прохода в Припять. "
            "В Припяти — отряд военных, лаборатория X-8, документы и финальные задания. "
            "Если нужен маршрут по месту, напиши текущую локацию и последний выполненный квест."
        )
    return (
        "Если в Чистом Небе пропал маркер или непонятно, куда идти, ориентируйся по цепочке: "
        "Болота -> Кордон -> Свалка -> Тёмная Долина -> Агропром -> Янтарь -> Рыжий лес -> Лиманск -> Госпиталь -> ЧАЭС. "
        "На Болотах ориентир — база Чистого Неба и точки ренегатов. На Кордоне — Деревня новичков, Волк и одиночки; военный блокпост лучше не штурмовать в лоб. "
        "На Свалке держись переходов к Бару/Тёмной Долине и сюжетных NPC. Дальше сюжет ведёт через Тёмную Долину, Агропром, Янтарь и Рыжий лес к Лиманску. "
        "Лево/право безопасно давать только от конкретного входа: напиши, с какой стороны вошёл на локацию и что видишь рядом."
    )


def unknown_or_unconfirmed_quest_answer(lowered: str, language: str) -> str | None:
    faction_base_assault = (
        ("\u0448\u0442\u0443\u0440\u043c" in lowered or "\u0430\u0442\u0430\u043a" in lowered)
        and "\u0431\u0430\u0437" in lowered
        and "\u0441\u0432\u043e\u0431\u043e\u0434" in lowered
        and ("\u0434\u043e\u043b\u0433" in lowered or "\u0434\u043e\u043b\u0433\u043e\u0432" in lowered)
        and ("\u0442\u0435\u043d\u044c" in lowered or "\u0447\u0435\u0440\u043d\u043e\u0431" in lowered or "\u0442\u0447" in lowered)
    )
    if not faction_base_assault:
        return None
    if language == "English":
        return (
            "I do not have a confirmed quest like “assault the Freedom base with Duty” in our Shadow of Chernobyl / Anthology knowledge base. "
            "It may be confused with separate Duty/Freedom side tasks or faction combat, but I should not replace it with another quest. "
            "If this exists in your build, send the exact quest title or NPC who gives it and I will add it."
        )
    return (
        "Такого подтверждённого квеста — «штурм базы Свободы с долговцами» — у нас в базе по Тени Чернобыля/Anthology сейчас нет. "
        "Похоже на путаницу с отдельными заданиями Долга/Свободы или обычной войной группировок, но я не должен подменять это другим квестом. "
        "Если он реально есть в вашей сборке — дай точное название задания или NPC, кто его выдаёт, и я добавлю."
    )


def local_fallback_answer(question: str) -> str:
    question = (question or "").strip()
    lowered = question.casefold()
    language = user_language_hint(question)
    story_priority = is_story_priority_question(question)
    if story_priority:
        quick_answer = quick_story_decision_answer(question)
        if quick_answer:
            return quick_answer
        full_context = full_sources.find_context(question, str(ROOT))
        if full_context:
            return local_story_answer_from_context(question, full_context)
        qa_answer = story_qa.find_answer(question, str(ROOT))
        if qa_answer:
            return qa_answer
        if has_explicit_story_game(question):
            return clarify_story_question_answer(question)
    quick_support = quick_support_decision_answer(question)
    if should_answer_support_before_story(question, quick_support):
        return quick_support
    if quick_support:
        return quick_support
    if is_support_priority_question(question):
        support_answer = general_knowledge.find_answer(question, str(ROOT), max_chars=MAX_ANSWER_CHARS)
        if support_answer:
            return support_answer
    support_answer = general_knowledge.find_answer(question, str(ROOT), max_chars=MAX_ANSWER_CHARS)
    if support_answer:
        return support_answer
    unconfirmed = unknown_or_unconfirmed_quest_answer(lowered, language)
    if unconfirmed:
        return unconfirmed
    quick_shadow_start = (
        ("\u0432\u043e\u043b\u043a" in lowered or "\u0432\u043e\u043b\u043a\u043e\u043c" in lowered)
        and ("\u0431\u0430\u043d\u0434\u0438\u0442" in lowered or "\u0431\u0430\u043d\u0434\u0438\u0442\u0430\u043c" in lowered)
    ) or (
        ("\u0442\u0435\u043d\u044c" in lowered or "\u0447\u0435\u0440\u043d\u043e\u0431" in lowered)
        and ("\u0448\u0443\u0441\u0442\u0440" in lowered or "\u043d\u0430\u0447\u0430\u043b" in lowered)
    )
    if quick_shadow_start:
        if language == "English":
            return (
                "Shadow of Chernobyl start: talk to Wolf in the rookie village, take the weapon, then go with the stalkers "
                "towards the bandit camp on Cordon. Shoot the bandits attacking/holding the camp, not Wolf's stalkers. "
                "Inside the building free Nimble/Shustry, take the flash drive from him, then return it to Sidorovich."
            )
        return (
            "\u0422\u0435\u043d\u044c \u0427\u0435\u0440\u043d\u043e\u0431\u044b\u043b\u044f, \u0441\u0430\u043c\u043e\u0435 \u043d\u0430\u0447\u0430\u043b\u043e: "
            "\u043f\u043e\u0433\u043e\u0432\u043e\u0440\u0438 \u0441 \u0412\u043e\u043b\u043a\u043e\u043c \u0432 \u0434\u0435\u0440\u0435\u0432\u043d\u0435 \u043d\u043e\u0432\u0438\u0447\u043a\u043e\u0432, \u0432\u043e\u0437\u044c\u043c\u0438 \u043e\u0440\u0443\u0436\u0438\u0435 \u0438 \u0438\u0434\u0438 \u043a \u0433\u0440\u0443\u043f\u043f\u0435 \u0441\u0442\u0430\u043b\u043a\u0435\u0440\u043e\u0432. "
            "\u041e\u043d\u0438 \u0432\u0435\u0434\u0443\u0442 \u043a \u0431\u0430\u0437\u0435 \u0431\u0430\u043d\u0434\u0438\u0442\u043e\u0432 \u043d\u0430 \u041a\u043e\u0440\u0434\u043e\u043d\u0435. "
            "\u0421\u0442\u0440\u0435\u043b\u044f\u0442\u044c \u043d\u0443\u0436\u043d\u043e \u043f\u043e \u0431\u0430\u043d\u0434\u0438\u0442\u0430\u043c, \u043d\u0435 \u043f\u043e \u0441\u0442\u0430\u043b\u043a\u0435\u0440\u0430\u043c \u0412\u043e\u043b\u043a\u0430. "
            "\u041f\u043e\u0441\u043b\u0435 \u0437\u0430\u0447\u0438\u0441\u0442\u043a\u0438 \u0437\u0430\u0439\u0434\u0438 \u0432 \u0437\u0434\u0430\u043d\u0438\u0435, \u043e\u0441\u0432\u043e\u0431\u043e\u0434\u0438 \u0428\u0443\u0441\u0442\u0440\u043e\u0433\u043e, \u0437\u0430\u0431\u0435\u0440\u0438 \u0443 \u043d\u0435\u0433\u043e \u0444\u043b\u0435\u0448\u043a\u0443 \u0438 \u0432\u0435\u0440\u043d\u0438 \u0435\u0451 \u0421\u0438\u0434\u043e\u0440\u043e\u0432\u0438\u0447\u0443."
        )
    quick_krot = (
        ("\u043a\u0440\u043e\u0442" in lowered or "\u043a\u0440\u043e\u0442\u0430" in lowered or "\u043a\u0440\u043e\u0442\u0443" in lowered)
        and "\u0430\u0433\u0440\u043e\u043f\u0440\u043e\u043c" in lowered
    )
    if quick_krot:
        return (
            "\u0422\u0435\u043d\u044c \u0427\u0435\u0440\u043d\u043e\u0431\u044b\u043b\u044f, \u0410\u0433\u0440\u043e\u043f\u0440\u043e\u043c: \u041a\u0440\u043e\u0442\u0430 \u043d\u0443\u0436\u043d\u043e \u0441\u043f\u0430\u0441\u0442\u0438 \u0432\u043e \u0432\u0440\u0435\u043c\u044f \u0431\u043e\u044f \u0441 \u0432\u043e\u0435\u043d\u043d\u044b\u043c\u0438. "
            "\u0418\u0434\u0438 \u043d\u0430 \u0442\u0435\u0440\u0440\u0438\u0442\u043e\u0440\u0438\u044e \u041d\u0418\u0418 \u0410\u0433\u0440\u043e\u043f\u0440\u043e\u043c\u0430, \u043f\u043e\u043c\u043e\u0433\u0438 \u0441\u0442\u0430\u043b\u043a\u0435\u0440\u0430\u043c \u043e\u0442\u0431\u0438\u0442\u044c \u0430\u0442\u0430\u043a\u0443 \u0438 \u043d\u0435 \u0442\u044f\u043d\u0438: \u0431\u043e\u0439 \u0438\u0434\u0451\u0442 \u0432 \u0440\u0435\u0430\u043b\u044c\u043d\u043e\u043c \u0432\u0440\u0435\u043c\u0435\u043d\u0438. "
            "\u041f\u043e\u0441\u043b\u0435 \u0441\u043f\u0430\u0441\u0435\u043d\u0438\u044f \u041a\u0440\u043e\u0442 \u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0435\u0442 \u043f\u0440\u043e \u0442\u0430\u0439\u043d\u0438\u043a \u0421\u0442\u0440\u0435\u043b\u043a\u0430 \u0438 \u043f\u043e\u0434\u0432\u0435\u0434\u0451\u0442 \u043a \u0432\u0445\u043e\u0434\u0443 \u0432 \u043f\u043e\u0434\u0437\u0435\u043c\u0435\u043b\u044c\u044f. "
            "\u0414\u0430\u043b\u044c\u0448\u0435 \u0438\u0434\u0438 \u0432 \u043f\u043e\u0434\u0437\u0435\u043c\u0435\u043b\u044c\u044f \u0410\u0433\u0440\u043e\u043f\u0440\u043e\u043c\u0430, \u043d\u0430\u0439\u0434\u0438 \u0443\u0431\u0435\u0436\u0438\u0449\u0435/\u0442\u0430\u0439\u043d\u0438\u043a \u0421\u0442\u0440\u0435\u043b\u043a\u0430 \u0438 \u0437\u0430\u0431\u0435\u0440\u0438 \u0441\u044e\u0436\u0435\u0442\u043d\u0443\u044e \u0444\u043b\u0435\u0448\u043a\u0443."
        )
    quick_kruglov = (
        ("\u043a\u0440\u0443\u0433\u043b\u043e\u0432" in lowered or "\u043a\u0440\u0443\u0433\u043b\u043e\u0432\u0430" in lowered)
        and (
            "\u0441\u0430\u0445\u0430\u0440\u043e\u0432" in lowered
            or "\u044f\u043d\u0442\u0430\u0440" in lowered
            or "\u0436\u0438\u0432" in lowered
            or "\u0434\u043e\u0432\u0435\u0441" in lowered
            or "\u0441\u043f\u0430\u0441" in lowered
        )
    )
    if quick_kruglov:
        return (
            "\u0422\u0435\u043d\u044c \u0427\u0435\u0440\u043d\u043e\u0431\u044b\u043b\u044f: \u041a\u0440\u0443\u0433\u043b\u043e\u0432\u0430 \u043b\u0443\u0447\u0448\u0435 \u0434\u043e\u0432\u0435\u0441\u0442\u0438 \u0436\u0438\u0432\u044b\u043c \u0434\u043e \u042f\u043d\u0442\u0430\u0440\u044f/\u0421\u0430\u0445\u0430\u0440\u043e\u0432\u0430: "
            "\u0442\u0430\u043a \u0442\u044b \u043f\u043e\u043b\u0443\u0447\u0438\u0448\u044c \u043d\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u0443\u044e \u0446\u0435\u043f\u043e\u0447\u043a\u0443 \u0441 \u0443\u0447\u0451\u043d\u044b\u043c\u0438, \u0437\u0430\u043c\u0435\u0440\u0430\u043c\u0438 \u0438 \u043f\u0441\u0438-\u0448\u043b\u0435\u043c\u043e\u043c. "
            "\u041d\u043e \u0435\u0441\u043b\u0438 \u043e\u043d \u0443\u043c\u0435\u0440, \u044d\u0442\u043e \u043d\u0435 \u0434\u043e\u043b\u0436\u043d\u043e \u043d\u0430\u0432\u0441\u0435\u0433\u0434\u0430 \u043b\u043e\u043c\u0430\u0442\u044c \u043f\u0440\u043e\u0445\u043e\u0436\u0434\u0435\u043d\u0438\u0435: \u043f\u0440\u043e\u0432\u0435\u0440\u044c \u0435\u0433\u043e \u0442\u0435\u043b\u043e, "
            "\u0437\u0430\u0431\u0435\u0440\u0438 \u0434\u0430\u043d\u043d\u044b\u0435/\u0444\u043b\u0435\u0448\u043a\u0443/\u041a\u041f\u041a \u0438 \u043d\u0435\u0441\u0438 \u0421\u0430\u0445\u0430\u0440\u043e\u0432\u0443. \u041c\u0438\u043d\u0443\u0441 \u0441\u043c\u0435\u0440\u0442\u0438 \u2014 \u043c\u0435\u043d\u044c\u0448\u0435 \u043d\u0430\u0433\u0440\u0430\u0434\u044b/\u0434\u0438\u0430\u043b\u043e\u0433\u043e\u0432. "
            "\u0415\u0441\u043b\u0438 \u0435\u0441\u0442\u044c \u0441\u0442\u0430\u0440\u044b\u0439 \u0441\u0435\u0439\u0432 \u2014 \u043b\u0443\u0447\u0448\u0435 \u043f\u0435\u0440\u0435\u0438\u0433\u0440\u0430\u0442\u044c \u0438 \u0441\u043f\u0430\u0441\u0442\u0438 \u0435\u0433\u043e."
        )

    quick_soc_red_forest_loot = (
        ("\u0440\u044b\u0436" in lowered or "\u0440\u0435\u0434 \u0444\u043e\u0440\u0435\u0441\u0442" in lowered or "red forest" in lowered)
        and ("\u0445\u0430\u0431\u0430\u0440" in lowered or "\u043b\u0443\u0442" in lowered or "\u0442\u0430\u0439\u043d\u0438\u043a" in lowered)
        and ("\u0442\u0435\u043d\u044c" in lowered or "\u0447\u0435\u0440\u043d\u043e\u0431" in lowered)
    )
    if quick_soc_red_forest_loot:
        return (
            "\u0422\u0435\u043d\u044c \u0427\u0435\u0440\u043d\u043e\u0431\u044b\u043b\u044f: \u043e\u0442\u0434\u0435\u043b\u044c\u043d\u043e\u0439 \u0441\u044e\u0436\u0435\u0442\u043d\u043e\u0439 \u00ab\u0434\u043e\u0431\u044b\u0447\u0438\u00bb \u0432 \u0420\u044b\u0436\u0435\u043c \u043b\u0435\u0441\u0443 \u0434\u043b\u044f \u043f\u0440\u043e\u0445\u043e\u0436\u0434\u0435\u043d\u0438\u044f \u043d\u0435\u0442. "
            "\u0415\u0441\u0442\u044c \u043e\u0431\u044b\u0447\u043d\u044b\u0439 \u043b\u0443\u0442/\u0442\u0430\u0439\u043d\u0438\u043a\u0438 \u043f\u043e \u043d\u0430\u0432\u043e\u0434\u043a\u0430\u043c, \u043d\u043e \u0431\u0435\u0437 \u043a\u043e\u043d\u043a\u0440\u0435\u0442\u043d\u043e\u0439 \u043d\u0430\u0432\u043e\u0434\u043a\u0438 \u0438\u0433\u0440\u0430 \u043d\u0435 \u043e\u0431\u044f\u0437\u044b\u0432\u0430\u0435\u0442 \u0442\u0430\u043c \u0447\u0442\u043e-\u0442\u043e \u0438\u0441\u043a\u0430\u0442\u044c. "
            "\u0414\u043b\u044f \u0441\u044e\u0436\u0435\u0442\u0430 \u0433\u043b\u0430\u0432\u043d\u043e\u0435 \u2014 \u043f\u0440\u043e\u0439\u0442\u0438 \u0420\u0430\u0434\u0430\u0440/X-10, \u043e\u0442\u043a\u043b\u044e\u0447\u0438\u0442\u044c \u0412\u044b\u0436\u0438\u0433\u0430\u0442\u0435\u043b\u044c \u0438 \u0438\u0434\u0442\u0438 \u0434\u0430\u043b\u044c\u0448\u0435 \u043a \u0427\u0410\u042d\u0421."
        )

    quick_skat3 = "\u0441\u043a\u0430\u0442-3" in lowered or "\u0441\u043a\u0430\u0442 3" in lowered
    if quick_skat3:
        return (
            "\u0417\u043e\u0432 \u041f\u0440\u0438\u043f\u044f\u0442\u0438, \u0421\u043a\u0430\u0442-3: \u0432\u0435\u0440\u0442\u043e\u043b\u0451\u0442 \u043d\u0430\u0445\u043e\u0434\u0438\u0442\u0441\u044f \u043d\u0430 \u044e\u0436\u043d\u043e\u043c \u043f\u043b\u0430\u0442\u043e \u0417\u0430\u0442\u043e\u043d\u0430. "
            "\u041e\u0431\u044b\u0447\u043d\u044b\u043c \u043f\u0443\u0442\u0451\u043c \u0442\u0443\u0434\u0430 \u043d\u0435 \u043f\u0440\u043e\u0439\u0442\u0438: \u043d\u0443\u0436\u043d\u043e \u043d\u0430\u0439\u0442\u0438 \u041d\u043e\u044f \u043d\u0430 \u0441\u0442\u0430\u0440\u043e\u0439 \u0431\u0430\u0440\u0436\u0435, \u043e\u043d \u043f\u043e\u043a\u0430\u0436\u0435\u0442 \u043c\u0430\u0440\u0448\u0440\u0443\u0442 \u043d\u0430 \u043f\u043b\u0430\u0442\u043e. "
            "\u041d\u0430 \u043c\u0435\u0441\u0442\u0435 \u043e\u0441\u043c\u043e\u0442\u0440\u0438 \u0421\u043a\u0430\u0442-3 \u0438 \u0437\u0430\u0431\u0435\u0440\u0438 \u0441\u044e\u0436\u0435\u0442\u043d\u044b\u0435 \u0434\u0430\u043d\u043d\u044b\u0435 \u0434\u043b\u044f \u0440\u0430\u0441\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u043d\u0438\u044f \u043e\u043f\u0435\u0440\u0430\u0446\u0438\u0438 \u00ab\u0424\u0430\u0440\u0432\u0430\u0442\u0435\u0440\u00bb."
        )
    quick_bullet = (
        ("\u043f\u0443\u043b\u044f" in lowered or "\u043f\u0443\u043b\u0435" in lowered or "\u043f\u0443\u043b\u0438" in lowered)
        and ("\u0442\u0435\u043d\u044c" in lowered or "\u0447\u0435\u0440\u043d\u043e\u0431" in lowered or "\u0442\u0451\u043c\u043d" in lowered or "\u0434\u043e\u043b\u0438\u043d" in lowered)
    )
    if quick_bullet:
        return (
            "Тень Чернобыля, Пуля/«Отбить долговца»: Пуля встречается у входа в Тёмную Долину со стороны Свалки и просит помочь освободить долговца. "
            "Если помочь — иди за Пулей к засаде, дождись конвоя и убей бандитов-конвоиров. Награда от Пули: деньги и прицел ПСО-1; спасённые долговцы потом уходят на заставу Долга на Свалке. "
            "Если проигнорировать — Пуля побежит один, напарник может погибнуть, а ты потеряешь награду/плюс к отношениям с Долгом. "
            "После этого можно ещё спасать Сергея Лохматого на базе бандитов: лучше сначала зачистить базу Борова, потому что пленников быстро убивают."
        )
    quick_pripyat_team = (
        ("\u0437\u043e\u0432" in lowered or "\u043f\u0440\u0438\u043f\u044f\u0442" in lowered)
        and (
            "\u043a\u043e\u043c\u0430\u043d\u0434" in lowered
            or "\u043e\u0442\u0440\u044f\u0434" in lowered
            or "\u0437\u0443\u043b\u0443\u0441" in lowered
            or "\u0432\u0430\u043d\u043e" in lowered
            or "\u0441\u043e\u043a\u043e\u043b\u043e\u0432" in lowered
            or "\u0431\u0440\u043e\u0434\u044f\u0433" in lowered
            or "\u043f\u0440\u0438\u043f\u044f\u0442\u044c-1" in lowered
        )
    )
    if quick_pripyat_team:
        return (
            "\u0417\u043e\u0432 \u041f\u0440\u0438\u043f\u044f\u0442\u0438, \u043a\u0432\u0435\u0441\u0442 \u00ab\u041f\u0440\u0438\u043f\u044f\u0442\u044c-1\u00bb / \u0441\u0431\u043e\u0440 \u043a\u043e\u043c\u0430\u043d\u0434\u044b: \u044d\u0442\u043e \u043d\u0435 X-8. "
            "\u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u0441\u043e\u0431\u0435\u0440\u0438 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u044b \u043e \u043f\u043e\u0434\u0437\u0435\u043c\u043d\u043e\u043c \u043f\u0443\u0442\u0438 \u043d\u0430 \u0437\u0430\u0432\u043e\u0434\u0435 \u042e\u043f\u0438\u0442\u0435\u0440 \u0438 \u043e\u0442\u0434\u0430\u0439 \u0438\u0445 \u0410\u0437\u043e\u0442\u0443 \u043d\u0430 \u042f\u043d\u043e\u0432\u0435. "
            "\u041f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u044c \u0441\u0435\u0431\u0435 \u043a\u043e\u0441\u0442\u044e\u043c \u0421\u0415\u0412\u0410, \u043f\u043e\u0442\u043e\u043c \u043f\u043e\u0433\u043e\u0432\u043e\u0440\u0438 \u0441 \u0417\u0443\u043b\u0443\u0441\u043e\u043c \u0443 \u042f\u043d\u043e\u0432\u0430. "
            "\u041a \u0417\u0443\u043b\u0443\u0441\u0443 \u043c\u043e\u0436\u043d\u043e \u043e\u0442\u043f\u0440\u0430\u0432\u0438\u0442\u044c \u0421\u043e\u043a\u043e\u043b\u043e\u0432\u0430 \u0441 \u043a\u043e\u0441\u0442\u044e\u043c\u043e\u043c, \u0412\u0430\u043d\u043e \u043f\u043e\u0441\u043b\u0435 \u0440\u0435\u0448\u0435\u043d\u0438\u044f \u0434\u043e\u043b\u0433\u043e\u0432 \u0438 \u043e\u043f\u043b\u0430\u0442\u044b/\u043f\u043e\u043a\u0443\u043f\u043a\u0438 \u043a\u043e\u0441\u0442\u044e\u043c\u0430, "
            "\u0438 \u0411\u0440\u043e\u0434\u044f\u0433\u0443 \u043f\u043e\u0441\u043b\u0435 \u0443\u0441\u0442\u0440\u043e\u0439\u0441\u0442\u0432\u0430 \u0435\u0433\u043e \u043c\u043e\u043d\u043e\u043b\u0438\u0442\u043e\u0432\u0446\u0435\u0432 \u043a \u0414\u043e\u043b\u0433\u0443 \u0438\u043b\u0438 \u0421\u0432\u043e\u0431\u043e\u0434\u0435. "
            "\u041a\u043e\u0433\u0434\u0430 \u043a\u043e\u043c\u0430\u043d\u0434\u0430 \u0433\u043e\u0442\u043e\u0432\u0430 \u2014 \u0432\u0435\u0440\u043d\u0438\u0441\u044c \u043a \u0417\u0443\u043b\u0443\u0441\u0443 \u0438 \u0438\u0434\u0438 \u0432 \u043f\u0443\u0442\u0435\u043f\u0440\u043e\u0432\u043e\u0434 \u00ab\u041f\u0440\u0438\u043f\u044f\u0442\u044c-1\u00bb."
        )

    quick_x8 = (
        ("x-8" in lowered or "\u0445-8" in lowered)
        and ("\u0437\u043e\u0432" in lowered or "\u043f\u0440\u0438\u043f\u044f\u0442" in lowered)
    )
    if quick_x8:
        return (
            "\u0417\u043e\u0432 \u041f\u0440\u0438\u043f\u044f\u0442\u0438, \u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u0438\u044f X-8: \u044d\u0442\u043e \u044d\u0442\u0430\u043f \u0443\u0436\u0435 \u0432 \u041f\u0440\u0438\u043f\u044f\u0442\u0438, \u0430 \u043d\u0435 \u0422\u0435\u043d\u044c \u0427\u0435\u0440\u043d\u043e\u0431\u044b\u043b\u044f. "
            "\u0418\u0434\u0438 \u043f\u043e \u0441\u044e\u0436\u0435\u0442\u043d\u044b\u043c \u0437\u0430\u0434\u0430\u043d\u0438\u044f\u043c \u0432 \u041f\u0440\u0438\u043f\u044f\u0442\u0438 \u043a \u0432\u0445\u043e\u0434\u0443 \u0432 X-8, \u0437\u0430\u0447\u0438\u0441\u0442\u0438 \u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u0438\u044e \u0438 \u0441\u043e\u0431\u0435\u0440\u0438 \u0441\u044e\u0436\u0435\u0442\u043d\u044b\u0435 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u044b/\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b\u044b. "
            "\u041e\u043d\u0438 \u043d\u0443\u0436\u043d\u044b \u0434\u043b\u044f \u0440\u0430\u0437\u0433\u0430\u0434\u043a\u0438 \u0433\u0430\u0443\u0441\u0441-\u043f\u0443\u0448\u043a\u0438/\u00ab\u043d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u043e\u0433\u043e \u043e\u0440\u0443\u0436\u0438\u044f\u00bb \u0443 \u041a\u0430\u0440\u0434\u0430\u043d\u0430 \u0438 \u0434\u043b\u044f \u0434\u0430\u043b\u044c\u043d\u0435\u0439\u0448\u0435\u0433\u043e \u0444\u0438\u043d\u0430\u043b\u044c\u043d\u043e\u0433\u043e \u044d\u0442\u0430\u043f\u0430 \u0417\u041f."
        )
    quick_azot_radio = (
        ("\u0430\u0437\u043e\u0442" in lowered or "\u0446\u0435\u043c\u0435\u043d\u0442" in lowered)
        and ("\u0438\u043d\u0441\u0442\u0440\u0443\u043c" in lowered or "\u0440\u0430\u0434\u0438\u043e" in lowered or "\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b" in lowered or "\u0446\u0435\u043c\u0435\u043d\u0442" in lowered)
    )
    if quick_azot_radio:
        return (
            "Зов Припяти, Азот и «радиоматериалы» на Цементном заводе: это не обычные инструменты для апгрейдов, а детали для Азота. "
            "Иди на локации Юпитер к Цементному заводу. Поднимайся внутрь завода/на верхние этажи по лестницам и проходам, осматривай ящики и полки на разных уровнях. "
            "Нужные детали лежат по этажам: текстолитовые основы, медная проволока, канифоль, конденсаторы, транзисторы. "
            "Если маркер тупит, ориентир такой: Цементный завод на Юпитере, обходи здание по лестницам снизу вверх и проверяй комнаты/ящики на каждом этаже, потом возвращайся к Азоту на Янов."
        )
    quick_merc_notebook = (
        ("\u043d\u043e\u0443\u0442" in lowered or "\u043d\u043e\u0443\u0442\u0431\u0443\u043a" in lowered or "\u043a\u043f\u043a" in lowered)
        and ("\u043d\u0430\u0451\u043c" in lowered or "\u043d\u0430\u0435\u043c" in lowered or "\u0441\u044b\u0447" in lowered or "\u043f\u0435\u0440\u0435\u0440\u0430\u0431\u043e\u0442" in lowered)
    )
    if quick_merc_notebook:
        return (
            "Зов Припяти, лагерь наёмников / ноутбук для Сыча: лагерь находится на станции переработки отходов на юге Затона. "
            "Есть два варианта. Силовой: зачистить наёмников, забрать ноутбук в здании и КПК с Крюка/Хребта, потом отнести Сычу. "
            "Стелс: ночью зайти с тыла через вентиляционную трубу/верхний проход, добраться до ноутбука, забрать его и уйти; КПК главарей так обычно не получить. "
            "Если хочешь без бойни — иди ночью, оружие не доставай лишний раз, используй присед+шаг и уходи тем же путём."
        )
    quick_merc_food = (
        ("\u043d\u0430\u0451\u043c" in lowered or "\u043d\u0430\u0435\u043c" in lowered or "\u0442\u0435\u0441\u0430\u043a" in lowered or "\u0442\u043e\u043f\u043e\u0440" in lowered or "hatchet" in lowered)
        and ("\u043f\u0440\u043e\u0432\u0438\u0437" in lowered or "\u0435\u0434" in lowered or "\u0435\u0434\u0443" in lowered or "\u043a\u043e\u043b\u0431\u0430\u0441" in lowered or "\u0445\u043b\u0435\u0431" in lowered or "\u043a\u043e\u043d\u0441\u0435\u0440\u0432" in lowered)
    )
    if quick_merc_food:
        return (
            "Зов Припяти, наёмники на подстанции / провизия: это отряд Тесака/Топора у цехов подстанции на Затоне. "
            "Им можно принести еду: всего нужно 6 единиц из подходящей еды — хлеб, колбаса или консервы/«Завтрак туриста»; можно смешивать. "
            "После этого они пропускают на территорию, и ты можешь спокойно забрать инструменты для тонкой работы. "
            "Да, позже этих наёмников можно нанять охранять бункер учёных на Юпитере, если они не стали враждебными. Не подходи к ним с оружием в руках."
        )
    quick_sokolov_suit = (
        ("\u0441\u043e\u043a\u043e\u043b\u043e\u0432" in lowered or "\u043a\u043e\u0441\u0442\u044e\u043c" in lowered)
        and ("\u0437\u043e\u0432" in lowered or "\u043f\u0440\u0438\u043f\u044f\u0442" in lowered or "\u044e\u043f\u0438\u0442\u0435\u0440" in lowered or "\u0431\u0443\u043d\u043a\u0435\u0440" in lowered)
    )
    if quick_sokolov_suit:
        return (
            "Зов Припяти, костюм для Соколова: костюм даёт профессор Озёрский в бункере учёных на Юпитере. "
            "Сначала поговори с Соколовым, потом с Озёрским. Обычно нужно выполнить для Озёрского задание с аномальным растением/образцом. "
            "После этого возвращайся к Озёрскому, получай костюм и отдавай его Соколову, чтобы он смог пойти в Припять. "
            "Ориентир: не ищи костюм у торговцев — иди в бункер учёных на Юпитере."
        )
    quick_topol_controller = (
        ("\u0442\u043e\u043f\u043e\u043b" in lowered or "\u043a\u043e\u043d\u0442\u0440\u043e\u043b" in lowered)
        and ("\u0433\u0440\u0443\u043f" in lowered or "\u0441\u043f\u0430\u0441" in lowered or "\u0443\u0431\u0438\u0432" in lowered or "\u0437\u043e\u0432" in lowered or "\u043f\u0440\u0438\u043f\u044f\u0442" in lowered)
    )
    if quick_topol_controller:
        return (
            "Зов Припяти, Тополь и контролёр: группу можно спасти, но нужно действовать быстро. "
            "Когда появляется контролёр, он берёт отряд под контроль и они начинают стрелять/гибнуть. Твоя цель — как можно быстрее убить контролёра, желательно с дистанции и мощным оружием/гранатами, не расстреливая своих. "
            "Если уже все погибли, обычно это результат проваленного боя — проще загрузиться до входа в опасную зону и сразу фокусить контролёра. "
            "Ориентир по тактике: не воюй с группой Тополя, ищи самого контролёра и снимай его первым."
        )
    quick_chimera_hunt = "\u0445\u0438\u043c\u0435\u0440" in lowered and ("\u0437\u0432\u0435\u0440\u043e\u0431" in lowered or "\u0433\u043e\u043d\u0442" in lowered or "\u043e\u0445\u043e\u0442" in lowered or "\u043d\u043e\u0447" in lowered)
    if quick_chimera_hunt:
        if "\u0437\u0432\u0435\u0440\u043e\u0431" in lowered or "\u0432\u0435\u043d\u0442\u0438\u043b" in lowered or "\u044e\u043f\u0438\u0442\u0435\u0440" in lowered:
            return (
                "Зов Припяти, Зверобой — «Ночная охота»: это химера у вентиляционного комплекса на Юпитере. "
                "Приходить нужно ночью: рабочее окно примерно с 21:00 до 06:00. Иди к вентиляционному комплексу, бери мощный дробовик/гранаты/РПГ или другое тяжёлое оружие, потому что химера здоровая и очень опасная. "
                "Можно занять безопасную позицию на высоте/у труб/за укрытием и стрелять оттуда. После убийства возвращайся к Зверобою на Янов за наградой."
            )
        return (
            "Зов Припяти, Гонта — «Охота на химеру» на Затоне: к Гонте нужно подойти около 3:00 ночи; обычно засчитывается окно примерно 02:45–04:00. "
            "Это другой квест, не Зверобой. Идёшь с Гонтой и Гарматой к Изумрудному, стараешься тихо подойти и быстро убить химеру, чтобы охотники выжили."
        )
    quick_flint = "\u0444\u043b\u0438\u043d\u0442" in lowered or "\u0441\u043e\u0440\u043e\u043a" in lowered or "\u0433\u043e\u043d\u0442" in lowered
    if quick_flint:
        return (
            "Зов Припяти, Сорока/Флинт: это квест на разоблачение предателя. На Затоне Гонта рассказывает про Сороку, а на Юпитере на станции Янов Флинт хвастается чужими подвигами. "
            "Не надо сразу стрелять: слушай рассказы Флинта, сопоставь их с историями сталкеров и Гонты, потом сдавай его как Сороку. "
            "Что будет: Флинта разоблачат, сталкеры получат справедливую развязку, а у игрока будет нормальный плюс к репутации."
        )
    quick_oasis = "\u043e\u0430\u0437\u0438\u0441" in lowered
    if quick_oasis:
        return (
            "Зов Припяти, Оазис: это загадка на Юпитере, а не обычная перестрелка. Иди в подземный комплекс/вентиляционный объект, проходи через зал с колоннами и подбирай правильную последовательность проходов. "
            "Когда путь выбран верно, появится проход к артефакту/сердцу Оазиса. Если телепортирует назад — последовательность неверная, повторяй и меняй проходы между колоннами. "
            "После получения артефакта возвращайся к учёным/по квесту."
        )
    quick_bloodsucker_tremor = "\u043a\u0440\u043e\u0432\u043e\u0441\u043e\u0441" in lowered or "\u0442\u0440\u0435\u043c\u043e\u0440" in lowered or "\u0433\u043b\u0443\u0445\u0430\u0440" in lowered
    if quick_bloodsucker_tremor:
        return (
            "Зов Припяти, кровососы/Глухарь/Тремор: это расследование на Затоне. Иди по цепочке Глухаря, проверь логово кровососов и доведи расследование до конца. "
            "Если вопрос про Тремора — он связан с развязкой дела о пропажах сталкеров. Если застрял, ищи следы Глухаря и возвращайся по разговорным подсказкам на Скадовск/Затон."
        )
    quick_pripyat_squad = "\u0432\u0430\u043d\u043e" in lowered or "\u0437\u0443\u043b\u0443\u0441" in lowered or "\u0431\u0440\u043e\u0434\u044f\u0433" in lowered or "\u043c\u043e\u043d\u043e\u043b\u0438\u0442" in lowered
    if quick_pripyat_squad and ("\u043f\u0440\u0438\u043f\u044f\u0442" in lowered or "\u0437\u043e\u0432" in lowered or "\u043e\u0442\u0440\u044f\u0434" in lowered):
        return (
            "Зов Припяти, отряд в Припять: для хорошего похода нужно закрывать личные проблемы кандидатов. "
            "Вано — решить вопрос с долгами/бандитами. Соколов — получить костюм через Озёрского. Зулус — поговорить и взять в подготовку похода. "
            "Бродяга/монолитовцы — помочь устроить их к Долгу или Свободе через лидеров группировок. После этого собирай отряд и иди к переходу в Припять."
        )
    quick_cs_robbery = (
        ("\u0447\u0438\u0441\u0442\u043e\u0435" in lowered and "\u043d\u0435\u0431\u043e" in lowered)
        and ("\u043e\u0433\u0440\u0430\u0431" in lowered or "\u0437\u0430\u0431\u0440\u0430\u043b" in lowered or "\u0431\u0430\u043d\u0434" in lowered or "\u0431\u0430\u043d\u0434\u043e\u0441" in lowered)
    )
    if quick_cs_robbery:
        return (
            "\u0427\u0438\u0441\u0442\u043e\u0435 \u041d\u0435\u0431\u043e: \u0441\u0446\u0435\u043d\u0430, \u0433\u0434\u0435 \u0431\u0430\u043d\u0434\u0438\u0442\u044b \u043e\u0433\u0440\u0430\u0431\u0438\u043b\u0438 \u0438 \u0437\u0430\u0431\u0440\u0430\u043b\u0438 \u0432\u0435\u0449\u0438, \u0441\u044e\u0436\u0435\u0442\u043d\u0430\u044f. "
            "\u0414\u043e\u0433\u043e\u0432\u043e\u0440\u0438\u0442\u044c\u0441\u044f \u0441 \u043d\u0438\u043c\u0438 \u0434\u043e \u044d\u0442\u043e\u0433\u043e \u043e\u0431\u044b\u0447\u043d\u043e \u043d\u0435\u043b\u044c\u0437\u044f: \u044d\u0442\u043e \u0441\u043a\u0440\u0438\u043f\u0442\u043e\u0432\u044b\u0439 \u044d\u0442\u0430\u043f. "
            "\u041f\u043e\u0441\u043b\u0435 \u043e\u0433\u0440\u0430\u0431\u043b\u0435\u043d\u0438\u044f \u0432\u044b\u0431\u0438\u0440\u0430\u0439\u0441\u044f \u0438\u0437 \u043b\u043e\u0432\u0443\u0448\u043a\u0438/\u043f\u043e\u0434\u0432\u0430\u043b\u0430, \u0438\u0449\u0438 \u044f\u0449\u0438\u043a/\u0441\u0445\u0440\u043e\u043d \u0441 \u0432\u0435\u0449\u0430\u043c\u0438 \u0438 \u0434\u0430\u043b\u044c\u0448\u0435 \u0438\u0434\u0438 \u043f\u043e \u043c\u0430\u0440\u043a\u0435\u0440\u0443 \u043a \u0421\u0435\u0440\u043e\u043c\u0443/\u0441\u0442\u0430\u043b\u043a\u0435\u0440\u0430\u043c. "
            "\u0415\u0441\u043b\u0438 \u043e\u0441\u0442\u0430\u043b\u0441\u044f \u0431\u0435\u0437 \u043e\u0440\u0443\u0436\u0438\u044f, \u043d\u0435 \u043b\u0435\u0437\u044c \u0432 \u043b\u043e\u0431: \u0441\u043d\u0430\u0447\u0430\u043b\u0430 \u0437\u0430\u0431\u0435\u0440\u0438 \u0441\u043d\u0430\u0440\u044f\u0433\u0443 \u0438 \u0434\u0435\u0440\u0436\u0438\u0441\u044c \u0441\u044e\u0436\u0435\u0442\u043d\u043e\u0433\u043e \u043c\u0430\u0440\u043a\u0435\u0440\u0430."
        )
    quick_cs_renegades = (
        ("\u0447\u0438\u0441\u0442\u043e\u0435" in lowered and "\u043d\u0435\u0431\u043e" in lowered)
        and ("\u0440\u0435\u043d\u0435\u0433\u0430\u0442" in lowered or "\u0431\u0430\u0437" in lowered or "\u0434\u043e\u0433\u043e\u0432\u043e\u0440" in lowered)
    )
    if quick_cs_renegades:
        return (
            "\u0427\u0438\u0441\u0442\u043e\u0435 \u041d\u0435\u0431\u043e, \u0440\u0435\u043d\u0435\u0433\u0430\u0442\u044b: \u044d\u0442\u043e \u0432\u0440\u0430\u0436\u0434\u0435\u0431\u043d\u0430\u044f \u0441\u044e\u0436\u0435\u0442\u043d\u0430\u044f \u0441\u0438\u043b\u0430 \u043d\u0430 \u0411\u043e\u043b\u043e\u0442\u0430\u0445. "
            "\u041c\u0438\u0440\u043d\u043e\u0433\u043e \u0432\u0430\u0440\u0438\u0430\u043d\u0442\u0430 \u00ab\u0434\u043e\u0433\u043e\u0432\u043e\u0440\u0438\u0442\u044c\u0441\u044f\u00bb \u043f\u043e \u043e\u0441\u043d\u043e\u0432\u043d\u043e\u043c\u0443 \u0441\u044e\u0436\u0435\u0442\u0443 \u043d\u0435\u0442: \u043e\u043d\u0438 \u0431\u0443\u0434\u0443\u0442 \u0441\u0442\u0440\u0435\u043b\u044f\u0442\u044c. "
            "\u041d\u0430 \u0431\u0430\u0437\u0435 \u0440\u0435\u043d\u0435\u0433\u0430\u0442\u043e\u0432 \u0442\u0435\u0431\u044f \u0436\u0434\u0451\u0442 \u0431\u043e\u0439: \u0438\u0434\u0438 \u0441 \u043e\u0442\u0440\u044f\u0434\u0430\u043c\u0438 \u0427\u0438\u0441\u0442\u043e\u0433\u043e \u041d\u0435\u0431\u0430/\u043f\u043e \u043c\u0430\u0440\u043a\u0435\u0440\u0443, \u0431\u0435\u0440\u0438 \u0443\u043a\u0440\u044b\u0442\u0438\u044f, \u0437\u0430\u0447\u0438\u0449\u0430\u0439 \u0442\u043e\u0447\u043a\u0443 \u0438 \u043f\u043e\u0441\u043b\u0435 \u0437\u0430\u0447\u0438\u0441\u0442\u043a\u0438 \u043f\u0440\u043e\u0432\u0435\u0440\u044f\u0439 \u0437\u0430\u0434\u0430\u043d\u0438\u0435/\u041a\u041f\u041a."
        )
    quick_cs_cordon_mg = (
        ("\u0447\u0438\u0441\u0442\u043e\u0435" in lowered and "\u043d\u0435\u0431\u043e" in lowered)
        and ("\u043a\u043e\u0440\u0434\u043e\u043d" in lowered or "\u0432\u043e\u044f\u043a" in lowered or "\u043f\u0443\u043b\u0435\u043c" in lowered or "\u0432\u043e\u0435\u043d" in lowered)
    )
    if quick_cs_cordon_mg:
        return (
            "\u0427\u0438\u0441\u0442\u043e\u0435 \u041d\u0435\u0431\u043e, \u041a\u043e\u0440\u0434\u043e\u043d \u0438 \u0432\u043e\u0435\u043d\u043d\u044b\u0439 \u043f\u0443\u043b\u0435\u043c\u0451\u0442: \u044d\u0442\u043e \u0442\u043e\u0442 \u0441\u0430\u043c\u044b\u0439 \u0436\u0451\u0441\u0442\u043a\u0438\u0439 \u0432\u0445\u043e\u0434 \u0441 \u0411\u043e\u043b\u043e\u0442. "
            "\u041e\u0442 \u043c\u0435\u0441\u0442\u0430 \u0432\u0445\u043e\u0434\u0430 \u0434\u0435\u0440\u0436\u0438\u0441\u044c \u043b\u0435\u0432\u043e\u0439 \u0441\u0442\u043e\u0440\u043e\u043d\u044b/\u0437\u0430\u0431\u043e\u0440\u0430, \u0431\u0435\u0433\u0438 \u043e\u0442 \u0432\u043e\u0435\u043d\u043d\u043e\u0433\u043e \u0431\u043b\u043e\u043a\u043f\u043e\u0441\u0442\u0430, \u0438\u0449\u0438 \u0440\u0430\u0437\u0440\u044b\u0432/\u043f\u0440\u043e\u0445\u043e\u0434 \u0432 \u0437\u0430\u0431\u043e\u0440\u0435 \u0441\u043b\u0435\u0432\u0430 \u0438 \u043f\u043e\u0441\u043b\u0435 \u043d\u0435\u0433\u043e \u0441\u0440\u0430\u0437\u0443 \u0443\u0445\u043e\u0434\u0438 \u043a \u0443\u043a\u0440\u044b\u0442\u0438\u044f\u043c. "
            "\u0422\u0432\u043e\u044f \u0446\u0435\u043b\u044c — \u0414\u0435\u0440\u0435\u0432\u043d\u044f \u043d\u043e\u0432\u0438\u0447\u043a\u043e\u0432/\u0431\u0443\u043d\u043a\u0435\u0440 \u0443 \u0412\u043e\u043b\u043a\u0430, \u043d\u0435 \u0441\u0430\u043c \u0431\u043b\u043e\u043a\u043f\u043e\u0441\u0442. "
            "\u0415\u0441\u043b\u0438 \u043d\u0435 \u0443\u0441\u043f\u0435\u0432\u0430\u0435\u0448\u044c \u0438 \u043f\u0443\u043b\u0435\u043c\u0451\u0442 \u0441\u043d\u043e\u0441\u0438\u0442 \u0441\u0440\u0430\u0437\u0443: \u0432\u0435\u0440\u043d\u0438\u0441\u044c \u043d\u0430 \u0411\u043e\u043b\u043e\u0442\u0430 \u0438 \u0437\u0430\u0439\u0434\u0438 \u043d\u0430 \u041a\u043e\u0440\u0434\u043e\u043d \u0447\u0435\u0440\u0435\u0437 \u0441\u0435\u0432\u0435\u0440\u043d\u044b\u0439/\u0441\u0435\u0432\u0435\u0440\u043e-\u0432\u043e\u0441\u0442\u043e\u0447\u043d\u044b\u0439 \u043f\u0435\u0440\u0435\u0445\u043e\u0434 \u0441 \u0411\u043e\u043b\u043e\u0442: \u0442\u0430\u043a \u043c\u043e\u0436\u043d\u043e \u043e\u0431\u043e\u0439\u0442\u0438 \u0441\u0435\u043a\u0442\u043e\u0440 \u043e\u0431\u0441\u0442\u0440\u0435\u043b\u0430 \u0438 \u0432\u044b\u0439\u0442\u0438 \u0431\u043b\u0438\u0436\u0435 \u043a \u0431\u0430\u0437\u0435 \u043e\u0434\u0438\u043d\u043e\u0447\u0435\u043a."
        )
    quick_unknown_weapon = (
        "\u043d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d" in lowered and "\u043e\u0440\u0443\u0436" in lowered
    ) or "\u0433\u0430\u0443\u0441" in lowered or "gauss" in lowered
    if quick_unknown_weapon:
        return (
            "\u0417\u043e\u0432 \u041f\u0440\u0438\u043f\u044f\u0442\u0438, \u00ab\u041d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u043e\u0435 \u043e\u0440\u0443\u0436\u0438\u0435\u00bb: \u0440\u0435\u0447\u044c \u043e \u0433\u0430\u0443\u0441\u0441-\u043f\u0443\u0448\u043a\u0435 \u043c\u043e\u043d\u043e\u043b\u0438\u0442\u043e\u0432\u0446\u0435\u0432. "
            "\u041f\u043e\u0441\u043b\u0435 \u0441\u0442\u044b\u0447\u043a\u0438 \u0437\u0430\u0431\u0435\u0440\u0438 \u043e\u0440\u0443\u0436\u0438\u0435 \u0438 \u043f\u043e\u043a\u0430\u0436\u0438 \u0435\u0433\u043e \u0442\u0435\u0445\u043d\u0438\u043a\u0443 \u041a\u0430\u0440\u0434\u0430\u043d\u0443 \u043d\u0430 \u00ab\u0421\u043a\u0430\u0434\u043e\u0432\u0441\u043a\u0435\u00bb. "
            "\u0414\u043b\u044f \u043f\u043e\u043b\u043d\u043e\u0439 \u0440\u0430\u0437\u0433\u0430\u0434\u043a\u0438 \u043d\u0443\u0436\u043d\u044b \u0441\u044e\u0436\u0435\u0442\u043d\u044b\u0435 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u044b/\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b\u044b \u0438\u0437 \u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u0438\u0438 X-8: \u0441 \u043d\u0438\u043c\u0438 \u041a\u0430\u0440\u0434\u0430\u043d \u0441\u043c\u043e\u0436\u0435\u0442 \u043e\u0431\u044a\u044f\u0441\u043d\u0438\u0442\u044c, \u0447\u0442\u043e \u044d\u0442\u043e \u0437\u0430 \u043e\u0440\u0443\u0436\u0438\u0435."
        )
    quick_third_person = (
        "\u0442\u0440\u0435\u0442\u044c" in lowered
        or "\u0442\u0440\u0435\u0442\u044c\u0435 \u043b\u0438\u0446\u043e" in lowered
        or "third person" in lowered
    )
    if quick_third_person:
        if language == "English":
            return "OpenAI is unavailable right now, so I answer from the local Anthology guides:\nTry camera keys: `Left Arrow`, `Down Arrow`, `Right Arrow`. In this build they switch `cam_1`, `cam_2`, `cam_3`. If it does not work, open `Settings -> Controls -> Camera` and check key bindings."
        return "OpenAI \u0441\u0435\u0439\u0447\u0430\u0441 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d, \u043f\u043e\u044d\u0442\u043e\u043c\u0443 \u043e\u0442\u0432\u0435\u0447\u0430\u044e \u0438\u0437 \u043b\u043e\u043a\u0430\u043b\u044c\u043d\u044b\u0445 \u0433\u0430\u0439\u0434\u043e\u0432 Anthology:\n\u041f\u043e\u043f\u0440\u043e\u0431\u0443\u0439 \u043a\u043b\u0430\u0432\u0438\u0448\u0438 \u043a\u0430\u043c\u0435\u0440\u044b: `Left Arrow`, `Down Arrow`, `Right Arrow`. \u0412 \u0442\u0435\u043a\u0443\u0449\u0435\u0439 \u0441\u0431\u043e\u0440\u043a\u0435 \u043e\u043d\u0438 \u043e\u0442\u0432\u0435\u0447\u0430\u044e\u0442 \u0437\u0430 `cam_1`, `cam_2`, `cam_3`. \u0415\u0441\u043b\u0438 \u043d\u0435 \u0440\u0430\u0431\u043e\u0442\u0430\u0435\u0442 — \u0437\u0430\u0439\u0434\u0438 \u0432 `\u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438 -> \u0423\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u0438\u0435 -> \u041a\u0430\u043c\u0435\u0440\u0430` \u0438 \u043f\u0440\u043e\u0432\u0435\u0440\u044c \u043d\u0430\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f."

    quick_jam = (
        "\u043a\u043b\u0438\u043d" in lowered
        or "unjam" in lowered
        or "jam" in lowered
    )
    if quick_jam:
        if language == "English":
            return "OpenAI is unavailable right now, so I answer from the local Anthology guides:\nWeapon jam/unjam is configured in MCM: `MCM -> WPO -> WPO weapon -> unjam key`. Also check `MCM -> MCM MENU -> All assigned keys -> WPO: inspect / unjam`."
        return "OpenAI \u0441\u0435\u0439\u0447\u0430\u0441 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d, \u043f\u043e\u044d\u0442\u043e\u043c\u0443 \u043e\u0442\u0432\u0435\u0447\u0430\u044e \u0438\u0437 \u043b\u043e\u043a\u0430\u043b\u044c\u043d\u044b\u0445 \u0433\u0430\u0439\u0434\u043e\u0432 Anthology:\n\u0421\u043d\u044f\u0442\u0438\u0435 \u043a\u043b\u0438\u043d\u0430 \u043d\u0430\u0441\u0442\u0440\u0430\u0438\u0432\u0430\u0435\u0442\u0441\u044f \u0447\u0435\u0440\u0435\u0437 MCM: `MCM -> WPO -> WPO \u043e\u0440\u0443\u0436\u0438\u0435 -> \u041a\u043b\u0430\u0432\u0438\u0448\u0430 \u0441\u043d\u044f\u0442\u0438\u044f \u043a\u043b\u0438\u043d\u0430`. \u0415\u0449\u0451 \u043f\u0440\u043e\u0432\u0435\u0440\u044c `MCM -> MCM MENU -> \u0412\u0441\u0435 \u043d\u0430\u0437\u043d\u0430\u0447\u0435\u043d\u043d\u044b\u0435 \u043a\u043b\u0430\u0432\u0438\u0448\u0438 -> WPO: \u041e\u0441\u043c\u043e\u0442\u0440 / \u0443\u0441\u0442\u0440\u0430\u043d\u0438\u0442\u044c \u043a\u043b\u0438\u043d`."

    early_story_mode = any(word in lowered for word in (
        "\u0442\u0435\u043d\u044c", "\u0447\u0435\u0440\u043d\u043e\u0431", "\u0447\u0438\u0441\u0442\u043e\u0435", "\u0447\u0438\u0441\u0442\u043e\u043c", "\u0447\u0438\u0441\u0442\u043e\u0433\u043e", "\u0447\u043d", "\u043d\u0435\u0431\u043e", "\u043d\u0435\u0431\u0435", "\u0437\u043e\u0432", "\u043f\u0440\u0438\u043f\u044f\u0442",
        "\u0441\u044e\u0436\u0435\u0442", "\u043a\u0432\u0435\u0441\u0442", "\u043a\u043e\u0440\u0434\u043e\u043d", "\u0441\u0432\u0430\u043b\u043a", "\u0430\u0433\u0440\u043e\u043f\u0440\u043e\u043c", "\u0431\u043e\u043b\u043e\u0442",
        "\u044f\u043d\u0442\u0430\u0440", "\u0440\u0430\u0434\u0430\u0440", "\u043b\u0438\u043c\u0430\u043d\u0441\u043a", "\u0447\u0430\u044d\u0441",
    ))
    early_route_mode = any(word in lowered for word in (
        "\u043a\u0443\u0434\u0430", "\u0438\u0434\u0442\u0438", "\u0431\u0435\u0436\u0430\u0442\u044c", "\u0434\u0432\u0438\u0433\u0430\u0442\u044c", "\u0441\u0442\u043e\u0440\u043e\u043d",
        "\u043b\u0435\u0432\u043e", "\u043f\u0440\u0430\u0432\u043e", "\u043f\u0440\u044f\u043c\u043e", "\u043d\u0430\u0437\u0430\u0434", "\u043c\u0430\u0440\u043a\u0435\u0440",
        "\u0437\u0430\u0441\u0442\u0440\u044f\u043b", "\u043f\u0435\u0440\u0435\u0445\u043e\u0434", "\u043b\u043e\u043a\u0430\u0446", "\u0434\u043e\u0440\u043e\u0433", "\u043f\u0443\u0442\u044c", "\u043e\u0440\u0438\u0435\u043d\u0442\u0438\u0440",
        "where", "go", "route", "marker", "stuck",
    ))
    if early_story_mode and early_route_mode:
        return story_route_answer(lowered, language)

    stopwords = {
        "\u0447\u0442\u043e", "\u043a\u0430\u043a", "\u0433\u0434\u0435", "\u0435\u0441\u043b\u0438", "\u0438\u043b\u0438",
        "\u0434\u043b\u044f", "\u043f\u0440\u0438", "\u044d\u0442\u043e", "\u0442\u0430\u043c", "\u0442\u0443\u0442",
        "\u043d\u0430\u0434\u043e", "\u043d\u0443\u0436\u043d\u043e", "\u0434\u0435\u043b\u0430\u0442\u044c", "\u0441\u0434\u0435\u043b\u0430\u0442\u044c",
        "\u043c\u043e\u0436\u043d\u043e", "\u043f\u043e\u0447\u0435\u043c\u0443", "\u043a\u043e\u0433\u0434\u0430", "\u043a\u0443\u0434\u0430",
        "\u043c\u043d\u0435", "\u043c\u0435\u043d\u044f", "\u0435\u0433\u043e", "\u0435\u0451", "\u043e\u043d\u0430",
        "\u043a\u0432\u0435\u0441\u0442", "\u043a\u0432\u0435\u0441\u0442\u0435", "\u0441\u044e\u0436\u0435\u0442", "\u0441\u044e\u0436\u0435\u0442\u043d\u0430\u044f",
        "\u043b\u0438\u043d\u0438\u044f", "\u043b\u0438\u043d\u0438\u0438",
        "the", "and", "for", "with", "what", "how", "where", "when", "why", "can", "should",
    }
    tokens = [
        token
        for token in re.findall(r"[A-Za-z\u0400-\u04FF0-9_\\-]+", lowered)
        if (len(token) >= 4 or re.fullmatch(r"[x\u0445]-\d+", token)) and token not in stopwords
    ]
    chunks = [
        chunk.strip()
        for chunk in re.split(r"\n(?=#{1,3} )", KNOWLEDGE)
        if chunk.strip()
    ]
    if not tokens:
        if language == "English":
            return "I need a more specific question. Mention the topic, error, menu, mod, or exact problem."
        return "Нужен более конкретный вопрос: укажи ошибку, меню, мод, настройку или проблему."

    best_score = 0
    best_chunk = ""
    second_score = 0
    wanted_game = wanted_story_game(lowered)
    for chunk in chunks:
        c = chunk.casefold()
        score = 0
        score += match_score(c, tokens)
        score += story_game_score(c[:600], wanted_game)
        if chunks and chunk.startswith("## "):
            title = chunk.splitlines()[0].casefold()
            score += match_score(title, tokens, title=True)
        if "\u0442\u0440\u0435\u0442\u044c" in lowered and ("\u0442\u0440\u0435\u0442\u044c" in c or "3" in c or "third" in c):
            score += 6
        if "\u043a\u043b\u0438\u043d" in lowered and "\u043a\u043b\u0438\u043d" in c:
            score += 8
        if "mcm" in lowered and "mcm" in c:
            score += 4
        if score > best_score:
            second_score = best_score
            best_score = score
            best_chunk = chunk
        elif score > second_score:
            second_score = score

    story_markers = {
        "\u0442\u0435\u043d\u044c", "\u0447\u0435\u0440\u043d\u043e\u0431", "\u0447\u0438\u0441\u0442\u043e\u0435", "\u043d\u0435\u0431\u043e", "\u0437\u043e\u0432", "\u043f\u0440\u0438\u043f\u044f\u0442",
        "\u0432\u043e\u043b\u043a", "\u0448\u0443\u0441\u0442\u0440", "\u043a\u0440\u043e\u0442", "\u0430\u0433\u0440\u043e\u043f\u0440\u043e\u043c", "\u0441\u0432\u0430\u043b\u043a",
        "\u0445-18", "x-18", "\u0445-16", "x-16", "\u0445-10", "x-10", "\u0440\u0430\u0434\u0430\u0440", "\u0447\u0430\u044d\u0441",
        "\u0441\u043a\u0430\u0442", "\u0431\u043e\u043b\u043e\u0442", "\u0440\u0435\u043d\u0435\u0433\u0430\u0442", "\u043b\u0435\u0441\u043d\u0438\u043a", "\u043a\u043e\u043c\u043f\u0430\u0441", "\u043b\u0438\u043c\u0430\u043d\u0441\u043a",
        "\u043c\u043e\u043d\u043e\u043b\u0438\u0442", "\u044d\u0432\u0430\u043a\u0443\u0430\u0446", "\u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440",
    }
    story_mode = any(any(variant in lowered for variant in token_variants(marker)) for marker in story_markers)
    required_score = max(10, len(tokens) * 5)
    if len(tokens) == 1:
        required_score = 12
    if story_mode:
        required_score = max(8, len(tokens) * 3)
    ambiguous_gap = 2 if story_mode else 4
    ambiguous = second_score and best_score - second_score < ambiguous_gap
    if not best_chunk or best_score < required_score or ambiguous:
        route_mode = any(word in lowered for word in (
            "\u043a\u0443\u0434\u0430", "\u0438\u0434\u0442\u0438", "\u0431\u0435\u0436\u0430\u0442\u044c", "\u0434\u0432\u0438\u0433\u0430\u0442\u044c", "\u0441\u0442\u043e\u0440\u043e\u043d",
            "\u043b\u0435\u0432\u043e", "\u043f\u0440\u0430\u0432\u043e", "\u043f\u0440\u044f\u043c\u043e", "\u043d\u0430\u0437\u0430\u0434", "\u043c\u0430\u0440\u043a\u0435\u0440",
            "\u0437\u0430\u0441\u0442\u0440\u044f\u043b", "\u043f\u0435\u0440\u0435\u0445\u043e\u0434", "\u043b\u043e\u043a\u0430\u0446", "where", "go", "route", "marker", "stuck",
        ))
        if story_mode and route_mode:
            return story_route_answer(lowered, language)
        if language == "English":
            return "I did not find a reliable exact answer in the local Anthology guides. Please ask a moderator or rephrase with the exact error/menu/mod name."
        return "Я не нашёл надёжный точный ответ в локальных гайдах Anthology. Лучше уточнить у модератора или переформулировать с точной ошибкой, меню или названием мода."

    cleaned = re.sub(r"^## .+?\n", "", best_chunk, count=1).strip()
    cleaned = trim_answer(cleaned)
    if language == "English":
        return "OpenAI is unavailable right now, so I answer from the local Anthology guides:\n" + cleaned
    return "OpenAI \u0441\u0435\u0439\u0447\u0430\u0441 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d, \u043f\u043e\u044d\u0442\u043e\u043c\u0443 \u043e\u0442\u0432\u0435\u0447\u0430\u044e \u0438\u0437 \u043b\u043e\u043a\u0430\u043b\u044c\u043d\u044b\u0445 \u0433\u0430\u0439\u0434\u043e\u0432 Anthology:\n" + cleaned

def strip_bot_mention(message: discord.Message) -> str:
    content = message.content or ""
    if bot.user:
        content = re.sub(rf"^\s*<@!?{bot.user.id}>\s*[,;:—-]?\s*", "", content).strip()
    return content.strip()


def is_triggered(message: discord.Message) -> bool:
    content = message.content or ""
    if bot.user and re.match(rf"^\s*<@!?{bot.user.id}>\b", content):
        return True
    lowered = content.casefold().lstrip()
    return bool(re.match(r"^(юра|yura)(\s+семецкий)?(\b|[,;:—-])", lowered))


def is_auto_question(message: discord.Message) -> bool:
    if str(getattr(message.channel, "id", "")) not in AUTO_REPLY_QUESTION_CHANNEL_IDS:
        return False
    content = (message.content or "").strip()
    if not content:
        return False
    return "?" in content or content.casefold().startswith(("как ", "что ", "где ", "почему ", "когда "))


def cleanup_question(message: discord.Message) -> str:
    question = strip_bot_mention(message)
    return cleanup_raw_question(question)


def cleanup_raw_question(question: str) -> str:
    cleaned = (question or "").strip()
    cleaned = re.sub(r"^\s*(юра|yura)(\s+семецкий)?\s*[,;:—-]?\s*", "", cleaned, flags=re.I).strip()
    return cleaned


def user_language_hint(question: str) -> str:
    cleaned = re.sub(r"(?i)\bx\s*-?\s*\d+\b", "", question or "")
    latin = len(re.findall(r"[A-Za-z]", cleaned))
    cyrillic = len(re.findall(r"[\u0400-\u04FF]", cleaned))
    if cyrillic >= 5:
        return "Russian"
    if latin and latin >= max(1, cyrillic * 2):
        return "English"
    return "Russian"


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    return [s.strip(" \t\r\n") for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]


def is_story_priority_question(question: str) -> bool:
    q = latest_player_question(question).casefold().replace("ё", "е")
    full_q = (question or "").casefold().replace("ё", "е")
    if any(term in full_q for term in (
        "сюжет забытый отряд", "сюжет смерти вопреки", "сюжет пространственная",
        "сюжет атрибут", "сюжет путь во мгле", "сюжет долина шорохов",
        "смерти вопреки: в паутине лжи", "смерти вопреки",
        "забытый отряд", "пространственная аномалия", "путь во мгле", "долина шорохов",
        "anomaly living legend", "living legend", "живая легенда", "живой легенде", "живую легенду", "цербер",
        "смертный грех", "смертного греха", "операция послесвечение", "послесвечение",
        "пустые границы", "темное присутствие", "тёмное присутствие", "тайны зоны", "секреты зоны",
    )):
        return True
    if (
        any(term in full_q for term in ("модпак", "модпаком", "модак", "оригинал"))
        and any(term in q for term in ("сюжет тот", "сюжет такой", "квесты", "квест", "задания", "меняет", "трогает"))
    ):
        return False
    anthology_story_terms = (
        "пространственная аномалия", "пространственная", "аномалия", "зверь", "лютый", "маркус", "шуруп",
        "дуболом", "лесник", "таченко", "мурад", "застава", "ставрид", "химик", "хромой", "левша",
        "миклуха", "стронглав", "петрович", "распутин", "хантер", "шмыга", "маскарад",
        "атрибут", "воланд", "никита", "шериф", "молаг", "мишель", "гаррота", "тесак", "квант",
        "цитра", "пророк", "сектант", "санатор", "катакомбы",
        "путь во мгле", "мгле", "саван", "борланд", "шаман", "патоген", "логопед", "колязин",
        "багрецов", "спектрум", "маятник", "курчатов", "мертвый город", "мёртвый город",
        "долина шорохов", "шорохов", "мутный", "максимильян", "радик", "тесла",
        "трус", "балбес", "бывалый", "сердце оазиса", "микросхема",
        "смерти вопреки", "паутина лжи", "паутине лжи", "топи", "варг", "анубис",
        "харольд", "хасан", "чех", "клык", "фугас",
        "забытый отряд", "змей", "бизон", "ржавый", "фома", "коста", "кривой",
        "старый", "ворон", "гарик", "лысый", "мертвое озеро", "мёртвое озеро",
        "группа бизона", "обитатели", "болотная тварь", "незваные гости", "чужой среди своих",
        "живая легенда", "живой легенде", "живую легенду", "смертный грех", "смертного греха", "операция послесвечение",
        "послесвечение", "пустые границы", "темное присутствие", "тёмное присутствие",
        "тайны зоны", "секреты зоны", "фантом", "чернобог", "шов", "пси-блокада",
    )
    explicit_story_game = (
        any(game in q for game in ("тень чернобыля", "тень черноб", "чистое небо", "зов припяти", *anthology_story_terms))
        or re.search(r"\b(тч|чн|зп)\b", q) is not None
        or "сюжет " in q
    )
    vague_probe = any(term in q for term in ("подскажи про", "проблема с", "как настроить", "что делать с"))
    support_control_terms = (
        "faq", "обт", "фпс", "fps", "производительност", "просад", "оптимизац",
        "лцу", "лазер", "миникарт", "мини-карт", "подстволь", "компас", "патрон",
        "счетчик", "счётчик", "клин", "bhs", "mcm", "mo2", "dx8", "dx9", "dx11",
        "модпак", "модпаком", "модак", "оригинал", "слабое желез", "слабом желез",
        "слабый пк", "скач", "ссылк", "установ", "лаунчер", "обнов", "кнопк",
        "нажмите", "нажать", "мыш", "правой", "правой кноп", "пкм", "сервер",
        "канал", "каналы", "discord", "vk", "вк", "github", "vpn",
    )
    concrete_story_terms = (
        "тень чернобыля", "тень черноб", "чистое небо", "зов припяти",
        "проводник", "доктор", "призрак", "стрелок", "декодер", "монолит",
        "волк", "шустрый", "сидорович", "круглов", "сахаров", "глухарь", "тремор",
        "кордон", "агропром", "янтар", "радар", "чаэс", "свалк", "затон", "юпитер", "припять",
        "x-8", "х-8", "x-10", "х-10", "x-16", "х-16", "x-18", "х-18",
        *anthology_story_terms,
    )
    if (not explicit_story_game) and any(term in q for term in support_control_terms):
        # "маркер пропал" часто дописывают к любому вопросу. Сам по себе маркер
        # не должен уводить FAQ/настройки/клавиши/скачивание в сюжетную базу.
        story_only = any(term in q for term in concrete_story_terms)
        if not story_only or any(term in q for term in support_control_terms):
            return False
    if (not explicit_story_game) and "маркер" in q and not any(term in q for term in concrete_story_terms):
        return False
    if "добавлено" in q and "сюжетн" in q and "линий" in q:
        return False
    if (not explicit_story_game) and ("локационн" in q or "локации" in q or "локаций" in q):
        return False
    if vague_probe and any(term in q for term in ("кордон", "тайник", "тайники", "тайников", "тайникам", "лаборатор", "сюжет", "задания")) and not any(
        specific in q for specific in (
            "тень чернобыля", "чистое небо", "зов припяти", "x-8", "х-8", "x-10", "х-10", "x-16", "х-16", "x-18", "х-18",
            "скат", "волк", "шустрый", "проводник", "глухарь", "коряга", "недоступный тайник",
        )
    ):
        return False
    if (not explicit_story_game) and any(term in q for term in ("профил", "стандартный профиль", "хард", "hard", "модпак", "оригинал")) and (
        "сюжет" in q or "квест" in q or "задани" in q
    ):
        return False
    if ("подскажи про" in q or "объясни" in q) and any(term in q for term in ("сюжетн", "сюжетов", "квестов", "квесты")) and not any(
        game in q for game in ("тень чернобыля", "чистое небо", "зов припяти", "кордон", "агропром", "затон", "юпитер", "припять")
    ):
        return False
    if ("подскажи про" in q or "объясни" in q) and "лаборатор" in q and not any(
        lab in q for lab in ("x-8", "х-8", "x-10", "х-10", "x-16", "х-16", "x-18", "х-18", "зов припяти", "тень чернобыля")
    ):
        return False
    if ("модпак" in q or "модпаком" in q or "модак" in q) and ("сюжет" in q or "квест" in q or "задани" in q):
        return False
    if (not explicit_story_game) and ("отлич" in q or "разниц" in q):
        return False
    if (not explicit_story_game) and any(term in q for term in ("faq", "фпс", "fps", "производительност", "просад", "оптимизац")):
        return False
    if ("оригинал" in q and ("модпак" in q or "модпаком" in q or "модак" in q)) and any(
        word in q for word in ("отлич", "разница", "чем", "слаб", "dx8", "dx9", "dx11")
    ):
        return False
    if "сюжетн" in q and "лини" in q and any(word in q for word in ("выбрать", "выбор", "старт", "новой игры", "фракц")):
        return False
    generic_problem = any(
        phrase in q
        for phrase in ("проблема с", "как настроить", "что делать с")
    )
    if generic_problem and not any(term in q for term in concrete_story_terms):
        return False
    return any(term in q for term in (
        "сюжет", "квест", "задание", "маркер", "проводник", "доктор", "призрак", "стрелок",
        "декодер", "монолит", "исполнитель желаний", "тайник", "лаборатор", "подземель",
        "тень чернобыля", "тень черноб", "чистое небо", "зов припяти",
        "кордон", "агропром", "янтар", "радар", "чаэс", "x-16", "х-16", "x-18", "х-18",
        "волк", "шустрый", "сидорович", "круглов", "сахаров",
        *anthology_story_terms,
    ))


def is_new_anthology_story_question(question: str) -> bool:
    q = latest_player_question(question).casefold().replace("ё", "е")
    if any(term in q for term in (
        "жекан", "зохан", "жохан", "живая легенда", "цербер", "фриплей", "freeplay",
        "смертный грех", "послесвечение", "потерянный в зоне", "война группировок", "warfare",
        "ironman", "azazel", "ииг", "unisg", "пустые границы", "темное присутствие", "тёмное присутствие",
    )):
        return True
    return any(term in q for term in (
        "пространственная аномалия", "зверь", "лютый", "маркус", "шуруп", "дуболом",
        "таченко", "застава", "ставрид", "стронглав", "маскарад",
        "атрибут", "воланд", "никита", "шериф", "молаг", "мишель", "гаррота",
        "квант", "цитра", "пророк", "сектант",
        "путь во мгле", "борланд", "саван", "патоген", "логопед", "спектрум", "маятник",
        "долина шорохов", "шорохов", "сердце оазиса", "микросхема", "мутный", "максимильян",
        "смерти вопреки", "паутина лжи", "паутине лжи", "варг", "анубис", "харольд", "хасан", "клык",
        "забытый отряд", "змей", "бизон", "ржавый", "фома", "коста", "кривой",
        "мертвое озеро", "мёртвое озеро", "группа бизона", "болотная тварь",
    ))


def quick_support_decision_answer(question: str) -> str | None:
    q = latest_player_question(question).casefold().replace("ё", "е")
    full_q = (question or "").casefold().replace("ё", "е")
    if any(term in q for term in ("жекан", "зохан", "зюхан", "жохан", "жехан")):
        return (
            "Забытый Отряд, Жекан: если пишут «Зохан/Зюхан/Жохан», обычно речь про Жекана — пленника из этой линии. "
            "Его этап завязан на ночь: дождись нужной ночной сцены, найди бандитов, зачисти их и освободи пленника. "
            "Если сцена не запускается — проверь, активна ли линия Жекана и не был ли заранее зачищен нужный лагерь/этап."
        )
    if "чернобог" in full_q and any(term in full_q for term in ("смертный грех", "смертном грехе", "грех")):
        return (
            "Смертный грех / Anomaly: Чернобог — ключевой противник и финальная фигура линии «Смертный грех», связанная с группировкой «Грех». "
            "Эта ветка идёт после «Живой легенды»: Рыжий лес/Лесник -> след «Греха» -> атаки на базы -> Лиманск -> Госпиталь -> Генераторы -> финальный конфликт. "
            "Это не Тень Чернобыля и не обычные бандиты: если игрок спрашивает про Чернобога, отвечай в рамках Anomaly/«Смертного греха»."
        )
    if "шов" in full_q and any(term in full_q for term in ("послесвеч", "afterglow", "операция")):
        return (
            "Операция Послесвечение / Anomaly: Шов — важный персонаж/узел этой линии, связанный с ИИГ/UNISG, Бродягой и финальными решениями после «Смертного греха». "
            "Если вопрос про Шова — не подменяй его Чистым Небом или Тенью Чернобыля: проверяй цепочку «Спецдоставка», «Под прикрытием», «Охраняемые секреты» и финальный разговор. "
            "Для точной подсказки по Шову нужен текущий квест или последняя сцена, потому что исходы у «Послесвечения» могут отличаться."
        )
    if "дожд" in q and any(term in q for term in ("нормаль", "сделать", "вернуть", "настро", "плох", "крив")):
        return (
            "По дождю/погоде: если дождь выглядит ненормально или слишком мешает, сначала проверь выбранный пресет погоды/графики и тяжёлые GFX/SSS-модули в MO2. "
            "Для слабого ПК лучше отключить тяжёлые графические моды, проверить рендер и настройки SSS. "
            "Если вопрос про конкретный визуальный баг дождя — пришли скрин и текущий профиль/рендер, потому что дождь может зависеть от погоды, шейдеров и выбранных графических модулей."
        )
    if (
        re.search(r"(^|[^a-zа-я0-9])pip([^a-zа-я0-9]|$)", full_q)
        or "picture in picture" in full_q
        or ("3dss" in full_q and any(term in full_q for term in ("прицел", "scope", "pip")))
    ):
        return (
            "PiP / Picture in Picture — это отдельный режим оптических прицелов для Anomaly/Anthology на базе 3DSS: "
            "картинка внутри прицела рендерится отдельно от обычного изображения. Поэтому при включённом PiP в момент прицеливания "
            "FPS может падать примерно в 2 раза из-за двойного рендера. "
            "В Anthology PiP ставится отдельно: архив `PiP for 3DSS in Anomaly.7z`, мод `[WPN][1.1][SCP][R.A.K Weapon Pack Adaptation Global Anomaly PiP for 3DSS (OBT)]`, "
            "обновлённый SSS 23.5 и папка `bin`. Если нужно выключить PiP — в MO2 просто отключи этот PiP-мод."
        )
    explicit_story_game = (
        any(game in q for game in ("тень чернобыля", "тень черноб", "чистое небо", "зов припяти"))
        or is_new_anthology_story_question(question)
    )
    vague_probe = any(term in q for term in ("подскажи про", "проблема с", "как настроить", "что делать с"))
    if (
        ("завис" in q or "слом" in q or "баг" in q or "отмен" in q or "сброс" in q)
        and ("задани" in q or "квест" in q)
        and not explicit_story_game
    ):
        return (
            "Если зависло обычное/циклическое задание и нужно его отменить: "
            "`MCM -> Job to be done - Redone -> Job to be done - Redone -> Настройки отмены заданий`. "
            "По умолчанию отмена висевшего задания делается клавишей `END` вместе с `Shift`: наведи курсор на задание в КПК и нажми `Shift + END`. "
            "Если речь про сюжетный квест — лучше не отменять вслепую, а уточнить название сюжета/этап."
        )
    if (
        any(term in full_q for term in ("модпак", "модпаком", "модак", "оригинал"))
        and any(term in q for term in ("сюжет тот", "сюжет такой", "сюжет тот же", "квесты", "квест", "задания", "меняет", "трогает"))
    ):
        return (
            "Да, в рамках «Оригинал/Модпак» сюжетная основа остаётся той же: модпак не должен переписывать сюжетные линии и основные квесты. "
            "Он меняет окружение прохождения — оружие, баланс, графику, сложность, интерфейс, MCM/MO2-модули и дополнительные механики. "
            "Если конкретный квест внезапно ведёт себя иначе или ломается — это уже баг/конфликт версии, а не задумка модпака."
        )
    if "добавлено" in q and "сюжетн" in q and "линий" in q:
        return (
            "В Anthology 2.1 ОБТ добавлены сюжетные линии из оригинальной трилогии и модификаций: например «Тень Чернобыля», "
            "«Зов Припяти», «Пространственная аномалия», «Путь во мгле», «Долина шорохов», «Забытый отряд», «Атрибут» и другие. "
            "Сюжет выбирается при старте новой игры через выбор локации/истории."
        )
    if ("2.0" in q and "2.1" in q) or ("антолог" in q and "2.1" in q and any(term in q for term in ("отлич", "разниц", "измен", "нового", "что добав"))):
        return (
            "Anthology 2.1 по сравнению с 2.0 — это не просто мелкий патч, а переход к более самостоятельной и разделённой сборке. "
            "Главное: 1) standalone-установка — больше не нужно отдельно скачивать чистую платформу Anomaly, сборка ставится единым установщиком; "
            "2) разделение на «Оригинал» и «Модпак»: «Оригинал» легче и ближе к базовой Anomaly с сюжетными линиями, а «Модпак» тяжелее и содержит оружейный пак, графику и расширенные механики; "
            "3) в лаунчер добавлены автоочистка кэша шейдеров и поддержка разных рендеров, базовая версия модпака поддерживает старые DX8/DX9/DX11, а модпак рассчитан на DX11; "
            "4) переработаны геймплей, баланс патронов/оружия, магазинное питание, размер арсенала и модели; "
            "5) в MO2 появились игровые профили, включая стандартный и HARD-профиль с BHS, холодом, опасной средой, экономикой и более жёстким интерфейсом; "
            "6) обновлены шейдеры, текстуры окружения и объектов, добавлены сезонные пресеты и переработанная броня/иконки. "
            "Если кратко: 2.1 — более автономная, гибкая и тяжёлая по возможностям версия, где слабым ПК лучше начинать с «Оригинала», а модпак выбирать при нормальном железе."
        )
    specific_support_problem = any(term in q for term in (
        "фпс", "fps", "производительност", "просад", "оптимизац", "лцу", "лазер",
        "миникарт", "подстволь", "компас", "патрон", "счетчик", "счётчик", "клин",
        "bhs", "mcm", "mo2", "dx8", "dx9", "dx11", "прицел", "шейдер", "рендер",
        "ошиб", "вылет", "фриз", "микрофриз", "лаунчер", "обнов",
    ))
    if ("faq" in q or "обт" in q) and not specific_support_problem:
        return (
            "FAQ по Anthology 2.1 ОБТ: если нужны ссылки на скачивание — открой Discord-сервер, нажми ПКМ по серверу и включи "
            "«Отобразить все каналы». Также ссылки обычно есть в закрепе ВК Anthology. Если вопрос не про скачивание, напиши конкретную проблему: "
            "установка, лаунчер, FPS, MCM, MO2, сюжет или ошибка."
        )
    if (
        "ссылк" in q
        or ("скач" in q and "где" in q)
        or any(term in q for term in ("пкм", "правой", "мыш", "кнопк", "нажмите", "нажать", "сервер", "канал"))
    ):
        return (
            "Ссылки на скачивание ищи в Discord Anthology: ПКМ по серверу -> включить «Отобразить все каналы», затем проверь нужный канал со ссылками. "
            "Также проверь закреп ВК Anthology. Если файл не качается или лаунчер не обновляется — попробуй VPN/Zapret и пришли точный HTTP-код ошибки."
        )
    if vague_probe and any(term in q for term in ("кордон", "тайник", "тайники", "тайников", "тайникам", "лаборатор", "сюжет", "задания")) and not any(
        specific in q for specific in (
            "тень чернобыля", "чистое небо", "зов припяти", "x-8", "х-8", "x-10", "х-10", "x-16", "х-16", "x-18", "х-18",
            "скат", "волк", "шустрый", "проводник", "глухарь", "коряга", "недоступный тайник",
        )
    ):
        return (
            "Уточни контекст: про какой сюжет/игру, квест, NPC или локацию речь. "
            "Слова вроде «Кордон», «тайники», «сюжет» или «задания» встречаются в разных местах, и без уточнения я могу подставить не тот квест."
        )
    if (not explicit_story_game) and ("локационн" in q or "локации" in q or "локаций" in q):
        return (
            "Локационный пак Anthology добавляет/использует большой набор наземных и подземных локаций. "
            "Часть локаций открывается строго по сюжетным линиям, а на многих заменены флора и растительность. "
            "Если вопрос про конкретный переход/маркер — укажи сюжет и текущую локацию."
        )
    if (not explicit_story_game) and "оригинал" in q and not ("модпак" in q or "модпаком" in q or "модак" in q):
        return (
            "«Оригинал» — облегчённая версия Anthology/Anomaly с сюжетными линиями без тяжёлого модпак-набора. "
            "Его стоит выбирать для слабого ПК или если нужна более стабильная базовая игра."
        )
    if any(term in q for term in ("сюжетн", "сюжетов", "квестов", "квесты")) and any(
        term in q for term in ("проблем", "настро", "что делать", "подскажи", "объясни")
    ):
        return (
            "Если речь про сюжетные линии/квесты в целом: выбери их при старте новой игры через фракцию и стартовую локацию. "
            "Если вопрос про конкретный квест — напиши название сюжета, NPC или задание, иначе ответ будет слишком общим."
        )
    if "вручную" in q or "ручной" in q:
        return (
            "Если в FAQ сказано делать вручную — значит действие не автоматическое: нужно самому выбрать нужный пункт/файл/настройку в лаунчере, MO2 или MCM. "
            "Для точной инструкции напиши, что именно делаешь вручную: установка, выбор сюжета, настройка мода или обновление."
        )
    if ("подскажи про" in q or "объясни" in q) and any(term in q for term in ("сюжетн", "сюжетов", "квестов", "квесты")) and not any(
        game in q for game in ("тень чернобыля", "чистое небо", "зов припяти", "кордон", "агропром", "затон", "юпитер", "припять")
    ):
        return (
            "Если вопрос про сюжетные линии в целом: в Anthology их выбирают при старте новой игры через фракцию/стартовую локацию. "
            "Если нужен конкретный квест — напиши игру/сюжетную линию, название задания или NPC, иначе я могу перепутать прохождение."
        )
    if ("подскажи про" in q or "объясни" in q) and "лаборатор" in q and not any(
        lab in q for lab in ("x-8", "х-8", "x-10", "х-10", "x-16", "х-16", "x-18", "х-18", "зов припяти", "тень чернобыля")
    ):
        return (
            "Уточни, про какую лабораторию речь: X-18, X-16, X-10 или X-8, и из какого сюжета. "
            "Без номера лаборатории я могу перепутать разные сюжетные этапы."
        )
    if (
        ("сюжет" in q and any(word in q for word in ("тот же", "такой же", "трог", "меняет", "измен", "лома", "слома")))
        or ("квест" in q and any(word in q for word in ("меняет", "измен", "трог", "друг", "отлич")))
        or ("задани" in q and any(word in q for word in ("меняет", "измен", "трог", "друг", "отлич")))
    ):
        return (
            "Модпак не должен переписывать саму сюжетную основу. Сюжетные линии и основная логика прохождения остаются базовыми, "
            "а модпак меняет то, как это играется: оружие, баланс, графику, сложность, интерфейс, механики и отдельные условия вокруг прохождения. "
            "Если где-то сюжет реально ломается — это уже баг конкретной версии/модуля, а не задумка модпака."
        )
    if ("модпак" in q or "модпаком" in q or "модак" in q) and "сюжет" in q:
        return (
            "Да, правильно: модпак не должен ломать или переписывать основную сюжетную базу. "
            "Сюжетная основа остаётся от Anomaly/выбранных сюжетных линий, а модпак меняет окружение вокруг прохождения: "
            "оружие, графику, баланс, механики, сложность, интерфейс и часть геймплейных условий. "
            "Если я раньше написал так, будто модпак меняет сам сюжет — это неверная формулировка."
        )
    if "лцу" in q or "лазер" in q:
        return (
            "ЛЦУ включается долгим нажатием на `L`. "
            "Если хочешь переназначить клавишу: `MCM -> Лазеры на основе BaS -> Лазеры на основе BaS -> Клавиша включения лазера`."
        )
    if re.search(r"\b(dх|dx)\s*\\??\s*(8|9|10|11)?\b", q) or re.search(r"\bдх\s*\\??\s*(8|9|10|11)?\b", q):
        if "модпак" in q or "модак" in q or "модпаком" in q:
            return (
                "Модпак нормально рассчитан на `DX11`. На `DX8/DX9` лучше играть в `Оригинал`, потому что модпак тяжелее и часть графики/механик завязана на DX11. "
                "`DX10` как отдельный рекомендуемый режим для модпака не используй — если спрашиваешь «на каком DX играть», ставь DX11 для модпака."
            )
        return (
            "По рендерам: `Оригинал` легче и может работать на `DX8/DX9/DX11`. `Модпак` рассчитан на `DX11`. "
            "Если ПК слабый — пробуй Оригинал на более лёгком DX; если играешь в модпак — выбирай DX11."
        )
    if "оруж" in q and ("пак" in q or "модпак" in q or "ствол" in q or "арсенал" in q):
        return (
            "В модпаке есть свежий оружейный пак R.A.K Weapon Pack Adaptation Global: он добавляет/перерабатывает оружие, анимации, магазины, "
            "баланс калибров, износ/починку деталей и распределение оружия у NPC. По FAQ арсенал в сборке — под `400 стволов`. "
            "Если вопрос по багам оружейки — лучше писать в Discord в раздел багов/вопросов по оружейному паку и прикладывать конкретный ствол, модуль, лог или скрин."
        )
    if "подстволь" in q:
        return "Подствольник: назначь удобную клавишу в настройках управления игры. Если клавиша не срабатывает — проверь конфликт назначений в управлении/MCM."
    if "осмотр" in q and "оруж" in q:
        return (
            "Осмотр оружия и устранение клина настраиваются тут: "
            "`MCM -> MCM MENU -> Все назначенные клавиши -> WPO: Осмотр / устранить клин`. "
            "По умолчанию осмотр/клин могут висеть на `F`, но если действие конфликтует с обыском трупа, зайди в это меню и назначь удобную отдельную клавишу."
        )
    if "миникарт" in q or "мини-карт" in q:
        return "Миникарта включается клавишей `Z`. Если хочешь другой режим — в настройках mini-map выбери режим нажатия/удержания."
    if "компас" in q:
        return (
            "Компас в профиле `HARD` — это тактический компас, он включается/выключается клавишей `Num5`. "
            "Если `Num5` не срабатывает, проверь конфликт клавиш в настройках управления/MCM и убедись, что ты играешь именно на HARD-профиле. "
            "В обычном профиле поведение компаса может отличаться, потому что часть хардкорного HUD завязана именно на HARD."
        )
    if "фильтр" in q or "противогаз" in q:
        return (
            "Фильтр противогаза ставится/снимается через назначенную клавишу: "
            "`MCM -> MCM MENU -> Все назначенные клавиши -> Клавиша снятия/установки фильтра`. "
            "По умолчанию обычно `T`. Если не работает — проверь, не конфликтует ли `T` с другой функцией, и переназначь кнопку."
        )
    if any(term in q for term in ("кэш", "кеш", "cache")):
        if "шейдер" in q or "shader" in q:
            return (
                "Кэш шейдеров очищается через лаунчер: нажми кнопку очистки кэша/обновления или просто запускай игру через кнопку `Играть` — "
                "в Anthology очистка шейдерного кэша встроена в лаунчер. Если после этого графика всё равно сломана, перезапусти игру и проверь выбранный рендер."
            )
        return (
            "Кэш очищается через лаунчер. Если вопрос про шейдеры — очисти шейдерный кэш в лаунчере/через запуск `Играть`, "
            "потом перезапусти игру, чтобы рендер пересобрал шейдеры заново."
        )
    if any(term in q for term in ("сколько места", "места занимает", "занимает сборка", "вес сборки", "размер сборки", "сколько гб", "гигабайт")):
        return (
            "По FAQ: `Оригинал` занимает примерно `~60 ГБ`, `Модпак` — примерно `~110 ГБ`. "
            "Для установки/распаковки Anthology держи запас: около `110 ГБ` свободно на системном диске для временной распаковки "
            "и около `117 ГБ` на диске, куда ставится игра. Лучше ставить на SSD/NVMe."
        )
    if "профил" in q and any(term in q for term in ("перейти", "переключ", "с харда", "с hard", "на стандарт")):
        return (
            "С `HARD` на `Стандарт` в той же игре лучше не переходить. Такой обратный переход может вызвать краши и поломку сохранений. "
            "Безопасная логика такая: со `Стандарта` на `HARD` перейти можно, а если хочешь играть на `Стандарте` после `HARD` — лучше начинать новую игру/отдельное сохранение."
        )
    if "профил" in q and any(term in q for term in ("игров", "обычн", "стандарт", "hard", "хард", "харда", "что за", "что такое", "разниц", "отлич")):
        return (
            "Игровые профили выбираются в MO2 во вкладке `Профили` под иконкой геймпада. "
            "`Anthology 2.1 Стандарт` — обычный профиль для спокойного прохождения. "
            "`Anthology 2.1 HARD` — сложный профиль для опытных игроков: BHS/конечности, холод/выживание, опасная среда, "
            "жёстче экономика/бартер, минималистичный HUD и тактический компас. "
            "Важно: со `Стандарта` на `HARD` перейти можно, а с `HARD` обратно на `Стандарт` в той же игре лучше не переходить — можно словить краши и поломку сохранений."
        )
    if any(term in q for term in ("hard", "хард", "харда")) and any(term in q for term in ("отлич", "разниц", "обычн", "стандарт", "профил")):
        return (
            "HARD-профиль отличается от обычного повышенной сложностью и набором хардкорных модулей: BHS/повреждения конечностей, холод/выживание, "
            "опасная среда, более жёсткая экономика/бартер, минималистичный HUD и тактический компас. "
            "Важно: со `Стандарта` на `HARD` переключаться можно, а с `HARD` обратно на `Стандарт` лучше не переходить в той же игре — это может ломать сохранения."
        )
    if "рюкзак" in q and ("анимац" in q or "animat" in q or "отключ" in q or "включ" in q):
        return (
            "Анимация именно рюкзака настраивается отдельно: "
            "`MCM -> Animat - Анимации -> Рюкзак -> Включить анимацию`. "
            "Если нужно убрать только рюкзак — трогай только этот пункт, остальные анимации можно не менять."
        )
    if any(term in q for term in ("анимац", "animat")) and not any(term in q for term in ("рюкзак", "разделк", "мутант", "подбор", "обыск", "шлем")):
        return (
            "Общие анимации лежат в `MCM -> Animat - Анимации`. "
            "Там уже выбирай конкретную вкладку: `Рюкзак`, `Разделка мутантов`, `Анимация подбора`, `Обыск тел` или `Головные уборы`. "
            "Если вопрос про один предмет — напиши его название, и менять нужно только его вкладку."
        )
    if "разделк" in q and "мутант" in q:
        return "Анимация разделки мутантов: `MCM -> Animat - Анимации -> Разделка мутантов -> Включить анимацию`."
    if "подбор" in q and ("анимац" in q or "предмет" in q):
        return "Анимация подбора предметов: `MCM -> Animat - Анимации -> Анимация подбора -> Включить анимацию`."
    if "обыск" in q and ("тел" in q or "труп" in q or "анимац" in q):
        return "Анимация обыска тел: `MCM -> Animat - Анимации -> Обыск тел -> Включить анимацию`."
    if "шлем" in q and ("анимац" in q or "сняти" in q or "надев" in q):
        return "Лишняя анимация шлема: `MCM -> Animat - Анимации -> Головные уборы -> Режим строгих шлемов`; сними галочку, если не хочешь долгую анимацию снятия/надевания."
    if "bhs" in q and any(term in q for term in ("стиль", "вид", "внешн", "положен", "позици", "hud", "худ")):
        if any(term in q for term in ("положен", "позици", "x", "y", "сдвин", "мест")):
            return (
                "Положение BHS меняется тут: "
                "`MCM -> Body Health System -> HUD -> Позиция HUD по оси X / Позиция HUD по оси Y`. "
                "Числа вводи руками и нажимай Enter, чтобы значение закрепилось."
            )
        return (
            "Внешний вид/стиль BHS меняется тут: "
            "`MCM -> Body Health System -> HUD -> Тип HUD`. "
            "Если нужно двигать сам блок на экране — рядом пункты `Позиция HUD по оси X / Y`."
        )
    if "bhs" in q or ("здоров" in q and "hud" in q):
        return "`H` открывает отображение BHS на харде без захода в инвентарь."
    if ("пуст" in q and "крафт" in q) or ("окно крафт" in q and any(term in q for term in ("пуст", "нет", "не отображ"))):
        return "Если пустое окно крафта — в MO2 подключи модуль `[HARD] SYS_Balance`."
    if ("рук" in q or "оруж" in q) and any(term in q for term in ("странн", "положен", "крив", "неправильн", "съехал", "съехало")):
        return "Если странное положение рук/оружия — в настройках видео выставь `FOV 65`, а `FOV интерфейса` — `0.65`."
    if (
        ("21:9" in q or "21 9" in q or "21-9" in q or "ультрашир" in q or "ultrawide" in q)
        and any(term in q for term in ("интерфейс", "hud", "худ", "монитор", "экран", "баг", "лома", "проблем"))
    ):
        return (
            "На 21:9/ультрашироких мониторах интерфейс/HUD может отображаться криво — часть UI в Anomaly/Anthology рассчитана в первую очередь на обычные 16:9. "
            "Временное решение: попробуй поставить разрешение/режим 16:9, оконный/безрамочный режим или другой масштаб интерфейса. "
            "Если проблема именно с руками/оружием/HUD — в настройках видео выставь `FOV 65` и `FOV интерфейса 0.65`. "
            "Если ломается конкретный элемент интерфейса, напиши какой именно: КПК, HUD, инвентарь, BHS или меню."
        )
    if "патрон" in q and any(term in q for term in ("провер", "клавиш", "настро", "проблем", "что делать", "счетчик", "счётчик")):
        return (
            "По патронам: проверка патронов — клавиша `-` / `минус`. "
            "Анимация проверки настраивается через `MCM -> Проверка патронов -> снять галочку с «Исправление занятых рук»`. "
            "Счётчик патронов возвращается там же: сними `Спрятать счётчик патронов`."
        )
    if ("счетчик" in q or "счётчик" in q) and "патрон" in q:
        return "Чтобы вернуть счётчик патронов: `MCM -> Проверка патронов` и сними галочку с `Спрятать счётчик патронов`."
    if "замедлен" in q or "bullet time" in q:
        return "Замедление времени: в MO2 подключи модуль `[GAM] Bullet Time`, затем назначь клавишу в настройках игры."
    if (
        ("завис" in q or "слом" in q or "баг" in q or "отмен" in q or "сброс" in q)
        and ("задани" in q or "квест" in q)
        and not explicit_story_game
    ):
        return (
            "Если зависло обычное/циклическое задание и нужно его отменить: "
            "`MCM -> Job to be done - Redone -> Job to be done - Redone -> Настройки отмены заданий`. "
            "По умолчанию отмена висевшего задания делается клавишей `END` вместе с `Shift`: наведи курсор на задание в КПК и нажми `Shift + END`. "
            "Если речь про сюжетный квест — лучше не отменять вслепую, а уточнить название сюжета/этап."
        )
    if "фпс" in q or "fps" in q or "производительност" in q or "просад" in q or "оптимизац" in q:
        return (
            "Если большие проблемы с FPS: сначала играй через «Оригинал» или снизь настройки в модпаке. "
            "Отключи тяжёлые графические моды/шейдеры: `Enhanced Shaders & Color Grading`, `Beefs NVGs Shaders`, `ScreenSpaceShaders Update 23.5` — "
            "именно их в FAQ советуют вырубать, если FPS болит. Ещё очисти кэш через лаунчер, снизь траву/тени/дальность, проверь DX11 для модпака "
            "и закрой лишние программы. Если фризы именно микрофризами — ограничь FPS на 1–2 кадра ниже герцовки монитора."
        )
    if "сюжетн" in q and "лини" in q and any(word in q for word in ("выбрать", "выбор", "старт", "новой игры", "фракц")):
        return (
            "Сюжетную линию выбирают при старте новой игры: выбери фракцию/группировку, открой список стартовых локаций "
            "и укажи нужную сюжетную линию. Если нужной линии нет — проверь выбранный профиль/режим и включённые моды в MO2."
        )
    if ("сюжет" in q or "сюжетн" in q) and ("отлич" in q or "разниц" in q):
        return (
            "Да, сюжетные отличия есть. «Оригинал» — это базовая Anomaly с сюжетными линиями и более лёгкой нагрузкой. "
            "«Модпак» оставляет сюжетную основу, но добавляет оружейный пак, графику, геймплейные механики и более тяжёлый профиль. "
            "То есть сюжет выбирается и проходится, но ощущения, баланс, сложность и набор механик в модпаке заметно отличаются."
        )
    if "модпак" in q or "модпаком" in q or "модак" in q or "оригинал" in q:
        if any(word in q for word in ("отлич", "разница", "что такое", "проблема", "настроить", "делать")):
            return (
                "Оригинал — облегчённая базовая Anomaly с сюжетными линиями, подходит для слабых ПК и может работать на DX8/DX9/DX11. "
                "Модпак — тяжёлая версия с оружейным паком, графикой, геймплейными улучшениями и новыми механиками; он рассчитан на DX11. "
                "Если ПК слабый или важна стабильность — начинай с Оригинала. Если хочешь максимум контента и железо тянет — выбирай Модпак."
            )
    if "клин" in q:
        return (
            "Снятие клина настраивается через `MCM -> WPO -> WPO оружие -> Клавиша снятия клина`. "
            "Также проверь `MCM -> MCM MENU -> Все назначенные клавиши -> WPO: Осмотр / устранить клин`."
        )
    if "dx11" in q or "dx 11" in q:
        return (
            "DX11 нужен для модпака: он тяжелее, но именно под него рассчитаны графика и часть новых фич. "
            "Если ПК слабый или DX11 работает плохо — используй «Оригинал» и более лёгкие настройки/рендеры."
        )
    if ("установ" in q and ("игр" in q or "антолог" in q or "anthology" in q or "сборк" in q)) or ("как установить" in q):
        return (
            "Установка Anthology по FAQ:\n"
            "1) Перед скачиванием добавь в исключения антивируса папку, куда качаешь файлы, и папку, куда будешь ставить `ANTHOLOGY`.\n"
            "2) Не ставь игру в `Загрузки`, `Windows`, `Program Files` и другие системные папки.\n"
            "3) Держи запас места: примерно `110 ГБ` на системном диске для распаковки и около `117 ГБ` на диске установки.\n"
            "4) Желательно ставить на `SSD/NVMe`.\n"
            "5) Распаковывай архивы только через `7Zip`, не WinRAR.\n"
            "6) Для слабого ПК скачай/распакуй `Anomaly-1.5.3-Anthology 2.1`, зайди в папку игры и запускай `Anomaly Launcher.exe` — это вариант `Оригинал` без модпака.\n"
            "7) Если ставишь модпак — после базы следуй инструкции/лаунчеру для модпака и запускай через нужный профиль.\n"
            "8) Если ОЗУ 16–32 ГБ — поставь фиксированный файл подкачки `40–50 ГБ`; при 64 ГБ можно оставить авто."
        )
    if "установ" in q or "скач" in q:
        return (
            "По установке: скачай все нужные архивы полностью, дождись окончания загрузки без `.crdownload`, сложи файлы в одну папку без лишних символов в пути "
            "и только потом запускай установку/обновление через лаунчер. Если ошибка повторяется — пришли точный текст ошибки."
        )
    if (
        ("слаб" in q or "слабом желез" in q or "слабый пк" in q or "картош" in q)
        and ("модпак" in q or "модпаком" in q or "модак" in q)
    ):
        return (
            "Если железо слабое — лучше играть в «Оригинал». Он легче и может работать даже на DX8/DX9/DX11. "
            "«Модпак» тяжелее: оружейный пак, графика и геймплейные улучшения, и он рассчитан только на DX11. "
            "Попробовать можно, но если начнутся просадки/вылеты — переходи на «Оригинал» или сильно режь графику."
        )
    if "лаунчер" in q or "обновляется" in q or "обновлен" in q or "обновл" in q:
        return (
            "Если лаунчер/обновление не качается, чаще всего нет связи с GitHub или скачивание режет сеть/провайдер. "
            "Попробуй включить/выключить VPN или Zapret, сменить страну/сервер VPN и нажать «Обновить» ещё раз. "
            "Если ошибка остаётся — открой логи лаунчера и пришли точный HTTP-код или текст ошибки."
        )
    if "mags" in q or "redux" in q or "магазин" in q or "магазины" in q:
        return (
            "Магазины включаются/отключаются через MO2 модуль `Mags Redux`. "
            "Если нужно отключить механику магазинов — выключи этот модуль в MO2 и запускай игру через нужный профиль. "
            "Если проблема с перезарядкой/магазинами в игре — проверь также назначение клавиши прямой зарядки в MCM."
        )
    if "интерфейс" in q and any(term in q for term in ("стиль", "помен", "смен", "измен", "hud", "худ")):
        return (
            "Стиль интерфейса/HUD меняется тут: `MCM -> Выбор худа -> Стиль худа`. "
            "Если речь именно про BHS — это отдельно: `MCM -> Body Health System -> HUD -> Тип HUD`."
        )
    if "худ" in q and any(term in q for term in ("стиль", "помен", "смен", "измен")) and "bhs" not in q:
        return "Стиль HUD меняется тут: `MCM -> Выбор худа -> Стиль худа`."
    if (
        ("пнв" in q or "ночн" in q or "night" in q or "nvg" in q)
        and any(term in q for term in ("прицел", "оптик", "режим", "выключ", "отключ", "включ", "переключ"))
    ):
        return (
            "ПНВ/ночной режим в продвинутой оптике переключается через настройки меток: "
            "`MCM -> Переключение меток -> Переключатель меток в коллиматорных прицелах -> Переключение метки коллиматорного прицела`. "
            "По умолчанию клавиша `BACKSPACE`: она меняет сетки коллиматоров, а также включает/выключает ПНВ и тепловизионные режимы на продвинутой оптике."
        )
    if "прицел" in q or "scopes" in q or "scope" in q or "сетк" in q:
        if "сетк" in q or "тепловиз" in q:
            return (
                "Если пропали/забагались сетки в прицелах: сначала в настройках графики поставь текстуры на максимум и полностью отключи сглаживание SMAA/другие виды сглаживания. "
                "Если проблема именно с тепловизионными прицелами — открой `MCM -> 3D Scopes` и полностью сними галочку "
                "с пункта `Уменьшить разрешение тепловизионных прицелов`. Если именно размываются 3D-прицелы — там же уменьши `Множитель увеличения` до `1`. "
                "После изменения лучше перезапустить игру, чтобы настройки прицелов применились чисто."
            )
        return (
            "Если размываются/плохо выглядят 3D-прицелы, зайди в `MCM -> 3D Scopes` и уменьши ползунок "
            "`Множитель увеличения` до `1`. Если багуются сетки/тепловизионные прицелы — там же сними галочку "
            "`Уменьшить разрешение тепловизионных прицелов`. После правки лучше перезапустить игру."
        )
    if "микрофриз" in q or "статтер" in q or "фриз" in q:
        return (
            "При микрофризах попробуй ограничить FPS на 1–2 кадра ниже герцовки монитора. "
            "Для FreeSync/G-Sync включи технологию в мониторе и панели видеодрайвера, затем поставь лимит FPS ниже частоты экрана. "
            "Для обычного монитора можно ограничить FPS через игру/RTSS и включить V-Sync."
        )
    if "график" in q or "шейдер" in q or "рендер" in q:
        return (
            "По графике сначала проверь выбранный рендер и профиль. Для слабого ПК лучше «Оригинал» и более лёгкий DX8/DX9/DX11, "
            "а модпак рассчитан на DX11 и тяжелее. Если проблема с шейдерами — попробуй отключить тяжёлые GFX-моды и очистить кэш через лаунчер."
            " Также есть видеогайд по настройке графики и SSS/Screen Space Shaders: он лежит в Discord-канале `гайды-и-другая-инфа-для-2_1`. "
            "Его стоит смотреть, если хочешь выжать максимум FPS без сильной потери картинки или разобраться с размытием."
        )
    if "ошиб" in q or "вылет" in q:
        return (
            "По ошибке нужен точный текст или лог. Открой логи в лаунчере/папке игры и пришли последнюю ошибку: HTTP-код для лаунчера "
            "или строки `FATAL ERROR` / `SCRIPT ERROR` для игры. Без текста ошибки можно только гадать."
        )
    if vague_probe or ("маркер" in q and not explicit_story_game):
        return (
            "Уточни, пожалуйста, что именно произошло: название сюжета/квеста, текущую локацию, NPC или точный пункт настройки. "
            "По одному обрывку я лучше попрошу контекст, чем подставлю случайный квест или неправильную инструкцию."
        )
    return None


def should_answer_support_before_story(question: str, answer: str | None) -> bool:
    if not answer:
        return False
    q = latest_player_question(question).casefold().replace("ё", "е")
    full_q = (question or "").casefold().replace("ё", "е")
    if any(term in full_q for term in (
        "сюжет забытый отряд", "сюжет смерти вопреки", "сюжет пространственная",
        "сюжет атрибут", "сюжет путь во мгле", "сюжет долина шорохов",
        "сюжет anomaly living legend", "living legend", "живая легенда", "цербер",
        "забытый отряд", "смерти вопреки", "паутина лжи", "паутине лжи",
        "пространственная аномалия", "путь во мгле", "долина шорохов",
    )):
        return False
    explicit_story_in_latest = (
        any(term in q for term in ("тень чернобыля", "тень черноб", "чистое небо", "зов припяти", "сюжет "))
        or re.search(r"\b(тч|чн|зп)\b", q) is not None
        or is_new_anthology_story_question(question)
    )
    explicit_story_in_context = (
        any(term in full_q for term in (
            "тень чернобыля", "тень черноб", "чистое небо", "зов припяти",
            "кордон", "агропром", "янтар", "x-16", "х-16", "x-18", "х-18", "x-8", "х-8",
            "волк", "петрух", "шуст", "призрак", "стрелок", "круглов", "сахаров",
            "пространственная", "зверь", "лютый", "атрибут", "воланд", "путь во мгле",
            "борланд", "долина шорохов", "сердце оазиса", "паутина лжи", "клык",
            "забытый отряд", "змей", "бизон", "мертвое озеро", "мёртвое озеро",
        ))
        or re.search(r"\b(тч|чн|зп)\b", full_q) is not None
    )
    concrete_support = any(term in q for term in (
        "фпс", "fps", "лцу", "лазер", "прицел", "сетк", "патрон", "компас", "миникарт",
        "подстволь", "клин", "bhs", "mcm", "mo2", "dx8", "dx9", "dx11", "2.0", "2.1",
        "скач", "ссылк", "faq", "обт", "установ", "лаунчер", "обнов",
        "завис", "отмен", "сброс", "крафт", "фильтр", "противогаз", "осмотр", "оруж",
        "рук", "график", "шейдер", "рендер", "21:9", "ультрашир",
    ))
    if explicit_story_in_context and any(term in full_q for term in (
        "сюжет ", "смерти вопреки", "паутина лжи", "паутине лжи",
        "пространственная", "атрибут", "путь во мгле", "долина шорохов", "забытый отряд",
    )):
        return False
    if concrete_support:
        return not explicit_story_in_latest
    modpack_followup = any(term in full_q for term in ("модпак", "модпаком", "модак", "оригинал")) and any(
        term in q for term in ("сюжет тот", "сюжет такой", "квесты", "квест", "задания", "меняет", "трогает")
    )
    return modpack_followup and not explicit_story_in_context


def is_support_priority_question(question: str) -> bool:
    q = latest_player_question(question).casefold().replace("ё", "е")
    generic_support_question = any(
        phrase in q
        for phrase in (
            "проблема с", "как настроить", "что делать с", "что делать если", "не работает",
            "сломалось", "ошибка", "вылетает", "лагает", "фризит",
        )
    )
    if generic_support_question and not is_story_priority_question(question):
        return True
    support_terms = (
        "модпак", "модпаком", "оригинал", "слабое желез", "слабый пк", "dx8", "dx9", "dx11",
        "faq", "фпс", "fps", "производительность", "просадки", "просадка", "оптимизация",
        "mcm", "mo2", "fov", "3d", "scopes", "прицел", "магазин", "mags", "redux",
        "обновляется", "лаунчер", "github", "vpn", "7zip", "7-zip", "crdownload",
        "установка", "скачать", "архив", "ошибка", "вылет", "фриз", "микрофриз",
        "настроить", "настройка", "графика", "шейдер", "рендер", "bhs", "клин", "лцу",
    )
    return any(term in q for term in support_terms)


def has_explicit_story_game(question: str) -> bool:
    q = latest_player_question(question).casefold().replace("ё", "е")
    anthology_story_terms = (
        "пространственная аномалия", "пространственная", "аномалия", "зверь", "лютый", "маркус", "шуруп",
        "дуболом", "лесник", "таченко", "мурад", "застава", "ставрид", "химик", "хромой", "левша",
        "миклуха", "стронглав", "петрович", "распутин", "хантер", "шмыга", "маскарад",
        "атрибут", "воланд", "никита", "шериф", "молаг", "мишель", "гаррота", "тесак", "квант",
        "цитра", "пророк", "сектант", "санатор", "катакомбы",
        "путь во мгле", "мгле", "саван", "борланд", "шаман", "патоген", "логопед", "колязин",
        "багрецов", "спектрум", "маятник", "курчатов", "мертвый город", "мёртвый город",
        "долина шорохов", "шорохов", "мутный", "максимильян", "радик", "тесла",
        "трус", "балбес", "бывалый", "сердце оазиса", "микросхема",
        "смерти вопреки", "паутина лжи", "паутине лжи", "топи", "варг", "анубис",
        "харольд", "хасан", "чех", "клык", "фугас",
        "забытый отряд", "змей", "бизон", "ржавый", "фома", "коста", "кривой",
        "старый", "ворон", "гарик", "лысый", "мертвое озеро", "мёртвое озеро",
        "группа бизона", "обитатели", "болотная тварь", "незваные гости", "чужой среди своих",
    )
    return (
        any(game in q for game in ("тень чернобыля", "тень черноб", "чистое небо", "зов припяти", *anthology_story_terms))
        or re.search(r"\b(тч|чн|зп)\b", q) is not None
        or "сюжет " in q
    )


def clarify_story_question_answer(question: str) -> str:
    return (
        "Я вижу, что вопрос про сюжет, но не нашёл точный квест/этап по такой формулировке. "
        "Укажи название задания, NPC, текущую локацию или последний выполненный шаг — тогда я не подменю ответ другим сюжетом."
    )


def local_story_answer_from_context(question: str, context: dict, max_chars: int = MAX_ANSWER_CHARS) -> str:
    title = context.get("title") or "Источник"
    source = context.get("source") or "гайд"
    text = context.get("text") or ""
    q = (question or "").casefold().replace("ё", "е")
    sentences = split_sentences(text)

    direct = ""
    if any(word in q for word in ("убить", "перебить", "застрелить", "атаковать")):
        if re.search(r"\b(убить|перебить|расправ|атак|рейд|бой|бандит)", text.casefold().replace("ё", "е")):
            direct = "Да, можно."
        else:
            direct = "В тексте гайда прямого варианта с убийством не подтверждено."
    elif any(word in q for word in ("спасти", "жив", "выживет")):
        low = text.casefold().replace("ё", "е")
        if any(mark in low for mark in ("не удалось", "погиб", "мертв", "мёртв", "умер")):
            direct = "Судя по гайду, нет — спасти не получится."
        elif any(mark in low for mark in ("спасти", "выручить", "выживет", "освободить")):
            direct = "Да, по гайду это можно сделать."
    elif any(word in q for word in ("можно", "можно ли", "получится")):
        direct = "По гайду — да, если выполнить описанный вариант." if sentences else ""

    consequence_words = (
        "если", "после", "когда", "в итоге", "тогда", "награ", "получ", "вариант",
        "выберите", "придется", "придётся", "вернит", "отпуст", "начнут", "обыск",
    )
    picked: list[str] = []
    for sentence in sentences:
        low = sentence.casefold().replace("ё", "е")
        if any(word in low for word in consequence_words):
            picked.append(sentence)
        if len(picked) >= 7:
            break
    if not direct:
        picked = sentences[:8]
    elif not picked:
        picked = sentences[:7]

    body = " ".join(picked).strip()
    if direct:
        answer = f"{direct} {body}"
    else:
        answer = body
    if len(answer) > max_chars:
        answer = answer[:max_chars].rsplit(".", 1)[0].strip() or answer[:max_chars].rstrip(" ,;:")
        answer += "..."
    return f"{title} ({source}): {answer}"


def is_direct_decision_question(question: str) -> bool:
    q = latest_player_question(question).casefold().replace("ё", "е")
    return any(mark in q for mark in (
        "можно", "нельзя", "стоит ли", "надо ли", "обязательно", "будет ли",
        "убить", "перебить", "застрелить", "атаковать", "спасти", "выживет",
        "что будет", "последств",
    ))


def quick_story_decision_answer(question: str) -> str | None:
    q = latest_player_question(question).casefold().replace("ё", "е")
    wolf_start = (
        ("волк" in q or "петрух" in q or "шуст" in q)
        and ("бандит" in q or "атп" in q or "соло" in q)
        and ("тень" in q or "черноб" in q or "кордон" in q)
    )
    if wolf_start and any(word in q for word in ("убить", "убью", "перебить", "соло", "один")):
        return (
            "Первые шаги (Тень Чернобыля): Да, можно пройти в соло. "
            "Сюжет от этого не ломается: ты просто сам зачищаешь бандитов на Кордоне/АТП, потом заходишь в двухэтажное здание, освобождаешь Шустрого, забираешь у него флешку и возвращаешься к Сидоровичу. "
            "Если перед штурмом сказать Петрухе, что справишься один, он после боя даст дополнительную награду — пистолет Фора-12. Главное не убить сталкеров Волка/Петрухи, стрелять нужно по бандитам."
        )
    return None


_legacy_quick_story_decision_answer = quick_story_decision_answer


def quick_story_decision_answer(question: str) -> str | None:
    q = latest_player_question(question).casefold().replace("ё", "е")
    full_q = (question or "").casefold().replace("ё", "е")
    if "чернобог" in full_q and any(term in full_q for term in ("смертный грех", "смертном грехе", "грех")):
        return (
            "Смертный грех / Anomaly: Чернобог — ключевой противник и финальная фигура линии «Смертный грех», связанная с группировкой «Грех». "
            "Эта ветка идёт после «Живой легенды»: Рыжий лес/Лесник -> след «Греха» -> атаки на базы -> Лиманск -> Госпиталь -> Генераторы -> финальный конфликт. "
            "Это не Тень Чернобыля и не обычные бандиты: если игрок спрашивает про Чернобога, отвечай в рамках Anomaly/«Смертного греха»."
        )
    if "шов" in full_q and any(term in full_q for term in ("послесвеч", "afterglow", "операция")):
        return (
            "Операция Послесвечение / Anomaly: Шов — важный персонаж/узел этой линии, связанный с ИИГ/UNISG, Бродягой и финальными решениями после «Смертного греха». "
            "Если вопрос про Шова — не подменяй его Чистым Небом или Тенью Чернобыля: проверяй цепочку «Спецдоставка», «Под прикрытием», «Охраняемые секреты» и финальный разговор. "
            "Для точной подсказки по Шову нужен текущий квест или последняя сцена, потому что исходы у «Послесвечения» могут отличаться."
        )
    clear_sky_machinegun = (
        any(term in full_q for term in ("чистое небо", "сюжет чн", " чн", " clear sky"))
        and "кордон" in full_q
        and any(term in full_q for term in ("пулемет", "пулемёт", "пулеметчик", "пулемётчик"))
    )
    if clear_sky_machinegun:
        return (
            "Чистое Небо, Кордон / пулемётчик: это не ветка Тень Чернобыля. "
            "На старте Кордона военный пулемётчик простреливает дорогу, поэтому не стой на открытом месте и не пытайся перестреливаться. "
            "Беги рывками по укрытиям/низине к безопасной зоне, держись подальше от прямой линии огня и двигайся к сюжетному переходу дальше по Кордону. "
            "Если постоянно убивает — снизь вес, убери оружие в руки/на спину для скорости, используй аптечку после попадания и не задерживайся у блокпоста."
        )
    if any(term in full_q for term in ("warfare", "война группировок")) and any(term in full_q for term in ("живая легенда", "живой легенде", "живую легенду", "смертный грех", "послесвечение", "сюжет")):
        return (
            "Anomaly / Warfare: сюжетный режим и Warfare/«Война группировок» несовместимы. "
            "Если хочешь проходить «Живую легенду», затем «Смертный грех» и «Операцию Послесвечение», нужно начинать новую игру в сюжетном режиме без Warfare. "
            "В Warfare эти цепочки могут не стартовать или работать неправильно, потому что это отдельный режим песочницы/войны группировок."
        )
    living_legend = any(term in full_q for term in ("живая легенда", "живой легенде", "живую легенду", "living legend", "anomaly living", "сюжет anomaly", "сюжет аномали"))
    mortal_sin = any(term in full_q for term in ("смертный грех", "смертного греха", "mortal sin", "sin storyline"))
    afterglow = any(term in full_q for term in ("операция послесвечение", "послесвечение", "operation afterglow", "afterglow"))
    empty_borders = any(term in full_q for term in ("пустые границы", "empty borders", "unisg", "ииг"))
    dark_presence = any(term in full_q for term in ("темное присутствие", "тёмное присутствие", "dark presence"))
    zone_secrets = any(term in full_q for term in ("тайны зоны", "секреты зоны"))
    if living_legend and any(term in full_q for term in ("стрелк", "групп", "найти", "искать", "призрак", "клык", "доктор")):
        return (
            "Живая легенда / Anomaly: это отдельная цепочка про поиск Стрелка и его группы — не путай её со старой лабораторной веткой из другого сюжета. "
            "Общий маршрут такой: сначала собираешь сведения о Стрелке через ключевых сталкеров и учёных, затем выходишь к этапу Барьера. "
            "На Барьере помоги отбить атаку Монолита и обязательно поговори с Цербером/Проводником после боя. "
            "После этого цепочка ведёт на Радар: нужно отключить Выжигатель мозгов и выйти к северным этапам поиска Стрелка. "
            "Если маркер пропал — ориентируйся не по ТЧ, а по текущей записи КПК в «Живой легенде»: последний важный узел обычно Барьер -> Цербер -> Радар/Выжигатель."
        )
    if living_legend and "фантом" in full_q:
        return (
            "Живая легенда / Anomaly, Фантом: это чемпион/сильный боец Монолита на позднем этапе линии. "
            "Перед боем нужно собрать союзный отряд. Дальше группа осаждает ДК «Энергетик», чтобы выманить и убить Фантома. "
            "Это не диалоговая сцена, а серьёзная перестрелка: подготовь броню, патроны, аптечки и не забудь оружие в пылу боя. "
            "После победы над Фантомом Стрелок раскрывает дальнейшие намерения, и сюжет уходит в финальную военную фазу."
        )
    if mortal_sin:
        if "чернобог" in full_q:
            return (
                "Смертный грех / Anomaly: Чернобог — ключевой противник и финальная фигура линии «Смертный грех», связанная с группировкой «Грех» и северными событиями. "
                "Эта ветка идёт после «Живой легенды»: Рыжий лес/Лесник -> след «Греха» -> атаки на базы -> Лиманск -> Госпиталь -> Генераторы -> финальный конфликт с Чернобогом. "
                "Если игрок спрашивает про него, не подменяй ответ Тенью Чернобыля или обычными бандитами: это именно сюжет Anomaly."
            )
        if any(term in full_q for term in ("стрелк", "групп", "найти", "искать")):
            return (
                "Смертный грех / Anomaly: это уже не поиск группы Стрелка. Эта линия открывается после «Живой легенды» и переводит сюжет в войну с группировкой «Грех». "
                "Сначала отчитайся своему лидеру фракции, затем иди по цепочке к Рыжему лесу и Леснику: он выводит на след подозрительных сталкеров/Греха. "
                "Дальше будут КПК командира, атаки «Греха» на базы, Лиманск, Госпиталь, Генераторы и финал с Чернобогом. "
                "Стрелок здесь важен позже как союзник/участник северных событий, но стартовая задача — не искать его группу, а раскрутить угрозу «Греха»."
            )
        return (
            "Смертный грех / Anomaly: линия начинается после «Живой легенды». Главная тема — война с группировкой «Грех» и Чернобогом. "
            "Маршрут в общих чертах: лидер твоей фракции -> Рыжий лес/Лесник -> лагерь подозрительных сталкеров и КПК командира -> атаки «Греха» -> Лиманск -> Госпиталь -> Генераторы -> финальный доклад. "
            "Если маркер пропал, назови текущий этап/локацию: у этой линии разные начальники в зависимости от стартовой группировки."
        )
    if afterglow:
        if "шов" in full_q:
            return (
                "Операция Послесвечение / Anomaly: Шов — один из важных персонажей/узлов этой линии, связанный с ИИГ/UNISG, Бродягой и финальными решениями после «Смертного греха». "
                "Если вопрос про Шова — отвечай в рамках «Послесвечения», а не ЧН/ТЧ: проверяй задания «Спецдоставка», «Под прикрытием», «Охраняемые секреты» и финальный разговор. "
                "Исходы у линии разные, поэтому для точной подсказки по Шову нужен текущий квест или последняя сцена."
            )
        if "бродяг" in full_q:
            return (
                "Операция Послесвечение / Anomaly: Бродяга — важный участник линии после «Смертного греха», связанный с конфликтом военных/СБУ, ИИГ/UNISG, Стрелка и секретов Зоны. "
                "Иди по цепочке Дегтярёва/военных и ИИГ: «Спецдоставка», «Под прикрытием», «Охраняемые секреты», затем события с Бродягой, Шовом и финальный выбор. "
                "Если Бродяга погиб/исчез — это может быть учтённым исходом, а не обязательно поломкой."
            )
        if any(term in full_q for term in ("стрелк", "групп", "найти", "искать")):
            return (
                "Операция Послесвечение / Anomaly: это линия после «Смертного греха», где военные, СБУ, Дегтярёв, ИИГ/UNISG, Стрелок, Бродяга и Шов сталкиваются из-за секретов Зоны. "
                "Если вопрос про Стрелка — он здесь не просто цель поиска, а один из ключевых участников финального конфликта. "
                "Иди по заданиям Дегтярёва/военных и цепочке ИИГ: важные узлы — «Спецдоставка», «Под прикрытием», «Охраняемые секреты», Бродяга, Шов и финальный разговор. "
                "Финал зависит от решений: можно выйти на разные исходы, поэтому для точной подсказки нужен текущий квест."
            )
        return (
            "Операция Послесвечение / Anomaly: это неоднозначная постсюжетная линия после «Смертного греха». "
            "В ней участвуют Дегтярёв, военные/СБУ, ИИГ/UNISG, Стрелок, Бродяга и Шов. "
            "Ключевые решения находятся в заданиях «Спецдоставка», «Под прикрытием», «Охраняемые секреты» и в финальном разговоре. "
            "Если Бродяга или Шов умирают — это учитывается сюжетом, а не всегда означает слом."
        )
    if empty_borders:
        return (
            "Пустые границы / ИИГ-UNISG: это отдельная перспектива ИИГ, которая открывается после завершения «Операции Послесвечение» и достижения «Коллаборационист». "
            "Это не обычная «Живая легенда за другую фракцию», а отдельная линия с другой стороной событий."
        )
    if dark_presence or zone_secrets:
        return (
            "Anomaly, постсюжет/тайны Зоны: после основных линий есть хвосты вроде «Тёмного присутствия» и связанные с ИИГ/секретами Зоны этапы. "
            "«Тёмное присутствие» связано с выжившим из «Греха» и походом в Тёмную долину/X-18. "
            "Если ты спрашиваешь про конкретный этап «Тайн Зоны», напиши название задания или NPC — я привяжу ответ к нужной ветке, а не к старым сюжетам."
        )
    if any(term in q for term in ("жекан", "зохан", "зюхан", "жохан", "жехан")):
        return (
            "Забытый Отряд, Жекан: похоже, ты про Жекана — пленника из этой сюжетной линии. "
            "Если игрок пишет «Зохан/Зюхан/Жохан», обычно имеется в виду именно Жекан. "
            "Его линия завязана на ночной этап: дождись нужной ночной сцены, найди бандитов, уничтожь их и освободи пленника. "
            "Если сцена не запускается — проверь, активна ли именно линия Жекана, не пропущен ли предыдущий этап, и лучше не зачищай место заранее, чтобы не сбить стадию."
        )
    if any(term in full_q for term in ("цербер", "cerber")) and any(term in full_q for term in ("живая легенда", "легенда", "anomaly", "аномали", "кто", "где")):
        return (
            "Живая легенда / Anomaly: Цербер — важный сюжетный персонаж этой линии, связанный с поиском Стрелка и продвижением на север. "
            "Это не мутант и не случайный NPC. По текущей базе он появляется на этапе Барьера: помоги сталкерам отбить атаку Монолита, "
            "после боя обязательно поговори с Цербером/Проводником ещё раз — он выводит дальше по цепочке к Радару и бункеру под Выжигателем мозгов. "
            "Если Цербер погиб или маркер пропал, откати сохранение перед Барьером: эта стадия легко ломается, если затянуть бой или привести лишних врагов."
        )
    if "борланд" in q and ("выкуп" in q or "выкупить" in q):
        return (
            "Путь во Мгле, Борланд/выкуп: это ранняя стадия с военными на Кордоне. "
            "Саван и Борланд сначала завязаны на блокпост, Колязина, документы/условия выхода и деньги. "
            "Проверь, поговорил ли ты с Борландом и Колязиным, выполнил ли условия Колязина или собрал нужную сумму. "
            "В моей текущей базе нет точной цены/кнопки выкупа, поэтому не буду выдумывать: ориентир — закрыть этап блокпоста через Колязина и диалоги, после этого идти в деревню новичков и дальше к АТП."
        )
    if "борланд" in q and ("антидот" in q or "лекар" in q or "противояд" in q):
        return (
            "Путь во Мгле, Борланд/антидот: в текущей локальной базе нет точного места антидота, поэтому я не должен подставлять случайный предмет из другого квеста. "
            "Проверь текущую задачу в КПК и последние диалоги с Борландом/ключевым NPC этой стадии. Если цель говорит найти антидот — иди строго по маркеру/описанию текущего этапа, "
            "а если маркер пропал, напиши текущую локацию и последний диалог — тогда можно будет привязать подсказку к месту."
        )
    in_soc = any(term in full_q for term in ("тень", "черноб", "тч"))
    if (
        in_soc
        and any(term in full_q for term in ("волк", "петрух", "шуст", "атп", "бандит"))
        and any(term in q for term in ("петрух", "награ", "один", "соло", "сам", "пойду"))
    ):
        return (
            "Первые шаги (Тень Чернобыля): да, если перед штурмом сказать Петрухе/группе, что справишься один, можно зачистить бандитов на АТП в соло. "
            "Сюжет не ломается: после зачистки освобождаешь Шустрого, забираешь флешку и возвращаешься к Сидоровичу. "
            "За вариант в одиночку Петруха после боя даёт дополнительную награду — Фора-12. Главное — стрелять по бандитам, а не по сталкерам Волка/Петрухи."
        )
    if (
        in_soc
        and "призрак" in q
        and any(term in q for term in ("труп", "тело", "найти", "находится", "где"))
        and not any(term in q for term in ("учен", "ученый", "учёный", "запис", "вертолет", "вертолёт", "крушен"))
    ):
        return (
            "Тень Чернобыля, тема «труп Призрака»: это не место крушения на Янтаре. Сам труп Призрака находится в лаборатории X-16, "
            "после отключения установки/рубильников и боя в нижней части комплекса. Ориентир: после опасного участка в X-16 дойди до зоны с контролёром/псевдогигантом, "
            "после боя осмотри труп Призрака — с него нужны сведения о Стрелке и полезная броня. Дальше выходи через дыру/проход в правой клетке в туннели, "
            "выбирайся наружу и возвращайся к Сахарову. Если ты сейчас на Янтаре снаружи — тебе сначала нужно идти в X-16, а не искать Призрака на поверхности."
        )
    if (
        in_soc
        and "призрак" in q
        and any(term in q for term in ("учен", "ученый", "учёный", "запис", "вертолет", "вертолёт", "крушен"))
    ):
        return (
            "Тень Чернобыля, Янтарь: ты говоришь не про труп Призрака, а про тело учёного/исследователя с информацией о лаборатории и упоминанием Призрака. "
            "Оно лежит у места крушения вертолёта на болотах Янтаря, перед полноценным походом в X-16. Сначала осмотри это место, забери сведения, затем вернись к учёным/Сахарову, "
            "экипируйся и уже после этого двигайся к заводу и лаборатории X-16. Самого Призрака найдёшь позже внутри X-16."
        )
    if (
        ("чист" in q or "небо" in q)
        and ("кордон" in q or "пулемет" in q or "пулеметчик" in q or "военн" in q)
        and any(word in q for word in ("куда", "как", "убежать", "пройти", "двигаться", "сторону", "убивают"))
    ):
        return (
            "Чистое Небо, Кордон: с пулемётчиком лучше не воевать — это scripted-зона военных. "
            "Твоя задача не перестреливаться, а быстро уйти с южного блокпоста/насыпи к северу в сторону деревни новичков и Сидоровича. "
            "Держись укрытий, не стой на открытом месте и не лезь обратно к военным: маркер должен вести выше по карте, к сталкерской части Кордона. "
            "Если маркер пропал — ориентир такой: от военных уходишь прочь от базы, вверх/севернее по Кордону, к безопасной деревне и торговцу."
        )
    if (
        ("проводник" in q or "доктор" in q or "декодер" in q)
        and ("тень" in q or "черноб" in q or "сюжет" in q)
    ):
        return (
            "Тень Чернобыля: если хочешь нормальную/истинную концовку — да, к Проводнику идти нужно. "
            "Цепочка такая: после X-16 и информации от Призрака идёшь к Проводнику на Кордон, он отправляет к Доктору "
            "в тайник Стрелка в подземельях Агропрома. Доктор подсказывает про декодер в Припяти. "
            "Без этой цепочки ты, скорее всего, уйдёшь к Исполнителю желаний/ложным концовкам и не откроешь правильный путь."
        )
    return _legacy_quick_story_decision_answer(question)


async def answer_from_story_context(question: str, author_name: str, context: dict) -> str:
    if is_direct_decision_question(question):
        return local_story_answer_from_context(question, context)
    if not OPENAI_ENABLED:
        return local_story_answer_from_context(question, context)
    language = user_language_hint(question)
    system_prompt = (
        f"You are {BOT_DISPLAY_NAME}, a live Discord helper for S.T.A.L.K.E.R. Anthology players. "
        "Use only the provided guide fragment. Do not copy it verbatim as a wall of text. "
        "Answer the player's exact question first. If it asks yes/no, start with yes/no/maybe and explain why. "
        "Then explain consequences, what will happen, where to go, and what to do next. "
        "If the fragment does not support a yes/no conclusion, say so honestly. "
        "Keep it concise, practical, and human. "
        "Answer in English if the player asks in English; otherwise answer in Russian. "
        f"Detected language: {language}."
    )
    guide = (
        f"Game/source: {context.get('source')}\n"
        f"Quest/section: {context.get('title')}\n"
        f"Guide fragment:\n{context.get('text', '')[:5000]}"
    )
    response = await openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{author_name}: {question}\n\n{guide}"},
        ],
        max_output_tokens=1000,
    )
    return trim_answer(response.output_text)


async def ask_yura(question: str, author_name: str) -> str:
    story_priority = is_story_priority_question(question)
    if story_priority:
        quick_answer = quick_story_decision_answer(question)
        if quick_answer:
            return quick_answer
        full_context = full_sources.find_context(question, str(ROOT))
        if full_context:
            try:
                return await answer_from_story_context(question, author_name, full_context)
            except Exception as exc:
                print(f"OpenAI story answer failed: {type(exc).__name__}: {exc}")
                return local_story_answer_from_context(question, full_context)
        qa_answer = story_qa.find_answer(question, str(ROOT))
        if qa_answer:
            return trim_answer(qa_answer)
        if has_explicit_story_game(question):
            return clarify_story_question_answer(question)
    quick_support = quick_support_decision_answer(question)
    if should_answer_support_before_story(question, quick_support):
        return quick_support
    if quick_support:
        return quick_support
    if is_support_priority_question(question):
        support_answer = general_knowledge.find_answer(question, str(ROOT), max_chars=MAX_ANSWER_CHARS)
        if support_answer:
            return trim_answer(support_answer)
    support_answer = general_knowledge.find_answer(question, str(ROOT), max_chars=MAX_ANSWER_CHARS)
    if support_answer:
        return trim_answer(support_answer)
    if not OPENAI_ENABLED:
        return local_fallback_answer(question)
    language = user_language_hint(question)
    system_prompt = (
        f"You are {BOT_DISPLAY_NAME}, the Discord assistant for A.N.T.H.O.L.O.G.Y / S.T.A.L.K.E.R. Anthology players. "
        "Answer only about Anthology, Anomaly, MO2, MCM, launcher/update issues, performance, installation, and server rules. "
        "Start with a concrete answer, then add a short explanation if useful. "
        "For story/navigation questions, behave like a guide: name the location chain, nearest landmark, transition, NPC, and what to do if the marker is missing. "
        "Use left/right/straight/back only when the entry point or landmark is known; otherwise orient by map names, transitions, buildings, bases, and NPCs. "
        "If the user asks about a specific quest, NPC, title, or consequence and the local knowledge has no exact match, say that this quest is not confirmed in our Anthology knowledge/build; do not substitute a different nearby quest. "
        "If the user asks in English, answer in English. If the user asks in Russian, answer in Russian. "
        "Do not invent download links, versions, or server facts. If the local knowledge does not contain the answer, say that it should be checked in the Anthology Discord and added to the knowledge base. "
        f"Detected user language: {language}.\n\n"
        "Local Anthology knowledge base:\n"
        f"{KNOWLEDGE}"
    )
    response = await openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{author_name}: {question}"},
        ],
        max_output_tokens=1000,
    )
    return trim_answer(response.output_text)


def looks_like_followup(question: str) -> bool:
    q = (question or "").casefold()
    if not q:
        return False
    words = re.findall(r"[a-zа-я0-9][a-zа-я0-9_+\\-]{1,}", q)
    followup_words = (
        "он", "она", "они", "его", "ее", "её", "их", "там", "туда", "дальше",
        "потом", "после", "спасти", "мертв", "мёртв", "нашел", "нашёл",
        "куда", "как быть", "что делать", "а если", "а можно", "можно ли",
    )
    has_explicit_topic = any(topic in q for topic in (
        "тень чернобыля", "зов припяти", "чистое небо", "глухарь", "тремор",
        "кардан", "азот", "соколов", "тополь", "стрелок", "круглов", "волк",
        "кордон", "затор", "юпитер", "припять", "агропром", "х-8", "x-8",
    ))
    return any(word in q for word in followup_words) and not has_explicit_topic


def looks_like_followup(question: str) -> bool:
    q = (question or "").casefold().replace("ё", "е").strip()
    if not q:
        return False
    strong_followup = (
        "мы все еще", "мы всё еще", "все еще говорим", "всё еще говорим",
        "изначальному вопрос", "моему вопрос", "не про", "это не",
        "а именно", "я спрашивал", "я спрашиваю", "я имел в виду", "имею в виду",
        "по этой теме", "по нему", "по ней", "по этому", "тот же", "тот самый",
    )
    if re.search(r"(^|[\s,.;:!?—-])я\s+про\b", q):
        return True
    if any(phrase in q for phrase in strong_followup):
        return True
    standalone_topics = (
        "модпак", "модпаком", "модак", "оригинал", "сюжетн", "сюжетные отлич",
        "чем отлич", "разница", "фпс", "fps", "лцу", "лазер", "патрон", "компас",
        "миникарт", "подстволь", "клин", "bhs", "mcm", "mo2", "прицел", "сетк",
    )
    if any(topic in q for topic in standalone_topics):
        return False
    if any(phrase in q for phrase in (
        "то есть", "а квест", "а сюжет", "сюжет тот", "сюжет такой", "сюжет слом",
        "он меняет", "он трогает", "переназнач", "а если хочу",
        "где это", "почему не", "сломается", "даст награду", "один пойду",
    )):
        return True
    if "?" in q or any(phrase in q for phrase in (
        "могу ли", "можно ли", "обязательно", "чем отличается", "что делать", "как сделать",
        "почему", "где", "куда", "когда", "сюжет", "квест", "задание", "модпак", "оригинал",
        "проводник", "доктор", "декодер",
    )):
        return False
    return any(phrase in q for phrase in (
        "а дальше", "куда потом", "что потом", "а если", "а он", "а она", "а они",
        "после этого", "и что", "как быть", "там что", "он умер", "он мертв", "я нашел",
    ))


def extract_conversation_topic(question: str, answer: str) -> str:
    combined = f"{question}\n{answer}"
    low = combined.casefold().replace("ё", "е")
    game = ""
    if "тень чернобыля" in low or re.search(r"\bтч\b", low):
        game = "Тень Чернобыля"
    elif "зов припяти" in low or re.search(r"\bзп\b", low):
        game = "Зов Припяти"
    elif "чистое небо" in low or re.search(r"\bчн\b", low):
        game = "Чистое Небо"
    important: list[str] = []
    for token in (
        "Призрак", "Стрелок", "Круглов", "Сахаров", "X-16", "Х-16", "X-18", "Х-18", "X-8", "Х-8",
        "Агропром", "Янтарь", "Кордон", "Затон", "Юпитер", "Припять", "СКАТ", "Глухарь", "Тремор",
        "Волк", "Шустрый", "Сидорович", "Проводник", "Доктор", "декодер", "3D-прицелы", "прицелы",
        "сетки", "ЛЦУ", "патроны", "модпак", "Оригинал", "Anthology 2.1",
    ):
        if token.casefold().replace("ё", "е") in low and token not in important:
            important.append(token)
    parts = []
    if game:
        parts.append(game)
    if important:
        parts.append(", ".join(important[:5]))
    return " / ".join(parts) if parts else compact_text(question, 160)


def context_key_from_message(message: discord.Message) -> str:
    if message.guild is None:
        return f"discord-dm:{getattr(message.author, 'id', 'author')}"
    return f"{getattr(message.channel, 'id', 'channel')}:{getattr(message.author, 'id', 'author')}"


def dm_context_key_from_user(user: discord.abc.User) -> str:
    return f"discord-dm:{getattr(user, 'id', 'user')}"


def context_is_recent(context_key: str | None) -> bool:
    if not context_key:
        return False
    previous = CONVERSATION_CONTEXT.get(context_key)
    if not previous:
        return False
    try:
        ts = float(previous.get("ts", "0"))
    except Exception:
        ts = 0.0
    return ts > 0 and (time.time() - ts) <= CONVERSATION_TTL_SECONDS


def is_reply_to_yura(message: discord.Message) -> bool:
    reference = getattr(message, "reference", None)
    resolved = getattr(reference, "resolved", None) if reference else None
    author = getattr(resolved, "author", None)
    return bool(bot.user and author and getattr(author, "id", None) == bot.user.id)


def should_continue_discord_dialogue(message: discord.Message, context_key: str | None) -> bool:
    if message.author.bot:
        return False
    content = (message.content or "").strip()
    if not content:
        return False
    if message.guild is None:
        return context_is_recent(context_key)
    if is_reply_to_yura(message):
        return True
    if not context_is_recent(context_key):
        return False
    lowered = content.casefold().replace("ё", "е")
    return (
        "?" in content
        or lowered.startswith(("а ", "а как", "а где", "а куда", "а что", "а если", "а можно", "почему", "как ", "где ", "куда ", "что "))
        or looks_like_followup(content)
    )


def with_conversation_context(question: str, context_key: str | None, force_context: bool = False) -> str:
    if not context_key:
        return question
    previous = CONVERSATION_CONTEXT.get(context_key, "")
    if previous and (force_context or looks_like_followup(question)):
        topic = previous.get("topic", "")
        prev_question = previous.get("question", "")
        prev_answer = previous.get("answer", "")
        return (
            f"Тема текущего разговора: {topic}\n"
            f"Предыдущий вопрос игрока: {prev_question}\n"
            f"Предыдущий ответ Юры: {prev_answer}\n\n"
            f"Уточнение игрока: {question}\n\n"
            "Важно: отвечай именно в рамках темы текущего разговора. Не подменяй сюжет/квест другим, если игрок уточняет или говорит «не про это»."
        )
    return question


def remember_conversation_context(context_key: str | None, question: str, answer: str) -> None:
    if not context_key:
        return
    compact_question = re.sub(r"\s+", " ", question or "").strip()
    compact_answer = compact_text(answer, 900)
    CONVERSATION_CONTEXT[context_key] = {
        "topic": extract_conversation_topic(compact_question, compact_answer),
        "question": compact_text(compact_question, 500),
        "answer": compact_answer,
        "ts": str(time.time()),
    }
    if len(CONVERSATION_CONTEXT) > 300:
        for key in list(CONVERSATION_CONTEXT)[:80]:
            CONVERSATION_CONTEXT.pop(key, None)


async def answer_discord(destination: discord.abc.Messageable, question: str, author_name: str, context_key: str | None = None, force_context: bool = False) -> None:
    if not question:
        await destination.send("\u041d\u0430\u043f\u0438\u0448\u0438 \u0432\u043e\u043f\u0440\u043e\u0441 \u043f\u043e\u0441\u043b\u0435 \u043e\u0431\u0440\u0430\u0449\u0435\u043d\u0438\u044f: `\u042e\u0440\u0430, ...` \u0438\u043b\u0438 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439 `/ask`.")
        return
    effective_question = with_conversation_context(question, context_key, force_context=force_context)
    async with destination.typing():
        try:
            answer = await ask_yura(effective_question, author_name)
        except Exception as exc:
            print(f"OpenAI answer failed: {type(exc).__name__}: {exc}")
            answer = local_fallback_answer(effective_question)
    remember_conversation_context(context_key, question, answer)
    await send_discord_answer(destination, answer)


async def answer_user_dm(message: discord.Message, question: str, force_context: bool = False) -> bool:
    """Answer a guild mention in the user's DM and keep a private per-user context."""
    try:
        dm_channel = message.author.dm_channel or await message.author.create_dm()
        await answer_discord(
            dm_channel,
            question,
            message.author.display_name,
            dm_context_key_from_user(message.author),
            force_context=force_context,
        )
        return True
    except discord.Forbidden:
        return False


async def send_discord_answer(destination: discord.abc.Messageable, answer: str) -> None:
    text = (answer or "").strip()
    if not text:
        text = "Я не нашёл точный ответ. Уточни сюжет/квест/локацию или точную ошибку."
    while len(text) > DISCORD_CHUNK_CHARS:
        cut = text[:DISCORD_CHUNK_CHARS]
        split_at = max(cut.rfind("\n"), cut.rfind(". "), cut.rfind("; "), cut.rfind(", "))
        if split_at < 700:
            split_at = DISCORD_CHUNK_CHARS
        chunk = text[:split_at].strip()
        await destination.send(chunk)
        text = text[split_at:].strip()
    if text:
        await destination.send(text)


def check_bridge_token(value: str | None) -> None:
    if not BRIDGE_TOKEN:
        return
    if not value or value.strip() != BRIDGE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bridge token.")


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


@app.post("/ask", response_class=PlainTextResponse)
async def bridge_ask(request: Request, x_anthology_bridge_token: str | None = Header(default=None)) -> str:
    check_bridge_token(x_anthology_bridge_token)
    client_host = getattr(request.client, "host", "local")
    now = time.time()
    wait = BRIDGE_RATE_SECONDS - (now - BRIDGE_LAST_REQUEST_BY_IP.get(client_host, 0.0))
    if wait > 0:
        raise HTTPException(status_code=429, detail=f"Too many requests. Retry after {wait:.1f}s.")
    BRIDGE_LAST_REQUEST_BY_IP[client_host] = now
    body = await request.body()
    question = cleanup_raw_question(body.decode("utf-8", errors="replace").strip())
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")
    context_key = f"bridge:{client_host}"
    effective_question = with_conversation_context(question[:700], context_key)
    try:
        answer = await ask_yura(effective_question, "Relay Chat")
    except Exception as exc:
        print(f"OpenAI bridge answer failed: {type(exc).__name__}: {exc}")
        answer = local_fallback_answer(effective_question)
    remember_conversation_context(context_key, question[:700], answer)
    return answer


@bot.event
async def on_ready() -> None:
    print(f"READY: {BOT_DISPLAY_NAME} logged in as {bot.user} and is ready.", flush=True)
    await bot.change_presence(
        activity=discord.Activity(type=discord.ActivityType.listening, name="Anthology questions")
    )
    for guild in bot.guilds:
        try:
            if guild.me and guild.me.nick != BOT_DISPLAY_NAME:
                await guild.me.edit(nick=BOT_DISPLAY_NAME, reason="Anthology bot display name")
        except discord.Forbidden:
            print(f"Cannot change nickname in guild {guild.name}: missing Manage Nicknames.")
        except Exception as exc:
            print(f"Cannot change nickname in guild {guild.name}: {type(exc).__name__}")

    try:
        if DISCORD_GUILD_ID:
            guild = discord.Object(id=int(DISCORD_GUILD_ID))
            bot.tree.copy_global_to(guild=guild)
            synced = await bot.tree.sync(guild=guild)
        else:
            synced = await bot.tree.sync()
        print(f"Synced {len(synced)} slash command(s).", flush=True)
    except Exception as exc:
        print(f"Slash command sync failed: {type(exc).__name__}: {exc}", flush=True)


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    triggered = is_triggered(message)
    context_key = context_key_from_message(message)
    continues_dialogue = should_continue_discord_dialogue(message, context_key)
    print(f"MESSAGE: guild={getattr(message.guild, 'id', None)} channel={getattr(message.channel, 'id', None)} author={message.author} content_len={len(message.content or '')} mentions_bot={bool(bot.user and bot.user in message.mentions)} triggered={triggered} continues_dialogue={continues_dialogue}", flush=True)
    if triggered or continues_dialogue:
        question = cleanup_question(message)
        if message.guild is not None:
            delivered = await answer_user_dm(message, question, force_context=continues_dialogue)
            if delivered:
                try:
                    await message.add_reaction("📩")
                except Exception:
                    pass
                return
            await message.channel.send(f"{message.author.mention}, не могу написать тебе в личку — открой ЛС от участников сервера. Пока отвечаю здесь.")
        await answer_discord(message.channel, question, message.author.display_name, context_key, force_context=continues_dialogue)
        return
    await bot.process_commands(message)


@bot.tree.command(name="ask", description="Ask Yura about Anthology.")
@app_commands.describe(question="Your Anthology / Anomaly / MO2 / MCM question")
async def ask_command(interaction: discord.Interaction, question: str) -> None:
    await interaction.response.defer(thinking=True, ephemeral=True)
    context_key = dm_context_key_from_user(interaction.user)
    effective_question = with_conversation_context(question, context_key)
    try:
        answer = await ask_yura(effective_question, interaction.user.display_name)
    except Exception as exc:
        print(f"OpenAI answer failed: {type(exc).__name__}: {exc}")
        answer = local_fallback_answer(effective_question)
    remember_conversation_context(context_key, question, answer)
    try:
        dm_channel = interaction.user.dm_channel or await interaction.user.create_dm()
        await send_discord_answer(dm_channel, answer)
        await interaction.followup.send("Ответил в личку.", ephemeral=True)
    except discord.Forbidden:
        await interaction.followup.send("Не могу написать в личку — открой ЛС от участников сервера. Ответ ниже:\n\n" + answer[:1800], ephemeral=True)


@bot.tree.command(name="reload_knowledge", description="Reload local knowledge files. Admin only.")
async def reload_knowledge_command(interaction: discord.Interaction) -> None:
    permissions = getattr(interaction.user, "guild_permissions", None)
    if not permissions or not permissions.manage_guild:
        await interaction.response.send_message("Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р Р‹Р РЋРІР‚С”Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР Р†Р вЂљРІР‚СљР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РІР‚вЂњР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћвЂ“Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†Р вЂљРЎвЂќР В Р’В Р В Р вЂ№Р В Р Р‹Р Р†Р вЂљРЎвЂќР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎС™Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР Р†Р вЂљРІР‚СљР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В Р вЂ№Р В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћвЂ“Р В Р’В Р вЂ™Р’В Р В Р’В Р В РІР‚в„–Р В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р Р‹Р РЋРІР‚С”Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р Р‹Р РЋРІР‚С”Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†РІР‚С›РЎС›Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†Р вЂљРЎвЂќР В Р’В Р В Р вЂ№Р В Р Р‹Р Р†Р вЂљРЎвЂќР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р Р‹Р РЋРІР‚С”Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В¶Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р Р‹Р РЋРІР‚С”Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В Р вЂ№Р В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В¦Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎС™Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†РІР‚С›РЎС›Р В Р’В Р вЂ™Р’В Р В Р’В Р В РІР‚в„–Р В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р Р‹Р РЋРІР‚С”Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†РІР‚С›РЎС›Р В Р’В Р вЂ™Р’В Р В Р’В Р В РІР‚в„–Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†Р вЂљРЎСљР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†РІР‚С›РЎС›Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В Р вЂ№Р В Р Р‹Р Р†РІР‚С›РЎС› Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р Р‹Р РЋРІР‚С”Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР Р†Р вЂљРІР‚СљР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В Р вЂ№Р В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†РІР‚С›РЎС›Р В Р’В Р вЂ™Р’В Р В Р’В Р В РІР‚в„–Р В Р’В Р В Р вЂ№Р В Р Р‹Р Р†РІР‚С›РЎС›Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р вЂ Р В РІР‚С™Р Р†Р вЂљРЎС™Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В Р вЂ№Р В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћвЂ“Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†Р вЂљРЎвЂќР В Р’В Р В Р вЂ№Р В Р Р‹Р Р†Р вЂљРЎвЂќР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р Р‹Р РЋРІР‚С”Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†РІР‚С›РЎС›Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†Р вЂљРЎвЂќР В Р’В Р В Р вЂ№Р В Р Р‹Р Р†Р вЂљРЎвЂќР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р Р‹Р РЋРІР‚С”Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В°Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р Р‹Р РЋРІР‚С”Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В Р вЂ№Р В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р Р‹Р РЋРІР‚С”Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†РІР‚С›РЎС›Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р Р‹Р Р†Р вЂљРЎвЂќР В Р’В Р В Р вЂ№Р В Р Р‹Р Р†Р вЂљРЎвЂќР В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В Р В Р’В Р вЂ™Р’В Р В Р’В Р Р†Р вЂљР’В Р В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р РЋРІвЂћСћР В Р’В Р В РІР‚В Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р Р‹Р РЋРІР‚С”Р В Р’В Р вЂ™Р’В Р В РІР‚в„ўР вЂ™Р’В Р В Р’В Р В РІР‚В Р В Р’В Р Р†Р вЂљРЎв„ўР В Р вЂ Р Р†Р вЂљРЎвЂєР РЋРЎвЂєР В Р’В Р вЂ™Р’В Р В Р вЂ Р В РІР‚С™Р Р†РІР‚С›РЎС›Р В Р’В Р Р†Р вЂљРІвЂћСћР В РІР‚в„ўР вЂ™Р’В° Manage Server.", ephemeral=True)
        return
    global KNOWLEDGE
    KNOWLEDGE = load_knowledge()
    await interaction.response.send_message("Knowledge base reloaded.", ephemeral=True)


async def run_http_server() -> None:
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def main() -> None:
    await asyncio.gather(
        bot.start(DISCORD_TOKEN),
        run_http_server(),
    )


if __name__ == "__main__":
    asyncio.run(main())
