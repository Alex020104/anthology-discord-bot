from __future__ import annotations

import asyncio
import logging
import os
import re
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
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "1600"))
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
CONVERSATION_CONTEXT: dict[str, str] = {}


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
    qa_answer = story_qa.find_answer(question, str(ROOT))
    if qa_answer:
        return qa_answer
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
        content = content.replace(f"<@{bot.user.id}>", "")
        content = content.replace(f"<@!{bot.user.id}>", "")
    return content.strip()


def is_triggered(message: discord.Message) -> bool:
    if bot.user and bot.user in message.mentions:
        return True
    lowered = (message.content or "").casefold()
    return any(name and name in lowered for name in BOT_TRIGGER_NAMES)


def is_auto_question(message: discord.Message) -> bool:
    if str(getattr(message.channel, "id", "")) not in AUTO_REPLY_QUESTION_CHANNEL_IDS:
        return False
    content = (message.content or "").strip()
    if not content:
        return False
    return "?" in content or content.casefold().startswith(("как ", "что ", "где ", "почему ", "когда "))


def cleanup_question(message: discord.Message) -> str:
    question = strip_bot_mention(message)
    lowered = question.casefold()
    for name in sorted(BOT_TRIGGER_NAMES, key=len, reverse=True):
        if lowered.startswith(name):
            question = question[len(name):].strip(" ,:;—-")
            break
    return question.strip()


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
        if len(picked) >= 4:
            break
    if not direct:
        picked = sentences[:4]
    elif not picked:
        picked = sentences[:4]

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
    q = (question or "").casefold().replace("ё", "е")
    return any(mark in q for mark in (
        "можно", "нельзя", "стоит ли", "надо ли", "обязательно", "будет ли",
        "убить", "перебить", "застрелить", "атаковать", "спасти", "выживет",
        "что будет", "последств",
    ))


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
        max_output_tokens=450,
    )
    return trim_answer(response.output_text)


async def ask_yura(question: str, author_name: str) -> str:
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
        max_output_tokens=450,
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


def context_key_from_message(message: discord.Message) -> str:
    return f"{getattr(message.channel, 'id', 'channel')}:{getattr(message.author, 'id', 'author')}"


def with_conversation_context(question: str, context_key: str | None) -> str:
    if not context_key:
        return question
    previous = CONVERSATION_CONTEXT.get(context_key, "")
    if previous and looks_like_followup(question):
        return f"{previous}\n\nУточнение игрока: {question}"
    return question


def remember_conversation_context(context_key: str | None, question: str, answer: str) -> None:
    if not context_key:
        return
    compact_question = re.sub(r"\s+", " ", question or "").strip()
    CONVERSATION_CONTEXT[context_key] = (
        f"Предыдущий вопрос игрока: {compact_question}"
    )
    if len(CONVERSATION_CONTEXT) > 300:
        for key in list(CONVERSATION_CONTEXT)[:80]:
            CONVERSATION_CONTEXT.pop(key, None)


async def answer_discord(destination: discord.abc.Messageable, question: str, author_name: str, context_key: str | None = None) -> None:
    if not question:
        await destination.send("\u041d\u0430\u043f\u0438\u0448\u0438 \u0432\u043e\u043f\u0440\u043e\u0441 \u043f\u043e\u0441\u043b\u0435 \u043e\u0431\u0440\u0430\u0449\u0435\u043d\u0438\u044f: `\u042e\u0440\u0430, ...` \u0438\u043b\u0438 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439 `/ask`.")
        return
    effective_question = with_conversation_context(question, context_key)
    async with destination.typing():
        try:
            answer = await ask_yura(effective_question, author_name)
        except Exception as exc:
            print(f"OpenAI answer failed: {type(exc).__name__}: {exc}")
            answer = local_fallback_answer(effective_question)
    remember_conversation_context(context_key, question, answer)
    await destination.send(answer)


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
    body = await request.body()
    question = body.decode("utf-8", errors="replace").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")
    context_key = f"bridge:{getattr(request.client, 'host', 'local')}"
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
    triggered = is_triggered(message) or is_auto_question(message)
    print(f"MESSAGE: guild={getattr(message.guild, 'id', None)} channel={getattr(message.channel, 'id', None)} author={message.author} content_len={len(message.content or '')} mentions_bot={bool(bot.user and bot.user in message.mentions)} triggered={triggered}", flush=True)
    if triggered:
        question = cleanup_question(message)
        await answer_discord(message.channel, question, message.author.display_name, context_key_from_message(message))
        return
    await bot.process_commands(message)


@bot.tree.command(name="ask", description="Ask Yura about Anthology.")
@app_commands.describe(question="Your Anthology / Anomaly / MO2 / MCM question")
async def ask_command(interaction: discord.Interaction, question: str) -> None:
    await interaction.response.defer(thinking=True)
    try:
        answer = await ask_yura(question, interaction.user.display_name)
    except Exception as exc:
        print(f"OpenAI answer failed: {type(exc).__name__}: {exc}")
        answer = local_fallback_answer(question)
    await interaction.followup.send(answer)


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
