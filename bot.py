from __future__ import annotations

import asyncio
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


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"

load_dotenv(ROOT / ".env")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()
BOT_DISPLAY_NAME = os.getenv("BOT_DISPLAY_NAME", "Юра Семецкий").strip()
BOT_TRIGGER_NAMES = [
    item.strip().casefold()
    for item in os.getenv("BOT_TRIGGER_NAMES", "anthology_bot,юра,юра семецкий,антология бот").split(",")
    if item.strip()
]
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID", "").strip()
BRIDGE_TOKEN = os.getenv("BRIDGE_TOKEN", "").strip()
PORT = int(os.getenv("PORT", "8787"))
MAX_KNOWLEDGE_CHARS = int(os.getenv("MAX_KNOWLEDGE_CHARS", "24000"))
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "1600"))

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is not set.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
app = FastAPI(title="Anthology Discord Bot Bridge")


def load_knowledge() -> str:
    parts: list[str] = []
    for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if text:
            parts.append(f"## {path.relative_to(KNOWLEDGE_DIR).as_posix()}\n{text}")
    knowledge = "\n\n".join(parts)
    if len(knowledge) > MAX_KNOWLEDGE_CHARS:
        return knowledge[:MAX_KNOWLEDGE_CHARS].rsplit("\n", 1)[0]
    return knowledge


KNOWLEDGE = load_knowledge()


def trim_answer(text: str) -> str:
    text = " ".join((text or "").replace("\r", "\n").split())
    if len(text) > MAX_ANSWER_CHARS:
        return text[: MAX_ANSWER_CHARS - 1].rstrip() + "…"
    return text


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


def cleanup_question(message: discord.Message) -> str:
    question = strip_bot_mention(message)
    lowered = question.casefold()
    for name in sorted(BOT_TRIGGER_NAMES, key=len, reverse=True):
        if lowered.startswith(name):
            question = question[len(name):].strip(" ,:;—-")
            break
    return question.strip()


def user_language_hint(question: str) -> str:
    latin = len(re.findall(r"[A-Za-z]", question or ""))
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", question or ""))
    if latin and latin >= max(1, cyrillic * 2):
        return "English"
    return "Russian"


async def ask_yura(question: str, author_name: str) -> str:
    language = user_language_hint(question)
    system_prompt = (
        f"You are {BOT_DISPLAY_NAME}, the Discord assistant for A.N.T.H.O.L.O.G.Y / S.T.A.L.K.E.R. Anthology players. "
        "Answer only about Anthology, Anomaly, MO2, MCM, launcher/update issues, performance, installation, and server rules. "
        "Start with a concrete answer, then add a short explanation if useful. "
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


async def answer_discord(destination: discord.abc.Messageable, question: str, author_name: str) -> None:
    if not question:
        await destination.send("Задай вопрос после упоминания. Например: `Юра как убрать клин?`")
        return
    async with destination.typing():
        try:
            answer = await ask_yura(question, author_name)
        except Exception as exc:
            answer = f"Юра пока споткнулся на ошибке: `{type(exc).__name__}`. Проверь OPENAI_API_KEY / модель / логи облака."
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
    return await ask_yura(question[:700], "Relay Chat")


@bot.event
async def on_ready() -> None:
    print(f"✅ {BOT_DISPLAY_NAME} logged in as {bot.user} and is ready.")
    await bot.change_presence(
        activity=discord.Activity(type=discord.ActivityType.listening, name="вопросы по Anthology")
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
        print(f"Synced {len(synced)} slash command(s).")
    except Exception as exc:
        print(f"Slash command sync failed: {type(exc).__name__}: {exc}")


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if is_triggered(message):
        question = cleanup_question(message)
        await answer_discord(message.channel, question, message.author.display_name)
        return
    await bot.process_commands(message)


@bot.tree.command(name="ask", description="Ask Юра Семецкий about Anthology.")
@app_commands.describe(question="Your Anthology / Anomaly / MO2 / MCM question")
async def ask_command(interaction: discord.Interaction, question: str) -> None:
    await interaction.response.defer(thinking=True)
    try:
        answer = await ask_yura(question, interaction.user.display_name)
    except Exception as exc:
        answer = f"Юра пока споткнулся на ошибке: `{type(exc).__name__}`. Проверь OPENAI_API_KEY / модель / логи облака."
    await interaction.followup.send(answer)


@bot.tree.command(name="reload_knowledge", description="Reload local knowledge files. Admin only.")
async def reload_knowledge_command(interaction: discord.Interaction) -> None:
    permissions = getattr(interaction.user, "guild_permissions", None)
    if not permissions or not permissions.manage_guild:
        await interaction.response.send_message("Нужны права Manage Server.", ephemeral=True)
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
