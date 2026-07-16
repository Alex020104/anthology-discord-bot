# Anthology Discord Bot — Юра Семецкий

Discord-бот помощник для сервера A.N.T.H.O.L.O.G.Y.

Он отвечает по гайдам из папки `knowledge/`, использует OpenAI Responses API и по умолчанию экономичную модель `gpt-5.6-luna`.

## Важно про токены

Не вставляй Discord token или OpenAI API key прямо в код. Используй переменные окружения:

- `DISCORD_TOKEN`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Токен из старого `bot.py`, который лежал в Downloads, уже был засвечен в файле. Его лучше перевыпустить в Discord Developer Portal.

## Локальный запуск

```powershell
py -3 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
.\.venv\Scripts\python bot.py
```

## Как спрашивать

В Discord:

- упомянуть бота: `@Anthology_Bot как включить 3 лицо?`
- написать имя: `Юра как убрать клин?`
- slash-команда: `/ask question: how to enable third person view?`

Бот отвечает на языке вопроса: русский вопрос — русский ответ, английский вопрос — английский ответ.

## Облачный деплой

Подходит любой сервис, где можно запустить Python worker:

- Render
- Railway
- Fly.io
- VPS

Для Render уже есть `render.yaml`.

Нужные Environment Variables:

```text
DISCORD_TOKEN=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.6-luna
BOT_DISPLAY_NAME=Юра Семецкий
```

## Discord Gateway Intents

В Discord Developer Portal включи:

- Message Content Intent
- Server Members Intent можно не включать, если бот только отвечает на вопросы

## Права бота на сервере

Минимально:

- View Channels
- Send Messages
- Read Message History
- Use Slash Commands

Если хочешь, чтобы бот сам ставил себе ник `Юра Семецкий`, дай ему право Manage Nicknames.

## Локальный helper сохраняется

Этот Discord-бот не заменяет локальный `anthology-ai-helper`, который используется Relay Chat внутри игры.

Локальный режим:

```text
Relay Chat -> http://127.0.0.1:8787/ask -> anthology-ai-helper
```

Discord режим:

```text
Discord -> Юра Семецкий -> OpenAI + knowledge/
```

Bridge режим:

```text
Relay Chat -> local anthology-ai-helper -> cloud Юра Семецкий -> OpenAI + knowledge/
```

Если облачный Юра недоступен, локальный helper отвечает сам из локальной базы.

Подробнее смотри:

```text
docs/local-and-discord-modes.md
```

Чтобы скопировать свежие гайды из локального helper в Discord-бота:

```powershell
.\scripts\sync_knowledge_from_local_helper.ps1
```
