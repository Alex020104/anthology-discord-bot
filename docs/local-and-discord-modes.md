# Локальный AI helper и Discord-бот

Нам нужны оба режима:

1. **Локальный helper**  
   Работает рядом с игрой и Relay Chat на компьютере игрока/разработчика.

   Путь в игре:

   ```text
   ANTHOLOGY\Anomaly-1.5.3-Anthology 2.1\anthology-ai-helper
   ```

   Relay Chat обращается к нему по:

   ```text
   http://127.0.0.1:8787/ask
   ```

   Это важно сохранить, потому что так AI работает внутри Реального чата/игры локально.

2. **Discord-бот Юра Семецкий**  
   Работает на облаке и отвечает в Discord.

   Репозиторий:

   ```text
   https://github.com/Alex020104/anthology-discord-bot
   ```

   Он не заменяет локальный helper. Это отдельный бот для Discord.

## Как обновлять знания

Источник знаний можно держать одинаковым:

- локальный helper: `projects/anthology-ai-helper/knowledge`
- Discord bot: `projects/anthology-discord-bot/knowledge`

Когда мы добавляем новые гайды в локальный helper, можно синхронизировать их в Discord-бот скриптом:

```powershell
.\scripts\sync_knowledge_from_local_helper.ps1
```

После этого нужно сделать commit/push в репозиторий Discord-бота, чтобы облако получило свежие гайды.

## Что нельзя делать

- Не удалять `anthology-ai-helper` из game payload.
- Не переводить Relay Chat напрямую только на Discord-бота.
- Не хранить Discord token или OpenAI API key в репозитории.

## Будущая связка

Bridge-режим:

- Relay Chat спрашивает локальный helper, как сейчас.
- Discord-бот отвечает в Discord.
- Если в локальном helper заданы переменные `ANTHOLOGY_CLOUD_AI_URL` и `ANTHOLOGY_CLOUD_AI_TOKEN`, локальный helper сначала спросит облачного Юру.
- Если облако недоступно, локальный helper вернётся к локальной базе.

Это безопаснее, чем заставлять игроков зависеть от облачного Discord-бота для локального игрового чата.

## Как включить bridge

На машине, где запускается Relay Chat/local helper, добавь переменные окружения:

```text
ANTHOLOGY_CLOUD_AI_URL=https://your-cloud-domain.example.com/ask
ANTHOLOGY_CLOUD_AI_TOKEN=тот_же_BRIDGE_TOKEN_что_на_облаке
```

После этого перезапусти Relay Chat/helper.

Если переменные не заданы, всё работает по-старому локально.
