# Развертывание Discord-бота на облаке

## 1. Перевыпусти Discord token

Старый токен был вставлен прямо в `bot.py`, поэтому его нужно считать засвеченным.

Discord Developer Portal -> Applications -> Anthology_Bot -> Bot -> Reset Token.

## 2. Включи intents

Discord Developer Portal -> Bot:

- Message Content Intent: ON

## 3. Переменные окружения

На облаке добавь:

```text
DISCORD_TOKEN=новый_токен_discord_бота
OPENAI_API_KEY=ключ_openai
OPENAI_MODEL=gpt-5.6-luna
BOT_DISPLAY_NAME=Юра Семецкий
BRIDGE_TOKEN=любая_длинная_случайная_строка
```

Опционально:

```text
DISCORD_GUILD_ID=id_твоего_discord_сервера
```

Если указать `DISCORD_GUILD_ID`, slash-команды появятся быстрее на конкретном сервере.

`BRIDGE_TOKEN` нужен, чтобы Relay Chat/local helper мог безопасно спрашивать облачного Юру через `/ask`.

## 4. Render

1. Создай новый Web Service / Background Worker из GitHub repo.
2. Build command:

```bash
pip install -r requirements.txt
```

3. Start command:

```bash
python bot.py
```

4. Добавь env vars из пункта 3.

## 5. Railway

1. New Project -> Deploy from GitHub repo.
2. Variables -> добавь env vars.
3. Start command:

```bash
python bot.py
```

## 6. Проверка

В Discord напиши:

```text
Юра как включить режим третьего лица?
```

или:

```text
@Anthology_Bot how to enable third person view?
```

HTTP bridge проверяется так:

```powershell
$headers = @{ "X-Anthology-Bridge-Token" = "твой_BRIDGE_TOKEN" }
Invoke-RestMethod -Method Post -Uri "https://your-cloud-domain/ask" -Headers $headers -Body "how to enable third person view"
```
