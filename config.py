# config.py — загрузка настроек из переменных окружения

import os
from pathlib import Path

from dotenv import load_dotenv

# Загружаем .env из корня проекта
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Telegram
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не задан в .env")

# OpenAI
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не задан в .env")

OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_IMAGE_MODEL: str = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1-mini")

# Пути к данным
PROMPTS_JSON_PATH: Path = BASE_DIR / "prompts.json"
MEMORY_JSON_PATH: Path = BASE_DIR / "memory.json"

# Лимиты контекста (по заданию — по 10 сообщений)
MAX_USER_MESSAGES: int = 10
MAX_ASSISTANT_MESSAGES: int = 10
