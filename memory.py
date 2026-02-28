# memory.py — хранение контекста диалога (последние N сообщений пользователя и ассистента)

import json
from pathlib import Path
from typing import Any

from config import (
    MAX_ASSISTANT_MESSAGES,
    MAX_USER_MESSAGES,
    MEMORY_JSON_PATH,
    PROMPTS_JSON_PATH,
)


def _load_json(path: Path) -> dict[str, Any]:
    """Загрузить JSON-файл или вернуть пустой словарь."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json(path: Path, data: dict[str, Any]) -> None:
    """Сохранить данные в JSON-файл."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _chat_key(chat_id: int) -> str:
    return str(chat_id)


# ---------------------------------------------------------------------------
# Промпты (режимы)
# ---------------------------------------------------------------------------


def load_prompts() -> dict[str, Any]:
    """Загрузить prompts.json: default_prompt и prompts."""
    data = _load_json(PROMPTS_JSON_PATH)
    prompts = data.get("prompts", {})
    default = data.get("default_prompt", "assistant")
    if default not in prompts:
        default = next(iter(prompts), "assistant")
    return {"default_prompt": default, "prompts": prompts}


def get_system_prompt(prompts_data: dict[str, Any], mode_key: str | None = None) -> str:
    """Вернуть system_prompt для выбранного режима."""
    mode_key = mode_key or prompts_data.get("default_prompt", "assistant")
    prompts = prompts_data.get("prompts", {})
    mode = prompts.get(mode_key, prompts.get("assistant", {}))
    return mode.get("system_prompt", "Ты — полезный помощник.")


# ---------------------------------------------------------------------------
# Память диалога по чатам
# ---------------------------------------------------------------------------

_memory_cache: dict[str, dict[str, Any]] = {}
_prompts_cache: dict[str, Any] | None = None


def _get_memory_data() -> dict[str, dict[str, Any]]:
    """Загрузить память из файла (с кэшем в памяти)."""
    global _memory_cache
    if not _memory_cache:
        _memory_cache = _load_json(MEMORY_JSON_PATH)
    return _memory_cache


def _persist_memory() -> None:
    """Сохранить память в файл."""
    global _memory_cache
    _save_json(MEMORY_JSON_PATH, _memory_cache)


def get_chat_state(chat_id: int) -> dict[str, Any]:
    """Получить состояние чата: mode, user_messages, assistant_messages."""
    key = _chat_key(chat_id)
    data = _get_memory_data()
    if key not in data:
        data[key] = {
            "mode": None,  # None = default_prompt из prompts.json
            "user_messages": [],
            "assistant_messages": [],
            "input_tokens": 0,   # накопительно, не сбрасывается /reset
            "output_tokens": 0,
        }
    return data[key]


def set_chat_mode(chat_id: int, mode_key: str) -> None:
    """Установить режим (промпт) для чата."""
    state = get_chat_state(chat_id)
    state["mode"] = mode_key
    _persist_memory()


def append_user_message(chat_id: int, text: str) -> None:
    """Добавить сообщение пользователя и обрезать до MAX_USER_MESSAGES."""
    state = get_chat_state(chat_id)
    state["user_messages"].append(text)
    state["user_messages"] = state["user_messages"][-MAX_USER_MESSAGES:]
    _persist_memory()


def append_assistant_message(chat_id: int, text: str) -> None:
    """Добавить ответ ассистента и обрезать до MAX_ASSISTANT_MESSAGES."""
    state = get_chat_state(chat_id)
    state["assistant_messages"].append(text)
    state["assistant_messages"] = state["assistant_messages"][-MAX_ASSISTANT_MESSAGES:]
    _persist_memory()


def get_messages_for_api(chat_id: int, system_prompt: str) -> list[dict[str, str]]:
    """
    Сформировать список сообщений для OpenAI API:
    [system], затем пары user/assistant из истории (хронологически), затем текущее user (если есть).
    """
    state = get_chat_state(chat_id)
    messages = [{"role": "system", "content": system_prompt}]
    user_msgs = state.get("user_messages", [])
    assistant_msgs = state.get("assistant_messages", [])
    n = min(len(user_msgs), len(assistant_msgs))
    for i in range(n):
        messages.append({"role": "user", "content": user_msgs[i]})
        messages.append({"role": "assistant", "content": assistant_msgs[i]})
    if len(user_msgs) > n:
        messages.append({"role": "user", "content": user_msgs[-1]})
    return messages


def reset_chat(chat_id: int) -> None:
    """Очистить историю сообщений чата (режим и статистику токенов не трогаем)."""
    key = _chat_key(chat_id)
    data = _get_memory_data()
    if key in data:
        data[key]["user_messages"] = []
        data[key]["assistant_messages"] = []
        _persist_memory()


def get_chat_stats(chat_id: int) -> tuple[int, int]:
    """Вернуть (input_tokens, output_tokens) для чата."""
    state = get_chat_state(chat_id)
    return (
        state.get("input_tokens") or 0,
        state.get("output_tokens") or 0,
    )


def add_tokens(chat_id: int, input_tokens: int, output_tokens: int) -> None:
    """Добавить токены к накопительной статистике чата."""
    state = get_chat_state(chat_id)
    state["input_tokens"] = (state.get("input_tokens") or 0) + input_tokens
    state["output_tokens"] = (state.get("output_tokens") or 0) + output_tokens
    _persist_memory()


def reset_chat_stats(chat_id: int) -> None:
    """Обнулить статистику токенов для чата."""
    state = get_chat_state(chat_id)
    state["input_tokens"] = 0
    state["output_tokens"] = 0
    _persist_memory()


def get_prompts_data() -> dict[str, Any]:
    """Загрузить промпты (с кэшем)."""
    global _prompts_cache
    if _prompts_cache is None:
        _prompts_cache = load_prompts()
    return _prompts_cache


def reload_prompts() -> dict[str, Any]:
    """Перезагрузить prompts.json (например, после изменения файла)."""
    global _prompts_cache
    _prompts_cache = load_prompts()
    return _prompts_cache
