# main.py — точка входа, роутеры и обработчики бота

import asyncio
import base64
import logging

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, BaseFilter
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    KeyboardButton,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardMarkup,
)
from openai import AsyncOpenAI

import config
from memory import (
    add_tokens,
    append_assistant_message,
    append_user_message,
    get_chat_stats,
    get_messages_for_api,
    get_prompts_data,
    get_system_prompt,
    get_chat_state,
    reset_chat,
    reset_chat_stats,
    set_chat_mode,
)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Лимит длины одного сообщения в Telegram
TELEGRAM_MESSAGE_LIMIT = 4096

router = Router()
openai_client: AsyncOpenAI | None = None

# Чаты, ожидающие ввод описания для генерации изображения
_chats_waiting_image_prompt: set[int] = set()
# Чаты в меню настроек изображения (после нажатия «Картинка»)
_chats_in_image_menu: set[int] = set()
# Настройки генерации изображения по чатам
_image_settings: dict[int, dict[str, str]] = {}

DEFAULT_IMAGE_SETTINGS = {
    "quality": "low",
    "size": "1024x1536",
    "background": "auto",
    "output_format": "png",
}


def get_openai_client() -> AsyncOpenAI:
    global openai_client
    if openai_client is None:
        openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return openai_client


# ---------------------------------------------------------------------------
# Клавиатура с командами (кнопки)
# ---------------------------------------------------------------------------

def build_main_keyboard() -> ReplyKeyboardMarkup:
    """Клавиатура с командами в виде удобочитаемых кнопок."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Перезапуск"), KeyboardButton(text="Режим")],
            [KeyboardButton(text="Очистить историю"), KeyboardButton(text="Статистика")],
            [KeyboardButton(text="Обнулить статистику"), KeyboardButton(text="Картинка")],
        ],
        resize_keyboard=True,
    )


def build_image_settings_keyboard() -> ReplyKeyboardMarkup:
    """Клавиатура настроек генерации изображения."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Качество"), KeyboardButton(text="Размер")],
            [KeyboardButton(text="Фон"), KeyboardButton(text="Формат")],
            [KeyboardButton(text="Ввести описание"), KeyboardButton(text="Выйти")],
        ],
        resize_keyboard=True,
    )


def get_image_settings(chat_id: int) -> dict[str, str]:
    """Получить настройки изображения для чата (с подстановкой значений по умолчанию)."""
    if chat_id not in _image_settings:
        _image_settings[chat_id] = DEFAULT_IMAGE_SETTINGS.copy()
    return _image_settings[chat_id]


def build_quality_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="low", callback_data="img_q:low")],
        [InlineKeyboardButton(text="medium", callback_data="img_q:medium")],
        [InlineKeyboardButton(text="high", callback_data="img_q:high")],
        [InlineKeyboardButton(text="auto", callback_data="img_q:auto")],
    ])


def build_size_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Square 1024×1024", callback_data="img_s:1024x1024")],
        [InlineKeyboardButton(text="Portrait 1024×1536", callback_data="img_s:1024x1536")],
        [InlineKeyboardButton(text="Landscape 1536×1024", callback_data="img_s:1536x1024")],
        [InlineKeyboardButton(text="auto", callback_data="img_s:auto")],
    ])


def build_background_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="transparent", callback_data="img_bg:transparent")],
        [InlineKeyboardButton(text="opaque", callback_data="img_bg:opaque")],
        [InlineKeyboardButton(text="auto", callback_data="img_bg:auto")],
    ])


def build_format_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="png", callback_data="img_fmt:png")],
        [InlineKeyboardButton(text="webp", callback_data="img_fmt:webp")],
        [InlineKeyboardButton(text="jpeg", callback_data="img_fmt:jpeg")],
    ])


def format_image_settings_text(chat_id: int) -> str:
    """Текст с текущими настройками изображения."""
    s = get_image_settings(chat_id)
    size_labels = {"1024x1024": "Square 1024×1024", "1024x1536": "Portrait 1024×1536", "1536x1024": "Landscape 1536×1024", "auto": "auto"}
    return (
        f"Качество: **{s['quality']}**\n"
        f"Размер: **{size_labels.get(s['size'], s['size'])}**\n"
        f"Фон: **{s['background']}**\n"
        f"Формат: **{s['output_format']}**"
    )


def build_mode_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура выбора режима ассистента (inline-кнопки)."""
    prompts_data = get_prompts_data()
    prompts = prompts_data.get("prompts", {})
    buttons = [
        [InlineKeyboardButton(text=data.get("name", key), callback_data=f"mode:{key}")]
        for key, data in prompts.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def split_message(text: str, max_length: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Разбить длинный текст на части не длиннее max_length (по границам строк где возможно)."""
    if len(text) <= max_length:
        return [text] if text else []
    chunks = []
    rest = text
    while rest:
        if len(rest) <= max_length:
            chunks.append(rest)
            break
        block = rest[:max_length]
        last_newline = block.rfind("\n")
        if last_newline > max_length // 2:
            cut = last_newline + 1
        else:
            cut = max_length
        chunks.append(rest[:cut])
        rest = rest[cut:].lstrip("\n")
    return chunks


async def keep_typing(bot: Bot, chat_id: int, stop_event: asyncio.Event) -> None:
    """Периодически отправлять «печатает...», пока не установлен stop_event."""
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            break
        await asyncio.wait(
            [stop_event.wait(), asyncio.sleep(4)],
            return_when=asyncio.FIRST_COMPLETED,
        )


# ---------------------------------------------------------------------------
# handlers: /start
# ---------------------------------------------------------------------------

@router.message(Command("start"))
@router.message(F.text == "Перезапуск")
async def cmd_start(message: Message) -> None:
    """Приветствие и подсказка по командам."""
    prompts_data = get_prompts_data()
    default_key = prompts_data.get("default_prompt", "assistant")
    state = get_chat_state(message.chat.id)
    current_mode = state.get("mode") or default_key
    mode_info = prompts_data["prompts"].get(current_mode, {})
    mode_name = mode_info.get("name", current_mode)
    text = (
        f"Привет! Я бот с поддержкой OpenAI.\n\n"
        f"Текущий режим: **{mode_name}**\n\n"
        "Используй кнопки ниже или команды:\n"
        "• Режим — сменить режим\n"
        "• Очистить историю — сбросить диалог\n\n"
        "Просто напиши сообщение — я отвечу с учётом контекста."
    )
    await message.answer(text, reply_markup=build_main_keyboard(), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# handlers: /mode — выбор режима (промпта)
# ---------------------------------------------------------------------------

@router.message(Command("mode"))
@router.message(F.text == "Режим")
async def cmd_mode(message: Message) -> None:
    """Показать список режимов и кнопки выбора режима ассистента."""
    prompts_data = get_prompts_data()
    prompts = prompts_data.get("prompts", {})
    lines = ["Выбери режим ассистента:\n"]
    for key, data in prompts.items():
        name = data.get("name", key)
        desc = data.get("description", "")
        lines.append(f"• **{name}** — {desc}")
    await message.answer(
        "\n".join(lines),
        reply_markup=build_mode_keyboard(),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# handlers: /reset — очистка памяти
# ---------------------------------------------------------------------------

@router.message(Command("reset"))
@router.message(F.text == "Очистить историю")
async def cmd_reset(message: Message) -> None:
    """Очистить историю диалога для этого чата (статистика токенов не сбрасывается)."""
    reset_chat(message.chat.id)
    await message.answer("История диалога очищена.", reply_markup=build_main_keyboard())


# ---------------------------------------------------------------------------
# handlers: /stats — статистика токенов и стоимости
# ---------------------------------------------------------------------------

COST_PER_1M_INPUT = 0.25
COST_PER_1M_OUTPUT = 2.00


@router.message(Command("stats"))
@router.message(F.text == "Статистика")
async def cmd_stats(message: Message) -> None:
    """Показать накопительную статистику: запросы, ответы модели, примерная стоимость."""
    input_tok, output_tok = get_chat_stats(message.chat.id)
    cost = (input_tok / 1_000_000 * COST_PER_1M_INPUT) + (
        output_tok / 1_000_000 * COST_PER_1M_OUTPUT
    )
    text = (
        "📊 **Статистика использования OpenAI**\n\n"
        f"Запросы пользователя (входящие токены): {input_tok:,}\n"
        f"Ответы модели (исходящие токены): {output_tok:,}\n\n"
        f"Стоимость исходя из вход/выход ${COST_PER_1M_INPUT:.2f}/${COST_PER_1M_OUTPUT:.2f} составила **${cost:.4f}**"
    )
    await message.answer(text, reply_markup=build_main_keyboard(), parse_mode="Markdown")


@router.message(Command("reset_stats"))
@router.message(F.text == "Обнулить статистику")
async def cmd_reset_stats(message: Message) -> None:
    """Обнулить накопительную статистику токенов для этого чата."""
    reset_chat_stats(message.chat.id)
    await message.answer("Статистика обнулена.", reply_markup=build_main_keyboard())


# ---------------------------------------------------------------------------
# handlers: генерация изображения по промпту
# ---------------------------------------------------------------------------

async def generate_image(prompt: str, settings: dict[str, str] | None = None) -> tuple[bytes | None, str | None]:
    """
    Сгенерировать изображение по текстовому описанию.
    settings: quality, size, background, output_format (для gpt-image-*).
    Возвращает (bytes, None) при ответе в base64 или (None, url) при ответе по URL.
    При ошибке — (None, None).
    """
    settings = settings or DEFAULT_IMAGE_SETTINGS
    try:
        client = get_openai_client()
        kwargs = {
            "model": config.OPENAI_IMAGE_MODEL,
            "prompt": prompt,
            "n": 1,
        }
        if "dall-e" in config.OPENAI_IMAGE_MODEL.lower():
            kwargs["size"] = "1024x1024"
        else:
            kwargs["quality"] = settings.get("quality", "low")
            kwargs["size"] = settings.get("size", "1024x1536")
            kwargs["background"] = settings.get("background", "auto")
            kwargs["output_format"] = settings.get("output_format", "png")
        response = await client.images.generate(**kwargs)
        if not response.data:
            return (None, None)
        img = response.data[0]
        if getattr(img, "b64_json", None):
            return (base64.b64decode(img.b64_json), None)
        if getattr(img, "url", None):
            return (None, img.url)
        return (None, None)
    except Exception as e:
        logger.exception("Image generation failed: %s", e)
        return (None, None)


@router.message(F.text == "Картинка")
async def cmd_image_menu(message: Message) -> None:
    """Показать меню настроек генерации изображения."""
    chat_id = message.chat.id
    _chats_in_image_menu.add(chat_id)
    get_image_settings(chat_id)
    text = (
        "Настройки генерации изображения:\n\n"
        f"{format_image_settings_text(chat_id)}\n\n"
        "Выберите параметр или нажмите «Ввести описание» для генерации."
    )
    await message.answer(
        text,
        reply_markup=build_image_settings_keyboard(),
        parse_mode="Markdown",
    )


class ImageMenuFilter(BaseFilter):
    """Фильтр: чат в режиме настроек изображения."""

    async def __call__(self, message: Message) -> bool:
        return message.chat.id in _chats_in_image_menu


@router.message(ImageMenuFilter(), F.text == "Качество")
async def cmd_image_quality(message: Message) -> None:
    """Показать выбор качества."""
    await message.answer("Выберите качество:", reply_markup=build_quality_keyboard())


@router.message(ImageMenuFilter(), F.text == "Размер")
async def cmd_image_size(message: Message) -> None:
    """Показать выбор размера."""
    await message.answer("Выберите размер:", reply_markup=build_size_keyboard())


@router.message(ImageMenuFilter(), F.text == "Фон")
async def cmd_image_background(message: Message) -> None:
    """Показать выбор фона."""
    await message.answer("Выберите фон:", reply_markup=build_background_keyboard())


@router.message(ImageMenuFilter(), F.text == "Формат")
async def cmd_image_format(message: Message) -> None:
    """Показать выбор формата."""
    await message.answer("Выберите формат файла:", reply_markup=build_format_keyboard())


@router.message(ImageMenuFilter(), F.text == "Ввести описание")
async def cmd_image_enter_prompt(message: Message) -> None:
    """Перейти к вводу описания изображения."""
    _chats_in_image_menu.discard(message.chat.id)
    _chats_waiting_image_prompt.add(message.chat.id)
    await message.answer(
        "Введите описание изображения в следующем сообщении:",
        reply_markup=build_main_keyboard(),
    )


@router.message(ImageMenuFilter(), F.text == "Выйти")
async def cmd_image_exit(message: Message) -> None:
    """Выйти из меню изображений (аналог /start)."""
    _chats_in_image_menu.discard(message.chat.id)
    await cmd_start(message)


@router.callback_query(F.data.startswith("img_q:"))
async def callback_image_quality(callback: CallbackQuery) -> None:
    """Сохранение выбора качества."""
    value = callback.data.removeprefix("img_q:")
    get_image_settings(callback.message.chat.id)["quality"] = value
    await callback.answer(f"Качество: {value}")
    text = f"Настройки обновлены.\n\n{format_image_settings_text(callback.message.chat.id)}"
    await callback.message.answer(text, reply_markup=build_image_settings_keyboard(), parse_mode="Markdown")


@router.callback_query(F.data.startswith("img_s:"))
async def callback_image_size(callback: CallbackQuery) -> None:
    """Сохранение выбора размера."""
    value = callback.data.removeprefix("img_s:")
    get_image_settings(callback.message.chat.id)["size"] = value
    await callback.answer(f"Размер: {value}")
    text = f"Настройки обновлены.\n\n{format_image_settings_text(callback.message.chat.id)}"
    await callback.message.answer(text, reply_markup=build_image_settings_keyboard(), parse_mode="Markdown")


@router.callback_query(F.data.startswith("img_bg:"))
async def callback_image_background(callback: CallbackQuery) -> None:
    """Сохранение выбора фона."""
    value = callback.data.removeprefix("img_bg:")
    get_image_settings(callback.message.chat.id)["background"] = value
    await callback.answer(f"Фон: {value}")
    text = f"Настройки обновлены.\n\n{format_image_settings_text(callback.message.chat.id)}"
    await callback.message.answer(text, reply_markup=build_image_settings_keyboard(), parse_mode="Markdown")


@router.callback_query(F.data.startswith("img_fmt:"))
async def callback_image_format(callback: CallbackQuery) -> None:
    """Сохранение выбора формата."""
    value = callback.data.removeprefix("img_fmt:")
    get_image_settings(callback.message.chat.id)["output_format"] = value
    await callback.answer(f"Формат: {value}")
    text = f"Настройки обновлены.\n\n{format_image_settings_text(callback.message.chat.id)}"
    await callback.message.answer(text, reply_markup=build_image_settings_keyboard(), parse_mode="Markdown")


async def _handle_image_request(message: Message, prompt: str) -> None:
    """Сгенерировать изображение по промпту и отправить в чат."""
    chat_id = message.chat.id
    bot = message.bot
    keyboard = build_main_keyboard()
    settings = get_image_settings(chat_id)
    await bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)

    image_bytes, image_url = await generate_image(prompt, settings)
    ext = settings.get("output_format", "png")

    if image_bytes:
        photo = BufferedInputFile(image_bytes, filename=f"image.{ext}")
        await message.answer_photo(photo=photo, reply_markup=keyboard)
    elif image_url:
        await message.answer_photo(photo=image_url, reply_markup=keyboard)
    else:
        await message.answer(
            "⚠️ Не удалось сгенерировать изображение. Проверьте ключ API и лимиты или попробуйте другой промпт.",
            reply_markup=keyboard,
        )


@router.message(Command("image"))
async def cmd_image(message: Message) -> None:
    """Сгенерировать изображение по промпту из команды /image <описание> (альтернатива кнопке)."""
    parts = (message.text or "").strip().split(maxsplit=1)
    prompt = (parts[1] if len(parts) > 1 else "").strip()
    if not prompt:
        await message.answer(
            "Укажите описание после команды: /image кот на луне",
            reply_markup=build_main_keyboard(),
        )
        return
    await _handle_image_request(message, prompt)


# ---------------------------------------------------------------------------
# handlers: callback — выбор режима по inline-кнопке
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("mode:"))
async def callback_mode(callback: CallbackQuery) -> None:
    """Обработка нажатия на кнопку выбора режима."""
    key = callback.data.removeprefix("mode:")
    prompts_data = get_prompts_data()
    if key not in prompts_data.get("prompts", {}):
        await callback.answer("Неизвестный режим.")
        return
    set_chat_mode(callback.message.chat.id, key)
    mode_name = prompts_data["prompts"][key].get("name", key)
    await callback.answer()
    await callback.message.answer(
        f"Режим изменён на: **{mode_name}**",
        reply_markup=build_main_keyboard(),
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# handlers: текстовое сообщение (режим или вопрос к боту)
# ---------------------------------------------------------------------------

def is_mode_key(text: str) -> str | None:
    """Проверить, является ли текст ключом одного из режимов."""
    if not text or len(text) > 50:
        return None
    key = text.strip().lower()
    prompts_data = get_prompts_data()
    if key in prompts_data.get("prompts", {}):
        return key
    return None


@router.message(F.text)
async def handle_text(message: Message) -> None:
    """Обработка текста: смена режима, промпт для картинки или запрос в OpenAI."""
    text = (message.text or "").strip()
    if not text:
        return

    # Режим «Картинка»: следующее сообщение — описание изображения
    if message.chat.id in _chats_waiting_image_prompt:
        _chats_waiting_image_prompt.discard(message.chat.id)
        await _handle_image_request(message, text)
        return

    # Проверка на выбор режима (пользователь мог написать "developer" и т.д.)
    mode_key = is_mode_key(text)
    if mode_key is not None:
        set_chat_mode(message.chat.id, mode_key)
        prompts_data = get_prompts_data()
        mode_name = prompts_data["prompts"][mode_key].get("name", mode_key)
        await message.answer(f"Режим изменён на: **{mode_name}**", reply_markup=build_main_keyboard(), parse_mode="Markdown")
        return

    try:
        await _handle_openai_request(message, text)
    except Exception as e:
        logger.exception("Unexpected error in handle_text: %s", e)
        await message.answer(
            "⚠️ Произошла непредвиденная ошибка. Попробуй перезапуск (кнопка «Перезапуск») или повтори позже.",
            reply_markup=build_main_keyboard(),
        )


async def _handle_openai_request(message: Message, text: str) -> None:
    """Отправить запрос в OpenAI и ответить пользователю."""
    # Обычное сообщение — отправляем в OpenAI
    chat_id = message.chat.id
    bot = message.bot
    append_user_message(chat_id, text)
    prompts_data = get_prompts_data()
    state = get_chat_state(chat_id)
    current_mode = state.get("mode") or prompts_data.get("default_prompt", "assistant")
    system_prompt = get_system_prompt(prompts_data, current_mode)
    messages = get_messages_for_api(chat_id, system_prompt)

    # Показываем «печатает...» пока ждём ответ
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(bot, chat_id, stop_typing))
    keyboard = build_main_keyboard()

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
        )
        content: str = (response.choices[0].message.content or "").strip()

        # Учёт токенов: API может вернуть prompt_tokens/completion_tokens или input_tokens/output_tokens
        if response.usage is not None:
            u = response.usage
            inp = getattr(u, "input_tokens", None) or getattr(u, "prompt_tokens", None) or 0
            out = getattr(u, "output_tokens", None) or getattr(u, "completion_tokens", None) or 0
            try:
                usage_dict = u.model_dump() if hasattr(u, "model_dump") else vars(u)
            except Exception:
                usage_dict = {"prompt_tokens": inp, "completion_tokens": out}
            logger.info(
                "OpenAI ответ: токены в запросе=%s, токены в ответе=%s | usage=%s",
                inp,
                out,
                usage_dict,
            )
            add_tokens(chat_id, inp, out)
            total_in, total_out = get_chat_stats(chat_id)
            logger.info(
                "Накопительно по чату %s: входящие=%s, исходящие=%s",
                chat_id,
                total_in,
                total_out,
            )
    except Exception as e:
        logger.exception("OpenAI request failed: %s", e)
        err_msg = str(e).strip()[:300]
        await message.answer(
            f"⚠️ Ошибка при обращении к OpenAI.\n\n"
            f"Причина: {err_msg}\n\n"
            "Проверь ключ API и лимиты на platform.openai.com. Можно попробовать позже или написать короче.",
            reply_markup=keyboard,
        )
        return
    finally:
        stop_typing.set()
        try:
            await asyncio.wait_for(typing_task, timeout=1.0)
        except asyncio.TimeoutError:
            typing_task.cancel()

    try:
        if content:
            append_assistant_message(chat_id, content)
            chunks = split_message(content)
            for i, chunk in enumerate(chunks):
                is_last = i == len(chunks) - 1
                await message.answer(
                    chunk,
                    reply_markup=keyboard if is_last else None,
                )
        else:
            await message.answer("Пустой ответ от модели.", reply_markup=keyboard)
    except Exception as e:
        logger.exception("Error sending response: %s", e)
        await message.answer(
            "⚠️ Ответ получен, но не удалось отправить его в чат. Попробуй перезапуск (/start) или напиши короче.",
            reply_markup=keyboard,
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def main() -> None:
    # Проверка наличия промптов при старте
    get_prompts_data()
    bot = Bot(token=config.BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    logger.info("Bot starting...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
