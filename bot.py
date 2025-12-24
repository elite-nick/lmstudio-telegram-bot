import logging
import sqlite3
import requests
import time
import os
import asyncio
import base64
from io import BytesIO
from dotenv import load_dotenv

from telegram import Update, User
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.request import HTTPXRequest

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not set")

# ------------------------------------------------------------------------------
# LM Studio
# ------------------------------------------------------------------------------
LM_BASE = "http://localhost:1234/v1"
LM_CHAT = f"{LM_BASE}/chat/completions"

DEFAULT_MODEL = "llava"
TOKEN_THRESHOLD = 12000

conversation_params = {
    "max_tokens": 400,
    "temperature": 0.7,
    "top_p": 1.0,
}

# ------------------------------------------------------------------------------
# Database
# ------------------------------------------------------------------------------
DB_FILE = "conversations.db"


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        c = conn.cursor()

        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            last_seen INTEGER
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            default_model TEXT,
            active_conversation_id INTEGER
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            model TEXT
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            ts INTEGER
        )""")


def upsert_user(user: User):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
        INSERT INTO users VALUES (?, ?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET last_seen=excluded.last_seen
        """, (user.id, user.username or "", int(time.time())))

        conn.execute("""
        INSERT INTO user_settings VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO NOTHING
        """, (user.id, DEFAULT_MODEL, None))


def get_settings(user_id):
    with sqlite3.connect(DB_FILE) as conn:
        row = conn.execute(
            "SELECT default_model, active_conversation_id FROM user_settings WHERE user_id=?",
            (user_id,),
        ).fetchone()
    return {"model": row[0], "cid": row[1]}


def set_active_conversation(user_id, cid):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "UPDATE user_settings SET active_conversation_id=? WHERE user_id=?",
            (cid, user_id),
        )


def create_conversation(user_id, model):
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversations (user_id, model) VALUES (?, ?)",
            (user_id, model),
        )
        return cur.lastrowid


def append_message(cid, role, content):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT INTO messages VALUES (NULL, ?, ?, ?, ?)",
            (cid, role, content, int(time.time())),
        )


def get_messages(cid):
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY ts",
            (cid,),
        ).fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
def trim_messages(messages, max_chars=TOKEN_THRESHOLD):
    total = 0
    result = []
    for m in reversed(messages):
        total += len(m["content"])
        if total > max_chars:
            break
        result.append(m)
    return list(reversed(result))


# ------------------------------------------------------------------------------
# LM Studio helpers
# ------------------------------------------------------------------------------
def call_lm_chat(messages, model):
    payload = {
        "model": model,
        "messages": messages,
        **conversation_params,
    }

    r = requests.post(
        LM_CHAT,
        json=payload,
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def image_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def call_lm_vision(image_b64, prompt, model):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 500,
    }

    r = requests.post(LM_CHAT, json=payload, timeout=240)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------------------
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = update.message.text

    upsert_user(user)
    s = get_settings(user.id)

    cid = s["cid"]
    if not cid:
        cid = create_conversation(user.id, s["model"])
        set_active_conversation(user.id, cid)

    append_message(cid, "user", text)

    sent = await update.message.reply_text("🤖 Im writing a reply, please wait a moment...")

    msgs = trim_messages(get_messages(cid))

    try:
        data = await asyncio.to_thread(call_lm_chat, msgs, s["model"])
        answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        await sent.edit_text(f"Error LM Studio:\n{e}")
        return

    append_message(cid, "assistant", answer)

    try:
        await sent.edit_text(answer, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=answer,
            parse_mode=ParseMode.MARKDOWN,
        )


async def image_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    upsert_user(user)

    photo = update.message.photo[-1]
    file = await photo.get_file()

    bio = BytesIO()
    await file.download_to_memory(out=bio)

    image_b64 = image_to_b64(bio.getvalue())
    prompt = update.message.caption or "Describe image"

    s = get_settings(user.id)

    sent = await update.message.reply_text("🖼 Analyzing image...")

    try:
        data = await asyncio.to_thread(
            call_lm_vision, image_b64, prompt, s["model"]
        )
        answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        await sent.edit_text(f"Vision error:\n{e}")
        return

    try:
        await sent.edit_text(answer)
    except Exception:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=answer,
        )


async def error_handler(update, context):
    logger.error("Unhandled error", exc_info=context.error)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    init_db()

    request = HTTPXRequest(
        read_timeout=120,
        write_timeout=120,
        connect_timeout=30,
    )

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(request)
        .build()
    )

    app.add_error_handler(error_handler)

    app.add_handler(MessageHandler(filters.PHOTO, image_chat))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    app.run_polling()


if __name__ == "__main__":
    main()