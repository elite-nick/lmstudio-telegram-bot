import logging
import sqlite3
import requests
import time
import os
import asyncio
import base64
import random
from io import BytesIO
from dotenv import load_dotenv
from collections import deque

from telegram import Update, User
from telegram.constants import ParseMode
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
    raise RuntimeError("BOT_TOKEN is not set")

# ------------------------------------------------------------------------------
# LM Studio
# ------------------------------------------------------------------------------
LM_BASE = "http://localhost:1234/v1"
LM_CHAT = f"{LM_BASE}/chat/completions"

DEFAULT_MODEL = "llava"
TOKEN_THRESHOLD = 12000

conversation_params = {
    "max_tokens": 700,
    "temperature": 0.4,
    "top_p": 1.0,
}

# ------------------------------------------------------------------------------
# Runtime state
# ------------------------------------------------------------------------------
USER_QUEUES = {}
USER_PROCESSING = set()

UNSUPPORTED_STICKER_MESSAGES = [
    "😅 This sticker is too lively for me — I only understand regular images or static stickers...",
    "🤖 Oops, I can’t recognize animated stickers yet...",
    "🎬 Looks cool! But I can’t read animated or video stickers."
]

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


def clear_user_history(user_id):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM conversations WHERE user_id=?", (user_id,))
        ids = [r[0] for r in c.fetchall()]

        for cid in ids:
            conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        conn.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))
        conn.execute(
            "UPDATE user_settings SET active_conversation_id=NULL WHERE user_id=?",
            (user_id,),
        )

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


def image_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()

# ------------------------------------------------------------------------------
# LM Studio
# ------------------------------------------------------------------------------
def call_lm_chat(messages, model):
    payload = {
        "model": model,
        "messages": messages,
        **conversation_params,
    }
    r = requests.post(LM_CHAT, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


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
        "max_tokens": 700,
    }
    r = requests.post(LM_CHAT, json=payload, timeout=240)
    r.raise_for_status()
    return r.json()

# ------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I’m LM Studio Telegram Bot!\n\n"
        "Here’s what I can do:\n"
        "• Chat via text (sometimes with emojis)\n"
        "• Analyze your images and stickers\n\n"
        "Commands:\n"
        "/status — current status (works only while generating a reply)\n"
        "/clear — clear our conversation history\n\n"
    )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid in USER_PROCESSING:
        q = len(USER_QUEUES.get(uid, []))
        await update.message.reply_text(
            f"🤖 I’m thinking right now.\n"
            f"📨 Messages in queue: {q}"
        )
    else:
        await update.message.reply_text("💤 I’m free and ready to reply.")


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    clear_user_history(update.effective_user.id)
    await update.message.reply_text("🧹 History cleared.")

# ------------------------------------------------------------------------------
# Text chat with queue + reply
# ------------------------------------------------------------------------------
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    uid = user.id

    upsert_user(user)

    if uid not in USER_QUEUES:
        USER_QUEUES[uid] = deque()

    USER_QUEUES[uid].append({
        "text": update.message.text,
        "message_id": update.message.message_id,
        "chat_id": update.effective_chat.id
    })

    if uid in USER_PROCESSING:
        await update.message.reply_text(
            "⏳ I’m already answering a previous question.\n"
            "I’ll respond to this one right after."
        )
        return

    asyncio.create_task(process_user_queue(uid, context))


async def process_user_queue(user_id, context: ContextTypes.DEFAULT_TYPE):
    USER_PROCESSING.add(user_id)

    try:
        while USER_QUEUES[user_id]:
            item = USER_QUEUES[user_id].popleft()

            text = item["text"]
            reply_to = item["message_id"]
            chat_id = item["chat_id"]

            s = get_settings(user_id)
            cid = s["cid"]
            if not cid:
                cid = create_conversation(user_id, s["model"])
                set_active_conversation(user_id, cid)

            append_message(cid, "user", text)

            sent = await context.bot.send_message(
                chat_id=chat_id,
                text="🤖 Thinking...",
                reply_to_message_id=reply_to
            )

            msgs = trim_messages(get_messages(cid))

            data = await asyncio.to_thread(
                call_lm_chat, msgs, s["model"]
            )
            answer = data["choices"][0]["message"]["content"]

            append_message(cid, "assistant", answer)

            await sent.edit_text(answer, parse_mode=ParseMode.MARKDOWN)

    finally:
        USER_PROCESSING.discard(user_id)

# ------------------------------------------------------------------------------
# Stickers (reply)
# ------------------------------------------------------------------------------
async def sticker_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sticker = update.message.sticker

    if sticker.is_animated or sticker.is_video:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=random.choice(UNSUPPORTED_STICKER_MESSAGES),
            reply_to_message_id=update.message.message_id
        )
        return

    file = await sticker.get_file()
    bio = BytesIO()
    await file.download_to_memory(out=bio)

    image_b64 = image_to_b64(bio.getvalue())
    prompt = (
        "I sent a sticker as a reply to your message. You can react to it. "
        "Respond while taking into account the context of the attached sticker image, "
        "which may contain anything drawn or written. You may also describe what is "
        "shown in the sticker image."
    )

    s = get_settings(update.effective_user.id)

    sent = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="😀 Analyzing sticker...",
        reply_to_message_id=update.message.message_id
    )

    data = await asyncio.to_thread(
        call_lm_vision, image_b64, prompt, s["model"]
    )
    answer = data["choices"][0]["message"]["content"]

    await sent.edit_text(answer)

# ------------------------------------------------------------------------------
# Images (reply)
# ------------------------------------------------------------------------------
async def image_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()

    bio = BytesIO()
    await file.download_to_memory(out=bio)

    image_b64 = image_to_b64(bio.getvalue())
    prompt = update.message.caption or "Describe the image"

    s = get_settings(update.effective_user.id)

    sent = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="🖼 Analyzing image...",
        reply_to_message_id=update.message.message_id
    )

    data = await asyncio.to_thread(
        call_lm_vision, image_b64, prompt, s["model"]
    )
    answer = data["choices"][0]["message"]["content"]

    await sent.edit_text(answer)

# ------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------
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

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))

    app.add_handler(MessageHandler(filters.Sticker.ALL, sticker_chat))
    app.add_handler(MessageHandler(filters.PHOTO, image_chat))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    app.run_polling()


if __name__ == "__main__":
    main()
