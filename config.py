"""
config.py — Persistent configuration manager for ChatBuddy.
Reads/writes a config.json file so settings survive Discloud restarts.
"""

import json
import os

CONFIG_FILE = "config.json"

DEFAULTS = {
    "api_key": None,
    "system_prompt": "You are a helpful Discord chatbot called ChatBuddy.",
    "model_endpoint": "gemini-2.0-flash",
    "temperature": 0.7,
    "chat_history_limit": 30,
    # Text model mode: "default" (systemInstruction supported) or "gemma"
    # (system prompt injected into user content instead).
    # Audio is a SEPARATE flag — see audio_enabled below.
    "model_mode": "default",
    # Audio clip mode — applies to all responses server-wide when True.
    # After generating a text reply the bot also calls the TTS endpoint
    # and sends a .wav attachment followed by the text transcript.
    "audio_enabled": False,
    # TTS model to use. Set via /set-audio-endpoint. NO hard-coded fallback —
    # must be explicitly configured before audio mode is useful.
    "audio_endpoint": "",
    "audio_settings": {"voice": "Aoede"},
    "ce_channels": {},        # {str(channel_id): bool} — per-channel [ce] toggle (default True)
    "allowed_channels": {},   # {str(channel_id): bool} — channel whitelist (default False)
    "chat_revival": None,     # {"channel_id": str, "interval_minutes": int, "system_instruct": str, "enabled": bool}
    "cr_leave_message": "Ok nice chatting to you all, see you later",
    "cr_active_minutes": 5,
    "cr_check_seconds": 30,
    # Stream of Consciousness (SoC) — thought extraction & cross-channel context
    "soc_channel_id": None,       # str(channel_id) — where thoughts are posted
    "soc_enabled": False,         # master toggle for thought extraction
    "soc_context_enabled": False, # read thoughts back as context
    "soc_context_count": 10,      # how many thought messages to read
}


def load_config() -> dict:
    """Load config from disk, falling back to defaults for any missing keys."""
    config = dict(DEFAULTS)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            config.update(stored)
        except (json.JSONDecodeError, OSError):
            pass  # Corrupted file — use defaults
    return config


def save_config(config: dict) -> None:
    """Atomically write the config dict to disk."""
    tmp_path = CONFIG_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, CONFIG_FILE)
