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
    # Dual model endpoints — one for each mode
    "model_endpoint_gemini": "gemini-2.0-flash",
    "model_endpoint_gemma": "",
    # Custom (non-Google) model support
    "api_key_custom": "",
    "model_endpoint_custom": "",
    "temperature": 0.7,
    "chat_history_limit": 30,
    # Text model mode: "gemini" | "gemma" | "custom"
    "model_mode": "gemini",
    # Audio clip mode
    "audio_enabled": False,
    "audio_endpoint": "",
    "audio_settings": {"voice": "Aoede"},
    "ce_channels": {},        # {str(channel_id): bool}
    "allowed_channels": {},   # {str(channel_id): bool}
    "chat_revival": None,
    "cr_leave_message": "Ok nice chatting to you all, see you later",
    "cr_active_minutes": 5,
    "cr_check_seconds": 30,
    # Stream of Consciousness (SoC)
    "soc_channel_id": None,
    "soc_enabled": False,
    "soc_context_enabled": False,
    "soc_context_count": 10,
    # Dynamic system prompt — appended after main prompt when enabled
    "dynamic_prompt": "",
    "dynamic_prompt_enabled": False,
    # Word game
    "word_game_prompt": "",         # prompt text with {secret-word} placeholder
    "word_game_enabled": False,
    "word_game_selector_prompt": "",  # system prompt for hidden word-selection turn
    "secret_word": "",
    "secret_word_allowed_roles": [],  # role IDs allowed to use /set-secret-word
    # Soul (dynamic auto-updating memory)
    "soul_enabled": False,
    "soul_limit": 2000,
    "soul_error_turn": "",
    "soul_channel_enabled": False,
    "soul_channel_id": "",  # stores 1-turn error message if update fails
    # Auto-chat mode
    "auto_chat_enabled": False,
    "auto_chat_channel_id": None,
    "auto_chat_interval": 30,        # seconds between checks
    "auto_chat_idle_minutes": 10,    # idle timeout
    "auto_chat_idle_message": "Going afk, ping me if you need me",
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
