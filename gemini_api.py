"""
gemini_api.py — Async Gemini API client for ChatBuddy.

Text model modes (model_mode):
  default  — Standard Gemini inference with systemInstruction support.
  gemma    — Gemma-compatible: system prompt injected into user content.

Audio clip mode (audio_enabled = True/False — fully independent of model_mode):
  When enabled, every text response is also converted to speech via the
  Gemini Live API WebSocket (tts.py).
"""

import aiohttp

from tts import generate_tts

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# ── friendly error messages ────────────────────────────────────────────────────
MSG_NO_KEY = (
    "⚠️ No API key has been configured yet. "
    "An administrator needs to run `/set-api-key` before I can respond."
)
MSG_RATE_LIMIT    = "I'm sorry, I'm out of API juice right now — please try again in a moment."
MSG_SAFETY_BLOCK  = (
    "⚠️ My response was blocked by the safety filter. "
    "Try rephrasing your message."
)
MSG_GENERIC_ERROR = "⚠️ Something went wrong while generating a response. Please try again later."


# ── helpers ───────────────────────────────────────────────────────────────────

def build_system_prompt(config: dict, *, include_word_game: bool = True) -> str:
    """
    Assemble the full system prompt from config.

    Hierarchy:
        Main system prompt
        + Dynamic prompt (if enabled)
        + Word game prompt with {secret-word} replaced (if enabled AND include_word_game)
    """
    parts = [config.get("system_prompt", "")]

    if config.get("dynamic_prompt_enabled") and config.get("dynamic_prompt", ""):
        parts.append(config["dynamic_prompt"])

    if include_word_game and config.get("word_game_enabled") and config.get("word_game_prompt", ""):
        secret = config.get("secret_word", "")
        game_prompt = config["word_game_prompt"].replace("{secret-word}", secret)
        parts.append(game_prompt)

    return "\n\n".join(p for p in parts if p)


def _build_user_text(
    prompt: str,
    context: str,
    system_prompt: str,
    gemma_mode: bool,
    speaker_name: str = "",
    speaker_id: str = "",
) -> str:
    """Assemble the full user-content string sent to the text model."""
    parts = []
    if gemma_mode and system_prompt:
        parts.append(
            "[BEHAVIORAL INSTRUCTIONS — follow these at all times]\n"
            f"{system_prompt}\n"
            "[END BEHAVIORAL INSTRUCTIONS]\n"
        )
    if context:
        parts.append(f"[CHAT CONTEXT]\n{context}\n[END CHAT CONTEXT]\n")
    if speaker_name:
        parts.append(f"[CURRENT SPEAKER]\n{speaker_name} (ID:{speaker_id})")
    parts.append(f"[USER MESSAGE]\n{prompt}")
    return "\n".join(parts)


def _extract_text(data: dict) -> str | None:
    """Pull the first text part out of a generateContent response."""
    candidates = data.get("candidates", [])
    if not candidates:
        return None
    candidate = candidates[0]
    if candidate.get("finishReason") == "SAFETY":
        return None
    for part in candidate.get("content", {}).get("parts", []):
        if "text" in part:
            return part["text"]
    return None


# ── main entry point ──────────────────────────────────────────────────────────

async def generate(
    prompt: str,
    context: str,
    config: dict,
    revival_system_instruct: str = "",
    speaker_name: str = "",
    speaker_id: str = "",
    system_prompt_override: str | None = None,
) -> tuple[str, bytes | None]:
    """
    Call the Gemini API and return (text_reply, wav_bytes_or_None).

    system_prompt_override: if provided, used instead of the auto-assembled
    system prompt.  Used by the word-game hidden turn.
    """
    api_key = config.get("api_key")
    if not api_key:
        return MSG_NO_KEY, None

    model_mode    = config.get("model_mode", "default")
    gemma_mode    = model_mode == "gemma"
    audio_enabled = config.get("audio_enabled", False)

    # Dual endpoints — pick based on mode
    if gemma_mode:
        text_endpoint = config.get("model_endpoint_gemma", "")
        if not text_endpoint:
            # Fallback to legacy key for migration
            text_endpoint = config.get("model_endpoint", "gemini-2.0-flash")
    else:
        text_endpoint = config.get("model_endpoint_gemini", "gemini-2.0-flash")
        if not text_endpoint:
            text_endpoint = config.get("model_endpoint", "gemini-2.0-flash")

    temperature = config.get("temperature", 0.7)

    # Build effective system prompt
    if system_prompt_override is not None:
        system_prompt = system_prompt_override
    else:
        # Normal assembly: include_word_game=True unless revival
        include_game = not bool(revival_system_instruct)
        system_prompt = build_system_prompt(config, include_word_game=include_game)

    if revival_system_instruct:
        system_prompt = (system_prompt + "\n\n" + revival_system_instruct).strip()

    # ── Step 1: text inference via REST generateContent ────────────────────
    user_text = _build_user_text(prompt, context, system_prompt, gemma_mode, speaker_name, speaker_id)

    text_body: dict = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }

    # Only non-gemma modes use the top-level systemInstruction field
    if not gemma_mode and system_prompt:
        text_body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    text_url = f"{API_BASE}/{text_endpoint}:generateContent?key={api_key}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(text_url, json=text_body) as resp:
                status = resp.status
                data   = await resp.json()

        if status == 429:
            return MSG_RATE_LIMIT, None

        if status != 200:
            err = str(data)
            if "SAFETY" in err.upper() or "blocked" in err.lower():
                return MSG_SAFETY_BLOCK, None
            print(f"[ChatBuddy] Text API error {status}: {data}")
            return MSG_GENERIC_ERROR, None

        if data.get("promptFeedback", {}).get("blockReason"):
            return MSG_SAFETY_BLOCK, None

        text_reply = _extract_text(data)
        if text_reply is None:
            return MSG_SAFETY_BLOCK, None

    except aiohttp.ClientError as e:
        print(f"[ChatBuddy] HTTP error during text inference: {e}")
        return MSG_GENERIC_ERROR, None
    except Exception as e:
        print(f"[ChatBuddy] Unexpected error during text inference: {e}")
        return MSG_GENERIC_ERROR, None

    # ── Step 2: TTS via WebSocket Live API (only when audio is enabled) ────
    if not audio_enabled:
        return text_reply, None

    tts_endpoint = config.get("audio_endpoint", "").strip()
    if not tts_endpoint:
        print("[ChatBuddy] audio_enabled=True but audio_endpoint is empty — skipping TTS.")
        return text_reply, None

    voice = config.get("audio_settings", {}).get("voice", "Aoede")

    wav_bytes = await generate_tts(api_key, tts_endpoint, voice, text_reply)

    if wav_bytes is None:
        return text_reply, None

    return text_reply, wav_bytes
