"""
tts.py — WebSocket-based TTS for ChatBuddy.

Uses the Gemini Live API (BidiGenerateContent WebSocket) to convert text to
speech audio.  This is the same protocol used by V2's voice_message.py but
stripped to just the pure TTS clip generation — no voice-chat, no VC logic.

Required model: any Gemini native-audio model, e.g.
  gemini-2.5-flash-native-audio-preview-12-2025

Returns raw WAV bytes on success, None on failure.
"""

import asyncio
import base64
import json
import struct

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

# Gemini Live API WebSocket endpoint
LIVE_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"
)

# Seconds to wait for the entire TTS generation to complete
TTS_TIMEOUT_SECONDS = 45

# System instruction that keeps the model in pure read-aloud mode
TTS_SYSTEM_INSTRUCTION = (
    "You are a text-to-speech engine. "
    "Read the following text aloud exactly as written. "
    "Do NOT add any commentary, thoughts, analysis, or extra words. "
    "Just speak the text naturally."
)


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw 16-bit PCM in a WAV header so Discord can play it."""
    data_size   = len(pcm_data)
    byte_rate   = sample_rate * channels * sample_width
    block_align = channels * sample_width
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate,
        byte_rate, block_align, sample_width * 8,
        b"data", data_size,
    )
    return header + pcm_data


async def _ws_tts(api_key: str, endpoint: str, voice: str, text: str) -> bytes | None:
    """Inner WebSocket TTS — must be called inside asyncio.wait_for."""
    ws_url = f"{LIVE_WS_URL}?key={api_key}"

    try:
        async with websockets.connect(ws_url, close_timeout=5) as ws:

            # --- Setup message ---
            setup_msg = {
                "setup": {
                    "model": f"models/{endpoint}",
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {
                                    "voiceName": voice
                                }
                            }
                        },
                    },
                    "systemInstruction": {
                        "parts": [{"text": TTS_SYSTEM_INSTRUCTION}]
                    },
                }
            }
            await ws.send(json.dumps(setup_msg))

            # Wait for setupComplete
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            setup_data = json.loads(raw)
            if "setupComplete" not in setup_data:
                print(f"[ChatBuddy] TTS WebSocket: unexpected setup response: {setup_data}")
                return None

            # --- Send the text turn ---
            content_msg = {
                "clientContent": {
                    "turns": [
                        {
                            "role": "user",
                            "parts": [{"text": f"Read this aloud: {text}"}],
                        }
                    ],
                    "turnComplete": True,
                }
            }
            await ws.send(json.dumps(content_msg))

            # --- Collect audio chunks ---
            audio_chunks: list[bytes] = []
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                except asyncio.TimeoutError:
                    print("[ChatBuddy] TTS WebSocket: recv timed out waiting for audio.")
                    break

                msg = json.loads(raw)
                server_content = msg.get("serverContent")
                if not server_content:
                    continue

                parts = server_content.get("modelTurn", {}).get("parts", [])
                for part in parts:
                    inline = part.get("inlineData", {})
                    if inline.get("mimeType", "").startswith("audio/"):
                        raw_b64 = inline.get("data", "")
                        if raw_b64:
                            audio_chunks.append(base64.b64decode(raw_b64))

                if server_content.get("turnComplete"):
                    break

            if not audio_chunks:
                print("[ChatBuddy] TTS WebSocket: turnComplete received but no audio parts collected.")
                return None

            pcm = b"".join(audio_chunks)
            return _pcm_to_wav(pcm)

    except Exception as e:
        print(f"[ChatBuddy] TTS WebSocket error: {e}")
        return None


async def generate_tts(api_key: str, endpoint: str, voice: str, text: str) -> bytes | None:
    """
    Convert *text* to a WAV audio clip using the Gemini Live API WebSocket.

    Parameters
    ----------
    api_key  : Gemini API key
    endpoint : Native-audio model ID, e.g. 'gemini-2.5-flash-native-audio-preview-12-2025'
    voice    : Prebuilt voice name, e.g. 'Aoede', 'Puck', 'Charon'
    text     : The text to speak aloud

    Returns
    -------
    bytes | None
        WAV file bytes on success, None on any failure (errors are printed to console).
    """
    if not _WS_AVAILABLE:
        print("[ChatBuddy] websockets package not installed — cannot generate TTS. Run: pip install websockets")
        return None

    try:
        return await asyncio.wait_for(
            _ws_tts(api_key, endpoint, voice, text),
            timeout=TTS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print(f"[ChatBuddy] TTS generation timed out after {TTS_TIMEOUT_SECONDS}s.")
        return None
    except Exception as e:
        print(f"[ChatBuddy] TTS generation unexpected error: {e}")
        return None
