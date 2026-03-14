"""
auto_chat.py — Auto-chat background task for ChatBuddy.
Periodically checks a channel and auto-replies to new messages without
requiring mentions or replies.  Goes idle after a configurable timeout.
"""

import asyncio
import io
import discord
from discord.ext import tasks

from config import save_config
from gemini_api import generate, build_system_prompt
from utils import format_context, chunk_message, resolve_custom_emoji, extract_thoughts


class AutoChatManager:
    """Manages the auto-chat background loop."""

    def __init__(self, bot, config: dict):
        self.bot = bot
        self.config = config
        self._task: tasks.Loop | None = None
        self._idle: bool = False
        self._seconds_since_last_reply: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start (or restart) the auto-chat loop based on current config."""
        self.stop()
        if not self.config.get("auto_chat_enabled"):
            return

        channel_id = self.config.get("auto_chat_channel_id")
        if not channel_id:
            return

        interval = self.config.get("auto_chat_interval", 30)
        self._idle = False
        self._seconds_since_last_reply = 0

        @tasks.loop(seconds=interval)
        async def _auto_chat_loop():
            await self._tick()

        @_auto_chat_loop.before_loop
        async def _before():
            await self.bot.wait_until_ready()

        self._task = _auto_chat_loop
        self._task.start()

    def stop(self):
        """Cancel the running loop if any.  Does NOT reset idle state."""
        if self._task and self._task.is_running():
            self._task.cancel()
        self._task = None

    @property
    def is_idle(self) -> bool:
        return self._idle

    def reactivate(self):
        """Wake up from idle mode (called from on_message when mentioned)."""
        if self._idle:
            print("[AutoChat] Reactivating from idle mode.")
            self._idle = False
            self._seconds_since_last_reply = 0
            self.start()

    # ------------------------------------------------------------------
    # Core tick
    # ------------------------------------------------------------------

    async def _tick(self):
        """Execute one auto-chat cycle."""
        if not self.config.get("auto_chat_enabled"):
            return
        if self._idle:
            return

        channel_id = self.config.get("auto_chat_channel_id")
        if not channel_id:
            return

        channel = self.bot.get_channel(int(channel_id))
        if channel is None:
            return

        interval = self.config.get("auto_chat_interval", 30)
        idle_minutes = self.config.get("auto_chat_idle_minutes", 10)

        # Fetch the most recent message
        last_msg = None
        async for msg in channel.history(limit=1):
            last_msg = msg

        if last_msg is None:
            return

        # If the last message is from the bot, don't reply
        if last_msg.author == self.bot.user:
            self._seconds_since_last_reply += interval
            # Check idle timeout
            if self._seconds_since_last_reply >= (idle_minutes * 60):
                print(f"[AutoChat] Idle timeout reached ({idle_minutes}m). Entering idle mode.")
                idle_msg = self.config.get("auto_chat_idle_message", "Going afk, ping me if you need me")
                if idle_msg:
                    await channel.send(idle_msg)
                self._idle = True
                # Cancel the loop but keep _idle=True so reactivate() can detect it
                if self._task and self._task.is_running():
                    self._task.cancel()
                self._task = None
            return

        # There's a new user message — reset idle counter and respond
        self._seconds_since_last_reply = 0

        try:
            async with channel.typing():
                # Gather chat history
                history_limit = self.config.get("chat_history_limit", 30)
                history_messages = []
                async for msg in channel.history(limit=history_limit):
                    history_messages.append(msg)
                history_messages.reverse()

                channel_key = str(channel_id)
                ce_channels = self.config.get("ce_channels", {})
                ce_enabled = ce_channels.get(channel_key, True)
                context = format_context(history_messages, ce_enabled=ce_enabled)

                # SoC context injection
                soc_context_enabled = self.config.get("soc_context_enabled", False)
                soc_channel_id = self.config.get("soc_channel_id")
                if soc_context_enabled and soc_channel_id:
                    soc_count = self.config.get("soc_context_count", 10)
                    soc_ch = self.bot.get_channel(int(soc_channel_id))
                    if soc_ch is not None:
                        soc_msgs = []
                        async for m in soc_ch.history(limit=soc_count):
                            soc_msgs.append(m)
                        soc_msgs.reverse()
                        # Apply [ce] to SoC context
                        filtered_soc = []
                        ce_idx = None
                        for i, m in enumerate(soc_msgs):
                            if m.content.strip().lower() == "[ce]":
                                ce_idx = i
                        if ce_idx is not None:
                            soc_msgs = soc_msgs[ce_idx + 1:]
                        if soc_msgs:
                            soc_lines = []
                            for m in soc_msgs:
                                ts = m.created_at.strftime("%Y-%m-%d %H:%M:%S")
                                soc_lines.append(f"[{ts}] {m.content}")
                            context += (
                                "\n[YOUR PREVIOUS THOUGHTS]\n"
                                + "\n".join(soc_lines)
                                + "\n[END YOUR PREVIOUS THOUGHTS]\n"
                            )

                response_text, audio_bytes, soul_logs = await generate(
                    prompt=last_msg.clean_content or "(empty message)",
                    context=context,
                    config=self.config,
                    speaker_name=last_msg.author.display_name,
                    speaker_id=str(last_msg.author.id),
                )

                # SoC thought extraction
                soc_enabled = self.config.get("soc_enabled", False)
                clean_text, thoughts_text = extract_thoughts(response_text)
                if thoughts_text and soc_enabled and soc_channel_id:
                    thought_ch = self.bot.get_channel(int(soc_channel_id))
                    if thought_ch is not None:
                        for c in chunk_message(thoughts_text):
                            await thought_ch.send(c)
                response_text = clean_text

                response_text = resolve_custom_emoji(response_text, channel.guild)

                if audio_bytes:
                    audio_file = discord.File(fp=io.BytesIO(audio_bytes), filename="auto_chat.wav")
                    await channel.send(file=audio_file)

                if response_text:
                    for chunk in chunk_message(response_text):
                        await channel.send(chunk)

                # Send soul logs to configured channel if present
                if soul_logs and self.config.get("soul_channel_enabled"):
                    ch_id = self.config.get("soul_channel_id")
                    if ch_id:
                        soul_ch = self.bot.get_channel(int(ch_id))
                        if soul_ch:
                            joined_logs = "\n".join(soul_logs)
                            for log_chunk in chunk_message(joined_logs, limit=1900):
                                await soul_ch.send(f"**🧠 Soul Updates:**\n{log_chunk}")

        except Exception as e:
            print(f"[AutoChat] Error during auto-reply: {e}")
