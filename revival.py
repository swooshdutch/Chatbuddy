"""
revival.py — Chat revival background task for ChatBuddy.
Periodically sends a conversation-starting message in a configured channel.
During the active window it also auto-replies to new messages without
needing a mention/reply.
"""

import asyncio
import io
import discord
from discord.ext import tasks

from config import save_config
from gemini_api import generate
from utils import format_context, chunk_message, resolve_custom_emoji, extract_thoughts


class RevivalManager:
    """Manages the chat-revival background loop."""

    def __init__(self, bot, config: dict):
        self.bot = bot
        self.config = config
        self._task: tasks.Loop | None = None
        self._revival_active: bool = False  # True while the active window is open

    def start(self):
        """Start (or restart) the revival loop based on current config."""
        self.stop()
        revival = self.config.get("chat_revival")
        if not revival:
            return

        interval = revival.get("interval_minutes", 30)

        @tasks.loop(minutes=interval)
        async def _revival_loop():
            await self._tick()

        @_revival_loop.before_loop
        async def _before():
            await self.bot.wait_until_ready()

        self._task = _revival_loop
        self._task.start()

    def stop(self):
        """Cancel the running loop if any."""
        if self._task and self._task.is_running():
            self._task.cancel()
        self._task = None

    # ------------------------------------------------------------------
    # Core tick
    # ------------------------------------------------------------------

    async def _tick(self):
        """Execute one revival cycle."""
        revival = self.config.get("chat_revival")
        if not revival:
            return

        # If revival is disabled, do nothing
        if not revival.get("enabled", True):
            return

        channel_id = int(revival["channel_id"])
        system_instruct = revival.get("system_instruct", "")

        channel = self.bot.get_channel(channel_id)
        if channel is None:
            return

        # Spam guard — don't fire if the last message is already from the bot
        last_msg = None
        async for msg in channel.history(limit=1):
            last_msg = msg
        if last_msg is not None and last_msg.author == self.bot.user:
            return

        channel_key = str(channel_id)
        allowed = self.config.get("allowed_channels", {})
        was_blocked = not allowed.get(channel_key, False)

        # NOTE: we intentionally do NOT enable the channel in allowed_channels.
        # The active-window polling loop sends messages directly and does not
        # consult allowed_channels.  Keeping the channel blocked ensures that
        # on_message() ignores mentions/replies during revival — only the
        # timed revival messages fire.  If the channel was already allowed,
        # on_message() continues to respond normally alongside revival.

        active_minutes = self.config.get("cr_active_minutes", 5)

        # Gather recent context from the channel
        history_limit = self.config.get("chat_history_limit", 30)
        history_messages = []
        async for msg in channel.history(limit=history_limit):
            history_messages.append(msg)
        history_messages.reverse()

        context = format_context(history_messages, ce_enabled=True)

        # ── SoC context injection ─────────────────────────────────────
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
                # Apply [ce] cutoff to SoC context
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

        # Generate the revival message
        response_text, audio_bytes, soul_logs = await generate(
            prompt="Start a new conversation to revive the chat.",
            context=context,
            config=self.config,
            revival_system_instruct=system_instruct,
        )

        if audio_bytes:
            audio_file = discord.File(fp=io.BytesIO(audio_bytes), filename="revival.wav")
            await channel.send(file=audio_file)

        if response_text:
            # ── SoC thought extraction ────────────────────────────────
            soc_enabled = self.config.get("soc_enabled", False)
            clean_text, thoughts_text = extract_thoughts(response_text)
            if thoughts_text and soc_enabled and soc_channel_id:
                thought_ch = self.bot.get_channel(int(soc_channel_id))
                if thought_ch is not None:
                    for c in chunk_message(thoughts_text):
                        await thought_ch.send(c)
            response_text = clean_text

            response_text = resolve_custom_emoji(response_text, channel.guild)
            footer = f"\n-# :loudspeaker: chat reviver active for : {active_minutes}m 0s"
            chunks = chunk_message(response_text)
            # Append the footer to the last chunk
            chunks[-1] += footer
            for chunk in chunks:
                await channel.send(chunk)

        # Log soul changes
        if soul_logs and self.config.get("soul_channel_enabled"):
            ch_id = self.config.get("soul_channel_id")
            if ch_id:
                soul_ch = self.bot.get_channel(int(ch_id))
                if soul_ch:
                    joined_logs = "\n".join(soul_logs)
                    for log_chunk in chunk_message(joined_logs, limit=1900):
                        await soul_ch.send(f"**🧠 Soul Updates:**\n{log_chunk}")

        # Start the auto-reply active window, then relock when done
        self.bot.loop.create_task(
            self._active_window(channel, channel_key, was_blocked, system_instruct)
        )

    # ------------------------------------------------------------------
    # Active window — auto-reply polling loop
    # ------------------------------------------------------------------

    async def _active_window(
        self,
        channel: discord.TextChannel,
        channel_key: str,
        was_blocked: bool,
        system_instruct: str,
    ):
        """
        Run for *cr_active_minutes* minutes, polling every *cr_check_seconds*
        seconds.  If the most recent message in the channel is NOT from the
        bot, treat it as input and generate a response.
        """
        active_minutes = self.config.get("cr_active_minutes", 5)
        check_seconds = self.config.get("cr_check_seconds", 30)

        total_seconds = active_minutes * 60
        elapsed = 0

        self._revival_active = True

        while elapsed < total_seconds:
            await asyncio.sleep(check_seconds)
            elapsed += check_seconds

            # Re-check that revival is still enabled (admin might disable mid-window)
            revival = self.config.get("chat_revival")
            if not revival or not revival.get("enabled", True):
                break

            # Calculate remaining time for the footer (e.g. "4m 26s")
            remaining_total = max(total_seconds - elapsed, 0)
            remaining_m = int(remaining_total // 60)
            remaining_s = int(remaining_total % 60)

            try:
                # Fetch the single most recent message
                last_msg = None
                async for msg in channel.history(limit=1):
                    last_msg = msg

                if last_msg is None:
                    continue

                # If the last message is already from the bot, nothing to do
                if last_msg.author == self.bot.user:
                    continue

                # Gather context and reply
                history_limit = self.config.get("chat_history_limit", 30)
                history_messages = []
                async for msg in channel.history(limit=history_limit):
                    history_messages.append(msg)
                history_messages.reverse()

                context = format_context(history_messages, ce_enabled=True)

                # ── SoC context injection ─────────────────────────────
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
                        # Apply [ce] cutoff to SoC context
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

                user_text = last_msg.clean_content
                if not user_text:
                    user_text = "(empty message)"

                response_text, audio_bytes, soul_logs = await generate(
                    prompt=user_text,
                    context=context,
                    config=self.config,
                    revival_system_instruct=system_instruct,
                )

                if audio_bytes:
                    audio_file = discord.File(fp=io.BytesIO(audio_bytes), filename="revival_reply.wav")
                    await channel.send(file=audio_file)

                if response_text:
                    # ── SoC thought extraction ────────────────────────
                    soc_enabled = self.config.get("soc_enabled", False)
                    clean_text, thoughts_text = extract_thoughts(response_text)
                    if thoughts_text and soc_enabled and soc_channel_id:
                        thought_ch = self.bot.get_channel(int(soc_channel_id))
                        if thought_ch is not None:
                            for c in chunk_message(thoughts_text):
                                await thought_ch.send(c)
                    response_text = clean_text

                    response_text = resolve_custom_emoji(response_text, channel.guild)
                    footer = f"\n-# :loudspeaker: chat reviver active for : {remaining_m}m {remaining_s}s"
                    chunks = chunk_message(response_text)
                    chunks[-1] += footer
                    for chunk in chunks:
                        await channel.send(chunk)

                # Log soul changes
                if soul_logs and self.config.get("soul_channel_enabled"):
                    ch_id = self.config.get("soul_channel_id")
                    if ch_id:
                        soul_ch = self.bot.get_channel(int(ch_id))
                        if soul_ch:
                            joined_logs = "\n".join(soul_logs)
                            for log_chunk in chunk_message(joined_logs, limit=1900):
                                await soul_ch.send(f"**🧠 Soul Updates:**\n{log_chunk}")

            except Exception as e:
                print(f"[ChatBuddy] Revival auto-reply error: {e}")

        self._revival_active = False

        # Send a leave/goodbye message regardless of channel permission state.
        # No channel state needs restoring — we never modified allowed_channels.
        leave_msg = self.config.get(
            "cr_leave_message", "Ok nice chatting to you all, see you later"
        )
        await channel.send(leave_msg)
