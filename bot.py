"""
bot.py — Main entry point for ChatBuddy, a Discord bot powered by Gemini.
"""

import os
import io
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from config import load_config, save_config
from gemini_api import generate
from utils import strip_mention, chunk_message, format_context, resolve_custom_emoji, extract_thoughts
from revival import RevivalManager

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "DISCORD_TOKEN is not set. "
        "Copy .env.template to .env and paste your bot token."
    )

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Runtime config (loaded from disk on startup)
bot_config: dict = {}

# Revival manager (initialised in on_ready)
revival_manager: RevivalManager | None = None

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@bot.event
async def on_ready():
    global bot_config, revival_manager
    bot_config = load_config()

    revival_manager = RevivalManager(bot, bot_config)
    revival_manager.start()

    try:
        synced = await bot.tree.sync()
        print(f"[ChatBuddy] Online as {bot.user} — synced {len(synced)} command(s)")
    except Exception as e:
        print(f"[ChatBuddy] Failed to sync commands: {e}")


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    # Channel whitelist gate
    channel_key = str(message.channel.id)
    allowed = bot_config.get("allowed_channels", {})
    if not allowed.get(channel_key, False):
        return

    is_mentioned = bot.user in message.mentions
    is_reply_to_bot = (
        message.reference is not None
        and message.reference.resolved is not None
        and isinstance(message.reference.resolved, discord.Message)
        and message.reference.resolved.author == bot.user
    )

    if not is_mentioned and not is_reply_to_bot:
        return

    async with message.channel.typing():
        user_text = strip_mention(message.content, bot.user.id)
        if not user_text:
            user_text = "(empty message)"

        history_limit = bot_config.get("chat_history_limit", 30)
        history_messages = []
        async for msg in message.channel.history(limit=history_limit, before=message):
            history_messages.append(msg)
        history_messages.reverse()

        ce_channels = bot_config.get("ce_channels", {})
        ce_enabled = ce_channels.get(channel_key, True)
        context = format_context(history_messages, ce_enabled=ce_enabled)

        # ── SoC context injection (after chat history) ────────────────
        soc_context_enabled = bot_config.get("soc_context_enabled", False)
        soc_channel_id = bot_config.get("soc_channel_id")
        if soc_context_enabled and soc_channel_id:
            soc_count = bot_config.get("soc_context_count", 10)
            soc_channel = bot.get_channel(int(soc_channel_id))
            if soc_channel is not None:
                soc_messages = []
                async for msg in soc_channel.history(limit=soc_count):
                    soc_messages.append(msg)
                soc_messages.reverse()
                if soc_messages:
                    soc_lines = []
                    for msg in soc_messages:
                        ts = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
                        soc_lines.append(f"[{ts}] {msg.content}")
                    context += (
                        "\n[YOUR PREVIOUS THOUGHTS]\n"
                        + "\n".join(soc_lines)
                        + "\n[END YOUR PREVIOUS THOUGHTS]\n"
                    )

        response_text, audio_bytes = await generate(
            user_text, context, bot_config,
            speaker_name=message.author.display_name,
            speaker_id=str(message.author.id),
        )

        # ── SoC thought extraction ────────────────────────────────────
        soc_enabled = bot_config.get("soc_enabled", False)
        clean_text, thoughts_text = extract_thoughts(response_text)
        if thoughts_text and soc_enabled and soc_channel_id:
            thought_channel = bot.get_channel(int(soc_channel_id))
            if thought_channel is not None:
                for chunk in chunk_message(thoughts_text):
                    await thought_channel.send(chunk)
        # Use the cleaned text (thoughts stripped) for the user-facing reply
        response_text = clean_text

        # Resolve custom emoji shortcodes before sending
        response_text = resolve_custom_emoji(response_text, message.guild)

        if audio_bytes:
            # Post audio clip first, then the text transcript
            audio_file = discord.File(fp=io.BytesIO(audio_bytes), filename="chatbuddy_voice.wav")
            await message.reply(file=audio_file, mention_author=False)
            chunks = chunk_message(response_text)
            for chunk in chunks:
                await message.channel.send(chunk)
        else:
            # Text-only path
            chunks = chunk_message(response_text)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await message.reply(chunk, mention_author=False)
                else:
                    await message.channel.send(chunk)


# ---------------------------------------------------------------------------
# Slash commands — Core settings
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-api-key", description="Set the Gemini API key")
@app_commands.describe(key="Your Gemini API key")
@app_commands.default_permissions(administrator=True)
async def set_api_key(interaction: discord.Interaction, key: str):
    bot_config["api_key"] = key
    save_config(bot_config)
    await interaction.response.send_message("✅ API key has been set and saved.", ephemeral=True)


@bot.tree.command(name="set-chat-history", description="Set how many messages of context the bot receives")
@app_commands.describe(limit="Number of previous messages (default: 30)")
@app_commands.default_permissions(administrator=True)
async def set_chat_history(interaction: discord.Interaction, limit: int):
    if limit < 1:
        await interaction.response.send_message("⚠️ Limit must be at least 1.", ephemeral=True)
        return
    bot_config["chat_history_limit"] = limit
    save_config(bot_config)
    await interaction.response.send_message(
        f"✅ Chat history limit set to **{limit}** messages.", ephemeral=True
    )


@bot.tree.command(name="set-temp", description="Set the model temperature")
@app_commands.describe(temperature="Temperature value (e.g. 0.7)")
@app_commands.default_permissions(administrator=True)
async def set_temp(interaction: discord.Interaction, temperature: float):
    if temperature < 0.0 or temperature > 2.0:
        await interaction.response.send_message(
            "⚠️ Temperature must be between 0.0 and 2.0.", ephemeral=True
        )
        return
    bot_config["temperature"] = temperature
    save_config(bot_config)
    await interaction.response.send_message(f"✅ Temperature set to **{temperature}**.", ephemeral=True)


@bot.tree.command(name="set-api-endpoint", description="Set the target Gemini text model endpoint")
@app_commands.describe(endpoint="Model name (e.g. gemini-2.0-flash, gemma-3-27b-it)")
@app_commands.default_permissions(administrator=True)
async def set_api_endpoint(interaction: discord.Interaction, endpoint: str):
    bot_config["model_endpoint"] = endpoint
    save_config(bot_config)
    await interaction.response.send_message(f"✅ Text model endpoint set to **{endpoint}**.", ephemeral=True)


@bot.tree.command(name="set-sys-instruct", description="Set the system instruction / prompt")
@app_commands.describe(prompt="The system prompt text")
@app_commands.default_permissions(administrator=True)
async def set_sys_instruct(interaction: discord.Interaction, prompt: str):
    prompt = prompt.replace("\\n", "\n")
    bot_config["system_prompt"] = prompt
    save_config(bot_config)
    await interaction.response.send_message("✅ System prompt updated and saved.", ephemeral=True)


@bot.tree.command(name="show-sys-instruct", description="Display the current system prompt")
@app_commands.default_permissions(administrator=True)
async def show_sys_instruct(interaction: discord.Interaction):
    prompt = bot_config.get("system_prompt", "(not set)")
    await interaction.response.send_message(
        f"📝 **Current system prompt:**\n```\n{prompt}\n```", ephemeral=True
    )


# ---------------------------------------------------------------------------
# Slash commands — Text model mode
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-model-mode", description="Toggle between default and Gemma text model modes")
@app_commands.describe(mode="default = standard Gemini, gemma = Gemma-compatible injection")
@app_commands.choices(mode=[
    app_commands.Choice(name="default", value="default"),
    app_commands.Choice(name="gemma",   value="gemma"),
])
@app_commands.default_permissions(administrator=True)
async def set_model_mode(interaction: discord.Interaction, mode: app_commands.Choice[str]):
    bot_config["model_mode"] = mode.value
    save_config(bot_config)
    info = ""
    if mode.value == "gemma":
        info = (
            "\n⚠️ Gemma mode: system prompt will be injected into user content "
            "instead of using `systemInstruction`."
        )
    await interaction.response.send_message(
        f"✅ Text model mode set to **{mode.value}**.{info}", ephemeral=True
    )


# ---------------------------------------------------------------------------
# Slash commands — Audio clip mode
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-audio-mode", description="Enable or disable audio clip mode server-wide")
@app_commands.describe(enabled="True = bot sends .wav voice clips with every response, False = text only")
@app_commands.default_permissions(administrator=True)
async def set_audio_mode(interaction: discord.Interaction, enabled: bool):
    # Guard: warn if audio mode is being enabled without an endpoint
    if enabled and not bot_config.get("audio_endpoint", "").strip():
        await interaction.response.send_message(
            "⚠️ No audio endpoint configured yet. "
            "Run `/set-audio-endpoint` first, then enable audio mode.",
            ephemeral=True,
        )
        return

    bot_config["audio_enabled"] = enabled
    save_config(bot_config)
    state = "**enabled** 🔊" if enabled else "**disabled** 🔇"
    voice = bot_config.get("audio_settings", {}).get("voice", "Aoede")
    endpoint = bot_config.get("audio_endpoint", "(not set)")
    await interaction.response.send_message(
        f"✅ Audio clip mode {state}.\n"
        f"• TTS model: `{endpoint}`\n"
        f"• Voice: **{voice}**",
        ephemeral=True,
    )


@bot.tree.command(name="set-audio-endpoint", description="Set the Gemini TTS model endpoint")
@app_commands.describe(endpoint="TTS model name (e.g. gemini-2.5-flash-preview-tts)")
@app_commands.default_permissions(administrator=True)
async def set_audio_endpoint(interaction: discord.Interaction, endpoint: str):
    bot_config["audio_endpoint"] = endpoint
    save_config(bot_config)
    await interaction.response.send_message(
        f"✅ Audio (TTS) endpoint set to **{endpoint}**.", ephemeral=True
    )


@bot.tree.command(name="set-audio-settings", description="Set the voice used for audio clip mode")
@app_commands.describe(voice="Voice name (e.g. Aoede, Puck, Charon, Kore, Fenrir, Leda, Orus, Zephyr)")
@app_commands.default_permissions(administrator=True)
async def set_audio_settings(interaction: discord.Interaction, voice: str):
    audio_settings = bot_config.get("audio_settings", {})
    audio_settings["voice"] = voice
    bot_config["audio_settings"] = audio_settings
    save_config(bot_config)
    await interaction.response.send_message(f"✅ Audio voice set to **{voice}**.", ephemeral=True)


# ---------------------------------------------------------------------------
# Slash commands — Channel / context settings
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-allowed-channel", description="Whitelist or blacklist a channel for the bot")
@app_commands.describe(
    channel="The channel to configure",
    enabled="True = bot responds in this channel, False = bot ignores this channel",
)
@app_commands.default_permissions(administrator=True)
async def set_allowed_channel(interaction: discord.Interaction, channel: discord.TextChannel, enabled: bool):
    allowed = bot_config.get("allowed_channels", {})
    allowed[str(channel.id)] = enabled
    bot_config["allowed_channels"] = allowed
    save_config(bot_config)
    state = "whitelisted" if enabled else "blacklisted"
    await interaction.response.send_message(f"✅ {channel.mention} has been **{state}**.", ephemeral=True)


@bot.tree.command(name="set-ce", description="Enable/disable [ce] context cutoff for a channel")
@app_commands.describe(
    channel="The channel to configure",
    enabled="True = [ce] cuts off context (default), False = [ce] is ignored",
)
@app_commands.default_permissions(administrator=True)
async def set_ce(interaction: discord.Interaction, channel: discord.TextChannel, enabled: bool):
    ce_channels = bot_config.get("ce_channels", {})
    ce_channels[str(channel.id)] = enabled
    bot_config["ce_channels"] = ce_channels
    save_config(bot_config)
    state = "enabled" if enabled else "disabled"
    await interaction.response.send_message(
        f"✅ `[ce]` context cutoff **{state}** for {channel.mention}.", ephemeral=True
    )


# ---------------------------------------------------------------------------
# Slash commands — Stream of Consciousness (SoC)
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-soc", description="Configure the Stream of Consciousness thoughts channel")
@app_commands.describe(
    channel="The channel where the bot's thoughts will be posted",
    enabled="True = extract thoughts to channel, False = disabled",
)
@app_commands.default_permissions(administrator=True)
async def set_soc(interaction: discord.Interaction, channel: discord.TextChannel, enabled: bool):
    bot_config["soc_channel_id"] = str(channel.id)
    if enabled:
        bot_config["soc_enabled"] = True
        save_config(bot_config)
        await interaction.response.send_message(
            f"✅ SoC thoughts channel set to {channel.mention} — **enabled**.\n"
            f"Text between `<my-thoughts>` and `</my-thoughts>` will be extracted and posted there.",
            ephemeral=True,
        )
    else:
        bot_config["soc_enabled"] = False
        save_config(bot_config)
        await interaction.response.send_message(
            f"✅ SoC thoughts channel set to {channel.mention} — **disabled**.",
            ephemeral=True,
        )


@bot.tree.command(name="set-soc-context", description="Enable cross-channel thought context from the SoC channel")
@app_commands.describe(
    enabled="True = read past thoughts as context, False = disabled",
    count="Number of recent thought messages to read (default: 10)",
)
@app_commands.default_permissions(administrator=True)
async def set_soc_context(interaction: discord.Interaction, enabled: bool, count: int = 10):
    if enabled and not bot_config.get("soc_channel_id"):
        await interaction.response.send_message(
            "⚠️ No SoC channel configured yet. Run `/set-soc` first to set a thoughts channel.",
            ephemeral=True,
        )
        return
    if count < 1:
        await interaction.response.send_message("⚠️ Count must be at least 1.", ephemeral=True)
        return
    bot_config["soc_context_enabled"] = enabled
    bot_config["soc_context_count"] = count
    save_config(bot_config)
    state = "enabled" if enabled else "disabled"
    await interaction.response.send_message(
        f"✅ SoC context **{state}** — reading last **{count}** thought messages.",
        ephemeral=True,
    )


# ---------------------------------------------------------------------------
# Slash commands — Chat revival
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-chat-revival", description="Configure periodic chat revival in a channel")
@app_commands.describe(
    channel="The channel for chat revival",
    minutes="Minutes between revival messages",
    system_instruct="Special system instruction for revival messages",
    enabled="True = revival is active, False = revival does nothing",
)
@app_commands.default_permissions(administrator=True)
async def set_chat_revival(
    interaction: discord.Interaction,
    channel: discord.TextChannel,
    minutes: int,
    system_instruct: str,
    enabled: bool,
):
    if minutes < 1:
        await interaction.response.send_message("⚠️ Interval must be at least 1 minute.", ephemeral=True)
        return

    system_instruct = system_instruct.replace("\\n", "\n")

    bot_config["chat_revival"] = {
        "channel_id": str(channel.id),
        "interval_minutes": minutes,
        "system_instruct": system_instruct,
        "enabled": enabled,
    }
    save_config(bot_config)

    if revival_manager:
        revival_manager.start()

    state = "enabled" if enabled else "disabled"
    await interaction.response.send_message(
        f"✅ Chat revival set for {channel.mention} every **{minutes}** minute(s) — **{state}**.\n"
        f"📝 Revival instruction: ```{system_instruct}```",
        ephemeral=True,
    )


@bot.tree.command(name="set-cr-leave-msg", description="Set the message the bot sends when chat revival time expires")
@app_commands.describe(message="The goodbye message to send after the revival window")
@app_commands.default_permissions(administrator=True)
async def set_cr_leave_msg(interaction: discord.Interaction, message: str):
    message = message.replace("\\n", "\n")
    bot_config["cr_leave_message"] = message
    save_config(bot_config)
    await interaction.response.send_message(
        f"✅ Chat revival leave message updated to:\n```{message}```", ephemeral=True
    )


@bot.tree.command(name="set-cr-params", description="Set chat revival active duration and check interval")
@app_commands.describe(
    minutes="How many minutes the bot can freely talk during revival",
    seconds="How often (in seconds) it checks for new messages during revival",
)
@app_commands.default_permissions(administrator=True)
async def set_cr_params(interaction: discord.Interaction, minutes: int, seconds: int):
    if minutes < 1:
        await interaction.response.send_message("⚠️ Active duration must be at least 1 minute.", ephemeral=True)
        return
    if seconds < 5:
        await interaction.response.send_message("⚠️ Check interval must be at least 5 seconds.", ephemeral=True)
        return

    bot_config["cr_active_minutes"] = minutes
    bot_config["cr_check_seconds"] = seconds
    save_config(bot_config)
    await interaction.response.send_message(
        f"✅ Chat revival params updated:\n"
        f"• Active duration: **{minutes}** minute(s)\n"
        f"• Check interval: **{seconds}** second(s)",
        ephemeral=True,
    )


# ---------------------------------------------------------------------------
# Slash commands — Help
# ---------------------------------------------------------------------------

@bot.tree.command(name="help", description="Show all available commands")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🤖 ChatBuddy — Command Reference",
        description="All commands except `/help` require **Administrator** permissions.",
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="⚙️ Core Settings",
        value=(
            "`/set-api-key` — Set the Gemini API key\n"
            "`/set-chat-history` — Set context message count (default: 30)\n"
            "`/set-temp` — Set model temperature (0.0 – 2.0)\n"
            "`/set-api-endpoint` — Set the Gemini **text** model endpoint\n"
            "`/set-sys-instruct` — Set the system prompt\n"
            "`/show-sys-instruct` — Display current system prompt\n"
            "`/set-model-mode` — Switch text mode: `default` or `gemma`"
        ),
        inline=False,
    )

    embed.add_field(
        name="🔊 Audio Clip Mode",
        value=(
            "`/set-audio-endpoint` — Set the TTS model (e.g. `gemini-2.5-flash-preview-tts`)\n"
            "`/set-audio-settings` — Choose the voice (Aoede, Puck, Charon, Kore, Fenrir…)\n"
            "`/set-audio-mode` — Enable/disable audio clips globally\n\n"
            "When **enabled**, every response gets a `.wav` voice clip + text transcript. "
            "Works with both `default` and `gemma` text modes."
        ),
        inline=False,
    )

    embed.add_field(
        name="📺 Channel Settings",
        value=(
            "`/set-allowed-channel` — Whitelist/blacklist a channel\n"
            "`/set-ce` — Enable/disable `[ce]` context cutoff per channel"
        ),
        inline=False,
    )

    embed.add_field(
        name="🧠 Stream of Consciousness (SoC)",
        value=(
            "`/set-soc` — Set thoughts output channel + enable/disable\n"
            "`/set-soc-context` — Enable cross-channel thought context + message count\n\n"
            "When enabled, text between `<my-thoughts>` and `</my-thoughts>` is "
            "extracted and posted to the SoC channel. With SoC Context on, the bot "
            "reads its recent thoughts as additional context for all responses."
        ),
        inline=False,
    )

    embed.add_field(
        name="🔁 Chat Revival",
        value=(
            "`/set-chat-revival` — Configure periodic chat revival + enable/disable\n"
            "`/set-cr-params` — Set active window duration & check interval\n"
            "`/set-cr-leave-msg` — Set the goodbye message after revival expires"
        ),
        inline=False,
    )

    embed.add_field(
        name="💡 Quick Setup — Audio Mode",
        value=(
            "1. `/set-audio-endpoint gemini-2.5-flash-preview-tts`\n"
            "2. `/set-audio-settings Aoede` *(or your preferred voice)*\n"
            "3. `/set-audio-mode true`\n\n"
            "To disable: `/set-audio-mode false`"
        ),
        inline=False,
    )

    embed.add_field(
        name="💡 Using [ce] — Context End",
        value=(
            "Type **`[ce]`** in chat to cut off context. "
            "The bot ignores all messages before the most recent `[ce]`."
        ),
        inline=False,
    )

    embed.add_field(
        name="💡 Chat Revival",
        value=(
            "When revival fires, the bot posts a conversation starter. "
            "It then auto-replies during an active window without needing a mention. "
            "Each reply shows remaining active time in the footer."
        ),
        inline=False,
    )

    embed.set_footer(text="Mention me or reply to my messages to chat!")
    await interaction.response.send_message(embed=embed)


# ---------------------------------------------------------------------------
# Dummy HTTP server for Back4app health checks
# ---------------------------------------------------------------------------

class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Bot is online")

def run_dummy_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), DummyHandler)
    server.serve_forever()

threading.Thread(target=run_dummy_server, daemon=True).start()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
bot.run(TOKEN)
