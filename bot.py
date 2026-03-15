"""
bot.py — Main entry point for ChatBuddy, a Discord bot powered by Gemini.
"""

import os
import io
import re
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from config import load_config, save_config
from gemini_api import generate, build_system_prompt
from utils import strip_mention, chunk_message, format_context, resolve_custom_emoji, extract_thoughts, extract_soul_updates
from revival import RevivalManager
from auto_chat import AutoChatManager

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

# Managers (initialised in on_ready)
revival_manager: RevivalManager | None = None
auto_chat_manager: AutoChatManager | None = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _read_soc_context(bot_ref, config: dict) -> str:
    """Read SoC channel messages and return formatted context string (or '')."""
    soc_context_enabled = config.get("soc_context_enabled", False)
    soc_channel_id = config.get("soc_channel_id")
    if not soc_context_enabled or not soc_channel_id:
        return ""

    soc_count = config.get("soc_context_count", 10)
    soc_channel = bot_ref.get_channel(int(soc_channel_id))
    if soc_channel is None:
        return ""

    soc_messages = []
    async for msg in soc_channel.history(limit=soc_count):
        soc_messages.append(msg)
    soc_messages.reverse()

    # Apply [ce] cutoff to SoC context
    ce_idx = None
    for i, m in enumerate(soc_messages):
        if m.content.strip().lower() == "[ce]":
            ce_idx = i
    if ce_idx is not None:
        soc_messages = soc_messages[ce_idx + 1:]

    if not soc_messages:
        return ""

    soc_lines = []
    for msg in soc_messages:
        ts = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
        soc_lines.append(f"[{ts}] {msg.content}")
    return (
        "\n[YOUR PREVIOUS THOUGHTS]\n"
        + "\n".join(soc_lines)
        + "\n[END YOUR PREVIOUS THOUGHTS]\n"
    )


async def _handle_soc_extraction(response_text: str, bot_ref, config: dict) -> str:
    """Extract thoughts, send to SoC channel, return clean text."""
    soc_enabled = config.get("soc_enabled", False)
    soc_channel_id = config.get("soc_channel_id")
    clean_text, thoughts_text = extract_thoughts(response_text)
    if thoughts_text and soc_enabled and soc_channel_id:
        thought_channel = bot_ref.get_channel(int(soc_channel_id))
        if thought_channel is not None:
            for chunk in chunk_message(thoughts_text):
                await thought_channel.send(chunk)
    return clean_text

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@bot.event
async def on_ready():
    global bot_config, revival_manager, auto_chat_manager
    bot_config = load_config()

    revival_manager = RevivalManager(bot, bot_config)
    revival_manager.start()

    auto_chat_manager = AutoChatManager(bot, bot_config)
    auto_chat_manager.start()

    try:
        synced = await bot.tree.sync()
        print(f"[ChatBuddy] Online as {bot.user} — synced {len(synced)} command(s)")
    except Exception as e:
        print(f"[ChatBuddy] Failed to sync commands: {e}")

@bot.command()
@commands.has_permissions(administrator=True)
async def purgecommands(ctx):
    """Nuke all guild-specific slash commands and resync global ones to clear 'ghosts'."""
    bot.tree.clear_commands(guild=ctx.guild)
    await bot.tree.sync(guild=ctx.guild)
    await ctx.send(f"✅ Wiped old guild slash commands and refreshed the tree for {ctx.guild.name}.")



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

    # Auto-chat reactivation: if mentioned/replied in the auto-chat channel
    # while idle, wake it up and respond normally
    auto_chat_channel = bot_config.get("auto_chat_channel_id")
    if auto_chat_channel and channel_key == str(auto_chat_channel):
        if auto_chat_manager and auto_chat_manager.is_idle:
            if is_mentioned or is_reply_to_bot:
                auto_chat_manager.reactivate()
                # Fall through to normal response below

    if not is_mentioned and not is_reply_to_bot:
        await bot.process_commands(message)
        return

    # If it is mentioned, also make sure we process commands in case it's a command too
    await bot.process_commands(message)

    async with message.channel.typing():
        user_text = strip_mention(message.content, bot.user.id)
        if not user_text:
            user_text = "(empty message)"

        # ── Normal response flow ────────────────────────────────────────
        history_limit = bot_config.get("chat_history_limit", 30)
        history_messages = []
        async for msg in message.channel.history(limit=history_limit, before=message):
            history_messages.append(msg)
        history_messages.reverse()

        ce_channels = bot_config.get("ce_channels", {})
        ce_enabled = ce_channels.get(channel_key, True)
        context = format_context(history_messages, ce_enabled=ce_enabled)

        # SoC context injection (after chat history)
        context += await _read_soc_context(bot, bot_config)

        response_text, audio_bytes, soul_logs = await generate(
            user_text, context, bot_config,
            speaker_name=message.author.display_name,
            speaker_id=str(message.author.id),
        )

        # SoC thought extraction
        response_text = await _handle_soc_extraction(response_text, bot, bot_config)

        # Resolve custom emoji shortcodes before sending
        response_text = resolve_custom_emoji(response_text, message.guild)

        if audio_bytes:
            audio_file = discord.File(fp=io.BytesIO(audio_bytes), filename="chatbuddy_voice.wav")
            await message.reply(file=audio_file, mention_author=False)
            chunks = chunk_message(response_text)
            for chunk in chunks:
                await message.channel.send(chunk)
        else:
            chunks = chunk_message(response_text)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await message.reply(chunk, mention_author=False)
                else:
                    await message.channel.send(chunk)

        # Send soul logs to configured channel if present
        if soul_logs and bot_config.get("soul_channel_enabled"):
            ch_id = bot_config.get("soul_channel_id")
            if ch_id:
                soul_ch = bot.get_channel(int(ch_id))
                if soul_ch:
                    joined_logs = "\n".join(soul_logs)
                    for log_chunk in chunk_message(joined_logs, limit=1900):
                        # Ensure we don't break code blocks if sending chunks
                        await soul_ch.send(f"**🧠 Soul Updates:**\n{log_chunk}")


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


@bot.tree.command(name="set-api-endpoint-gemini", description="Set the Gemini text model endpoint")
@app_commands.describe(endpoint="Model name (e.g. gemini-2.0-flash)")
@app_commands.default_permissions(administrator=True)
async def set_api_endpoint_gemini(interaction: discord.Interaction, endpoint: str):
    bot_config["model_endpoint_gemini"] = endpoint
    save_config(bot_config)
    await interaction.response.send_message(f"✅ Gemini endpoint set to **{endpoint}**.", ephemeral=True)


@bot.tree.command(name="set-api-endpoint-gemma", description="Set the Gemma text model endpoint")
@app_commands.describe(endpoint="Model name (e.g. gemma-3-27b-it)")
@app_commands.default_permissions(administrator=True)
async def set_api_endpoint_gemma(interaction: discord.Interaction, endpoint: str):
    bot_config["model_endpoint_gemma"] = endpoint
    save_config(bot_config)
    await interaction.response.send_message(f"✅ Gemma endpoint set to **{endpoint}**.", ephemeral=True)


@bot.tree.command(name="set-sys-instruct", description="Set the system instruction / prompt")
@app_commands.describe(prompt="The system prompt text")
@app_commands.default_permissions(administrator=True)
async def set_sys_instruct(interaction: discord.Interaction, prompt: str):
    prompt = prompt.replace("\\n", "\n")
    bot_config["system_prompt"] = prompt
    save_config(bot_config)
    await interaction.response.send_message("✅ System prompt updated and saved.", ephemeral=True)


@bot.tree.command(name="show-sys-instruct", description="Display the full effective system prompt")
@app_commands.default_permissions(administrator=True)
async def show_sys_instruct(interaction: discord.Interaction):
    prompt = build_system_prompt(bot_config, include_word_game=True)
    if not prompt:
        prompt = "(not set)"

    full_text = f"📝 **Current effective system prompt:**\n```\n{prompt}\n```"
    chunks = chunk_message(full_text)
    await interaction.response.send_message(chunks[0], ephemeral=True)
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk, ephemeral=True)


# ---------------------------------------------------------------------------
# Slash commands — Text model mode
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-model-mode", description="Switch between Gemini, Gemma, and Custom text model modes")
@app_commands.describe(mode="gemini = standard Gemini, gemma = Gemma-compatible injection, custom = external API")
@app_commands.choices(mode=[
    app_commands.Choice(name="gemini",  value="gemini"),
    app_commands.Choice(name="gemma",   value="gemma"),
    app_commands.Choice(name="custom",  value="custom"),
])
@app_commands.default_permissions(administrator=True)
async def set_model_mode(interaction: discord.Interaction, mode: app_commands.Choice[str]):
    bot_config["model_mode"] = mode.value
    save_config(bot_config)
    if mode.value == "gemma":
        ep = bot_config.get("model_endpoint_gemma", "(not set)")
        info = (
            f"\n⚠️ Gemma mode: system prompt injected into user content.\n"
            f"• Endpoint: `{ep}`"
        )
    elif mode.value == "custom":
        ep = bot_config.get("model_endpoint_custom", "(not set)")
        key_status = "set" if bot_config.get("api_key_custom", "").strip() else "not set"
        info = (
            f"\n⚠️ Custom mode: system prompt injected into user content.\n"
            f"• Endpoint: `{ep}`\n"
            f"• Custom API key: **{key_status}**"
        )
    else:
        ep = bot_config.get("model_endpoint_gemini", "gemini-2.0-flash")
        info = f"\n• Endpoint: `{ep}`"
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
# Slash commands — Dynamic system prompt
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-dynamic-system-prompt", description="Set an extra dynamic system prompt (appended after main)")
@app_commands.describe(
    prompt="The dynamic prompt text",
    enabled="True = active, False = disabled",
)
@app_commands.default_permissions(administrator=True)
async def set_dynamic_system_prompt(interaction: discord.Interaction, prompt: str, enabled: bool):
    prompt = prompt.replace("\\n", "\n")
    bot_config["dynamic_prompt"] = prompt
    bot_config["dynamic_prompt_enabled"] = enabled
    save_config(bot_config)
    state = "enabled" if enabled else "disabled"
    await interaction.response.send_message(
        f"✅ Dynamic system prompt **{state}** and saved.", ephemeral=True
    )


# ---------------------------------------------------------------------------
# Slash commands — Word game
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-word-game", description="Set the word game rules prompt + enable/disable")
@app_commands.describe(
    prompt="Game rules prompt (use {secret-word} as placeholder)",
    enabled="True = word game active, False = disabled",
)
@app_commands.default_permissions(administrator=True)
async def set_word_game(interaction: discord.Interaction, prompt: str, enabled: bool):
    prompt = prompt.replace("\\n", "\n")
    bot_config["word_game_prompt"] = prompt
    bot_config["word_game_enabled"] = enabled
    save_config(bot_config)
    state = "enabled" if enabled else "disabled"
    await interaction.response.send_message(
        f"✅ Word game **{state}**.\n"
        f"Prompt contains `{{secret-word}}`: **{'yes' if '{secret-word}' in prompt else 'no'}**",
        ephemeral=True,
    )


@bot.tree.command(name="set-word-game-selector-prompt", description="Set the hidden-turn prompt for selecting a secret word")
@app_commands.describe(prompt="Instruction appended to main prompt for the hidden word-selection turn")
@app_commands.default_permissions(administrator=True)
async def set_word_game_selector_prompt(interaction: discord.Interaction, prompt: str):
    prompt = prompt.replace("\\n", "\n")
    bot_config["word_game_selector_prompt"] = prompt
    save_config(bot_config)
    await interaction.response.send_message(
        "✅ Word game selector prompt saved.", ephemeral=True
    )


@bot.tree.command(name="set-secret-word", description="Trigger a hidden turn to pick a new secret word")
@app_commands.describe(prompt="Theme or constraint for the secret word (e.g. 'animals', 'foods')")
async def set_secret_word(interaction: discord.Interaction, prompt: str):
    # --- Role-based permission check ---
    allowed_roles = [str(r) for r in bot_config.get("secret_word_allowed_roles", [])]
    is_admin = False
    has_role = False
    
    if getattr(interaction, "guild", None) and isinstance(interaction.user, discord.Member):
        is_admin = interaction.user.guild_permissions.administrator
        has_role = any(str(role.id) in allowed_roles for role in interaction.user.roles)
        
    if not is_admin and not has_role:
        await interaction.response.send_message(
            "⚠️ You don't have permission to use this command. "
            "Ask an admin to grant your role access via `/set-secret-word-permission`.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(ephemeral=True)

    # Build hidden-turn system prompt: main + selector only
    main_prompt = bot_config.get("system_prompt", "")
    selector = bot_config.get("word_game_selector_prompt", "")
    hidden_sys = (main_prompt + "\n\n" + selector).strip() if selector else main_prompt

    hidden_response, _, _ = await generate(
        prompt=prompt,
        context="",
        config=bot_config,
        system_prompt_override=hidden_sys,
    )

    # Parse {secret-word:WORD} from the response
    word_match = re.search(r"\{secret-word:(.+?)\}", hidden_response)
    if word_match:
        secret = word_match.group(1).strip()
        bot_config["secret_word"] = secret
        save_config(bot_config)
        await interaction.followup.send("✅ A new secret word has been set!", ephemeral=True)
    else:
        await interaction.followup.send(
            "⚠️ Could not parse a secret word from the hidden turn. "
            "Make sure the selector prompt instructs the model to output `{secret-word:WORD}`.",
            ephemeral=True,
        )


@bot.tree.command(name="set-secret-word-permission", description="Grant or revoke a role's access to /set-secret-word")
@app_commands.describe(
    role="The role to configure",
    allowed="True = grant access, False = revoke access",
)
@app_commands.default_permissions(administrator=True)
async def set_secret_word_permission(interaction: discord.Interaction, role: discord.Role, allowed: bool):
    roles_list: list = bot_config.get("secret_word_allowed_roles", [])
    role_id = str(role.id)
    if allowed:
        if role_id not in roles_list:
            roles_list.append(role_id)
        action = "granted"
    else:
        if role_id in roles_list:
            roles_list.remove(role_id)
        action = "revoked"
    bot_config["secret_word_allowed_roles"] = roles_list
    save_config(bot_config)
    await interaction.response.send_message(
        f"✅ `/set-secret-word` access **{action}** for role **{role.name}**.",
        ephemeral=True,
    )


# ---------------------------------------------------------------------------
# Slash commands — Auto-chat mode
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-auto-chat-mode", description="Configure auto-chat mode for a channel")
@app_commands.describe(
    channel="The channel for auto-chat (one at a time)",
    enabled="True = auto-chat active, False = disabled",
    interval="Seconds between checks (default: 30)",
    idle_minutes="Minutes of inactivity before idle mode (default: 10)",
)
@app_commands.default_permissions(administrator=True)
async def set_auto_chat_mode(
    interaction: discord.Interaction,
    channel: discord.TextChannel,
    enabled: bool,
    interval: int = 30,
    idle_minutes: int = 10,
):
    if interval < 5:
        await interaction.response.send_message("⚠️ Interval must be at least 5 seconds.", ephemeral=True)
        return
    if idle_minutes < 1:
        await interaction.response.send_message("⚠️ Idle timeout must be at least 1 minute.", ephemeral=True)
        return

    bot_config["auto_chat_channel_id"] = str(channel.id)
    bot_config["auto_chat_enabled"] = enabled
    bot_config["auto_chat_interval"] = interval
    bot_config["auto_chat_idle_minutes"] = idle_minutes
    save_config(bot_config)

    if auto_chat_manager:
        auto_chat_manager.start()

    state = "enabled" if enabled else "disabled"
    await interaction.response.send_message(
        f"✅ Auto-chat **{state}** for {channel.mention}.\n"
        f"• Check interval: **{interval}s**\n"
        f"• Idle timeout: **{idle_minutes}m**",
        ephemeral=True,
    )


@bot.tree.command(name="set-auto-idle-message", description="Set the message posted when auto-chat enters idle mode")
@app_commands.describe(message="The idle message (default: 'Going afk, ping me if you need me')")
@app_commands.default_permissions(administrator=True)
async def set_auto_idle_message(interaction: discord.Interaction, message: str):
    message = message.replace("\\n", "\n")
    bot_config["auto_chat_idle_message"] = message
    save_config(bot_config)
    await interaction.response.send_message(
        f"✅ Auto-chat idle message set to:\n```{message}```", ephemeral=True
    )


# ---------------------------------------------------------------------------
# Slash commands — Soul feature
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-soul", description="Enable or disable the dynamic soul prompt and set its limit")
@app_commands.describe(
    enabled="True = active, False = disabled",
    limit="Max physical character limit of the soul text (default 2000)",
)
@app_commands.default_permissions(administrator=True)
async def set_soul(interaction: discord.Interaction, enabled: bool, limit: int = 2000):
    if limit < 100:
        await interaction.response.send_message("⚠️ Limit must be at least 100.", ephemeral=True)
        return
    bot_config["soul_enabled"] = enabled
    bot_config["soul_limit"] = limit
    save_config(bot_config)
    state = "enabled" if enabled else "disabled"
    await interaction.response.send_message(
        f"✅ Soul feature **{state}** with a limit of **{limit}** characters.\n"
        f"Bot uses `<!soul-update: text>` and `<!soul-override: text>` to update it.",
        ephemeral=True,
    )


@bot.tree.command(name="show-soul", description="View the current contents of the soul")
async def show_soul(interaction: discord.Interaction):
    if not os.path.exists("soul.md"):
        await interaction.response.send_message("📝 Soul is currently **empty** (file does not exist).", ephemeral=True)
        return
    
    with open("soul.md", "r", encoding="utf-8") as f:
        soul_text = f.read().strip()
        
    if not soul_text or soul_text == "{}":
        await interaction.response.send_message("📝 Soul is currently **empty**.", ephemeral=True)
        return

    full_text = f"📝 **Current Soul:**\n```\n{soul_text}\n```"
    chunks = chunk_message(full_text)
    await interaction.response.send_message(chunks[0], ephemeral=True)
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk, ephemeral=True)


async def _read_soul() -> dict:
    if not os.path.exists("soul.md"):
        return {}
    try:
        with open("soul.md", "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    except:
        pass
    return {}

async def _write_soul(interaction: discord.Interaction, soul_data: dict) -> bool:
    new_json = json.dumps(soul_data, indent=2, ensure_ascii=False)
    soul_limit = bot_config.get("soul_limit", 2000)
    if len(new_json) > soul_limit:
        await interaction.response.send_message(
            f"⚠️ Manual edit rejected: too large ({len(new_json)} > {soul_limit} limit).", 
            ephemeral=True
        )
        return False
    with open("soul.md", "w", encoding="utf-8") as f:
        f.write(new_json)
    return True

@bot.tree.command(name="wipe-soul", description="Wipe the entire soul file empty")
@app_commands.default_permissions(administrator=True)
async def wipe_soul(interaction: discord.Interaction):
    with open("soul.md", "w", encoding="utf-8") as f:
        f.write("{}")
    await interaction.response.send_message("✅ Soul successfully wiped.", ephemeral=True)

@bot.tree.command(name="edit-soul-delete-entry", description="Delete an entry from the soul")
@app_commands.describe(entry_name="The ID of the entry to delete")
@app_commands.default_permissions(administrator=True)
async def edit_soul_delete_entry(interaction: discord.Interaction, entry_name: str):
    soul_data = await _read_soul()
    if entry_name in soul_data:
        soul_data.pop(entry_name, None)
        if await _write_soul(interaction, soul_data):
            await interaction.response.send_message(f"✅ Deleted entry **{entry_name}**.", ephemeral=True)
    else:
        await interaction.response.send_message(f"⚠️ Entry **{entry_name}** not found.", ephemeral=True)

@bot.tree.command(name="edit-soul-add-entry", description="Add text to an entry (appends if exists)")
@app_commands.describe(
    entry_name="The ID of the entry",
    entry_text="Text to append or create"
)
@app_commands.default_permissions(administrator=True)
async def edit_soul_add_entry(interaction: discord.Interaction, entry_name: str, entry_text: str):
    soul_data = await _read_soul()
    entry_text = entry_text.replace("\\n", "\n")
    if entry_name in soul_data:
        soul_data[entry_name] += "\n" + entry_text
    else:
        soul_data[entry_name] = entry_text
    if await _write_soul(interaction, soul_data):
        await interaction.response.send_message(f"✅ Appended/added text to **{entry_name}**.", ephemeral=True)

@bot.tree.command(name="edit-soul-overwrite", description="Replace the text of an entry")
@app_commands.describe(
    entry_name="The ID of the entry",
    entry_text="Text to replace with"
)
@app_commands.default_permissions(administrator=True)
async def edit_soul_overwrite(interaction: discord.Interaction, entry_name: str, entry_text: str):
    soul_data = await _read_soul()
    entry_text = entry_text.replace("\\n", "\n")
    soul_data[entry_name] = entry_text
    if await _write_soul(interaction, soul_data):
        await interaction.response.send_message(f"✅ Overwrote entry **{entry_name}**.", ephemeral=True)


@bot.tree.command(name="set-soul-channel", description="Set the channel to log soul updates + enable/disable")
@app_commands.describe(
    channel="The channel to log updates to",
    enabled="True = active, False = disabled",
)
@app_commands.default_permissions(administrator=True)
async def set_soul_channel(interaction: discord.Interaction, channel: discord.TextChannel, enabled: bool):
    bot_config["soul_channel_id"] = str(channel.id)
    bot_config["soul_channel_enabled"] = enabled
    save_config(bot_config)
    state = "enabled" if enabled else "disabled"
    await interaction.response.send_message(f"✅ Soul logging **{state}** in {channel.mention}.", ephemeral=True)


# ---------------------------------------------------------------------------
# Slash commands — Custom model settings
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-api-key-custom", description="Set the API key for the custom (non-Google) model")
@app_commands.describe(key="Your custom model API key")
@app_commands.default_permissions(administrator=True)
async def set_api_key_custom(interaction: discord.Interaction, key: str):
    bot_config["api_key_custom"] = key
    save_config(bot_config)
    await interaction.response.send_message("✅ Custom API key has been set and saved.", ephemeral=True)


@bot.tree.command(name="set-api-endpoint-custom", description="Set the endpoint for the custom (non-Google) model")
@app_commands.describe(endpoint="Full URL or model name for your custom model")
@app_commands.default_permissions(administrator=True)
async def set_api_endpoint_custom(interaction: discord.Interaction, endpoint: str):
    bot_config["model_endpoint_custom"] = endpoint
    save_config(bot_config)
    await interaction.response.send_message(
        f"✅ Custom model endpoint set to **{endpoint}**.", ephemeral=True
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
        description="All commands except `/help` and `/set-secret-word` require **Administrator** permissions.",
        color=discord.Color.blurple(),
    )

    embed.add_field(
        name="⚙️ Core Settings",
        value=(
            "`/set-api-key` — Set the Gemini API key\n"
            "`/set-chat-history` — Set context message count (default: 30)\n"
            "`/set-temp` — Set model temperature (0.0 – 2.0)\n"
            "`/set-api-endpoint-gemini` — Set the Gemini model endpoint\n"
            "`/set-api-endpoint-gemma` — Set the Gemma model endpoint\n"
            "`/set-api-key-custom` — Set the API key for a custom (non-Google) model\n"
            "`/set-api-endpoint-custom` — Set the endpoint for a custom model\n"
            "`/set-sys-instruct` — Set the main system prompt\n"
            "`/show-sys-instruct` — Display the full effective system prompt\n"
            "`/set-model-mode` — Switch between `gemini`, `gemma`, and `custom`"
        ),
        inline=False,
    )

    embed.add_field(
        name="📝 Dynamic & Game Prompts",
        value=(
            "`/set-dynamic-system-prompt` — Set an extra prompt (appended after main) + enable/disable\n"
            "`/set-soul` — Enable/disable the self-updating soul memory\n"
            "`/show-soul` — View current soul memory\n"
            "`/edit-soul` — Manually adjust the soul memory\n"
            "`/set-soul-channel` — Set the channel to log soul updates + enable/disable\n"
            "*Note: For Soul to work best, instruct the bot in its system prompt to save memories using `<!soul-add-new[id]: text>`, `<!soul-update[id]: text>`, `<!soul-override[id]: text>`, or `<!soul-delete[id]>`.*\n\n"
            "`/set-word-game` — Set word game rules (`{secret-word}` placeholder) + enable/disable\n"
            "`/set-word-game-selector-prompt` — Set the hidden-turn prompt for word selection\n"
            "`/set-secret-word` — Trigger a hidden turn to pick a new secret word (role-gated)\n"
            "`/set-secret-word-permission` — Grant/revoke a role's access to `/set-secret-word`"
        ),
        inline=False,
    )

    embed.add_field(
        name="🔊 Audio Clip Mode",
        value=(
            "`/set-audio-endpoint` — Set the TTS model\n"
            "`/set-audio-settings` — Choose the voice\n"
            "`/set-audio-mode` — Enable/disable audio clips globally"
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
            "Extracts `<my-thoughts>` blocks to a dedicated channel. "
            "`[ce]` works in the SoC channel too."
        ),
        inline=False,
    )

    embed.add_field(
        name="💬 Auto-Chat Mode",
        value=(
            "`/set-auto-chat-mode` — Auto-reply in a channel without needing mentions\n"
            "`/set-auto-idle-message` — Set the message posted when entering idle\n\n"
            "Checks every N seconds. Goes idle if the bot's own message is the latest "
            "for the configured timeout. A mention/reply reactivates it."
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

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

def run_dummy_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), DummyHandler)
    server.serve_forever()

threading.Thread(target=run_dummy_server, daemon=True).start()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
bot.run(TOKEN)
