"""
utils.py — Utility functions for ChatBuddy.
Mention stripping, message chunking, context formatting, and emoji resolution.
"""

import re
import os
from typing import List

import discord


def strip_mention(text: str, bot_id: int) -> str:
    """Remove the bot's mention tag(s) from the message text."""
    # Matches both <@bot_id> and <@!bot_id>
    pattern = rf"<@!?{bot_id}>"
    return re.sub(pattern, "", text).strip()


def chunk_message(text: str, limit: int = 2000) -> List[str]:
    """
    Split *text* into chunks of at most *limit* characters.
    Prefers splitting at newlines, then spaces, to keep messages readable.
    """
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break

        # Try to split at a newline within the limit
        split_pos = text.rfind("\n", 0, limit)
        if split_pos == -1:
            # Fall back to a space
            split_pos = text.rfind(" ", 0, limit)
        if split_pos == -1:
            # Hard cut if no good break point
            split_pos = limit

        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")

    return chunks


def format_context(messages: List[discord.Message], ce_enabled: bool = True) -> str:
    """
    Format a list of Discord messages into a rich context string.

    Each line:
        [YYYY-MM-DD HH:MM:SS] DisplayName (ID:123456789012345678): message content

    Including the user ID lets the LLM construct <@ID> mentions when instructed.
    Using raw `content` (not `clean_content`) preserves Discord formatting tokens
    such as <@id>, <:emoji:id>, etc. so the model sees how they actually appear.

    If *ce_enabled* is True, any message whose content is exactly "[ce]"
    (case-insensitive) acts as a context boundary — all messages before it
    (and the [ce] message itself) are discarded.
    """
    if ce_enabled:
        # Find the index of the LAST [ce] message
        ce_index = None
        for i, msg in enumerate(messages):
            if msg.content.strip().lower() == "[ce]":
                ce_index = i
        if ce_index is not None:
            messages = messages[ce_index + 1:]  # everything after the last [ce]

    lines: List[str] = []
    for msg in messages:
        timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
        display_name = msg.author.display_name
        user_id = msg.author.id
        content = msg.content  # raw content — preserves Discord tokens
        lines.append(f"[{timestamp}] {display_name} (ID:{user_id}): {content}")
    return "\n".join(lines)


def resolve_custom_emoji(text: str, guild: discord.Guild | None) -> str:
    """
    Replace custom emoji shortcodes in *text* with real Discord emoji markup.

    Scans for patterns like :emoji_name: and, if a matching custom emoji exists
    in the guild, replaces it with <:emoji_name:id> (or <a:emoji_name:id> for
    animated emoji).  Standard Unicode emoji and unknown shortcodes are left
    untouched.

    If *guild* is None (e.g. DMs), the text is returned unchanged.
    """
    if guild is None or not guild.emojis:
        return text

    # Build a lookup: lowercase emoji name -> emoji object
    emoji_map = {e.name.lower(): e for e in guild.emojis}
    if not emoji_map:
        return text

    # First, temporarily protect already-resolved Discord emoji from being
    # re-processed.  These look like <:name:id> or <a:name:id>.
    # We swap them out, do our replacements, then swap them back.
    _PLACEHOLDER = "\x00EMOJI{}\x00"
    protected: list[str] = []

    def _protect(match: re.Match) -> str:
        protected.append(match.group(0))
        return _PLACEHOLDER.format(len(protected) - 1)

    text = re.sub(r"<a?:\w+:\d+>", _protect, text)

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        # Skip purely-numeric matches — these are timestamp fragments like :34:
        if name.isdigit():
            return match.group(0)
        emoji = emoji_map.get(name.lower())
        if emoji is None:
            return match.group(0)  # not a guild emoji — leave as-is
        if emoji.animated:
            return f"<a:{emoji.name}:{emoji.id}>"
        return f"<:{emoji.name}:{emoji.id}>"

    # Match :word_chars: (2+ chars, matching Discord's minimum emoji name length)
    text = re.sub(r":([A-Za-z0-9_]{2,}):", _replace, text)

    # Restore protected emoji
    for i, original in enumerate(protected):
        text = text.replace(_PLACEHOLDER.format(i), original)

    return text


def extract_thoughts(text: str) -> tuple[str, str | None]:
    """
    Extract content between <my-thoughts> and </my-thoughts> tags.

    Returns (clean_text, thoughts_text):
      - clean_text: everything AFTER the last </my-thoughts> closing tag,
        with all thought blocks removed.  Users see only this.
      - thoughts_text: the concatenated inner content of all thought blocks
        (or None if no tags were found).
    """
    pattern = re.compile(r"<my-thoughts>(.*?)</my-thoughts>", re.DOTALL)
    matches = pattern.findall(text)

    if not matches:
        return text, None

    # Collect all thought content
    thoughts_text = "\n".join(m.strip() for m in matches)

    # Remove all thought blocks (tags + content) from the response
    clean_text = pattern.sub("", text).strip()

    return clean_text, thoughts_text


def extract_soul_updates(text: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Extract soul update commands from the response text.
    
    Returns (clean_text, updates):
      - clean_text: text with all soul tags removed.
      - updates: list of tuples (action, content) where action is 'update' or 'override'.
    """
    updates = []
    
    # Match <!soul-update: ...> or <!soul-override: ...>
    # Use DOTALL to grab multi-line content if necessary
    pattern = re.compile(r"<!soul-(update|override):\s*(.*?)>", re.DOTALL | re.IGNORECASE)
    
    for match in pattern.finditer(text):
        action = match.group(1).lower()
        content = match.group(2).strip()
        updates.append((action, content))
        
    clean_text = pattern.sub("", text).strip()
    return clean_text, updates


def handle_soul_updates(response_text: str, config: dict) -> str:
    """Extract soul tags, apply them immediately to soul.md if valid, and return cleaned text."""
    soul_enabled = config.get("soul_enabled", False)
    if not soul_enabled:
        return response_text

    clean_text, updates = extract_soul_updates(response_text)
    if not updates:
        return clean_text

    soul_limit = config.get("soul_limit", 2000)
    
    # Read current
    current_text = ""
    if os.path.exists("soul.md"):
        try:
            with open("soul.md", "r", encoding="utf-8") as f:
                current_text = f.read().strip()
        except Exception as e:
            print(f"[Soul] Error reading soul.md: {e}")

    # Process updates (only the last one really takes effect if multiple)
    new_text = current_text
    applied = False
    
    for action, content in updates:
        if action == "override":
            new_text = content
            applied = True
        elif action == "update":
            if new_text:
                new_text += f"\n{content}"
            else:
                new_text = content
            applied = True

    if applied:
        if len(new_text) > soul_limit:
            # Reject update
            print(f"[Soul] Update rejected: {len(new_text)} chars > {soul_limit} limit.")
            # We record a 1-turn error injection
            from config import save_config
            config["soul_error_turn"] = (
                f"System Error: Failed to update soul because it exceeded the {soul_limit} "
                f"character limit (attempted {len(new_text)} chars). "
                f"Faulty output rejected."
            )
            save_config(config)
        else:
            # Commit update
            try:
                with open("soul.md", "w", encoding="utf-8") as f:
                    f.write(new_text)
                print(f"[Soul] Updated soul.md ({len(new_text)} chars).")
            except Exception as e:
                print(f"[Soul] Failed to write soul.md: {e}")

    return clean_text
