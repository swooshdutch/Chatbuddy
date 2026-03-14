"""
utils.py — Utility functions for ChatBuddy.
Mention stripping, message chunking, context formatting, and emoji resolution.
"""

import re
import os
import json
from typing import List, Dict, Tuple

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


def extract_soul_updates(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    """
    Extracts soul tags directly from generated text.
    Returns:
      (clean_text, [(action, id, content), ...])
    Wait for markdown escapes and optional whitespace around [id]:
       <!soul-update[id]: text>
       <!soul-delete[id]>
    """
    updates = []
    
    # Match <!soul-add-new[id]: ...>, <!soul-update[id]: ...>, <!soul-override[id]: ...>, <!soul-delete[id]>
    # Handle possible markdown escaping or spaces (e.g. \<\!soul-update\[1\]:)
    pattern = re.compile(
        r"\\?<\s*\\?!\s*soul-(add-new|update|override|delete)\s*\\?\[\s*(.+?)\s*\\?\](?:\s*:\s*(.*?))?\s*\\?>", 
        re.DOTALL | re.IGNORECASE
    )
    
    for match in pattern.finditer(text):
        action = match.group(1).lower()
        entry_id = match.group(2).strip()
        content = match.group(3)
        if content:
            content = content.strip()
        else:
            content = ""
        updates.append((action, entry_id, content))
        
    clean_text = pattern.sub("", text).strip()
    return clean_text, updates


def handle_soul_updates(response_text: str, config: dict) -> tuple[str, list[str]]:
    """
    Extracts tags, enforces limit, and applies to JSON in soul.md.
    Returns (clean_text, logs), where `logs` is a list of log strings to print to soul-channel.
    """
    clean_text, updates = extract_soul_updates(response_text)
    logs = []

    soul_enabled = config.get("soul_enabled", False)
    if not soul_enabled or not updates:
        return clean_text, logs

    soul_limit = config.get("soul_limit", 2000)
    soul_file = "soul.md"

    # Read current soul.md
    soul_data: Dict[str, str] = {}
    if os.path.exists(soul_file):
        try:
            with open(soul_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    soul_data = json.loads(content)
        except json.JSONDecodeError:
            # Conversion for legacy raw text to JSON
            print("[Soul] Converting legacy text to JSON entry '0'.")
            soul_data = {"0": content}
        except Exception as e:
            print(f"[Soul] Error reading soul file: {e}")

    # Process updates
    made_changes = False
    for action, entry_id, content in updates:
        # Ignore empty ids
        if not entry_id:
            continue
            
        previous_data = soul_data.copy()

        if action == "add-new":
            if entry_id in soul_data:
                soul_data[entry_id] += "\n" + content
                logs.append(f"**Appended to existing [{entry_id}] (via add-new)**:\n`{content}`")
            else:
                soul_data[entry_id] = content
                logs.append(f"**Added New [{entry_id}]**:\n`{content}`")
            made_changes = True

        elif action == "update":
            if entry_id in soul_data:
                soul_data[entry_id] += "\n" + content
                logs.append(f"**Updated [{entry_id}]**:\nAppended: `{content}`")
            else:
                soul_data[entry_id] = content
                logs.append(f"**Created [{entry_id}]**:\n`{content}`")
            made_changes = True

        elif action == "override":
            soul_data[entry_id] = content
            logs.append(f"**Overwrote [{entry_id}]**:\n`{content}`")
            made_changes = True
            
        elif action == "delete":
            if entry_id in soul_data:
                del soul_data[entry_id]
                logs.append(f"**Deleted [{entry_id}]**")
                made_changes = True

        # Validation per action step
        if made_changes:
            new_json_str = json.dumps(soul_data, indent=2, ensure_ascii=False)
            if len(new_json_str) > soul_limit:
                # Reject this specific update, rollback state
                soul_data = previous_data
                made_changes = False
                if logs:
                    logs.pop() # remove associated log for rejection
                print(f"[Soul] Update rejected: {len(new_json_str)} chars > {soul_limit} limit.")
                # We record a 1-turn error injection
                from config import save_config
                config["soul_error_turn"] = (
                    f"System Error: Failed to apply soul action '{action}' on ID '{entry_id}' because it exceeded the {soul_limit} "
                    f"character JSON file limit (attempted {len(new_json_str)} chars). "
                    f"Faulty output rejected."
                )
                save_config(config)

    # Save to file if we hold valid changes
    if made_changes:
        new_json_str = json.dumps(soul_data, indent=2, ensure_ascii=False)
        try:
            with open(soul_file, "w", encoding="utf-8") as f:
                f.write(new_json_str)
            print(f"[Soul] Updated soul.md ({len(new_json_str)} chars JSON).")
        except Exception as e:
            print(f"[Soul] Failed to write soul.md: {e}")

    return clean_text, logs