# ChatBuddy v3.1

ChatBuddy is a Discord chatbot powered by Google Gemini (and optionally Gemma or custom external models) that participates in server conversations through mentions, auto-chat mode, and scheduled chat revival — with support for audio responses, a word guessing game, stream-of-consciousness thought extraction, and fully configurable behaviour via slash commands.

---

## Quick Start

### 1. Prerequisites

- **Python 3.10+**
- A **Discord Bot Token** (from the [Discord Developer Portal](https://discord.com/developers/applications))
- A **Google Gemini API key** (from [Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy or create a `.env` file in the project root:

```env
DISCORD_TOKEN=your_discord_bot_token_here
```

### 4. Run the Bot

```bash
python bot.py
```

The bot will come online and sync its slash commands with Discord automatically.

---

## Setup Walkthrough

Once the bot is online in your server, use these slash commands to configure it (all require **Administrator** permissions unless noted):

### Step 1 — Set Your API Key

```
/set-api-key key:<your Gemini API key>
```

### Step 2 — Whitelist a Channel

The bot only responds in whitelisted channels:

```
/set-allowed-channel channel:#general enabled:True
```

### Step 3 — Set a System Prompt

Give the bot its personality:

```
/set-sys-instruct prompt:You are a friendly and witty chatbot named ChatBuddy.
```

### Step 4 — Start Chatting

Mention the bot (`@ChatBuddy`) or reply to one of its messages. That's it!

---

## Command Reference

### ⚙️ Core Settings

| Command | Description |
|---|---|
| `/set-api-key` | Set the Gemini API key |
| `/set-chat-history` | Set how many messages of context the bot receives (default: 30) |
| `/set-temp` | Set model temperature (0.0 – 2.0) |
| `/set-api-endpoint-gemini` | Set the Gemini model endpoint |
| `/set-api-endpoint-gemma` | Set the Gemma model endpoint |
| `/set-api-key-custom` | Set the API key for a custom (non-Google) model |
| `/set-api-endpoint-custom` | Set the endpoint for a custom model (model name or full URL) |
| `/set-sys-instruct` | Set the main system prompt |
| `/show-sys-instruct` | Display the full effective system prompt |
| `/set-model-mode` | Switch between `gemini`, `gemma`, and `custom` |

### 📝 Dynamic & Game Prompts

| Command | Description |
|---|---|
| `/set-dynamic-system-prompt` | Set an extra prompt appended after the main prompt + enable/disable |
| `/set-soul` | Enable/disable the self-updating soul memory |
| `/show-soul` | View current soul memory |
| `/edit-soul` | Manually adjust the soul memory |
| `/set-soul-channel` | Set the channel to log soul updates + enable/disable |
| `/set-word-game` | Set word game rules (use `{secret-word}` placeholder) + enable/disable |
| `/set-word-game-selector-prompt` | Set the hidden-turn prompt the model uses to pick a word |
| `/set-secret-word` | Trigger a hidden turn to pick a new secret word *(role-gated)* |
| `/set-secret-word-permission` | Grant or revoke a role's access to `/set-secret-word` |

**💡 Note on Soul Memory:** 
To use the Soul feature effectively, ensure you provide instructions in the main system prompt (via `/set-sys-instruct`) telling the bot *when* and *what* it should remember. The bot interacts with the soul file by outputting these exact tags in its response:
- `<!soul-add-new[id]: text>` to add a completely new memory entry.
- `<!soul-update[id]: text>` to append to an existing memory.
- `<!soul-override[id]: text>` to completely overwrite a memory ID.
- `<!soul-delete[id]>` to remove a memory entirely.

### 🔊 Audio Clip Mode

| Command | Description |
|---|---|
| `/set-audio-endpoint` | Set the TTS model |
| `/set-audio-settings` | Choose the voice (e.g. Aoede, Puck, Charon) |
| `/set-audio-mode` | Enable/disable audio clips globally |

### 📺 Channel Settings

| Command | Description |
|---|---|
| `/set-allowed-channel` | Whitelist or blacklist a channel for the bot |
| `/set-ce` | Enable/disable `[ce]` context cutoff per channel |

### 🧠 Stream of Consciousness (SoC)

| Command | Description |
|---|---|
| `/set-soc` | Set the thoughts output channel + enable/disable |
| `/set-soc-context` | Enable cross-channel thought context + set message count |

Wrap thoughts in `<my-thoughts>` tags in the system prompt — the bot will extract them to the SoC channel. `[ce]` works in the SoC channel too.

### 💬 Auto-Chat Mode

| Command | Description |
|---|---|
| `/set-auto-chat-mode` | Auto-reply in a channel without requiring mentions |
| `/set-auto-idle-message` | Set the message posted when the bot enters idle mode |

The bot checks every N seconds for new messages. After the configured idle timeout with no new user messages, it goes idle and stops checking. A mention or reply to the bot reactivates it.

### 🔁 Chat Revival

| Command | Description |
|---|---|
| `/set-chat-revival` | Configure periodic chat revival + enable/disable |
| `/set-cr-params` | Set active window duration & check interval |
| `/set-cr-leave-msg` | Set the goodbye message after revival expires |

---

## Model Modes

| Mode | Description |
|---|---|
| `gemini` | Standard Google Gemini — uses `systemInstruction` field |
| `gemma` | Gemma-compatible — system prompt injected into user content |
| `custom` | External / non-Google API — uses a separate API key and endpoint |

For **custom mode**, set your endpoint and key:

```
/set-api-endpoint-custom endpoint:https://your-api.example.com/v1/generateContent
/set-api-key-custom key:your_custom_key
/set-model-mode mode:custom
```

If the custom endpoint starts with `http`, it's used as a full URL. Otherwise, it's treated as a model name under the standard Gemini API base.

---

## Deployment

The bot includes a built-in HTTP health-check server (for platforms like Back4app / Discloud). It listens on the port defined by the `PORT` environment variable (default `8080`).

### Docker

```bash
docker build -t chatbuddy .
docker run -e DISCORD_TOKEN=your_token -e PORT=8080 chatbuddy
```
