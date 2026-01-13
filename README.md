# Phone Agent

An AI-powered phone assistant that answers calls with Claude's brain and Deepgram's voice.

## How It Works

```
Caller → Twilio → Your Server → Deepgram STT → Claude → Deepgram TTS → Caller
```

1. **Twilio** receives the call and streams audio to your server
2. **Deepgram STT** transcribes the caller's speech in real-time
3. **Claude** generates intelligent, contextual responses
4. **Deepgram TTS** converts responses to natural speech
5. Audio streams back through **Twilio** to the caller

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
DEEPGRAM_API_KEY=your_deepgram_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
AGENT_NAME=Will's Assistant
```

### 3. Run the Server

```bash
uv run python -m src.main
```

Server runs on `http://localhost:8000`

### 4. Expose to Internet

For Twilio to reach your server, you need a public URL. Use ngrok:

```bash
ngrok http 8000
```

Note the `https://xxxx.ngrok.io` URL.

### 5. Configure Twilio

1. Go to [Twilio Console](https://console.twilio.com/)
2. Get a phone number (or use existing)
3. Configure the number's webhook:
   - **Voice & Fax → A Call Comes In**
   - **Webhook URL:** `https://your-ngrok-url/incoming-call`
   - **HTTP Method:** POST

### 6. Call Your Number!

Call the Twilio number and talk to your AI assistant.

## Customization

### Change the Assistant's Personality

Edit `system_prompt.md` to customize:
- Name and personality
- What it can/can't do
- How it handles different situations

### Change the Voice

In `src/config.py`, change `tts_model` to a different Deepgram Aura voice:
- `aura-asteria-en` - Female, warm (default)
- `aura-luna-en` - Female, soft
- `aura-stella-en` - Female, professional
- `aura-athena-en` - Female, British
- `aura-hera-en` - Female, authoritative
- `aura-orion-en` - Male, warm
- `aura-arcas-en` - Male, professional
- `aura-perseus-en` - Male, friendly
- `aura-angus-en` - Male, Irish
- `aura-orpheus-en` - Male, deep
- `aura-helios-en` - Male, British
- `aura-zeus-en` - Male, authoritative

## Project Structure

```
phone-agent/
├── src/
│   ├── main.py           # FastAPI app & Twilio endpoints
│   ├── config.py         # Configuration management
│   ├── call_handler.py   # Call orchestration
│   ├── stt.py            # Deepgram speech-to-text
│   ├── tts.py            # Deepgram text-to-speech
│   └── brain.py          # Claude integration
├── transcripts/          # Call transcripts (auto-created)
├── system_prompt.md      # AI personality & instructions
├── .env                  # Your API keys (create from .env.example)
└── pyproject.toml        # Dependencies
```

## Features

- **Real-time transcription** - Deepgram's low-latency STT
- **Streaming responses** - Claude responds sentence-by-sentence for faster first-word-out
- **Barge-in support** - Interrupting the AI stops its speech
- **Call transcripts** - Every call is logged to `transcripts/`
- **Customizable personality** - Edit `system_prompt.md`

## Cost Estimate

With $200 Deepgram credit:
- **STT:** ~$0.0043/min → ~46,500 minutes
- **TTS:** ~$0.015/1000 chars → millions of characters

Plus Anthropic API costs (~$0.003/1K input, $0.015/1K output tokens for Claude Sonnet).

A typical 5-minute call costs roughly:
- Deepgram: ~$0.05
- Claude: ~$0.02-0.05
- **Total: ~$0.10 per call**

## Troubleshooting

### "Cannot connect to Deepgram"
- Check your `DEEPGRAM_API_KEY` is correct
- Ensure you have credit in your Deepgram account

### "No audio from assistant"
- Check Twilio webhook is configured for POST
- Verify ngrok is running and URL is correct
- Check logs for TTS errors

### "Assistant doesn't respond"
- Check `ANTHROPIC_API_KEY` is correct
- Look at server logs for Claude API errors

### "Audio is choppy"
- This can happen with high latency connections
- Consider deploying closer to Twilio's servers

## License

MIT
