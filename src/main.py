"""
Phone Agent - AI-powered phone assistant.

FastAPI application that handles:
- Twilio webhook for incoming calls
- WebSocket endpoint for real-time audio
- Integration with Deepgram (STT/TTS) and Claude (brain)
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, Response

from src.call_handler import CallHandler
from src.config import settings
from src.tts import get_tts

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "DEBUG"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Quiet down noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Phone Agent starting up...")
    logger.info(f"Server will run on {settings.host}:{settings.port}")

    # Pre-warm TTS client
    tts = await get_tts()

    yield

    # Cleanup
    await tts.close()
    logger.info("Phone Agent shut down")


app = FastAPI(
    title="Phone Agent",
    description="AI-powered phone assistant using Deepgram and Claude",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def home():
    """Simple status page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phone Agent</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #333; margin-bottom: 10px; }
            .status { color: #4CAF50; font-weight: bold; }
            .info { color: #666; margin-top: 20px; }
            code {
                background: #f0f0f0;
                padding: 2px 6px;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Phone Agent</h1>
            <p class="status">✓ Server is running</p>
            <div class="info">
                <p><strong>Endpoints:</strong></p>
                <ul>
                    <li><code>POST /incoming-call</code> - Twilio webhook</li>
                    <li><code>WS /media-stream</code> - Audio WebSocket</li>
                </ul>
                <p><strong>Configure Twilio:</strong></p>
                <ol>
                    <li>Set your webhook URL to <code>https://your-domain/incoming-call</code></li>
                    <li>Make sure to use POST method</li>
                </ol>
            </div>
        </div>
    </body>
    </html>
    """


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def incoming_call(request: Request):
    """
    Twilio webhook for incoming calls.

    Returns TwiML that tells Twilio to:
    1. Connect the call audio to our WebSocket endpoint
    2. Stream audio bidirectionally
    """
    # Get the host from the request to build WebSocket URL
    host = request.headers.get("host", "localhost:8000")

    # Determine WebSocket protocol based on original request
    # If behind a proxy with HTTPS, use wss://
    forwarded_proto = request.headers.get("x-forwarded-proto", "http")
    ws_proto = "wss" if forwarded_proto == "https" else "ws"

    # Extract caller info from Twilio's request
    form_data = {}
    if request.method == "POST":
        form_data = await request.form()

    caller = form_data.get("From", "Unknown")
    called = form_data.get("To", "")

    logger.info(f"Incoming call from {caller} to {called}")

    # TwiML response to start media stream
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_proto}://{host}/media-stream">
            <Parameter name="caller" value="{caller}" />
            <Parameter name="called" value="{called}" />
        </Stream>
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.

    Handles bidirectional audio streaming:
    - Receives caller audio from Twilio
    - Sends assistant audio back to Twilio
    """
    await websocket.accept()
    logger.info("Media stream WebSocket connected")

    # Create and run call handler
    handler = CallHandler(websocket)
    await handler.handle()

    logger.info("Media stream WebSocket disconnected")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Run the server."""
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
