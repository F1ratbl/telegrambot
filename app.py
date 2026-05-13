from __future__ import annotations

import logging

from flask import Flask

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is installed in normal runtime
    def load_dotenv() -> bool:
        return False

from src.ai.agent import EconomyAgent
from src.audio.speech import SpeechToTextClient, TextToSpeechClient
from src.bot.memory import InMemoryConversationMemory
from src.bot.telegram import TelegramClient
from src.bot.webhook import create_telegram_blueprint
from src.config import Settings
from src.tools.knowledge_base import KnowledgeBaseTool
from src.tools.charting import PriceChartTool
from src.tools.market_data import MarketDataClient
from src.tools.news import NewsSearchClient
from src.tools.visual_generation import EconomyVisualGenerator


load_dotenv()


def create_app() -> Flask:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    settings = Settings.from_env()
    memory = InMemoryConversationMemory(
        max_messages=settings.memory_max_messages,
        ttl_seconds=settings.memory_ttl_seconds,
    )
    market_data = MarketDataClient(settings)
    knowledge_base = KnowledgeBaseTool(settings)
    news_search = NewsSearchClient(settings)
    speech_to_text = SpeechToTextClient(settings)
    text_to_speech = TextToSpeechClient(settings)
    price_chart = PriceChartTool(settings)
    visual_generator = EconomyVisualGenerator(settings)
    agent = EconomyAgent(
        settings=settings,
        market_data=market_data,
        knowledge_base=knowledge_base,
        news_search=news_search,
        memory=memory,
    )
    telegram = TelegramClient(settings)

    app = Flask(__name__)
    app.register_blueprint(
        create_telegram_blueprint(
            settings=settings,
            agent=agent,
            telegram=telegram,
            speech_to_text=speech_to_text,
            text_to_speech=text_to_speech,
            price_chart=price_chart,
            visual_generator=visual_generator,
        )
    )

    @app.get("/")
    @app.get("/health")
    def health() -> tuple[dict[str, object], int]:
        return (
            {
                "ok": True,
                "service": "telegram-economy-ai",
                "gemini_model": settings.gemini_model,
                "qdrant_configured": settings.qdrant_enabled,
                "telegram_configured": settings.telegram_enabled,
                "voice_configured": settings.voice_enabled,
                "image_generation_configured": settings.image_enabled,
            },
            200,
        )

    return app


app = create_app()
