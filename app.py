from __future__ import annotations

import logging

from flask import Flask

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is installed in normal runtime
    def load_dotenv() -> bool:
        return False

from src.ai.agent import EconomyAgent
from src.bot.memory import InMemoryConversationMemory
from src.bot.telegram import TelegramClient
from src.bot.webhook import create_telegram_blueprint
from src.config import Settings
from src.tools.knowledge_base import KnowledgeBaseTool
from src.tools.market_data import MarketDataClient


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
    agent = EconomyAgent(
        settings=settings,
        market_data=market_data,
        knowledge_base=knowledge_base,
        memory=memory,
    )
    telegram = TelegramClient(settings)

    app = Flask(__name__)
    app.register_blueprint(create_telegram_blueprint(settings, agent, telegram))

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
            },
            200,
        )

    return app


app = create_app()
