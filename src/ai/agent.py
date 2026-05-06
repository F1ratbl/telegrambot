from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.ai.prompts import SYSTEM_PROMPT
from src.ai.tool_declarations import FUNCTION_DECLARATIONS
from src.bot.memory import InMemoryConversationMemory
from src.config import Settings
from src.tools.knowledge_base import KnowledgeBaseTool
from src.tools.market_data import MarketDataClient


logger = logging.getLogger(__name__)


class EconomyAgent:
    def __init__(
        self,
        settings: Settings,
        market_data: MarketDataClient,
        knowledge_base: KnowledgeBaseTool,
        memory: InMemoryConversationMemory,
    ) -> None:
        self.settings = settings
        self.market_data = market_data
        self.knowledge_base = knowledge_base
        self.memory = memory
        self._client = None

    def reply(self, user_message: str, chat_id: str | None = None, user_name: str | None = None) -> str:
        if not user_message.strip():
            return "Ekonomiyle ilgili sorunuzu yazarsaniz yardimci olayim."
        if not self.settings.google_api_key:
            return (
                "Gemini API anahtari henuz tanimli degil. GOOGLE_API_KEY veya "
                "GEMINI_API_KEY ortam degiskenini ekledikten sonra cevap uretebilirim."
            )

        try:
            return self._reply_with_gemini(user_message.strip(), chat_id, user_name)
        except Exception:
            logger.exception("Gemini agent failed.")
            return (
                "Su an ekonomi asistaninin model veya veri baglantisinda bir sorun var. "
                "Biraz sonra tekrar deneyebilir misiniz?"
            )

    def _reply_with_gemini(self, user_message: str, chat_id: str | None, user_name: str | None) -> str:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.settings.google_api_key)

        config = self._build_config(types, include_tools=True)
        contents = self._build_contents(types, user_message, chat_id, user_name)

        for _ in range(self.settings.gemini_max_tool_rounds):
            response = self._client.models.generate_content(
                model=self.settings.gemini_model,
                contents=contents,
                config=config,
            )
            function_calls = self._extract_function_calls(response)
            if not function_calls:
                answer = self._extract_text(response)
                self.memory.remember_exchange(chat_id, user_message, answer)
                return answer

            contents.append(response.candidates[0].content)
            response_parts = []
            for function_call in function_calls:
                tool_result = self._execute_tool(function_call)
                response_parts.append(self._function_response_part(types, function_call, tool_result))
            contents.append(types.Content(role="user", parts=response_parts))

        final_response = self._client.models.generate_content(
            model=self.settings.gemini_model,
            contents=contents,
            config=self._build_config(types, include_tools=False),
        )
        answer = self._extract_text(final_response)
        self.memory.remember_exchange(chat_id, user_message, answer)
        return answer

    def _build_config(self, types: Any, include_tools: bool) -> Any:
        kwargs: dict[str, Any] = {
            "system_instruction": SYSTEM_PROMPT,
            "max_output_tokens": self.settings.gemini_max_output_tokens,
        }
        if include_tools:
            kwargs["tools"] = [types.Tool(function_declarations=FUNCTION_DECLARATIONS)]
        if self.settings.gemini_temperature is not None:
            kwargs["temperature"] = self.settings.gemini_temperature
        if self.settings.gemini_thinking_level and hasattr(types, "ThinkingConfig"):
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.settings.gemini_thinking_level
            )
        return types.GenerateContentConfig(**kwargs)

    def _build_contents(
        self,
        types: Any,
        user_message: str,
        chat_id: str | None,
        user_name: str | None,
    ) -> list[Any]:
        contents = []
        for message in self.memory.snapshot(chat_id):
            contents.append(
                types.Content(
                    role=message.role,
                    parts=[types.Part(text=message.text)],
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=self._current_turn_text(user_message, user_name))],
            )
        )
        return contents

    def _current_turn_text(self, user_message: str, user_name: str | None) -> str:
        try:
            now = datetime.now(ZoneInfo(self.settings.timezone))
        except ZoneInfoNotFoundError:
            now = datetime.utcnow()
        timestamp = now.strftime("%Y-%m-%d %H:%M %Z").strip()
        display_name = user_name or "bilinmiyor"
        return (
            f"Telegram kullanicisi: {display_name}\n"
            f"Tarih/saat: {timestamp}\n"
            f"Kullanici mesaji: {user_message}"
        )

    def _extract_function_calls(self, response: Any) -> list[Any]:
        calls = getattr(response, "function_calls", None)
        if calls:
            return list(calls)

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return []
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        return [part.function_call for part in parts if getattr(part, "function_call", None)]

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "Cevap olusturamadim."
        parts = getattr(candidates[0].content, "parts", None) or []
        chunks = [part.text for part in parts if getattr(part, "text", None)]
        return "\n".join(chunks).strip() or "Cevap olusturamadim."

    def _execute_tool(self, function_call: Any) -> dict[str, Any]:
        name = getattr(function_call, "name", "")
        args = dict(getattr(function_call, "args", {}) or {})
        try:
            if name == "get_market_snapshot":
                return self.market_data.get_snapshot(symbols=args.get("symbols"))
            if name == "search_knowledge_base":
                return self.knowledge_base.search(
                    query=str(args.get("query", "")),
                    limit=args.get("limit"),
                )
            return {"status": "error", "message": f"Unknown tool: {name}"}
        except Exception as exc:
            logger.exception("Tool execution failed: %s", name)
            return {"status": "error", "message": str(exc), "tool": name}

    def _function_response_part(self, types: Any, function_call: Any, result: dict[str, Any]) -> Any:
        kwargs: dict[str, Any] = {
            "name": function_call.name,
            "response": {"result": result},
        }
        call_id = getattr(function_call, "id", None)
        if call_id:
            kwargs["id"] = call_id
        try:
            return types.Part.from_function_response(**kwargs)
        except TypeError:
            kwargs.pop("id", None)
            return types.Part.from_function_response(**kwargs)
