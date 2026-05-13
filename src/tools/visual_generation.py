from __future__ import annotations

import base64
from io import BytesIO
import logging
import re
from typing import Any

from src.config import Settings


logger = logging.getLogger(__name__)


class EconomyVisualGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    def parse_request(self, text: str) -> str | None:
        lowered = text.lower()
        markers = [
            "infografik",
            "görsel",
            "gorsel",
            "şema",
            "sema",
            "görselle anlat",
            "gorselle anlat",
            "resim oluştur",
            "resim olustur",
        ]
        if not any(marker in lowered for marker in markers):
            return None
        return text.strip()

    def generate(self, request_text: str) -> tuple[bytes, str]:
        if not self.settings.image_enabled:
            return self._render_fallback_infographic(request_text), "Ekonomi semasi"

        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.settings.google_api_key)

        prompt = self._build_prompt(request_text)
        try:
            response = self._client.models.generate_content(
                model=self.settings.gemini_image_model,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
            )
            image = self._extract_image_bytes(response)
            if image:
                return image, "Ekonomi gorseli"
            logger.warning("Gemini image model returned no image; using fallback infographic.")
        except Exception as exc:
            logger.warning("Gemini image generation failed; using fallback infographic: %s", exc)
        return self._render_fallback_infographic(request_text), "Ekonomi semasi"

    def _build_prompt(self, request_text: str) -> str:
        return (
            "Turkce ekonomi ve finans egitimi icin sade, profesyonel bir infografik uret. "
            "Gorselde yaniltici fiyat, sahte grafik veya kesin yatirim tavsiyesi olmasin. "
            "Kisa basliklar, net oklar ve finansal analiz mantigina uygun akış kullan. "
            "Telegram'da okunacak sekilde temiz, modern ve yuksek kontrastli tasarla. "
            f"Konu: {request_text}"
        )

    def _extract_image_bytes(self, response: Any) -> bytes | None:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                inline_data = getattr(part, "inline_data", None)
                data = getattr(inline_data, "data", None)
                if data:
                    if isinstance(data, bytes):
                        return data
                    if isinstance(data, str):
                        try:
                            return base64.b64decode(data)
                        except Exception:
                            return data.encode("utf-8")
        return None

    def _render_fallback_infographic(self, request_text: str) -> bytes:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        topic = _clean_topic(request_text)
        title = _shorten(topic, 58)
        steps = _fallback_steps(topic)

        fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        ax.axis("off")

        ax.text(
            0.5,
            0.92,
            title,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="#111827",
            wrap=True,
        )
        ax.text(
            0.5,
            0.855,
            "Ekonomi yorumu icin hizli okuma semasi",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#475569",
        )

        y_positions = [0.68, 0.50, 0.32]
        colors = ["#dbeafe", "#dcfce7", "#fee2e2"]
        border_colors = ["#2563eb", "#16a34a", "#dc2626"]
        for index, (label, body) in enumerate(steps):
            x = 0.12
            y = y_positions[index]
            width = 0.76
            height = 0.12
            box = FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.018,rounding_size=0.02",
                linewidth=1.6,
                edgecolor=border_colors[index],
                facecolor=colors[index],
            )
            ax.add_patch(box)
            ax.text(
                x + 0.035,
                y + height * 0.68,
                label,
                ha="left",
                va="center",
                fontsize=12.5,
                fontweight="bold",
                color="#111827",
            )
            ax.text(
                x + 0.035,
                y + height * 0.34,
                body,
                ha="left",
                va="center",
                fontsize=10.5,
                color="#334155",
                wrap=True,
            )
            if index < 2:
                ax.annotate(
                    "",
                    xy=(0.5, y_positions[index + 1] + 0.145),
                    xytext=(0.5, y - 0.018),
                    arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.6},
                )

        ax.text(
            0.5,
            0.12,
            "Kesin yatirim tavsiyesi degildir; karar icin veri, haber kaynagi ve riskler birlikte okunmalidir.",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#64748b",
            wrap=True,
        )

        output = BytesIO()
        fig.savefig(output, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return output.getvalue()


def _clean_topic(text: str) -> str:
    clean = re.sub(
        r"\b(infografik|görsel|gorsel|şema|sema|çiz|ciz|oluştur|olustur|görselle|gorselle|anlat|yap)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"\s+", " ", clean).strip(" .,:;!?")
    return clean or "Ekonomi konusu"


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _fallback_steps(topic: str) -> list[tuple[str, str]]:
    lowered = topic.lower()
    if any(marker in lowered for marker in ["bedelli", "sermaye"]):
        return [
            ("Ne oluyor?", "Sirket yeni pay ihraciyla kasasina nakit koymaya calisir."),
            ("Olasi arti", "Borc azaltma, yatirim veya isletme sermayesi icin kaynak saglayabilir."),
            ("Ana risk", "Mevcut ortaklar katilmazsa paylari sulanabilir; fonun nerede kullanilacagi kritik."),
        ]
    if any(marker in lowered for marker in ["tahvil", "convertible", "donusturulebilir"]):
        return [
            ("Ne oluyor?", "Sirket borclanarak bugun finansman saglar; vadede geri odeme veya donusum olabilir."),
            ("Olasi arti", "Likiditeyi guclendirir ve buyume yatirimlarini finanse edebilir."),
            ("Ana risk", "Faiz yuku, vade baskisi ve donusum varsa ileride sulanma riski dogurabilir."),
        ]
    if any(marker in lowered for marker in ["enflasyon", "faiz", "merkez bank", "fed", "tcmb"]):
        return [
            ("Veri", "Enflasyon ve faiz beklentileri piyasa fiyatlamasinin temel girdileridir."),
            ("Piyasa etkisi", "Faiz beklentisi tahvil, kur, altin ve hisse carpanlarini etkileyebilir."),
            ("Risk", "Beklentiden sapma volatilite yaratir; karar metni ve ileri yonlendirme izlenmelidir."),
        ]
    return [
        ("Konu", "Once haberin sirket, varlik veya makro veri uzerindeki dogrudan etkisi okunur."),
        ("Olasi etki", "Gelir, kar, nakit akisi, borcluluk, faiz, kur veya beklenti kanali incelenir."),
        ("Risk", "Haber tek basina karar icin yeterli degildir; fiyatlama, zamanlama ve belirsizlikler kontrol edilir."),
    ]
