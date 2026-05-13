from __future__ import annotations

import base64
from io import BytesIO
import ast
import json
import logging
import os
import re
import tempfile
from typing import Any

from src.config import Settings


logger = logging.getLogger(__name__)


class EconomyVisualGenerator:
    huggingface_text_to_image_urls = (
        "https://router.huggingface.co/hf-inference/models/{model}",
        "https://api-inference.huggingface.co/models/{model}",
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        self._native_google_image_unavailable = False

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

        prompt = self._build_prompt(request_text)
        if self.settings.huggingface_image_enabled:
            image = self._generate_with_huggingface(prompt)
            if image:
                return image, "Ekonomi gorseli"

        if not self.settings.google_api_key:
            return self._render_fallback_infographic(request_text), "Ekonomi semasi"

        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.settings.google_api_key)

        if self._uses_text_guided_visual_mode():
            image = self._generate_text_guided_infographic(request_text, types)
            if image:
                return image, "Ekonomi semasi"
            return self._render_fallback_infographic(request_text), "Ekonomi semasi"

        try:
            if not self._native_google_image_unavailable:
                image = self._generate_with_google(prompt, types)
                if image:
                    return image, "Ekonomi gorseli"
                logger.warning("Gemini image model returned no image; trying text-guided infographic.")
        except Exception as exc:
            if _looks_like_paid_tier_image_error(exc):
                self._native_google_image_unavailable = True
            logger.warning("Gemini image generation failed; trying text-guided infographic: %s", exc)
        image = self._generate_text_guided_infographic(request_text, types)
        if image:
            return image, "Ekonomi semasi"
        return self._render_fallback_infographic(request_text), "Ekonomi semasi"

    def _generate_with_google(self, prompt: str, types: Any) -> bytes | None:
        model = self.settings.gemini_image_model.strip()
        if model.lower().startswith("imagen-"):
            response = self._client.models.generate_images(
                model=model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="16:9",
                    output_mime_type="image/png",
                ),
            )
            return self._extract_image_bytes(response)

        response = self._client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(response_modalities=["Image"]),
        )
        return self._extract_image_bytes(response)

    def _generate_text_guided_infographic(self, request_text: str, types: Any) -> bytes | None:
        model = self.settings.gemini_visual_text_model.strip()
        if not model:
            return None

        prompt = _build_visual_plan_prompt(request_text)
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=[prompt],
                config=_build_visual_plan_config(types),
            )
        except TypeError:
            response = self._client.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
        except Exception as exc:
            logger.warning("Gemini text visual planning failed; using static fallback infographic: %s", exc)
            return None

        spec = _parse_visual_plan(_extract_text(response), request_text)
        if spec is None:
            preview = _shorten(re.sub(r"\s+", " ", _extract_text(response) or ""), 220)
            logger.warning(
                "Gemini text visual planning returned an unusable plan; using static fallback infographic. "
                "Preview: %s",
                preview or "<empty>",
            )
            return None
        return self._render_fallback_infographic(request_text, spec)

    def _generate_with_huggingface(self, prompt: str) -> bytes | None:
        import requests

        model = self.settings.huggingface_image_model.strip()
        if not model:
            return None

        headers = {
            "Authorization": f"Bearer {self.settings.huggingface_api_key}",
            "Accept": "image/png",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "width": 1024,
                "height": 576,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "negative_prompt": (
                    "blurry, unreadable text, fake price chart, investment advice, noisy layout, "
                    "decorative clutter"
                ),
            },
            "options": {"wait_for_model": True},
        }
        for url_template in self.huggingface_text_to_image_urls:
            url = url_template.format(model=model)
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=max(self.settings.request_timeout_seconds, 20),
                )
                content_type = response.headers.get("content-type", "")
                if response.status_code >= 400:
                    logger.warning(
                        "Hugging Face image generation failed with %s at %s: %s",
                        response.status_code,
                        url,
                        response.text[:500],
                    )
                    continue
                if content_type.startswith("image/") and response.content:
                    return response.content
                logger.warning("Hugging Face returned non-image response from %s: %s", url, response.text[:500])
            except Exception as exc:
                logger.warning("Hugging Face image generation failed at %s; trying next fallback: %s", url, exc)
        return None

    def _build_prompt(self, request_text: str) -> str:
        return (
            "Create a clean Turkish finance education illustration. Avoid fake numbers, fake maps, "
            "fake price charts, random flags, random labels, tiny unreadable text, and investment advice. "
            "Use simple visual metaphors, clear spacing, and at most three short Turkish labels. "
            "The image must directly explain the requested concept, not a generic market dashboard. "
            f"Konu: {request_text}"
        )

    def _uses_text_guided_visual_mode(self) -> bool:
        model = self.settings.gemini_image_model.strip().lower()
        return model in {
            "gemini-text-infographic",
            "text-guided-infographic",
            "local-gemini-infographic",
        }

    def _extract_image_bytes(self, response: Any) -> bytes | None:
        generated_images = getattr(response, "generated_images", None) or []
        for generated_image in generated_images:
            image = getattr(generated_image, "image", None)
            image_bytes = getattr(image, "image_bytes", None)
            if image_bytes:
                return image_bytes

        image = _extract_from_parts(getattr(response, "parts", None) or [])
        if image:
            return image

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            image = _extract_from_parts(getattr(content, "parts", None) or [])
            if image:
                return image
        return None

    def _render_fallback_infographic(self, request_text: str, spec: dict[str, Any] | None = None) -> bytes:
        _ensure_matplotlib_config_dir()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        topic = _clean_topic(request_text)
        title = _shorten(str(spec.get("title") if spec else topic), 58)
        subtitle = _shorten(
            str(spec.get("subtitle") if spec else "Ekonomi yorumu icin hizli okuma semasi"),
            82,
        )
        footer = _shorten(
            str(
                spec.get("footer")
                if spec
                else "Kesin yatirim tavsiyesi degildir; karar icin veri, haber kaynagi ve riskler birlikte okunmalidir."
            ),
            125,
        )
        steps = spec.get("steps") if spec else None
        if not isinstance(steps, list) or len(steps) != 3:
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
            subtitle,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#475569",
        )

        y_positions = [0.68, 0.50, 0.32]
        colors = ["#dbeafe", "#dcfce7", "#fee2e2"]
        border_colors = ["#2563eb", "#16a34a", "#dc2626"]
        for index, step in enumerate(steps[:3]):
            if isinstance(step, dict):
                label = str(step.get("label") or f"Adim {index + 1}")
                body = str(step.get("body") or "")
            else:
                label, body = step
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
            footer,
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


def _build_visual_plan_prompt(request_text: str) -> str:
    return (
        "You are planning a Turkish finance education infographic that will be rendered as PNG by code. "
        "Return only valid JSON, no markdown, no explanation, no code fence. Do not invent prices, percentages, company facts, dates, "
        "logos, flags, or investment advice. Use short Turkish text that fits inside infographic boxes. "
        "Schema: {\"title\": string, \"subtitle\": string, \"steps\": [{\"label\": string, "
        "\"body\": string}, {\"label\": string, \"body\": string}, {\"label\": string, "
        "\"body\": string}], \"footer\": string}. "
        "Constraints: title max 48 chars, subtitle max 72 chars, each label max 18 chars, "
        "each body max 82 chars, footer max 120 chars. "
        f"Topic/request: {request_text}"
    )


def _ensure_matplotlib_config_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = os.path.join(tempfile.gettempdir(), "telegramaibot-matplotlib")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        return
    os.environ["MPLCONFIGDIR"] = cache_dir


def _build_visual_plan_config(types: Any) -> Any:
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "subtitle": {"type": "string"},
            "steps": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["label", "body"],
                },
            },
            "footer": {"type": "string"},
        },
        "required": ["title", "subtitle", "steps", "footer"],
    }
    kwargs: dict[str, Any] = {
        "response_mime_type": "application/json",
        "response_json_schema": schema,
        "temperature": 0.1,
        "max_output_tokens": 1200,
    }
    if hasattr(types, "ThinkingConfig"):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    try:
        return types.GenerateContentConfig(**kwargs)
    except Exception:
        kwargs.pop("response_json_schema", None)
        return types.GenerateContentConfig(**kwargs)


def _parse_visual_plan(text: str | None, request_text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = _strip_json_fence(text)
    data = _load_relaxed_json(cleaned)
    if data is None:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return _parse_labeled_visual_plan(cleaned, request_text)
        data = _load_relaxed_json(match.group(0))
        if data is None:
            return _parse_labeled_visual_plan(cleaned, request_text)
    if isinstance(data, list):
        data = {"steps": data}
    if not isinstance(data, dict):
        return _parse_labeled_visual_plan(cleaned, request_text)

    title = _clean_plan_text(_get_first(data, "title", "baslik", "başlık"), _clean_topic(request_text), 58)
    subtitle = _clean_plan_text(
        _get_first(data, "subtitle", "alt_baslik", "altBaşlık", "alt başlık", "altbaslik"),
        "Ekonomi yorumu icin hizli okuma semasi",
        82,
    )
    footer = _clean_plan_text(
        _get_first(data, "footer", "dipnot", "not", "uyarı", "uyari"),
        "Kesin yatirim tavsiyesi degildir; karar icin veri ve riskler birlikte okunmalidir.",
        125,
    )
    steps = _coerce_plan_steps(
        _get_first(data, "steps", "blocks", "items", "adimlar", "adımlar", "maddeler")
    )
    if steps is None:
        return _parse_labeled_visual_plan(cleaned, request_text)
    return _normalize_visual_plan({
        "title": title,
        "subtitle": subtitle,
        "steps": steps,
        "footer": footer,
    }, request_text)


def _load_relaxed_json(text: str) -> Any | None:
    candidates = [
        text,
        re.sub(r",\s*([}\]])", r"\1", text),
        text.replace("“", '"').replace("”", '"').replace("’", "'"),
    ]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    try:
        value = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(value, (dict, list)):
        return value
    return None


def _get_first(data: dict[Any, Any], *keys: str) -> Any:
    normalized = {_normalize_key(key): value for key, value in data.items()}
    for key in keys:
        value = normalized.get(_normalize_key(key))
        if value not in (None, ""):
            return value
    return None


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9ğüşöçıİĞÜŞÖÇ]+", "", str(value).casefold())


def _parse_labeled_visual_plan(text: str, request_text: str) -> dict[str, Any] | None:
    lines = [line.strip(" \t\r\n-*•") for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None

    title: str | None = None
    subtitle: str | None = None
    footer: str | None = None
    steps: list[dict[str, str]] = []

    for line in lines:
        key_match = re.match(r"^(title|başlık|baslik|subtitle|alt başlık|alt_baslik|footer|dipnot|not)\s*[:=-]\s*(.+)$", line, flags=re.IGNORECASE)
        if key_match:
            key = _normalize_key(key_match.group(1))
            value = key_match.group(2).strip()
            if key in {"title", "başlık", "baslik"}:
                title = value
            elif key in {"subtitle", "altbaşlık", "altbaslik"}:
                subtitle = value
            else:
                footer = value
            continue

        step = _parse_step_line(line)
        if step:
            steps.append(step)

    if len(steps) < 3:
        steps = _steps_from_plain_lines(lines)
    if len(steps) < 3:
        return None

    return _normalize_visual_plan(
        {
            "title": title or _clean_topic(request_text),
            "subtitle": subtitle or "Ekonomi yorumu icin hizli okuma semasi",
            "steps": steps[:3],
            "footer": footer or "Kesin yatirim tavsiyesi degildir; karar icin veri ve riskler birlikte okunmalidir.",
        },
        request_text,
    )


def _parse_step_line(line: str) -> dict[str, str] | None:
    cleaned = re.sub(
        r"^(step|adım|adim|madde)?\s*\d{1,2}\s*[.)\-:]\s*",
        "",
        line,
        flags=re.IGNORECASE,
    ).strip()
    if cleaned == line and not re.match(r"^\d{1,2}\s*[.)\-:]", line):
        return None
    for separator in [" - ", " – ", " — ", ": "]:
        if separator in cleaned:
            label, body = cleaned.split(separator, 1)
            return {
                "label": _clean_plan_text(label, "Adim", 20),
                "body": _clean_plan_text(body, "", 88),
            }
    return {"label": "Adim", "body": _clean_plan_text(cleaned, "", 88)}


def _steps_from_plain_lines(lines: list[str]) -> list[dict[str, str]]:
    ignored_prefixes = ("title", "başlık", "baslik", "subtitle", "alt", "footer", "dipnot", "not")
    candidates = [
        line
        for line in lines
        if len(line) >= 12 and not line.lower().startswith(ignored_prefixes)
    ]
    steps: list[dict[str, str]] = []
    for index, line in enumerate(candidates[:3]):
        if ": " in line:
            label, body = line.split(": ", 1)
        else:
            label, body = f"Adim {index + 1}", line
        steps.append({
            "label": _clean_plan_text(label, f"Adim {index + 1}", 20),
            "body": _clean_plan_text(body, "", 88),
        })
    return steps


def _normalize_visual_plan(spec: dict[str, Any], request_text: str) -> dict[str, Any] | None:
    steps = _coerce_plan_steps(spec.get("steps"))
    if steps is None:
        return None
    return {
        "title": _clean_plan_text(spec.get("title"), _clean_topic(request_text), 58),
        "subtitle": _clean_plan_text(spec.get("subtitle"), "Ekonomi yorumu icin hizli okuma semasi", 82),
        "steps": steps,
        "footer": _clean_plan_text(
            spec.get("footer"),
            "Kesin yatirim tavsiyesi degildir; karar icin veri ve riskler birlikte okunmalidir.",
            125,
        ),
    }


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _clean_plan_text(value: Any, default: str, limit: int) -> str:
    if value is None:
        return _shorten(default, limit)
    text = re.sub(r"\s+", " ", str(value)).strip(" \t\r\n\"'")
    if not text:
        text = default
    return _shorten(text, limit)


def _coerce_plan_steps(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None

    steps: list[dict[str, str]] = []
    for index, item in enumerate(value[:3]):
        if isinstance(item, dict):
            label = _clean_plan_text(item.get("label") or item.get("title"), f"Adim {index + 1}", 20)
            body = _clean_plan_text(item.get("body") or item.get("text") or item.get("description"), "", 88)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            label = _clean_plan_text(item[0], f"Adim {index + 1}", 20)
            body = _clean_plan_text(item[1], "", 88)
        elif isinstance(item, str):
            parsed = _parse_step_line(f"{index + 1}. {item}")
            if not parsed:
                return None
            label = parsed["label"]
            body = parsed["body"]
        else:
            return None
        if not body:
            return None
        steps.append({"label": label, "body": body})
    return steps


def _extract_text(response: Any) -> str | None:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    chunks: list[str] = []
    _append_text_parts(chunks, getattr(response, "parts", None) or [])
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        _append_text_parts(chunks, getattr(content, "parts", None) or [])
    joined = "\n".join(chunk for chunk in chunks if chunk)
    return joined or None


def _append_text_parts(chunks: list[str], parts: list[Any]) -> None:
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            chunks.append(text)


def _looks_like_paid_tier_image_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in [
            "only available on paid",
            "free tier",
            "not available",
            "billing",
            "upgrade your account",
        ]
    )


def _clean_topic(text: str) -> str:
    clean = re.sub(
        r"\b(infografik|görsel|gorsel|şema|sema|çiz|ciz|oluştur|olustur|görselle|gorselle|anlat|yap)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"\s+", " ", clean).strip(" .,:;!?")
    return clean or "Ekonomi konusu"


def _extract_from_parts(parts: list[Any]) -> bytes | None:
    for part in parts:
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

        as_image = getattr(part, "as_image", None)
        if not callable(as_image):
            continue
        try:
            image = as_image()
        except Exception:
            continue
        if image is None:
            continue
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
    return None


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _fallback_steps(topic: str) -> list[tuple[str, str]]:
    lowered = topic.lower()
    if any(marker in lowered for marker in ["hisse bölünmesi", "hisse bolunmesi", "stock split"]):
        return [
            ("Ne oluyor?", "Pay adedi artar; pay basina fiyat ayni oranda teorik olarak duser."),
            ("Deger etkisi", "Yatirimcinin toplam portfoy degeri bolunme aninda teorik olarak degismez."),
            ("Piyasa yorumu", "Likidite ve ilgi artabilir; ama bolunme tek basina temel deger yaratmaz."),
        ]
    if any(marker in lowered for marker in ["dilution", "sulanma"]):
        return [
            ("Ne oluyor?", "Yeni pay veya donusebilir menkul kiymetler ortaklik yuzdesini azaltabilir."),
            ("Olasi arti", "Sirket kasasina giren kaynak buyume, borc azaltma veya yatirim icin kullanilabilir."),
            ("Ana risk", "Kaynak verimli kullanilmazsa pay basina kar ve mevcut ortak degeri baskilanabilir."),
        ]
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
