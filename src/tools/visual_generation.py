from __future__ import annotations

import base64
from io import BytesIO
import logging
import re
from typing import Any

from src.config import Settings


logger = logging.getLogger(__name__)


class EconomyVisualGenerator:
    replicate_api_base_url = "https://api.replicate.com/v1"
    huggingface_text_to_image_urls = (
        "https://router.huggingface.co/hf-inference/models/{model}",
        "https://api-inference.huggingface.co/models/{model}",
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    def parse_request(self, text: str, has_reference_image: bool = False) -> str | None:
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
        if has_reference_image:
            markers.extend(
                [
                    "çiz",
                    "ciz",
                    "çevir",
                    "cevir",
                    "dönüştür",
                    "donustur",
                    "olarak",
                    "stil",
                    "style",
                    "bunu",
                ]
            )
        if not any(marker in lowered for marker in markers):
            return None
        return text.strip()

    def generate(
        self,
        request_text: str,
        reference_image: bytes | None = None,
        reference_mime_type: str = "image/jpeg",
    ) -> tuple[bytes, str]:
        if not self.settings.image_enabled:
            return self._render_fallback_infographic(request_text), "Ekonomi semasi"

        prompt = self._build_prompt(request_text, has_reference_image=bool(reference_image))
        if self.settings.replicate_image_enabled:
            image = self._generate_with_replicate(prompt, reference_image, reference_mime_type)
            if image:
                return image, "Ekonomi gorseli"

        if self.settings.huggingface_image_enabled and reference_image is None:
            image = self._generate_with_huggingface(prompt)
            if image:
                return image, "Ekonomi gorseli"

        if not self.settings.google_api_key:
            return self._render_fallback_infographic(request_text), "Ekonomi semasi"

        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.settings.google_api_key)

        try:
            image = self._generate_with_google(prompt, types, reference_image, reference_mime_type)
            if image:
                return image, "Ekonomi gorseli"
            logger.warning("Gemini image model returned no image; using fallback infographic.")
        except Exception as exc:
            logger.warning("Gemini image generation failed; using fallback infographic: %s", exc)
        return self._render_fallback_infographic(request_text), "Ekonomi semasi"

    def _generate_with_google(
        self,
        prompt: str,
        types: Any,
        reference_image: bytes | None = None,
        reference_mime_type: str = "image/jpeg",
    ) -> bytes | None:
        model = self.settings.gemini_image_model.strip()
        if model.lower().startswith("imagen-") and reference_image is None:
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

        contents: list[Any] = [prompt]
        if reference_image:
            contents.append(_google_image_part(types, reference_image, reference_mime_type))

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=["Image"]),
        )
        return self._extract_image_bytes(response)

    def _generate_with_replicate(
        self,
        prompt: str,
        reference_image: bytes | None = None,
        reference_mime_type: str = "image/jpeg",
    ) -> bytes | None:
        import requests

        model = self.settings.replicate_image_model.strip()
        token = self.settings.replicate_api_token
        if not model or not token:
            return None

        owner, name = _split_replicate_model(model)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": "wait=60",
            "Cancel-After": "90s",
        }
        payload = {
            "input": {
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "resolution": "1 MP",
                "output_format": "png",
                "output_quality": 90,
                "safety_tolerance": 2,
            }
        }
        if reference_image:
            payload["input"]["input_image"] = _data_uri(reference_image, reference_mime_type)
        url = f"{self.replicate_api_base_url}/models/{owner}/{name}/predictions"
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=max(self.settings.request_timeout_seconds, 65),
            )
            if response.status_code >= 400:
                logger.warning("Replicate image generation failed with %s: %s", response.status_code, response.text[:500])
                return None
            prediction = response.json()
            image_url = _replicate_output_url(prediction.get("output"))
            if image_url:
                return self._download_replicate_image(image_url)

            get_url = (prediction.get("urls") or {}).get("get")
            if not get_url:
                logger.warning("Replicate returned no output URL and no polling URL.")
                return None
            return self._poll_replicate_prediction(get_url, headers)
        except Exception as exc:
            logger.warning("Replicate image generation failed; trying next fallback: %s", exc)
            return None

    def _poll_replicate_prediction(self, get_url: str, headers: dict[str, str]) -> bytes | None:
        import time
        import requests

        poll_headers = {"Authorization": headers["Authorization"]}
        deadline = time.monotonic() + max(self.settings.request_timeout_seconds, 75)
        while time.monotonic() < deadline:
            response = requests.get(get_url, headers=poll_headers, timeout=max(self.settings.request_timeout_seconds, 20))
            if response.status_code >= 400:
                logger.warning("Replicate polling failed with %s: %s", response.status_code, response.text[:500])
                return None
            prediction = response.json()
            image_url = _replicate_output_url(prediction.get("output"))
            if image_url:
                return self._download_replicate_image(image_url)
            status = prediction.get("status")
            if status in {"failed", "canceled"}:
                logger.warning("Replicate prediction ended with status %s: %s", status, str(prediction.get("error"))[:500])
                return None
            time.sleep(2)
        logger.warning("Replicate prediction did not finish before timeout.")
        return None

    def _download_replicate_image(self, image_url: str) -> bytes | None:
        import requests

        response = requests.get(image_url, timeout=max(self.settings.request_timeout_seconds, 30))
        if response.status_code >= 400:
            logger.warning("Replicate image download failed with %s: %s", response.status_code, response.text[:500])
            return None
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            logger.warning("Replicate output URL returned non-image content type %s.", content_type or "<empty>")
            return None
        return response.content or None

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

    def _build_prompt(self, request_text: str, has_reference_image: bool = False) -> str:
        if has_reference_image:
            return (
                "Transform the supplied reference image according to the Turkish instruction. "
                "Preserve the main subject identity, pose, and composition when possible. "
                "Apply an economics/finance/editorial illustration style only if requested. "
                "Do not add fake price charts, fake financial numbers, logos, watermarks, or investment advice. "
                "Keep the result clean, professional, and suitable for a finance education bot. "
                f"Talimat: {request_text}"
            )
        return (
            "Create a clean Turkish finance education illustration. Avoid fake numbers, fake maps, "
            "fake price charts, random flags, random labels, tiny unreadable text, and investment advice. "
            "Use simple visual metaphors, clear spacing, and at most three short Turkish labels. "
            "The image must directly explain the requested concept, not a generic market dashboard. "
            f"Konu: {request_text}"
        )

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


def _split_replicate_model(model: str) -> tuple[str, str]:
    model = model.strip().strip("/")
    if "/" not in model:
        raise ValueError("REPLICATE_IMAGE_MODEL must be in owner/model format.")
    owner, name = model.split("/", 1)
    if not owner or not name or "/" in name:
        raise ValueError("REPLICATE_IMAGE_MODEL must be in owner/model format.")
    return owner, name


def _replicate_output_url(output: Any) -> str | None:
    if isinstance(output, str) and output.startswith(("http://", "https://")):
        return output
    if isinstance(output, list):
        for item in output:
            image_url = _replicate_output_url(item)
            if image_url:
                return image_url
    if isinstance(output, dict):
        for key in ("url", "image", "output"):
            image_url = _replicate_output_url(output.get(key))
            if image_url:
                return image_url
    return None


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


def _google_image_part(types: Any, image: bytes, mime_type: str) -> Any:
    part = getattr(types, "Part", None)
    from_bytes = getattr(part, "from_bytes", None)
    if callable(from_bytes):
        return from_bytes(data=image, mime_type=mime_type)
    return image


def _data_uri(image: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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
