from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False

from src.config import Settings


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Register the Telegram webhook URL.")
    parser.add_argument("url", help="Full webhook URL, e.g. https://app.vercel.app/telegram/webhook")
    parser.add_argument("--drop-pending-updates", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_env()
    if not settings.telegram_bot_token:
        print("TELEGRAM_BOT_TOKEN is required.", file=sys.stderr)
        return 1

    import requests

    payload = {
        "url": args.url,
        "drop_pending_updates": args.drop_pending_updates,
        "allowed_updates": ["message", "edited_message"],
    }
    if settings.telegram_webhook_secret:
        payload["secret_token"] = settings.telegram_webhook_secret

    response = requests.post(
        f"https://api.telegram.org/bot{settings.telegram_bot_token}/setWebhook",
        json=payload,
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    print(response.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
