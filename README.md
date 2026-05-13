# Telegram Economy AI Bot

Flask tabanli Telegram webhook botu. Gemini modeliyle dogal sohbet eder, ekonomi disi sorularda cevabi sabit return ile degil modelin kendisi uretir, piyasa verisini tool olarak ceker ve Qdrant uzerindeki bilgi tabanini yine tool olarak kullanir.

## Mimari

- `app.py`: Vercel'in gorecegi Flask giris noktasi.
- `src/ai/agent.py`: Gemini function calling dongusu, kapsam promptu, sohbet bellegi.
- `src/tools/market_data.py`: Endeks, kur, emtia ve kripto snapshot tool'u.
- `src/tools/knowledge_base.py`: Qdrant arama tool'u.
- `src/qdrant/knowledge_store.py`: Qdrant koleksiyon, embedding ve upsert/search katmani.
- `src/bot/webhook.py`: Telegram webhook handler.

## Ortam Degiskenleri

`.env.example` dosyasini `.env` olarak kopyalayip doldurun:

```bash
TELEGRAM_BOT_TOKEN=...
TELEGRAM_WEBHOOK_SECRET=uzun-rastgele-bir-secret
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
QDRANT_URL=https://...qdrant.tech
QDRANT_API_KEY=...
QDRANT_COLLECTION=economy_knowledge
```

## Lokal Calistirma

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
flask --app app run --port 5000
```

Telegram webhook icin lokal tünel kullanin, sonra:

```bash
python scripts/set_telegram_webhook.py https://YOUR_DOMAIN/telegram/webhook --drop-pending-updates
```

## Bilgi Tabani Ingestion

`knowledgebase/` altina `.md` veya `.txt` ekonomi dokumanlari koyun.

```bash
python scripts/ingest_knowledgebase.py --path knowledgebase
```

Bu script dokumanlari parcalar, Gemini `gemini-embedding-001` ile 768 boyutlu embedding uretir ve Qdrant koleksiyonuna yazar.

## Vercel Deploy

Root dizindeki `app.py` Flask `app` instance'ini export eder. Vercel projesine environment variable'lari ekleyip deploy etmeniz yeterli.

Deploy sonrasi webhook:

```bash
python scripts/set_telegram_webhook.py https://YOUR_VERCEL_DOMAIN/telegram/webhook
```

## Notlar

- Piyasa verisi varsayilan olarak Yahoo Finance chart endpointinden gelir; borsa/veri saglayici durumuna gore gecikmeli olabilir.
- Serverless ortamda dahili sohbet bellegi warm instance icinde calisir. Uzun sureli ve garantili sohbet hafizasi gerekiyorsa ayni `memory` arayuzu Redis/Upstash gibi kalici bir depo ile degistirilebilir.
- Bot yatirim tavsiyesi vermek yerine egitim, analiz, senaryo ve risk odakli cevap vermeye yonlendirilir.
