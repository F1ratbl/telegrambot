# Knowledge Base

Bu klasor botun Qdrant'a yuklenecek kalici ekonomi bilgisidir. Gunluk haberleri buraya koymayin; burasi aciga satis, hisse bolunmesi, sermaye artirimi, KAP/SEC bildirimi okuma gibi surekli kullanilacak kavramlar icindir.

Yeni dosya ekledikten sonra:

```bash
python scripts/ingest_knowledgebase.py --path knowledgebase
```

`README.md` is skipped by the ingestion script.

Onerilen yapi:

```text
knowledgebase/
  tr/
    aciga-satis.md
    hisse-bolunmesi.md
    sermaye-artirimi.md
    kap-bildirimi-nasil-okunur.md
    haber-yorumlama-rehberi.md
  en/
    convertible-note.md
    dilution.md
```

Icerik yazarken uzun resmi metinleri kopyalamak yerine kisa ozet, yorum cercevesi ve kaynak linki ekleyin.
