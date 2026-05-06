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
from src.qdrant.knowledge_store import KnowledgeChunk, QdrantKnowledgeBase


SUPPORTED_SUFFIXES = {".md", ".txt"}
SKIPPED_NAMES = {"README.md"}


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest local economy knowledge files into Qdrant.")
    parser.add_argument("--path", default="knowledgebase", help="Folder or file to ingest.")
    parser.add_argument("--chunk-size", type=int, default=1400)
    parser.add_argument("--overlap", type=int, default=180)
    args = parser.parse_args()

    settings = Settings.from_env()
    source_path = Path(args.path)
    chunks = list(load_chunks(source_path, chunk_size=args.chunk_size, overlap=args.overlap))
    if not chunks:
        print("No .md or .txt knowledge files found.")
        return 0

    store = QdrantKnowledgeBase(settings)
    count = store.upsert_chunks(chunks)
    print(f"Ingested {count} chunks into Qdrant collection '{settings.qdrant_collection}'.")
    return 0


def load_chunks(path: Path, chunk_size: int, overlap: int) -> list[KnowledgeChunk]:
    files = [path] if path.is_file() else list(iter_knowledge_files(path))
    chunks: list[KnowledgeChunk] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = extract_title(text, fallback=file_path.stem)
        for index, chunk_text in enumerate(chunk_text_by_paragraphs(text, chunk_size, overlap)):
            chunks.append(
                KnowledgeChunk(
                    title=title,
                    content=chunk_text,
                    source=str(file_path),
                    chunk_index=index,
                    metadata={"filename": file_path.name},
                )
            )
    return chunks


def iter_knowledge_files(path: Path):
    if not path.exists():
        return
    for file_path in sorted(path.rglob("*")):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_SUFFIXES
            and file_path.name not in SKIPPED_NAMES
            and not any(part.startswith(".") for part in file_path.parts)
        ):
            yield file_path


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def chunk_text_by_paragraphs(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = _tail(current, overlap)
        if len(paragraph) > chunk_size:
            chunks.extend(_chunk_long_text(paragraph, chunk_size, overlap))
            current = ""
        else:
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def _chunk_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size].strip())
        start += max(1, chunk_size - overlap)
    return [chunk for chunk in chunks if chunk]


def _tail(text: str, size: int) -> str:
    if size <= 0:
        return ""
    return text[-size:].strip()


if __name__ == "__main__":
    raise SystemExit(main())
