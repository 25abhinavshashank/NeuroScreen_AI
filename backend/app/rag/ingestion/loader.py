
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from docx import Document as DocxDocument
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[4]

KNOWLEDGE_BASE_DIR = (
    PROJECT_ROOT / "knowledge_base"
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".json",
    ".xml",
}


@dataclass
class LoadedDocument:
    text: str
    source: str
    file_name: str
    file_type: str


def load_pdf(path: Path) -> str:

    reader = PdfReader(str(path))

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


def load_docx(path: Path) -> str:

    document = DocxDocument(str(path))

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def load_text(path: Path) -> str:

    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def load_json(path: Path) -> str:

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    )


def load_file(path: Path) -> str:

    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(path)

    if extension == ".docx":
        return load_docx(path)

    if extension in {
        ".txt",
        ".md",
        ".xml",
    }:
        return load_text(path)

    if extension == ".json":
        return load_json(path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


def discover_files() -> List[Path]:

    files = []

    for path in KNOWLEDGE_BASE_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files.append(path)

    return sorted(files)


def load_knowledge_base() -> List[LoadedDocument]:

    loaded_documents = []

    files = discover_files()

    print(
        f"Knowledge-base files discovered: "
        f"{len(files)}"
    )

    for path in files:

        try:

            text = load_file(path)

            text = text.strip()

            if not text:
                print(
                    f"Skipping empty file: {path.name}"
                )
                continue

            relative_path = path.relative_to(
                PROJECT_ROOT
            )

            loaded_documents.append(
                LoadedDocument(
                    text=text,
                    source=str(relative_path),
                    file_name=path.name,
                    file_type=path.suffix.lower(),
                )
            )

            print(
                f"Loaded: {relative_path}"
            )

        except Exception as exc:

            print(
                f"Failed to load {path}: {exc}"
            )

    return loaded_documents