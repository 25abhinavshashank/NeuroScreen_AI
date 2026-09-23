
from __future__ import annotations

from typing import List

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from backend.app.rag.ingestion.loader import (
    LoadedDocument,
)


CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


def build_splitter():

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


def chunk_documents(
    documents: List[LoadedDocument],
) -> List[dict]:

    splitter = build_splitter()

    chunks = []

    global_chunk_id = 0

    for document in documents:

        split_texts = splitter.split_text(
            document.text
        )

        for local_index, chunk_text in enumerate(
            split_texts
        ):

            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            chunks.append(
                {
                    "chunk_id": (
                        f"chunk_{global_chunk_id}"
                    ),
                    "source": document.source,
                    "file_name": document.file_name,
                    "file_type": document.file_type,
                    "chunk_index": local_index,
                    "text": chunk_text,
                }
            )

            global_chunk_id += 1

    return chunks