from __future__ import annotations

import hashlib
import io
from typing import Any

import chromadb
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    SOURCE_XLSX_URL,
    TOP_K,
)


def _clean(value: Any) -> str:

    if pd.isna(value):
        return ""

    return str(value).strip()


def _row_to_text(row: pd.Series) -> str:

    parts = []

    for column, value in row.items():

        value = _clean(value)

        if value:
            parts.append(
                f"{column}: {value}"
            )

    return "\n".join(parts)


def _chunk_text(text: str) -> list[str]:

    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + CHUNK_SIZE,
            len(text),
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - CHUNK_OVERLAP,
            start + 1,
        )

    return chunks


def download_workbook() -> bytes:

    response = requests.get(
        SOURCE_XLSX_URL,
        timeout=30,
        headers={
            "User-Agent": "Prunely/1.0"
        },
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()

    if len(response.content) < 1000:
        raise RuntimeError(
            "The source workbook download "
            "was unexpectedly small."
        )

    if (
        "html" in content_type
        and not response.content.startswith(b"PK")
    ):
        raise RuntimeError(
            "The Google Sheet is not publicly "
            "accessible as an XLSX export."
        )

    return response.content


class KnowledgeBase:

    def __init__(self) -> None:

        self.embedding_model = (
            SentenceTransformer(
                EMBEDDING_MODEL
            )
        )

        self.client = (
            chromadb.PersistentClient(
                path=CHROMA_PATH
            )
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine"
                },
            )
        )

        self.source_hash = ""

        self._ensure_index()


    def _ensure_index(self) -> None:

        workbook = download_workbook()

        source_hash = hashlib.sha256(
            workbook
        ).hexdigest()

        self.source_hash = source_hash

        indexed_hash = None

        try:

            metadata = (
                self.collection.metadata
                or {}
            )

            indexed_hash = metadata.get(
                "source_hash"
            )

        except Exception:

            pass

        if (
            self.collection.count() > 0
            and indexed_hash == source_hash
        ):
            return

        self._rebuild(
            workbook,
            source_hash,
        )


    def _rebuild(
        self,
        workbook: bytes,
        source_hash: str,
    ) -> None:

        try:

            self.client.delete_collection(
                COLLECTION_NAME
            )

        except Exception:

            pass

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "source_hash": source_hash,
                    "source_url": SOURCE_XLSX_URL,
                    "embedding_model": EMBEDDING_MODEL,
                },
            )
        )

        sheets = pd.read_excel(
            io.BytesIO(workbook),
            sheet_name=None,
            engine="openpyxl",
        )

        documents = []
        metadatas = []
        ids = []

        for sheet_name, frame in sheets.items():

            frame = (
                frame
                .dropna(how="all")
                .reset_index(drop=True)
            )

            for row_idx, row in frame.iterrows():

                row_text = _row_to_text(row)

                if not row_text:
                    continue

                chunks = _chunk_text(
                    row_text
                )

                plant_id = (
                    _clean(
                        row.get(
                            "plant_id",
                            "",
                        )
                    )
                    if hasattr(row, "get")
                    else ""
                )

                for chunk_idx, chunk in enumerate(
                    chunks
                ):

                    record_key = (
                        f"{source_hash}:"
                        f"{sheet_name}:"
                        f"{row_idx + 2}:"
                        f"{chunk_idx}:"
                        f"{chunk}"
                    )

                    doc_id = hashlib.sha256(
                        record_key.encode(
                            "utf-8"
                        )
                    ).hexdigest()

                    documents.append(chunk)

                    ids.append(doc_id)

                    metadatas.append(
                        {
                            "document_id":
                                source_hash[:16],

                            "source_file":
                                "Prunely knowledge workbook",

                            "source_type":
                                "public_google_sheet_xlsx",

                            "source_url":
                                SOURCE_XLSX_URL,

                            "sheet_name":
                                str(sheet_name),

                            "row_number":
                                int(row_idx + 2),

                            "plant_id":
                                plant_id,

                            "chunk_number":
                                int(chunk_idx),
                        }
                    )

        if not documents:

            raise RuntimeError(
                "No usable rows were found "
                "in the source workbook."
            )

        embeddings = (
            self.embedding_model.encode(
                documents,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
        )

        batch_size = 100

        for start in range(
            0,
            len(documents),
            batch_size,
        ):

            end = start + batch_size

            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
            )


    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[dict[str, Any]]:

        query = query.strip()

        if not query:
            return []

        embedding = (
            self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].tolist()
        )

        result = (
            self.collection.query(
                query_embeddings=[embedding],
                n_results=max(
                    1,
                    min(
                        top_k,
                        self.collection.count(),
                    ),
                ),
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )
        )

        documents = result.get(
            "documents",
            [[]],
        )[0]

        metadatas = result.get(
            "metadatas",
            [[]],
        )[0]

        distances = result.get(
            "distances",
            [[]],
        )[0]

        hits = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):

            hits.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "distance": float(distance),
                }
            )

        return hits


    def format_search_results(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> str:

        hits = self.search(
            query,
            top_k=top_k,
        )

        if not hits:
            return (
                "No matching source records were found."
            )

        lines = [
            f"Knowledge-base search for: {query}"
        ]

        for i, hit in enumerate(
            hits,
            start=1,
        ):

            meta = hit["metadata"]

            lines.append(
                f"\n[SOURCE {i}] "
                f"sheet={meta.get('sheet_name')} | "
                f"row={meta.get('row_number')} | "
                f"plant_id={meta.get('plant_id')} | "
                f"distance={hit['distance']:.4f}"
            )

            lines.append(
                hit["text"]
            )

        return "\n".join(lines)
