from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CHROMA_DIR = BASE_DIR / "chroma_db"
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

_RETRIEVAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "have", "has", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or",
    "the", "to", "what", "when", "where", "which", "who", "why", "will", "with", "you",
    "kya", "hai", "hain", "ka", "ki", "ke", "ko", "se", "me", "mein", "par",
}

EXPANSIONS = {
    "سمسٹر": "semester",
    "فریز": "freeze",
    "کورس": "course",
    "سبجیکٹ": "course",
    "پروگرامنگ": "programming",
    "استاد": "teacher",
    "ٹیچر": "teacher",
    "گریڈ": "grade",
    "نتیجہ": "result",
    "سی جی پی اے": "cgpa",
    "حاضری": "attendance",
    "امتحان": "exam examination",
    "داخلہ": "admission eligibility",
    "فیس": "fee installment",
    "قسط": "installment fee",
    "نقل": "cheating penalty",
    "hazri": "attendance",
    "haziri": "attendance",
    "chutti": "leave attendance",
    "dakhla": "admission eligibility",
    "qist": "installment fee",
    "qiston": "installment fee",
    "kist": "installment fee",
    "imtihan": "exam examination",
    "paper": "exam examination",
    "naqal": "cheating penalty",
    "cheat": "cheating",
    "for ease": "freeze",
    "forease": "freeze",
    "freeze": "freeze",
    "freezing": "freeze",
    "gpas": "gpa cgpa",
    "cgpa": "cgpa gpa",
    "seventh": "semester 7",
    "sixth": "semester 6",
    "fifth": "semester 5",
    "fourth": "semester 4",
    "third": "semester 3",
    "second": "semester 2",
    "eighth": "semester 8",
    "first": "semester 1",
    "saatwan": "semester 7",
    "chattha": "semester 6",
    "panchwan": "semester 5",
    "chotha": "semester 4",
    "teesra": "semester 3",
    "doosra": "semester 2",
    "promote": "promotion progression",
    "promotion": "promotion progression",
    "drop": "course drop withdrawal",
    "dropped": "course drop withdrawal",
    "withdraw": "course drop withdrawal",
    "fail": "failed drop repeat course grade f",
    "failed": "fail drop repeat course grade f",
    "failing": "fail drop repeat course grade f",
    "فیل": "fail drop repeat course grade f",
    "repeat": "repeat course failed cease",
    "repeating": "repeat course failed cease",
}


@lru_cache(maxsize=1)
def get_knowledge_sections() -> list[str]:
    sections: list[str] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        sections.extend(
            section.strip()
            for section in re.split(r"\n(?=## )", text)
            if section.strip()
        )
    return sections


def _get_local_embedding(text: str) -> list[float] | None:
    try:
        res = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=3,
        )
        if res.status_code == 200:
            return res.json().get("embedding")
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def get_chroma_collection():
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_or_create_collection("biit_policies")
        if col.count() == 0:
            sections = get_knowledge_sections()
            docs, ids, metadatas, embeddings = [], [], [], []
            for i, sec in enumerate(sections):
                heading = sec.splitlines()[0].replace("#", "").strip()
                emb = _get_local_embedding(sec)
                if emb:
                    docs.append(sec)
                    ids.append(f"sec_{i}")
                    metadatas.append({"heading": heading})
                    embeddings.append(emb)
            if docs:
                col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
                logger.info("Indexed %d policy sections into persistent ChromaDB", len(docs))
        return col
    except Exception:
        logger.exception("ChromaDB initialization failed, falling back to lexical retrieval")
        return None


def lexical_search(query: str) -> str:
    normalized = query.casefold()
    for source, replacement in EXPANSIONS.items():
        if source in normalized:
            normalized += f" {replacement}"

    raw_query_terms = set(re.findall(r"[a-z0-9]+", normalized))
    query_terms = (raw_query_terms - _RETRIEVAL_STOPWORDS) or raw_query_terms
    if not query_terms:
        return ""

    ranked: list[tuple[int, str]] = []
    for section in get_knowledge_sections():
        section_lower = section.casefold()
        section_terms = set(re.findall(r"[a-z0-9]+", section_lower))
        heading = section.splitlines()[0].casefold() if section else ""
        heading_terms = set(re.findall(r"[a-z0-9]+", heading))
        score = len(query_terms & section_terms) + 4 * len(query_terms & heading_terms)
        for term in query_terms:
            if term in heading:
                score += 5
        if score:
            ranked.append((score, section))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return ""
    return ranked[0][1][:1500].strip()


def retrieve_policy_knowledge(query: str) -> str:
    """Retrieve policy knowledge using offline Hybrid Search (ChromaDB + Lexical fallback)."""
    # 1. Try dense vector search using local ChromaDB
    try:
        col = get_chroma_collection()
        if col is not None and col.count() > 0:
            q_emb = _get_local_embedding(query)
            if q_emb:
                results = col.query(query_embeddings=[q_emb], n_results=2)
                docs = results.get("documents", [[]])[0]
                if docs:
                    combined = "\n\n".join(docs)
                    return combined[:1500].strip()
    except Exception:
        logger.exception("Vector retrieval encountered an issue, defaulting to lexical search")

    # 2. Resilient lexical retrieval
    return lexical_search(query)
