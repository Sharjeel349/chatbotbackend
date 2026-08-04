"""Index Markdown knowledge files into the local Chroma collection."""

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CHROMA_DIR = BASE_DIR / "chroma_db"


def load_documents():
    documents = []
    ids = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        sections = [section.strip() for section in text.split("\n## ") if section.strip()]
        for index, section in enumerate(sections):
            content = section if index == 0 else f"## {section}"
            source = str(path.relative_to(BASE_DIR))
            documents.append(
                Document(page_content=content, metadata={"source": source})
            )
            ids.append(
                hashlib.sha256(f"{source}:{index}".encode("utf-8")).hexdigest()
            )
    return documents, ids


def main():
    documents, ids = load_documents()
    if not documents:
        raise SystemExit("No Markdown files found in FYP2/knowledge")

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    )
    vectorstore.add_documents(documents=documents, ids=ids)
    print(f"Indexed {len(documents)} university knowledge sections.")


if __name__ == "__main__":
    main()
