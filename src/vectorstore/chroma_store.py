import chromadb
from chromadb.config import Settings
from src.vectorstore.base import BaseVectorStore
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "textbook_chunks"

# All metadata fields stored per chunk
_META_FIELDS = ["source", "chapter", "section", "subsection",
                "page_number", "heading_path", "chunk_type", "token_count",
                "chunk_index"]


class ChromaStore(BaseVectorStore):

    def __init__(self, persist_path: str = None):
        config = get_config()
        persist_path = persist_path or config["vectorstore"]["persist_path"]

        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaStore ready at '{persist_path}' — {self.count()} vectors")

    # ─── Write ────────────────────────────────────────────────────────────────

    def add(self, chunks: list[dict]) -> None:
        """
        Upsert enriched textbook chunks into ChromaDB.

        Expected fields per chunk:
          chunk_id, text, embedding,
          source, chapter, section, subsection,
          page_number, heading_path, chunk_type, token_count
        """
        if not chunks:
            logger.warning("No chunks to add")
            return

        ids        = [c["chunk_id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        documents  = [c["text"] for c in chunks]
        metadatas  = [
            {
                "source":       c.get("source", ""),
                "chapter":      c.get("chapter", ""),
                "section":      c.get("section", ""),
                "subsection":   c.get("subsection", ""),
                "page_number":  int(c.get("page_number", 0)),
                "heading_path": c.get("heading_path", ""),
                "chunk_type":   c.get("chunk_type", "text"),
                "token_count":  int(c.get("token_count", 0)),
                "chunk_index":  int(c.get("chunk_index", 0)),
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Upserted {len(chunks)} chunks into ChromaDB")

    # ─── Read ─────────────────────────────────────────────────────────────────

    def search(self, query_embedding: list[float], top_k: int = None,
               filters: dict = None) -> list[dict]:
        """
        Similarity search with optional metadata filters.

        filters example: {"source": "calculus.pdf", "chapter": "Chapter 3"}

        Returns list of:
          { text, source, chapter, section, subsection, page_number,
            heading_path, chunk_type, token_count, score }
        """
        config = get_config()
        top_k  = top_k or config["retrieval"]["top_k"]

        where = None
        if filters:
            conditions = [{k: {"$eq": v}} for k, v in filters.items()]
            where = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            hits.append({
                "text":         doc,
                "source":       meta.get("source", ""),
                "chapter":      meta.get("chapter", ""),
                "section":      meta.get("section", ""),
                "subsection":   meta.get("subsection", ""),
                "page_number":  meta.get("page_number", 0),
                "heading_path": meta.get("heading_path", ""),
                "chunk_type":   meta.get("chunk_type", "text"),
                "token_count":  meta.get("token_count", 0),
                "chunk_index":  meta.get("chunk_index", 0),
                "score":        round(1 - dist, 4)
            })

        logger.info(f"Retrieved {len(hits)} chunks")
        return hits

    def search_by_type(self, query_embedding: list[float],
                       chunk_type: str, top_k: int = 5) -> list[dict]:
        """Priority search for specific content types (definition, formula, table, figure)."""
        return self.search(query_embedding, top_k=top_k,
                           filters={"chunk_type": chunk_type})

    def get_by_source(self, source: str) -> list[dict]:
        """
        Fetch every stored chunk for one document, ordered by chunk_index
        (document order). No embedding/similarity involved — this is a
        direct metadata lookup, used to reconstruct a full section around
        a similarity-search hit (see src.retrieval.context_expander).

        Returns list of:
          { text, source, chapter, section, subsection, page_number,
            heading_path, chunk_type, token_count, chunk_index }
        (no "score" — these aren't ranked by similarity)
        """
        results = self.collection.get(
            where={"source": {"$eq": source}},
            include=["documents", "metadatas"]
        )

        rows = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            rows.append({
                "text":         doc,
                "source":       meta.get("source", ""),
                "chapter":      meta.get("chapter", ""),
                "section":      meta.get("section", ""),
                "subsection":   meta.get("subsection", ""),
                "page_number":  meta.get("page_number", 0),
                "heading_path": meta.get("heading_path", ""),
                "chunk_type":   meta.get("chunk_type", "text"),
                "token_count":  meta.get("token_count", 0),
                "chunk_index":  meta.get("chunk_index", 0),
            })

        rows.sort(key=lambda r: r["chunk_index"])
        return rows

    # ─── Utils ────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB collection reset")
