from pathlib import Path
from typing import List

from aiofiles import open as aio_open
from annoy import AnnoyIndex
from loguru import logger

from ragathon.data.models import ChunkedTextSet, MatchedChunk, SearchResult
from ragathon.indexing import CorpusItem, Indexer
from ragathon.llms.common import Embedder

VECTOR_INDEX_FILE_NAME = "annoy.ann"


class VectorIndex(Indexer):
    def __init__(self, storage_dir: Path, embedder: Embedder) -> None:
        self._storage_dir: Path = storage_dir
        self._corpus: List[CorpusItem] = []
        self._embedder: Embedder = embedder

        self._embedding_size: int = self._embedder.get_embedding_size()
        self._index: AnnoyIndex = AnnoyIndex(self._embedding_size, "angular")

        self._index_path: Path = self._storage_dir / VECTOR_INDEX_FILE_NAME
        self._corpus_path: Path = self._storage_dir / "corpus.jsonl"

    async def create(self, data_set: ChunkedTextSet) -> None:
        """Create an embedding index from the given data set.

        Args:
            data_set (ChunkedTextSet): The data set to create the index from.
        """
        texts = [chunk.text for chunk in data_set.chunks]
        logger.debug(f"Embedding {len(texts)} chunks...")

        list_of_embeddings = await self._embedder.embed(texts=texts)

        for i, embedding in enumerate(list_of_embeddings):
            self._index.add_item(i, embedding)

        self._index.build(n_trees=10, n_jobs=-1)

        self._corpus = [
            CorpusItem(
                index=index,
                chunk_id=chunk.id,
                section_id=chunk.section_id,
                text=chunk.text,
            )
            for index, chunk in enumerate(data_set.chunks)
        ]

        await self._save()

    async def load(self) -> None:
        """Load the embedding index from the storage directory."""

        if not self._index_path.exists() or not self._corpus_path.exists():
            raise FileNotFoundError("Index or corpus file not found.")

        self._index.load(str(self._index_path))

        async with aio_open(self._corpus_path, "r") as f:
            while True:
                line = await f.readline()
                if not line:
                    break
                item = CorpusItem.model_validate_json(json_data=line)
                self._corpus.append(item)

    async def search(self, query: str, k: int) -> SearchResult:
        """Search for the given query.

        Args:
            query (str): The query to search for.
            k (int): The number of results to return.

        Returns:
            SearchResult: The search result.
        """

        if self._index is None:
            raise ValueError(
                "The embedding index has not been loaded. Please create or load it first"
            )

        logger.debug(f"Searching for query: {query}")
        query_embeddings = await self._embedder.embed(texts=[query])
        query_embedding = query_embeddings[0]

        indices, distances = self._index.get_nns_by_vector(
            vector=query_embedding, n=k, search_k=1000, include_distances=True
        )

        result: SearchResult = SearchResult(query=query)

        for i, (idx, distance) in enumerate(zip(indices, distances)):
            corpus_item = self._corpus[idx]
            # Convert distance to similarity score (Annoy uses angular distance)
            score = 1.0 / (1.0 + distance)
            matched_chunk = MatchedChunk(
                chunk_id=corpus_item.chunk_id,
                section_id=corpus_item.section_id,
                chunk_text=corpus_item.text,
                rank=i + 1,
                score=float(score),
            )
            result.matches.append(matched_chunk)

        return result

    async def exists(self) -> bool:
        """Check if the BM25 index exists.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        file_names = [
            "corpus.jsonl",
            VECTOR_INDEX_FILE_NAME,
        ]

        return all((self._storage_dir / file_name).exists() for file_name in file_names)

    async def _save(self) -> None:
        """Save the index and corpus to the storage directory."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the index
        self._index.save(str(self._index_path))

        # Save the corpus
        async with aio_open(self._corpus_path, "w") as f:
            for item in self._corpus:
                line = item.model_dump_json()
                await f.write(line + "\n")
