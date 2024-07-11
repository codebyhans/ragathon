from pathlib import Path
from typing import List, Optional

import bm25s
import nltk
from loguru import logger
from nltk.stem import SnowballStemmer

from ragathon.data.models import ChunkedTextSet, MatchedChunk, SearchResult
from ragathon.indexing import CorpusItem, Indexer

BM25_INDEX_FILE_NAME = "params.index.json"


class BM25Index(Indexer):
    def __init__(self, storage_dir: Path, language: str) -> None:
        self._storage_dir = storage_dir
        self._language = language
        self._model: Optional[bm25s.BM25] = None
        self._stemmer = SnowballStemmer(language=language)
        self._stop_words: List[str] = nltk.corpus.stopwords.words(language)

    async def create(self, data_set: ChunkedTextSet) -> None:
        """Create a BM25 index from the given data set.

        Args:
            data_set (ChunkedTextSet): The data set to create the index from.
        """
        texts = [chunk.text for chunk in data_set.chunks]

        texts_tokens = bm25s.tokenize(
            texts=texts, stopwords=self._stop_words, stemmer=self._stem
        )

        self._model = bm25s.BM25()
        self._model.index(texts_tokens)

        # Build the corpus that will be saved with the model
        corpus = [
            CorpusItem(
                index=index,
                chunk_id=chunk.id,
                section_id=chunk.section_id,
                text=chunk.text,
            ).model_dump()
            for index, chunk in enumerate(data_set.chunks)
        ]

        assert self._model is not None
        self._model.save(save_dir=self._storage_dir, corpus=corpus)

    async def load(self) -> None:
        """Load the BM25 model from the storage directory."""
        self._model = bm25s.BM25.load(save_dir=self._storage_dir, load_corpus=True)

    async def search(self, query: str, k: int) -> SearchResult:
        """Search for the given query.

        Args:
            query (str): The query to search for.
            k (int): The number of results to return.

        Returns:
            SearchResult: The search result.
        """

        if self._model is None:
            raise ValueError(
                "The BM25 model has not been loaded. Please create or load it first"
            )

        logger.debug(f"Searching for query: {query}")
        query_tokens = bm25s.tokenize(
            texts=query, stopwords=self._stop_words, stemmer=self._stem
        )

        # Perform the search
        assert self._model is not None
        results, scores = self._model.retrieve(
            query_tokens=query_tokens,  # pyre-ignore[6]
            k=k,
            show_progress=True,
        )

        # Build the search result
        result: SearchResult = SearchResult(query=query)

        for i in range(results.shape[1]):
            corpus_item = CorpusItem(**results[0, i])
            score = scores[0, i]

            matched_chunk = MatchedChunk(
                chunk_id=corpus_item.chunk_id,
                section_id=corpus_item.section_id,
                chunk_text=corpus_item.text,
                rank=i + 1,
                score=score,
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
            "corpus.mmindex.json",
            "data.csc.index.npy",
            "indices.csc.index.npy",
            "indptr.csc.index.npy",
            "params.index.json",
            "vocab.index.json",
        ]

        return all((self._storage_dir / file_name).exists() for file_name in file_names)

    def _stem(self, tokens: List[str]) -> List[str]:
        return [
            self._stemmer.stem(token)
            for token in tokens
            if token not in self._stop_words
        ]
