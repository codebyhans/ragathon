from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from ragathon.data.models import ChunkedTextSet, SearchResult


class CorpusItem(BaseModel):
    index: int = Field(..., description="Index of the document.")
    """Index of the document."""

    chunk_id: str = Field(..., description="ID of the chunk.")
    """ID of the chunk."""

    section_id: str = Field(..., description="ID of the section.")
    """ID of the section."""

    text: str = Field(..., description="Text of the chunk.")
    """Text of the chunk."""


class Indexer(ABC):
    """Abstract class for indexing models."""

    @abstractmethod
    async def create(self, data_set: ChunkedTextSet) -> None:
        """Create the index from the given data set.

        Args:
            data_set (ChunkedTextSet): The data set to create the index from.
        """
        raise NotImplementedError

    @abstractmethod
    async def load(self) -> None:
        """Load the index from the storage directory."""
        raise NotImplementedError

    @abstractmethod
    async def search(self, query: str, k: int) -> SearchResult:
        """Search for the given query.

        Args:
            query (str): The query to search for.
            k (int): The number of results to return.

        Returns:
            SearchResult: The search result.
        """
        raise NotImplementedError

    @abstractmethod
    async def exists(self) -> bool:
        """Check if the index exists.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        raise NotImplementedError
