from abc import ABC, abstractmethod
from enum import StrEnum
from typing import AsyncGenerator, List, Optional, Sequence, Type, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

T = TypeVar("T")


class MessageRole(StrEnum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    role: MessageRole = Field(
        default=MessageRole.USER,
        description="The role of this message.",
    )

    content: str = Field(
        description="The content of this message.",
    )


class LLM(ABC):
    """Represents an interface to a Large Language Model (LLM)."""

    @abstractmethod
    async def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Sends a sequences of messages to the LLM and yields text fragment generator.

        Args:
            messages: A sequence of chat messages.
            temperature: The temperature to use when generating the response. Defaults to None.

        Yields:
            A generator of text fragments returned by the LLM.
        """
        raise NotImplementedError

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float] = None,
    ) -> str:
        """Sends a sequences of messages to the LLM and returns the response.

        Args:
            messages: A sequence of chat messages.
            temperature: The temperature to use when generating the response. Defaults to None.

        Returns:
            The response text returned by the LLM.
        """

        raise NotImplementedError

    @abstractmethod
    async def structured_completion(
        self,
        messages: Sequence[ChatMessage],
        response_model: Type[T],
        temperature: Optional[float] = None,
    ) -> T:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Releases any resources associated with the LLM client.

        This method should be called when the LLM client is no longer needed.
        """
        raise NotImplementedError


class Embedder(ABC):
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """Computes the embeddings of the given texts.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embedding_size(self) -> int:
        """The size of the embeddings produced by this embedder."""
        raise NotImplementedError
