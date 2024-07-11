from datetime import datetime
from typing import AsyncGenerator, List, Optional, Sequence, Type, TypeVar, cast

import instructor
import numpy as np
from azure.core.credentials import AccessToken
from azure.identity.aio import ClientSecretCredential
from loguru import logger
from numpy.typing import NDArray
from openai import AsyncAzureOpenAI, AsyncStream
from openai.lib.azure import AsyncAzureADTokenProvider
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.create_embedding_response import CreateEmbeddingResponse

from ragathon.config import Settings
from ragathon.llms.common import LLM, ChatMessage, Embedder

T = TypeVar("T")


class AzureEntraIDClientCredentialsTokenGenerator:
    """A token generator using Azure Entra ID via OAuth2 Client Credentials Flow.

    This class is a callable that generates an access token using the Azure OAuth2
    client credentials flow. The token is cached and refreshed every 25 minutes
    by default.

    Notice that this class follows the structure defined in the
    `AsyncAzureADTokenProvider` specification.
    """

    def __init__(
        self,
        credentials: ClientSecretCredential,
        refresh_token_in_secs: int,
        scope: str = "https://cognitiveservices.azure.com/.default",
    ) -> None:
        """Initialize the token generator.

        Args:
            tenant_id: The tenant of the Azure Entra ID.
            client_id: The client ID of the Azure Application Registration.
            client_secret: The client secret of the Azure Application Registration.
            scope: The scope of the token. Defaults to "https://cognitiveservices.azure.com/.default".
            refresh_token_in_secs: The time in seconds before the token is refreshed.
        """

        self._credentials = credentials
        self._current_token: Optional[AccessToken] = None
        self._refresh_at: Optional[int] = None
        self._refresh_token_in_secs = refresh_token_in_secs
        self._scope = scope

    async def __call__(self) -> str:
        """Generate a token."""
        now_unix_time = self.now_unix_time
        if self._current_token is None:
            await self._generate_token()
        elif self._refresh_at is not None and (self._refresh_at - now_unix_time) <= 0:
            await self._generate_token()
        if self._current_token is None:
            raise ValueError("Token is not available")

        return self._current_token.token

    @property
    def now_unix_time(self) -> int:
        return int(datetime.now().timestamp())

    async def _generate_token(self) -> None:
        now_unix_time = self.now_unix_time
        self._current_token = await self._credentials.get_token(self._scope)
        self._refresh_at = now_unix_time + self._refresh_token_in_secs


class AzureOpenAIBasedLLM(LLM):
    """A Large Language Model that uses Azure OpenAI.

    Notice that this class follows the structure defined in the `LLM` protocol.
    """

    def __init__(
        self,
        api_base_endpoint: str,
        api_version: str,
        token_generator: AsyncAzureADTokenProvider,  # pyre-ignore[11] see https://github.com/facebook/pyre-check/issues/232
        deployment_id: str,
        model_name: str,
        default_temperature: float,
    ) -> None:
        """Initialize the LLM.

        Args:
            api_base_endpoint: The base endpoint of the Azure OpenAI API.
            api_version: The version of the Azure OpenAI API.
            token_generator: The token generator to use for authentication.
            deployment_id: The deployment ID of the Azure OpenAI model.
            model_name: The name of the Azure OpenAI model.
            default_temperature: The default temperature to use when generating the response.
        """

        self._model_name = model_name
        self._default_temperature = default_temperature
        self._client = AsyncAzureOpenAI(
            azure_deployment=deployment_id,
            azure_endpoint=api_base_endpoint,
            api_version=api_version,
            azure_ad_token_provider=token_generator,
        )

    async def chat_stream(  # pyre-fixme[15]
        self, messages: Sequence[ChatMessage], temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        transformed_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]

        logger.debug(f"Sending messages to LLM:\n {transformed_messages}")

        current_temperature = (
            temperature if temperature is not None else self._default_temperature
        )

        chunk_stream: AsyncStream[
            ChatCompletionChunk
        ] = await self._client.chat.completions.create(
            model=self._model_name,
            messages=transformed_messages,
            temperature=current_temperature,
            stream=True,
        )

        async for chunk in chunk_stream:
            chunk_text = await self._get_chunk_text_from_openai_api(chunk=chunk)
            if chunk_text is not None:
                yield chunk_text

    async def chat(
        self, messages: Sequence[ChatMessage], temperature: Optional[float] = None
    ) -> str:
        transformed_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]

        logger.debug(f"Sending messages to LLM:\n {transformed_messages}")

        current_temperature = (
            temperature if temperature is not None else self._default_temperature
        )

        completion: ChatCompletion = await self._client.chat.completions.create(
            model=self._model_name,
            messages=transformed_messages,
            temperature=current_temperature,
            stream=False,
        )

        response_text = completion.choices[0].message.content
        assert response_text is not None
        return response_text

    async def structured_completion(
        self,
        messages: Sequence[ChatMessage],
        response_model: Type[T],
        temperature: Optional[float] = None,
    ) -> T:
        current_temperature = (
            temperature if temperature is not None else self._default_temperature
        )

        instructor_client = instructor.from_openai(
            client=self._client, mode=instructor.Mode.TOOLS
        )

        transformed_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]

        logger.debug(f"Sending messages to LLM:\n {transformed_messages}")

        response = await instructor_client.chat.completions.create(
            model=self._model_name,
            messages=transformed_messages,  # pyre-ignore[6]
            temperature=current_temperature,
            stream=False,
            response_model=response_model,  # pyre-ignore[6]
        )

        return response

    async def close(self) -> None:
        await self._client.close()

    async def _get_chunk_text_from_openai_api(
        self, chunk: ChatCompletionChunk
    ) -> Optional[str]:
        """Get the delta output from the LLM."""
        if chunk.choices and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is not None:
                return delta.content
        return None


class AzureOpenAIBasedEmbedder(Embedder):
    """An Embedder that uses Azure OpenAI."""

    def __init__(
        self,
        api_base_endpoint: str,
        api_version: str,
        token_generator: AsyncAzureADTokenProvider,
        deployment_id: str,
        model_name: str,
        embedding_size: int,
    ) -> None:
        self._model_name = model_name
        self._client = AsyncAzureOpenAI(
            azure_deployment=deployment_id,
            azure_endpoint=api_base_endpoint,
            api_version=api_version,
            azure_ad_token_provider=token_generator,
        )
        self._embedding_size = embedding_size

    async def embed(self, texts: List[str]) -> List[NDArray[np.float32]]:
        model_result: CreateEmbeddingResponse = await self._client.embeddings.create(
            model=self._model_name, input=texts, dimensions=self._embedding_size
        )

        return [
            np.array(embedding.embedding, dtype=np.float32)
            for embedding in model_result.data
        ]

    def get_embedding_size(self) -> int:
        return self._embedding_size


def create_token_generator(
    settings: Settings,
) -> AzureEntraIDClientCredentialsTokenGenerator:
    client_secret_credential = ClientSecretCredential(
        tenant_id=settings.AZURE_IDENTITY_TENANT_ID,
        client_id=settings.AZURE_IDENTITY_CLIENT_ID,
        client_secret=settings.AZURE_IDENTITY_CLIENT_SECRET,
    )

    token_generator = AzureEntraIDClientCredentialsTokenGenerator(
        credentials=client_secret_credential,
        refresh_token_in_secs=60 * 10,  # TODO: Make this configurable?
    )

    return token_generator


def instantiate_llm_for_rag(settings: Settings) -> LLM:
    token_generator = create_token_generator(settings=settings)

    llm_rag_azure_open_ai = AzureOpenAIBasedLLM(
        api_base_endpoint=settings.AZURE_OPEN_AI_API_ENDPOINT,
        api_version=settings.AZURE_OPEN_AI_API_VERSION,
        token_generator=token_generator,
        deployment_id=settings.AZURE_OPEN_AI_LLM_RAG_DEPLOYMENT_NAME,
        model_name=settings.AZURE_OPEN_AI_LLM_RAG_MODEL_NAME,
        default_temperature=0.1,  # TODO: Make this configurable?
    )

    llm_rag: LLM = cast(LLM, llm_rag_azure_open_ai)

    return llm_rag


def instantiate_embedder(settings: Settings) -> Embedder:
    token_generator = create_token_generator(settings=settings)

    embedder = AzureOpenAIBasedEmbedder(
        api_base_endpoint=settings.AZURE_OPEN_AI_API_ENDPOINT,
        api_version=settings.AZURE_OPEN_AI_API_VERSION,
        token_generator=token_generator,
        deployment_id=settings.AZURE_OPEN_AI_EMBEDDING_DEPLOYMENT_NAME,
        model_name=settings.AZURE_OPEN_AI_EMBEDDING_MODEL_NAME,
        embedding_size=3072,  # TODO: Make this configurable?
    )

    return embedder
