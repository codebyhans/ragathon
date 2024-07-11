import asyncio
from pathlib import Path
from typing import List

from aiofiles import open as aio_open
from aiofiles.os import makedirs as aio_makedirs
from loguru import logger
from ragathon.chunking.token import NaiveTokenBasedTextSplitter
from ragathon.config import Settings, init_settings
from ragathon.data.models import (
    ChunkedText,
    ChunkedTextSet,
    ChunkingMethod,
    MarkdownDocument,
    SearchResult,
    SyntheticQuestionSet,
)
from ragathon.indexing.bm25 import BM25Index
from ragathon.indexing.vector import VectorIndex

from ragathon.llms import LLM
from ragathon.llms.azure import instantiate_llm_for_rag, instantiate_embedder
from ragathon.llms.common import ChatMessage, MessageRole
from ragathon.pipelines.common import (
    GenerationMetrics,
    QueryInfo,
    RAGPipeline,
    RAGPipelineConfig,
    RAGPipelineOutput,
    RetrievalMetrics,
    RetrievedChunk,
)

from ragathon.llms.azure import AzureOpenAIBasedEmbedder
from ragathon.utils.date import utcnow


class NaiveChunkingVectorPipeline(RAGPipeline):
    def __init__(self, markdown_file_path: Path, output_dir: Path, embedder: str):
        self._config = RAGPipelineConfig(
            version="1",
            chunking_method="naive",
            max_chunk_size=128,
            chunk_overlap=50,
            retrieval_method="vector",
            generation_model_name="gpt4o",
        )

        self._output_dir = self._config.get_dir(parent_dir=output_dir)

        self._markdown_file_path = markdown_file_path

        self._index_storage_dir = self._output_dir / "indices" / markdown_file_path.stem
        app_settings: Settings = init_settings()
        embedder = instantiate_embedder(settings=app_settings)
        self._index = VectorIndex(storage_dir=self._index_storage_dir,embedder=embedder)

        chunked_file_name = (
            f"{markdown_file_path.stem}-chunked-{self._config.chunking_method}.json"
        )
        self._chunked_file_path = self._output_dir / chunked_file_name
        self._llm: LLM = instantiate_llm_for_rag(settings=app_settings)

    async def build_or_load(self) -> None:
        await self._store_config()

        if not self._index_storage_dir.exists():
            logger.info(
                f"Index not found at at {self._index_storage_dir}. Rebuilding index..."
            )
            doc: MarkdownDocument = await self._load_document()
            chunked_set: ChunkedTextSet = await self._load_or_create_chunked_set(
                doc=doc
            )
            await self._index.create(data_set=chunked_set)

        await self._index.load()

    async def run(self, query: str, max_retrieve_docs: int) -> RAGPipelineOutput:
        output = RAGPipelineOutput(
            config=self._config,
            query_info=QueryInfo(original_query=query),
            retrieved_chunks=[],
            generated_answer="",
            retrieval_metrics=RetrievalMetrics(
                retrieval_method=self._config.retrieval_method,
                max_retrieve_docs=max_retrieve_docs,
                total_chunks_retrieved=0,
                retrieval_time_ms=0,
            ),
            generation_metrics=GenerationMetrics(
                llm_name=self._config.generation_model_name,
                input_token_count=0,
                output_token_count=0,
                generation_time_ms=0,
            ),
            started_at=utcnow(),
            completed_at=None,
        )

        # Run retrieval part
        await self._run_retrieval_part(
            query=query, max_retrieve_docs=max_retrieve_docs, output=output
        )

        # Run generation part
        await self._run_generation_part(query=query, output=output)

        output.completed_at = utcnow()

        return output

    async def _run_retrieval_part(
        self, query: str, max_retrieve_docs: int, output: RAGPipelineOutput
    ) -> None:
        retrieval_start = utcnow()
        search_result: SearchResult = await self._index.search(
            query=query, k=max_retrieve_docs
        )
        retrieval_end = utcnow()

        # Update retrieval metrics
        output.retrieval_metrics.total_chunks_retrieved = len(search_result.matches)
        output.retrieval_metrics.retrieval_time_ms = int(
            (retrieval_end - retrieval_start).total_seconds() * 1000
        )

        # Update retrieved chunks
        output.retrieved_chunks = [
            RetrievedChunk(
                chunk_id=match.chunk_id,
                section_id=match.section_id,
                document_id=None,
                text=match.chunk_text,
                rank=match.rank,
                score=match.score,
            )
            for match in search_result.matches
        ]

    async def _run_generation_part(self, query: str, output: RAGPipelineOutput) -> str:
        if len(output.retrieved_chunks) == 0:
            raise ValueError("No retrieved chunks available for generation")

        generation_start = utcnow()
        output.generated_answer = await self._generate_answer(
            query=query, retrieved_chunks=output.retrieved_chunks
        )
        generation_end = utcnow()

        # Update generation metrics
        output.generation_metrics.input_token_count = 0
        output.generation_metrics.output_token_count = 0
        output.generation_metrics.generation_time_ms = int(
            (generation_end - generation_start).total_seconds() * 1000
        )

    async def _generate_answer(
        self, query: str, retrieved_chunks: List[RetrievedChunk]
    ) -> str:
        # Generate answer using LLM
        context = "\n---\n".join([chunk.text for chunk in retrieved_chunks])
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant. Answer the question based on the provided context. You will answer questions about GPDR",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"## Context:\n{context}\n\n## Question:\n{query}",
            ),
        ]

        return await self._llm.chat(messages=messages)

    async def _load_document(self) -> MarkdownDocument:
        async with aio_open(self._markdown_file_path, mode="r") as file:
            content = await file.read()
            return MarkdownDocument.deserialize_from_json(json_data=content)

    async def _load_or_create_chunked_set(
        self, doc: MarkdownDocument
    ) -> ChunkedTextSet:
        if self._chunked_file_path.exists():
            logger.info(
                f"Chunked file {self._chunked_file_path} already exists. Loading..."
            )
            async with aio_open(self._chunked_file_path, mode="r") as file:
                content = await file.read()
                return ChunkedTextSet.model_validate_json(json_data=content)

        logger.info(f"Chunked file {self._chunked_file_path} not found. Creating...")

        splitter = NaiveTokenBasedTextSplitter(
            max_token_size=self._config.max_chunk_size,
            overlap=self._config.chunk_overlap,
        )

        output_data = ChunkedTextSet(chunking_method=ChunkingMethod.NAIVE)
        for section in doc.sections:
            raw_chunks = splitter.split_text(text=section.text)
            for raw_chunk in raw_chunks:
                chunked_text = ChunkedText(section_id=section.id, text=raw_chunk)
                output_data.chunks.append(chunked_text)

        logger.info(
            f"Saving {len(output_data.chunks)} chunks to {self._chunked_file_path}"
        )
        async with aio_open(self._chunked_file_path, mode="w") as file:
            await file.write(output_data.model_dump_json(indent=2))

        return output_data

    async def _store_config(self) -> None:
        config_file_path = self._output_dir / "config.json"
        await aio_makedirs(name=config_file_path.parent, exist_ok=True)
        async with aio_open(config_file_path, mode="w") as file:
            await file.write(self._config.model_dump_json(indent=2))


# Usage example:
async def main():
    pipeline = NaiveChunkingVectorPipeline(
        markdown_file_path=Path("data/gdpr-handbook/processed/handbook-cleaned.json"),
         embedder="data/gdpr-handbook/processed/handbook-cleaned-questions.json" ,
        output_dir=Path("data/pipelines"),
    )
    await pipeline.build_or_load()

    # Load the questions
    questions_file_path = Path(
        "data/gdpr-handbook/processed/handbook-cleaned-questions.json"
    )
    async with aio_open(questions_file_path, mode="r") as file:
        content = await file.read()
        q_set = SyntheticQuestionSet.model_validate_json(json_data=content)

    pipeline_output_file = pipeline._output_dir / "result.jsonl"
    pipeline_result: List[RAGPipelineOutput] = []

    if pipeline_output_file.exists():
        async with aio_open(pipeline_output_file, mode="r") as file:
            async for line in file:
                pipeline_result.append(
                    RAGPipelineOutput.model_validate_json(json_data=line)
                )

    async def save_pipeline_result():
        async with aio_open(pipeline_output_file, mode="w") as file:
            for result in pipeline_result:
                await file.write(result.model_dump_json() + "\n")

    for question in q_set.questions:
        if any(
            [
                result.query_info.original_query == question.question
                for result in pipeline_result
            ]
        ):
            logger.info(f"Skipping question: {question.question}")
            continue

        result = await pipeline.run(query=question.question, max_retrieve_docs=5)
        pipeline_result.append(result)
        await save_pipeline_result()


if __name__ == "__main__":
    asyncio.run(main())
