from pathlib import Path
from typing import Dict, List

import anyio
from aiofiles import open as aio_open
from loguru import logger

from ragathon.config import Settings, init_settings
from ragathon.data.models import (
    QueryRetrievalMetric,
    SyntheticQuestionSet,
)
from ragathon.llms.azure import instantiate_llm_for_rag
from ragathon.llms.common import LLM
from ragathon.pipelines.common import RAGPipelineOutput
from ragathon.utils.strings import generate_id


class PipelineEvaluator:
    def __init__(self, pipeline_results_path: Path, questions_file_path: Path) -> None:
        self._pipeline_result_path: Path = pipeline_results_path
        self._questions_file_path: Path = questions_file_path

        app_settings: Settings = init_settings()
        self._llm: LLM = instantiate_llm_for_rag(settings=app_settings)

    async def run(self) -> None:
        outputs = await self._load_outputs()
        questions = await self._load_questions()

        question_to_output = {out.query_info.original_query: out for out in outputs}

        logger.debug(
            f"Loaded {len(outputs)} outputs and {len(questions.questions)} questions."
        )

        # Compute the reciprocal rank metric for each questions
        rr_metrics = self._compute_reciprocal_rank(
            q_set=questions,
            question_to_output=question_to_output,
            k=1,
        )
        mean_reciprocal_rank = sum([metric.value for metric in rr_metrics]) / len(
            rr_metrics
        )

        logger.debug(f"Mean Reciprocal Rank: {mean_reciprocal_rank}")

    def _compute_reciprocal_rank(
        self,
        q_set: SyntheticQuestionSet,
        question_to_output: Dict[str, RAGPipelineOutput],
        k: int,
    ) -> List[QueryRetrievalMetric]:
        metrics: List[QueryRetrievalMetric] = []

        for item in q_set.questions:
            metric = QueryRetrievalMetric(
                query_id=generate_id(parts=[item.question]),
                query=item.question,
                measure_name="Reciprocal Rank",
                k=k,
                value=0,
            )
            metrics.append(metric)

            if item.question in question_to_output:
                output = question_to_output[item.question]

                found_phrase = False

                # Find all relevant chunks for the given question
                for phrase in item.phrases:
                    for chunk in output.retrieved_chunks[:k]:
                        if phrase in chunk.text:
                            metric.value = 1.0 / chunk.rank
                            found_phrase = True
                            break
                    if found_phrase:
                        break

            else:
                # Reciprocal rank is 0 if no output is found because 1/infinity = 0
                metric.value = 0.0
                logger.warning(f"No output found for question: {item.question}")

        return metrics

    async def _load_outputs(self) -> List[RAGPipelineOutput]:
        outputs: List[RAGPipelineOutput] = []

        async with aio_open(self._pipeline_result_path, mode="r") as f:
            async for line in f:
                outputs.append(RAGPipelineOutput.model_validate_json(json_data=line))

        return outputs

    async def _load_questions(self) -> SyntheticQuestionSet:
        async with aio_open(self._questions_file_path, mode="r") as file:
            content = await file.read()
            return SyntheticQuestionSet.model_validate_json(json_data=content)


# Usage example:
async def main():
    eval = PipelineEvaluator(
        pipeline_results_path=Path(
            "data/pipelines/naive-100-overlap-210-retriever-vector-generator-gpt4o-v1/result.jsonl"
        ),
        questions_file_path=Path(
            "data/gdpr-handbook/processed/handbook-cleaned-questions.json"
        ),
    )
    await eval.run()


if __name__ == "__main__":
    anyio.run(main)
