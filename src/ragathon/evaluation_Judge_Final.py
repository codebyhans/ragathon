from pathlib import Path
from typing import Dict, List, cast

import anyio
from aiofiles import open as aio_open
from loguru import logger
from pydantic import BaseModel, Field

from ragathon.config import Settings, init_settings
from ragathon.data.models import (
    QueryRetrievalMetric,
    SyntheticQuestionSet,
)
from ragathon.llms.azure import AzureOpenAIBasedLLM, create_token_generator
from ragathon.llms.common import LLM, ChatMessage, MessageRole
from ragathon.pipelines.common import RAGPipelineOutput
from ragathon.utils.strings import generate_id


class LLMJudgement(BaseModel):
    justification: str = Field(
        ..., description="The justification for the judgement score."
    )
    score: float = Field(..., description="The judgement score for the given answer.")


def instantiate_llm_for_eval(settings: Settings) -> LLM:
    token_generator = create_token_generator(settings=settings)

    llm_eval_azure_open_ai = AzureOpenAIBasedLLM(
        api_base_endpoint=settings.AZURE_OPEN_AI_API_ENDPOINT,
        api_version=settings.AZURE_OPEN_AI_API_VERSION,
        token_generator=token_generator,
        deployment_id=settings.AZURE_OPEN_AI_LLM_EVAL_DEPLOYMENT_NAME,
        model_name=settings.AZURE_OPEN_AI_LLM_EVAL_DEPLOYMENT_NAME,
        default_temperature=0.1,  # TODO: Make this configurable?
    )

    llm_eval: LLM = cast(LLM, llm_eval_azure_open_ai)

    return llm_eval


class PipelineEvaluator:
    def __init__(self, pipeline_results_path: Path, questions_file_path: Path) -> None:
        self._pipeline_result_path: Path = pipeline_results_path
        self._questions_file_path: Path = questions_file_path

        app_settings: Settings = init_settings()
        self._llm: LLM = instantiate_llm_for_eval(settings=app_settings)

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

        llm_metrics = await self._compute_llm_judgement_metrics(
            q_set=questions,
            question_to_output=question_to_output,
        )

        mean_llm_score = sum([metric.value for metric in llm_metrics]) / len(
            llm_metrics
        )

        logger.debug(f"Mean LLM Judgement Score: {mean_llm_score}")

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

    async def _compute_llm_judgement_metrics(
        self,
        q_set: SyntheticQuestionSet,
        question_to_output: Dict[str, RAGPipelineOutput],
    ) -> List[QueryRetrievalMetric]:
        metrics: List[QueryRetrievalMetric] = []

        for item in q_set.questions:
            metric = QueryRetrievalMetric(
                query_id=generate_id(parts=[item.question]),
                query=item.question,
                measure_name="LLM Judgement",
                value=0,
            )
            metrics.append(metric)

            if item.question in question_to_output:
                output = question_to_output[item.question]

                messages = [
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content="""You are an AI judge tasked with assessing the correctness of an answer given a reference answer and question. Your role is to carefully analyze the given answer in comparison to the reference answer, considering the context of the question.

Please follow these steps to assess the given answer:

1. Carefully read and understand the question, reference answer, and given answer.
2. Compare the given answer to the reference answer, focusing on:
   a. Factual correctness
   b. Completeness of information
   c. Relevance to the question
   d. Clarity and coherence
3. Identify any errors, omissions, or extraneous information in the given answer.
4. Consider the overall quality and accuracy of the given answer in relation to the reference answer.

Based on your analysis, provide a score and justification for your assessment. Use the following scoring system:

- 5: Excellent - Fully correct and complete
- 4: Good - Mostly correct with minor omissions or errors
- 3: Fair - Partially correct but with significant omissions or errors
- 2: Poor - Mostly incorrect or irrelevant
- 1: Very Poor - Completely incorrect or unrelated to the question

First, provide your justification for the score, explaining your reasoning and citing specific examples from both the reference and given answers. Then, give your final score.
""",
                    ),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=f"""Here is the question:
<question>
{item.question}
</question>

Here is the reference answer:
<reference_answer>
{item.reference_answers[0]}
</reference_answer>

Here is the given answer to be assessed:
<given_answer>
{output.generated_answer}
</given_answer>
""",
                    ),
                ]

                response = await self._llm.structured_completion(
                    messages=messages,
                    response_model=LLMJudgement,
                    temperature=0.2,
                )

                logger.debug(
                    f"LLM Judgement for question: {item.question}:\n{response}"
                )

                # Add the judgement score to the metric
                metric.value = response.score

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
            "data/pipelines/naive-210-overlap-100-retriever-vector-generator-gpt4o-v1/result.jsonl"
        ),
        questions_file_path=Path(
            "data/gdpr-handbook/processed/handbook-cleaned-questions.json"
        ),
    )
    await eval.run()


if __name__ == "__main__":
    anyio.run(main)
