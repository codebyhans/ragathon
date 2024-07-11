from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import anyio
import click
import pytrec_eval
from aiofiles import open as aio_open
from click_params import IntListParamType
from pydantic import BaseModel
from ragathon.config import Settings, init_settings
from ragathon.data.models import (
    AnnotationSet,
    EvaluatedQuery,
    MetricScore,
    RetrievalEvaluation,
)
from ragathon.indexing import Indexer
from ragathon.indexing.bm25 import BM25Index
from ragathon.indexing.vector import VECTOR_INDEX_FILE_NAME, VectorIndex
from ragathon.llms.azure import instantiate_embedder
from ragathon.llms.common import Embedder


class ScoredQuery(BaseModel):
    query_id: str
    metric_name: str
    k: int
    score: float


def compute_reciprocal_rank(
    ground_truth: Dict[str, Dict[str, float]],
    predictions: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, List[ScoredQuery]]:
    # Based on https://github.com/beir-cellar/beir/blob/main/beir/retrieval/custom_metrics.py

    k_max = max(k_values)

    # Create a dictionary to store the top k hits for each query
    # The dictionary is structured as follows:
    # {
    #     query_id: [
    #         (doc_id_1, score_1),
    #         (doc_id_2, score_2),
    #         ...
    #     ]
    # }
    top_hits: Dict[str, List[Tuple[str, float]]] = {}

    for query_id, pred_scores in predictions.items():
        sorted_doc_scores = sorted(
            pred_scores.items(), key=lambda item: item[1], reverse=True
        )[0:k_max]
        top_hits[query_id] = sorted_doc_scores

    results: Dict[str, List[ScoredQuery]] = defaultdict(list)
    for query_id in top_hits:
        query_relevant_docs = set(
            [
                doc_id
                for doc_id in ground_truth[query_id]
                if ground_truth[query_id][doc_id] > 0
            ]
        )
        relevant_found = False
        previous_score = -42.0
        for k in sorted(k_values):
            if relevant_found:
                # If we've already found a relevant doc for a smaller k, use the same score.
                # This is because the reciprocal rank will be the same for all larger k values.
                # By breaking out of the outer k-loop once a relevant document is found, we
                # improve performance.
                scored_query = ScoredQuery(
                    query_id=query_id,
                    metric_name=f"Reciprocal Rank@{k}",
                    k=k,
                    score=previous_score,
                )
                results[scored_query.query_id].append(scored_query)

            else:
                for rank, (doc_id, _) in enumerate(top_hits[query_id][:k]):
                    if doc_id in query_relevant_docs:
                        scored_query = ScoredQuery(
                            query_id=query_id,
                            metric_name=f"Reciprocal Rank@{k}",
                            k=k,
                            score=1.0 / (rank + 1),
                        )
                        results[scored_query.query_id].append(scored_query)
                        relevant_found = True
                        previous_score = scored_query.score
                        break
                if not relevant_found:
                    # No relevant doc found within top k, score is 0
                    scored_query = ScoredQuery(
                        query_id=query_id,
                        metric_name=f"Reciprocal Rank@{k}",
                        k=k,
                        score=0.0,
                    )
                    results[scored_query.query_id].append(scored_query)

    return results


class RetrievalEvaluatorCLI:
    def __init__(
        self,
        index_dir: Path,
        annotation_set_file_path: Path,
        k_values: List[int],
        output_path: Path,
    ) -> None:
        self._index_dir: Path = index_dir
        self._annotation_set_file_path: Path = annotation_set_file_path
        self._output_path: Path = output_path
        self._k_values: List[int] = k_values

        if not self._annotation_set_file_path.exists():
            raise FileNotFoundError(f"File {self._annotation_set_file_path} not found.")

        if not self._index_dir.exists():
            raise FileNotFoundError(f"Directory {self._index_dir} not found.")

    async def run(self) -> None:
        annotations: AnnotationSet = await self._load_annotations()

        vector_index_file_path = self._index_dir / VECTOR_INDEX_FILE_NAME
        if vector_index_file_path.exists():
            app_settings: Settings = init_settings()
            embedder: Embedder = instantiate_embedder(settings=app_settings)
            index: Indexer = VectorIndex(storage_dir=self._index_dir, embedder=embedder)
        else:
            index: Indexer = BM25Index(storage_dir=self._index_dir, language="danish")

        await self._run_internal(annotations=annotations, index=index)

    async def _run_internal(self, annotations: AnnotationSet, index: Indexer) -> None:
        await index.load()

        max_k = max(self._k_values)

        evaluated_queries: List[EvaluatedQuery] = []

        for item in annotations.items:
            evaluated_query = EvaluatedQuery(
                annotation_id=item.id,
                question=item.question,
                relevant_chunk_ids=item.relevant_chunk_ids,
            )
            evaluated_queries.append(evaluated_query)

            search_result = await index.search(query=item.question, k=max_k)
            for match in search_result.matches:
                evaluated_query.retrieved_chunks.append(match)

        evaluation = self._compute_metrics(
            evaluated_queries=evaluated_queries,
        )

        async with aio_open(self._output_path, mode="w") as file:
            await file.write(evaluation.model_dump_json(indent=2))

    async def _load_annotations(self) -> AnnotationSet:
        async with aio_open(self._annotation_set_file_path, mode="r") as file:
            content = await file.read()
            return AnnotationSet.model_validate_json(json_data=content)

    def _compute_metrics(
        self,
        evaluated_queries: List[EvaluatedQuery],
    ) -> RetrievalEvaluation:
        # Create dictionaries required by pytrec_eval
        ground_truth: Dict[str, Dict[str, float]] = {}
        predictions: Dict[str, Dict[str, float]] = {}

        for evaluated_query in evaluated_queries:
            qid = evaluated_query.annotation_id

            # Create ground truth for this annotation
            ground_truth[qid] = {
                chunk_id: 1 for chunk_id in evaluated_query.relevant_chunk_ids
            }

            # Create predictions for this annotation
            retrieved_chunks = evaluated_query.retrieved_chunks
            scores = [match.score for match in retrieved_chunks]
            normalized_scores = self._normalize_scores(scores=scores)
            predictions[qid] = {
                retrieved.chunk_id: score
                for retrieved, score in zip(retrieved_chunks, normalized_scores)
            }

        # Define the measures that pytrec_eval should compute
        measures = {
            f"map_cut.{','.join([str(k) for k in self._k_values])}",
            f"ndcg_cut.{','.join([str(k) for k in self._k_values])}",
            f"recall.{','.join([str(k) for k in self._k_values])}",
            f"P.{','.join([str(k) for k in self._k_values])}",
        }

        # Define the mapping from pytrec_eval keys to metric names
        metric_name_to_pyrec_key = dict()
        for k in self._k_values:
            metric_name_to_pyrec_key[f"MAP@{k}"] = f"map_cut_{k}"
            metric_name_to_pyrec_key[f"NDCG@{k}"] = f"ndcg_cut_{k}"
            metric_name_to_pyrec_key[f"Recall@{k}"] = f"recall_{k}"
            metric_name_to_pyrec_key[f"Precision@{k}"] = f"P_{k}"

        # Compute the metrics using pytrec_eval
        evaluator = pytrec_eval.RelevanceEvaluator(
            query_relevance=ground_truth, measures=measures
        )
        pytrec_results = evaluator.evaluate(scores=predictions)

        # Compute the reciprocal rank metric as pytrec_eval does not support it
        rr_result = compute_reciprocal_rank(
            ground_truth=ground_truth,
            predictions=predictions,
            k_values=self._k_values,
        )

        # Container for the sum of the scores for each metric
        metric_sums: Dict[str, float] = defaultdict(lambda: 0.0)

        for evaluated_query in evaluated_queries:
            qid = evaluated_query.annotation_id
            scores = pytrec_results[qid]

            # Add the pytrec_eval scores
            for metric_name, pyrec_key in metric_name_to_pyrec_key.items():
                ms = MetricScore(name=metric_name, value=scores[pyrec_key])
                evaluated_query.metric_scores[ms.name] = ms

                # Sum the scores for this particular metric to compute the mean later
                metric_sums[metric_name] += ms.value

            # Add the reciprocal rank scores
            for scored_query in rr_result[qid]:
                evaluated_query.metric_scores[scored_query.metric_name] = MetricScore(
                    name=scored_query.metric_name, value=scored_query.score
                )

                # Sum the reciprocal rank scores so we can compute MRR later
                metric_sums[scored_query.metric_name] += scored_query.score

        # Compute the mean scores for each metric
        mean_metric_scores = {
            metric_name: sum / len(evaluated_queries)
            for metric_name, sum in metric_sums.items()
        }

        return RetrievalEvaluation(
            k_values=self._k_values,
            measures={"MAP", "NDCG", "Recall", "Precision", "Reciprocal Rank"},
            queries=evaluated_queries,
            mean_metric_scores=mean_metric_scores,
        )

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        max_score = max(scores)
        if max_score <= 1.0:
            # Scores are already normalized
            return scores

        min_score = min(scores)
        if min_score == max_score:
            # All scores are equal
            return [1.0 for _ in scores]

        return [(score - min_score) / (max_score - min_score) for score in scores]


@click.command()
@click.option(
    "-x",
    "--index-dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
    required=True,
    help="Location of the BM25 index directory.",
)
@click.option(
    "-a",
    "--annotation-set-file-path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
    help="Location of the file containing the annotations.",
)
@click.option(
    "-k",
    "--k-values",
    type=IntListParamType(separator=","),
    default="1,2,3,4,5,6,8,10,15,20",
    required=False,
    help="List of k values to use for the evaluation.",
)
@click.option(
    "-o",
    "--output-path",
    required=True,
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path
    ),
    help="Where to save the output file.",
)
def main(**kwargs) -> None:  # pyre-ignore [2]
    async def run_main() -> None:
        cli = RetrievalEvaluatorCLI(**kwargs)
        await cli.run()

    anyio.run(run_main)


if __name__ == "__main__":
    main()
