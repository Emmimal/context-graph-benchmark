"""
Benchmark harness.

Runs every architecture against every scenario, turn by turn, in order.
Both FACT and DISTRACTOR turns are ingested by every architecture (nobody
gets to skip the noise). QUERY turns are answered and graded immediately,
then also ingested as a turn (a query is part of the conversation too).

Grading is deterministic substring/exact matching against each scenario's
hand-authored ground_truth -- no LLM-as-judge, so there is no hidden
non-determinism or API call in the grading step either.
"""

import json
import statistics
from dataclasses import dataclass, asdict

from scenarios import all_scenarios, TurnType, Scenario
from arch_raw_dump import RawHistoryDump
from arch_vector_rag import VectorOnlyRAG
from arch_context_graph import ContextGraph

ARCHITECTURES = [RawHistoryDump, VectorOnlyRAG, ContextGraph]


@dataclass
class QueryResult:
    scenario: str
    architecture: str
    query_text: str
    query_type: str
    ground_truth: str
    raw_answer: str
    correct: bool
    tokens_used: int
    turn_distance: int  # turns since the most recent required fact


def _grade(ground_truth: str, raw_answer: str) -> bool:
    return ground_truth.strip().lower() in raw_answer.strip().lower()


def _turn_distance(scenario: Scenario, query_turn) -> int:
    fact_lookup = {t.fact_id: t.turn_id for t in scenario.fact_turns}
    relevant_turn_ids = [fact_lookup[fid] for fid in query_turn.required_fact_ids if fid in fact_lookup]
    if not relevant_turn_ids:
        return -1
    most_recent_fact_turn = max(relevant_turn_ids)
    return query_turn.turn_id - most_recent_fact_turn


def run_benchmark() -> list[QueryResult]:
    results: list[QueryResult] = []
    for scenario in all_scenarios():
        for arch_cls in ARCHITECTURES:
            arch = arch_cls()
            for turn in scenario.turns:
                if turn.turn_type == TurnType.QUERY:
                    raw_answer, tokens = arch.answer_query(turn)
                    correct = _grade(turn.ground_truth, raw_answer)
                    results.append(QueryResult(
                        scenario=scenario.name,
                        architecture=arch.name,
                        query_text=turn.text,
                        query_type=turn.query_type,
                        ground_truth=turn.ground_truth,
                        raw_answer=raw_answer,
                        correct=correct,
                        tokens_used=tokens,
                        turn_distance=_turn_distance(scenario, turn),
                    ))
                    # The query itself becomes part of the conversation
                    # history going forward, same as in a real agent loop.
                    arch.ingest(turn)
                else:
                    arch.ingest(turn)
    return results


def summarize(results: list[QueryResult]) -> dict:
    by_arch = {}
    for arch_cls in ARCHITECTURES:
        arch_results = [r for r in results if r.architecture == arch_cls.name]
        n = len(arch_results)
        correct = sum(r.correct for r in arch_results)
        token_values = [r.tokens_used for r in arch_results]

        by_type = {}
        for qtype in sorted(set(r.query_type for r in arch_results)):
            type_results = [r for r in arch_results if r.query_type == qtype]
            type_n = len(type_results)
            type_correct = sum(r.correct for r in type_results)
            by_type[qtype] = {
                "n": type_n,
                "correct": type_correct,
                "accuracy": round(type_correct / type_n, 4) if type_n else None,
            }

        by_arch[arch_cls.name] = {
            "n_queries": n,
            "correct": correct,
            "accuracy": round(correct / n, 4) if n else None,
            "avg_tokens_per_query": round(statistics.mean(token_values), 1) if token_values else None,
            "median_tokens_per_query": round(statistics.median(token_values), 1) if token_values else None,
            "max_tokens_per_query": max(token_values) if token_values else None,
            "min_tokens_per_query": min(token_values) if token_values else None,
            "total_tokens_all_queries": sum(token_values),
            "accuracy_by_query_type": by_type,
        }
    return by_arch


def results_to_jsonable(results: list[QueryResult]) -> list[dict]:
    return [asdict(r) for r in results]


if __name__ == "__main__":
    results = run_benchmark()
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
