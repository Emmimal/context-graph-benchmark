"""
Regression tests for the benchmark.

These exist so that future edits to scenarios.py or any architecture file
cannot silently change the headline numbers without a human noticing. Run
with: pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scenarios import all_scenarios, TurnType
from arch_raw_dump import RawHistoryDump
from arch_vector_rag import VectorOnlyRAG
from arch_context_graph import ContextGraph
from benchmark import run_benchmark, summarize, ARCHITECTURES


def test_scenarios_are_well_formed():
    for scenario in all_scenarios():
        assert len(scenario.fact_turns) > 0, f"{scenario.name} has no facts"
        assert len(scenario.query_turns) > 0, f"{scenario.name} has no queries"
        for q in scenario.query_turns:
            assert q.ground_truth, f"query in {scenario.name} missing ground truth"
            assert q.query_type in ("direct", "distant", "join")
            assert len(q.required_fact_ids) >= 1


def test_distractors_outnumber_facts_per_scenario():
    # Sanity check on realism: every scenario should have more chatter than
    # signal, otherwise the benchmark doesn't represent a real long-running
    # multi-agent conversation.
    for scenario in all_scenarios():
        n_distractor = sum(1 for t in scenario.turns if t.turn_type == TurnType.DISTRACTOR)
        n_fact = len(scenario.fact_turns)
        assert n_distractor > n_fact, f"{scenario.name} doesn't have realistic noise levels"


def test_benchmark_is_deterministic():
    r1 = run_benchmark()
    r2 = run_benchmark()
    s1 = summarize(r1)
    s2 = summarize(r2)
    assert s1 == s2, "Benchmark is not deterministic across runs"


def test_context_graph_beats_raw_dump_on_tokens():
    results = run_benchmark()
    summary = summarize(results)
    graph_tokens = summary["context_graph"]["avg_tokens_per_query"]
    raw_tokens = summary["raw_history_dump"]["avg_tokens_per_query"]
    assert graph_tokens < raw_tokens, (
        "Context graph should use meaningfully fewer tokens per query than "
        "raw history dump -- if this fails, the token-savings claim in the "
        "article is no longer supported by the benchmark."
    )


def test_context_graph_handles_fact_supersession():
    # Regression test for the specific bug found during development: a
    # graph that doesn't supersede old edges on a (subject, predicate)
    # match will return a stale fact instead of the current one.
    from scenarios import build_scenario_support_escalation

    scenario = build_scenario_support_escalation()
    arch = ContextGraph()
    for turn in scenario.turns:
        if turn.turn_type == TurnType.QUERY and "current priority" in turn.text:
            answer, _ = arch.answer_query(turn)
            assert "critical" in answer.lower(), (
                f"Expected superseded priority 'critical', got '{answer}' -- "
                "graph is returning a stale fact."
            )
        arch.ingest(turn)


def test_all_architectures_produce_some_correct_answers():
    # Catches the case where an architecture is silently broken (e.g.
    # returns UNKNOWN for everything) rather than genuinely underperforming.
    results = run_benchmark()
    summary = summarize(results)
    for arch_cls in ARCHITECTURES:
        acc = summary[arch_cls.name]["accuracy"]
        assert acc > 0.3, f"{arch_cls.name} accuracy ({acc}) suspiciously low -- check for a bug"


def test_join_queries_are_genuinely_harder_or_equal_for_flat_architectures():
    # The article's structural claim is that join queries are the case
    # flat architectures (no relationship structure) structurally struggle
    # with relative to their own direct/distant performance. We check the
    # *direction* of this effect rather than asserting graph perfection.
    results = run_benchmark()
    summary = summarize(results)
    for arch_name in ("raw_history_dump", "vector_only_rag"):
        by_type = summary[arch_name]["accuracy_by_query_type"]
        join_acc = by_type["join"]["accuracy"]
        direct_acc = by_type["direct"]["accuracy"]
        assert join_acc <= direct_acc, (
            f"{arch_name}: expected join accuracy <= direct accuracy "
            f"(got join={join_acc}, direct={direct_acc}) -- if this fails, "
            "the article's structural claim about join queries needs to "
            "be re-examined, not asserted."
        )


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
