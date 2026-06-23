"""
Empirical measurement of how token cost per query scales with conversation
length, for each architecture. This directly tests the claim that raw-dump
scales superlinearly (it resends the whole growing transcript on every
query) while the context graph's per-query cost stays flat (it only ever
returns the matched triple(s), regardless of how much has been discussed).

We don't assert the Big-O claim -- we generate a longer synthetic
conversation by repeating a benign extended distractor pattern around a
fixed set of facts, and literally measure tokens-per-query at increasing
conversation lengths, then fit a curve to see what scaling actually shows
up. This keeps the claim checkable rather than asserted.
"""

import sys
import os

from scenarios import Turn, TurnType
from arch_raw_dump import RawHistoryDump
from arch_vector_rag import VectorOnlyRAG
from arch_context_graph import ContextGraph


def build_growing_conversation(n_distractors: int) -> list[Turn]:
    """
    One fact at the start, then n_distractors filler turns, then one query
    at the end asking for the fact. This isolates how each architecture's
    per-query token cost grows purely as a function of conversation length,
    holding the actual information content constant.
    """
    turns = []
    tid = 0

    tid += 1
    turns.append(Turn(tid, TurnType.FACT, "Agent_A",
                       "Agent_A decided the deployment region is us-west-2.",
                       subject="Service_X", predicate="DEPLOYS_TO", object="us-west-2",
                       fact_id="f_region"))

    filler_pool = [
        "Checking in on progress, all looks fine so far.",
        "No blockers to report at this time.",
        "Will follow up again shortly with an update.",
        "Build is green, nothing unusual to flag.",
        "Standing by for the next milestone.",
    ]
    for i in range(n_distractors):
        tid += 1
        turns.append(Turn(tid, TurnType.DISTRACTOR, "Agent_B", filler_pool[i % len(filler_pool)]))

    tid += 1
    turns.append(Turn(tid, TurnType.QUERY, "Agent_C",
                       "What deployment region was decided for Service_X?",
                       query_type="distant", required_fact_ids=("f_region",),
                       ground_truth="us-west-2"))
    return turns


def measure_scaling():
    lengths = [10, 50, 100, 200, 400, 800]
    architectures = [RawHistoryDump, VectorOnlyRAG, ContextGraph]

    print(f"{'n_distractors':>15} | {'raw_dump':>10} | {'vector_rag':>10} | {'context_graph':>14}")
    print("-" * 60)

    rows = []
    for n in lengths:
        turns = build_growing_conversation(n)
        row = {"n": n}
        for arch_cls in architectures:
            arch = arch_cls()
            tokens_at_query = None
            for turn in turns:
                if turn.turn_type == TurnType.QUERY:
                    _, tokens_at_query = arch.answer_query(turn)
                    arch.ingest(turn)
                else:
                    arch.ingest(turn)
            row[arch_cls.name] = tokens_at_query
        rows.append(row)
        print(f"{n:>15} | {row['raw_history_dump']:>10} | {row['vector_only_rag']:>10} | {row['context_graph']:>14}")

    # Rough growth-rate check: compare ratio of tokens at longest vs
    # shortest length against the ratio of conversation lengths themselves.
    first, last = rows[0], rows[-1]
    length_ratio = last["n"] / first["n"]
    print()
    print(f"Conversation length grew {length_ratio:.1f}x (from {first['n']} to {last['n']} distractor turns)")
    for arch_name in ("raw_history_dump", "vector_only_rag", "context_graph"):
        token_ratio = last[arch_name] / first[arch_name]
        print(f"  {arch_name}: tokens grew {token_ratio:.2f}x")

    return rows


if __name__ == "__main__":
    measure_scaling()
