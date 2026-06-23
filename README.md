# context-graph-benchmark

A pure-Python structured memory layer for multi-agent LLM systems — stores decisions as entity-relationship triples, retrieves by graph traversal instead of text similarity, zero API calls.

![Python Version](https://img.shields.io/badge/python-3.12-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![API Calls](https://img.shields.io/badge/API%20calls-zero-brightgreen)

---

Vector RAG retrieves chunks that look similar to your query. It cannot retrieve relationships *between* facts. When one agent makes a decision and another agent needs to recall it twenty turns later — or when the answer requires combining two separately-stated facts — similarity search has no mechanism to do that, regardless of how good the embedding model is.

This project benchmarks three memory architectures on that exact problem and measures where the ceiling is for each one.

Read the full write-up on Towards Data Science → **[Vector RAG Isn't Enough — I Built a Context Graph Layer for Multi-Agent Memory](https://towardsdatascience.com/author/emmimalp-alexander/)**

---

## Benchmark Results

Five scenarios, 18 graded queries, fully deterministic, zero LLM calls, reproduced identically on two separate machines.

| Architecture | Accuracy | Avg tokens/query | Direct | Distant | Join |
|---|---|---|---|---|---|
| Raw History Dump | 61.1% | 490.9 | 66.7% | 71.4% | 40.0% |
| Vector-Only RAG | 50.0% | 75.9 | 66.7% | 57.1% | 20.0% |
| **Context Graph** | **88.9%** | **26.9** | **100%** | **85.7%** | **80.0%** |

The join column is the number that matters. Join queries require combining two separately-stated facts — "which component does the module owned by Agent_X depend on?" Vector similarity has no native mechanism to construct that answer when the two facts live in different chunks. A graph walks it directly.

### Token Cost vs Conversation Length

| Filler turns | Raw Dump | Vector RAG | Context Graph |
|---|---|---|---|
| 10 | 157 | 54 | 23 |
| 50 | 659 | 54 | 23 |
| 100 | 1,287 | 54 | 23 |
| 200 | 2,542 | 54 | 23 |
| 400 | 5,052 | 54 | 23 |
| 800 | 10,072 | 54 | 23 |

Conversation length grew 80x. Raw dump tokens grew 64.15x (O(N), linear). Vector RAG and context graph both grew 1.00x (O(1) per query, flat regardless of conversation length).

---

## What This Is

Three memory architectures, each implemented in pure Python with no LLM calls anywhere:

```
Conversation turns (FACT + DISTRACTOR + QUERY)
        |
        ├── RawHistoryDump      → flat transcript, resent in full every query
        ├── VectorOnlyRAG       → TF-IDF chunks, top-K cosine similarity
        └── ContextGraph        → (subject, predicate, object) triples
                                   NetworkX MultiDiGraph
                                   two-hop traversal for join queries
```

A benchmark harness runs all three against five scripted multi-agent scenarios and grades answers by deterministic substring match against hand-authored ground truth.

---

## Project Structure

```
context-graph-benchmark/
├── src/
│   ├── scenarios.py            # Five scripted scenarios, Turn dataclass, all_scenarios()
│   ├── arch_raw_dump.py        # Architecture 1: flat transcript baseline
│   ├── arch_vector_rag.py      # Architecture 2: TF-IDF vector retrieval
│   ├── arch_context_graph.py   # Architecture 3: NetworkX entity-relationship graph
│   ├── benchmark.py            # Harness: run_benchmark(), summarize(), QueryResult
│   ├── measure_scaling.py      # Token cost vs conversation length measurement
│   ├── tokenizer.py            # Zero-dependency token estimator (~4 chars/token)
│   └── export_neo4j.py         # Optional: export in-memory graph to Neo4j Cypher
└── tests/
    └── test_benchmark.py       # Regression tests locking the headline numbers
```

---

## Installation

```bash
git clone https://github.com/Emmimal/context-graph-benchmark.git
cd context-graph-benchmark
pip install networkx scikit-learn
```

Two dependencies. No API keys. No database required for the core benchmark.

```bash
# Optional — only needed for the Neo4j export path
pip install neo4j
```

---

## Running the Benchmark

```bash
python src/benchmark.py
```

Runs all five scenarios against all three architectures, prints a JSON summary with accuracy, token counts, and per-query-type breakdowns.

```bash
python src/measure_scaling.py
```

Measures how per-query token cost scales from 10 to 800 filler turns, with growth-rate ratios printed at the end.

```bash
python src/export_neo4j.py
```

Builds one scenario's graph in-memory and prints the generated Cypher statements. Requires no running Neo4j instance — useful for reviewing the export format before connecting a real database.

---

## Running the Tests

```bash
pip install pytest
pytest tests/ -v
```

Six regression tests that lock the headline numbers and structural claims:

| Test | What it guards |
|---|---|
| `test_scenarios_are_well_formed` | Every scenario has facts, queries, and ground truth |
| `test_distractors_outnumber_facts_per_scenario` | Realistic noise levels in every scenario |
| `test_benchmark_is_deterministic` | Two back-to-back runs produce identical output |
| `test_context_graph_beats_raw_dump_on_tokens` | Token-savings claim stays supported |
| `test_context_graph_handles_fact_supersession` | Stale-fact bug stays fixed |
| `test_all_architectures_produce_some_correct_answers` | No architecture is silently broken |
| `test_join_queries_are_genuinely_harder_or_equal_for_flat_architectures` | Structural claim about join queries holds |

If any of the headline benchmark numbers change, at least one of these tests will fail before the article ships.

---

## How the Context Graph Works

Facts are ingested as `(subject, predicate, object)` triples into a NetworkX `MultiDiGraph`. Distractor turns are never stored.

```python
graph.ingest(Turn(
    turn_id=3,
    turn_type=TurnType.FACT,
    speaker="Agent_Planner",
    text="Agent_Planner decided the project uses PostgreSQL.",
    subject="Project_Alpha",
    predicate="USES_DATABASE",
    object="PostgreSQL",
    fact_id="f_db"
))
```

**Fact supersession** is built in. When a new fact restates the same `(subject, predicate)` pair — e.g. a ticket priority changing from "high" to "critical" — the old edge is removed before the new one is written. Without this, a graph returns stale facts with full structural confidence, which is worse than a fuzzy retrieval returning a stale chunk, because a graph *looks* authoritative.

**Join query traversal** is a two-hop walk rather than similarity search:

```
query: "what does the module owned by Agent_Implementer depend on?"

hop 1: Agent_Implementer <--ASSIGNED_TO-- AuthModule
hop 2: AuthModule --DEPENDS_ON--> RateLimiter
answer: RateLimiter
```

This is the case where flat architectures structurally cannot answer: no single chunk holds both facts, and similarity search has no mechanism to combine them even if both chunks are retrieved.

---

## The Five Benchmark Scenarios

| Scenario | Domain | Facts | Queries |
|---|---|---|---|
| `pipeline_review` | Software planning | Database, auth module, rate limiter decisions | Direct, join, distant |
| `research_pipeline` | Research coordination | Article draft, reviewer assignments | Direct, distant, join |
| `incident_response` | On-call / ops | Service owner, deployment region, severity | Direct, distant, join, direct |
| `support_escalation` | Customer support | Ticket priority (with supersession), assigned agent | Direct, distant, join |
| `data_pipeline` | Data engineering | Dataset anomaly, pipeline owner | Direct, distant, join, distant |

Across all five: 6 direct queries, 7 distant queries, 5 join queries. Distractors outnumber facts in every scenario — verified by `test_distractors_outnumber_facts_per_scenario`.

---

## Design Decisions and Limitations

**No LLM calls anywhere.** Not for extraction, not for query answering, not for grading. A real LLM would introduce variance that would measure model differences as much as architecture differences. Every result here is reproducible byte-for-byte across machines.

**TF-IDF, not neural embeddings.** `TfidfVectorizer` has no random state — deterministic by construction. It is a real sparse-retrieval method used in production RAG, not a weakened stand-in. The structural weakness being tested (inability to combine two separately-stated facts) applies to any vector retrieval method regardless of embedding quality.

**Token estimation, not tiktoken.** `tiktoken` downloads BPE rank files from a remote URL on first use — a hidden network dependency. The estimator in `tokenizer.py` uses ~4 chars/token (OpenAI's own suggested approximation). Applied identically across all three architectures, so it cannot bias the comparison between them.

**The alias table is a stand-in for entity linking.** In production, resolving "the authentication module" to `AuthModule` is an LLM call. The hardcoded alias table makes the benchmark deterministic but only covers phrasing anticipated in advance. This is disclosed, not hidden — the vocabulary-mismatch problem exists for any architecture and is moved to write-time (rather than solved) by the graph.

**Two queries left intentionally broken.** The data-pipeline scenario contains two queries that refer to entities by description rather than name ("the dataset that currently has an anomaly" instead of `Upstream_Orders`). Fixing these with alias expansion would overfit the benchmark to the test queries. They stay broken to represent a real limitation.

**Neo4j export is optional, not a performance upgrade.** The in-memory NetworkX graph is faster for a single process than a networked database call. The reasons to move to Neo4j in production are transactional guarantees and concurrent multi-agent writes — not query speed.

---

## When to Use This

Worth building if you have multi-agent pipelines where one agent's decision needs to be correctly retrieved by a different agent many turns later, or systems where questions routinely require combining two or more separately-stated facts.

Skip it for single-agent single-turn tasks, always-single-fact queries (vector RAG gets you most of the accuracy at less engineering cost), or teams with no tolerance for an extraction step that a flat store avoids.

---



## License

MIT
