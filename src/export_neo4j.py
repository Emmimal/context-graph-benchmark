"""
Production export path: NetworkX -> Neo4j.

This is intentionally a separate, optional module. The core tutorial and
benchmark have zero external infrastructure dependencies (no database, no
API, no network calls) -- this file is only relevant to a reader who wants
to take the prototype to production scale, and it requires a running Neo4j
instance to actually execute.

Design note for the article: this is presented as a *path*, not a
performance claim. We are not benchmarking Neo4j here -- a real graph
database adds transactional guarantees, concurrent multi-agent writes, and
persistence across process restarts, none of which the in-memory NetworkX
prototype has. Those are the actual reasons to make this jump in production,
not raw query speed (NetworkX in-memory will often be *faster* for a single
process than a networked database call -- the value of Neo4j is durability
and concurrency, and the article should say so rather than imply this
export step is a speed upgrade).
"""

from arch_context_graph import ContextGraph


def export_to_cypher(graph: ContextGraph) -> list[str]:
    """
    Generate Cypher statements that recreate the in-memory graph in Neo4j.
    Returns a list of statements rather than executing them, so the reader
    can review, batch, or adapt before running against a real instance.
    """
    statements = []

    for node in graph.graph.nodes:
        safe_label = "Entity"
        # Use MERGE, not CREATE, so re-running this export is idempotent
        # and doesn't duplicate nodes on repeated syncs.
        statements.append(
            f'MERGE (n:{safe_label} {{name: {_cypher_str(node)}}})'
        )

    for u, v, data in graph.graph.edges(data=True):
        predicate = data["predicate"]
        fact_id = data.get("fact_id", "")
        statements.append(
            f'MATCH (a:Entity {{name: {_cypher_str(u)}}}), '
            f'(b:Entity {{name: {_cypher_str(v)}}}) '
            f'MERGE (a)-[r:{predicate} {{fact_id: {_cypher_str(fact_id)}}}]->(b)'
        )

    return statements


def _cypher_str(value: str) -> str:
    # Minimal escaping for embedding a Python string as a Cypher string
    # literal. Sufficient for this export use case; a production
    # integration should use parameterized queries (see run_against_neo4j
    # below) rather than string interpolation, which is shown here only
    # because it's clearer to read in an article.
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def run_against_neo4j(graph: ContextGraph, uri: str, user: str, password: str) -> None:
    """
    Example of how you'd actually push the exported graph into a running
    Neo4j instance, using parameterized queries (the production-safe way,
    as opposed to the string-interpolated Cypher above which is shown only
    for readability).

    Requires: pip install neo4j
    Requires a running Neo4j instance -- this function is not exercised by
    the benchmark or test suite, since the rest of this project has no
    external infrastructure dependency by design.
    """
    from neo4j import GraphDatabase  # import here: optional dependency

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            for node in graph.graph.nodes:
                session.run(
                    "MERGE (n:Entity {name: $name})",
                    name=str(node),
                )
            for u, v, data in graph.graph.edges(data=True):
                predicate = data["predicate"]
                session.run(
                    f"MATCH (a:Entity {{name: $u}}), (b:Entity {{name: $v}}) "
                    f"MERGE (a)-[r:{predicate} {{fact_id: $fact_id}}]->(b)",
                    u=str(u), v=str(v), fact_id=data.get("fact_id", ""),
                )
    finally:
        driver.close()


if __name__ == "__main__":
    # Demo: build a small graph and show the generated Cypher without
    # requiring a Neo4j instance to be running.
    from scenarios import build_scenario_pipeline_review, TurnType

    scenario = build_scenario_pipeline_review()
    graph = ContextGraph()
    for turn in scenario.turns:
        graph.ingest(turn)

    statements = export_to_cypher(graph)
    print(f"Generated {len(statements)} Cypher statements:\n")
    for stmt in statements[:8]:
        print(stmt)
    print("...")
