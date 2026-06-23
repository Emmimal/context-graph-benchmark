"""
Scenario engine for the memory-architecture benchmark.

Design intent
-------------
Every scenario is a fixed, hand-authored script of turns. No LLM, no API call,
no randomness in content generation. This is what makes the benchmark fair:
all three memory architectures see *exactly* the same sequence of turns, and
the only thing that varies is how each architecture stores and retrieves
information from that sequence.

A scenario is a list of Turn objects. Each turn is one of:
    - FACT      : an agent asserts a piece of structured information
                   (e.g. "Agent_Planner decided to use Postgres for storage")
    - DISTRACTOR: irrelevant chatter that bloats a raw transcript but should
                  not be needed to answer any query (simulates the realistic
                  noise of a long-running multi-agent conversation)
    - QUERY     : a later agent needs a fact asserted earlier, possibly many
                  turns and distractors ago, to complete its task. Each query
                  has a single unambiguous ground-truth answer we can grade
                  against.

Scenario design principles (so the benchmark isn't rigged):
    1. Distractors outnumber facts, like a real conversation.
    2. Queries are spread across "distance" (turns since the fact was stated)
       so we can see how each architecture degrades with distance, not just
       whether it works at turn+1.
    3. Some queries require combining two separate facts (a "join"), which is
       the case raw-dump and vector-RAG both structurally struggle with and
       a graph is structurally suited for. This is included because it's a
       real differentiator, not because it's flattering — it actually is the
       graph's structural argument, and we report it as a distinct query
       type ("join") rather than folding it into the headline number, so the
       result is checkable rather than asserted.
    4. Some queries are answerable from a *single recent fact* with no
       distance at all. This matters because it gives vector-RAG and raw-dump
       a fair shot at scoring well, instead of every query being
       hand-picked to favor the graph.
"""

from dataclasses import dataclass, field
from enum import Enum


class TurnType(Enum):
    FACT = "fact"
    DISTRACTOR = "distractor"
    QUERY = "query"


@dataclass
class Turn:
    turn_id: int
    turn_type: TurnType
    speaker: str
    text: str
    # Structured payload, only populated for FACT turns.
    # subject/predicate/object mirrors a graph triple, but it is equally
    # available to the raw-dump and vector architectures as plain text below
    # (the `text` field) -- nobody gets a free structured-data advantage.
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    fact_id: str | None = None  # unique key for grading lookups

    # Only populated for QUERY turns.
    query_type: str | None = None          # "direct", "distant", "join"
    required_fact_ids: tuple = field(default_factory=tuple)
    ground_truth: str | None = None


@dataclass
class Scenario:
    name: str
    description: str
    turns: list[Turn]

    @property
    def fact_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.turn_type == TurnType.FACT]

    @property
    def query_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.turn_type == TurnType.QUERY]


def _t(counter: list[int]) -> int:
    counter[0] += 1
    return counter[0]


def build_scenario_pipeline_review() -> Scenario:
    """
    A 3-agent software-review scenario: Planner, Implementer, Reviewer.
    Facts are decisions made early; queries happen much later after
    unrelated chatter, mimicking a real long-running agent session.
    """
    c = [0]
    turns: list[Turn] = []

    def fact(speaker, text, subject, predicate, obj, fact_id):
        turns.append(Turn(_t(c), TurnType.FACT, speaker, text,
                           subject=subject, predicate=predicate, object=obj,
                           fact_id=fact_id))

    def distractor(speaker, text):
        turns.append(Turn(_t(c), TurnType.DISTRACTOR, speaker, text))

    def query(speaker, text, query_type, required, ground_truth):
        turns.append(Turn(_t(c), TurnType.QUERY, speaker, text,
                           query_type=query_type,
                           required_fact_ids=tuple(required),
                           ground_truth=ground_truth))

    # --- Early facts ---
    fact("Agent_Planner",
         "Planner decided the project will use PostgreSQL for the storage layer.",
         "Project_Alpha", "USES_STORAGE", "PostgreSQL", "f_storage")
    fact("Agent_Planner",
         "Planner assigned the authentication module to Agent_Implementer.",
         "AuthModule", "ASSIGNED_TO", "Agent_Implementer", "f_auth_owner")
    distractor("Agent_Implementer", "Sounds good, I'll get started on that shortly.")
    distractor("Agent_Planner", "Let me know if you need any clarification on requirements.")
    fact("Agent_Implementer",
         "Implementer set the authentication token expiry to 15 minutes.",
         "AuthModule", "HAS_TOKEN_EXPIRY", "15 minutes", "f_token_expiry")
    distractor("Agent_Reviewer", "I'm reviewing the open PRs from last week now.")
    distractor("Agent_Implementer", "The build is currently green on the main branch.")
    distractor("Agent_Planner", "We should sync on sprint goals tomorrow.")
    fact("Agent_Implementer",
         "Implementer reported AuthModule depends on the RateLimiter component.",
         "AuthModule", "DEPENDS_ON", "RateLimiter", "f_auth_depends_ratelimiter")
    distractor("Agent_Reviewer", "Coffee break, back in five.")
    distractor("Agent_Implementer", "Pushed a minor formatting fix, nothing functional.")
    distractor("Agent_Planner", "Reminder: standup moved to 10am tomorrow.")
    distractor("Agent_Reviewer", "Noted, thanks for the heads up.")
    fact("Agent_Reviewer",
         "Reviewer flagged that RateLimiter currently has no test coverage.",
         "RateLimiter", "HAS_TEST_COVERAGE", "none", "f_ratelimiter_coverage")
    distractor("Agent_Implementer", "I'll batch that with tomorrow's commits.")
    distractor("Agent_Planner", "Let's keep scope tight for this sprint.")
    distractor("Agent_Reviewer", "Agreed, no scope creep please.")
    distractor("Agent_Implementer", "Working through the remaining edge cases now.")
    distractor("Agent_Planner", "Appreciate the update.")
    fact("Agent_Planner",
         "Planner decided Project_Alpha's deployment target is AWS us-east-1.",
         "Project_Alpha", "DEPLOYS_TO", "AWS us-east-1", "f_deploy_target")
    distractor("Agent_Reviewer", "Double-checked the linter config, all clean.")
    distractor("Agent_Implementer", "No blockers on my end currently.")
    distractor("Agent_Planner", "Great, steady progress overall.")
    distractor("Agent_Reviewer", "I'll start the security pass after lunch.")
    distractor("Agent_Implementer", "Sounds good, ping me if anything comes up.")

    # --- Queries: spread across distance and type ---

    # Direct / near query (low distance) -- should be easy for everyone.
    query("Agent_Reviewer",
          "What is the test coverage status of RateLimiter?",
          "direct", ["f_ratelimiter_coverage"], "none")

    distractor("Agent_Planner", "Let's also check in on the docs progress.")
    distractor("Agent_Implementer", "Docs are about 70% done.")
    distractor("Agent_Reviewer", "I'll take a pass on those once code review wraps.")
    distractor("Agent_Planner", "Sounds good, thanks all.")
    distractor("Agent_Implementer", "One more thing -- the staging env is currently down.")
    distractor("Agent_Reviewer", "I'll file a ticket for staging.")

    # Distant query -- fact was asserted long before, lots of distractors since.
    query("Agent_Implementer",
          "What storage technology did the team decide on for this project?",
          "distant", ["f_storage"], "PostgreSQL")

    distractor("Agent_Planner", "Let's revisit estimates at the end of the week.")
    distractor("Agent_Reviewer", "Sure, I'll have my section ready.")

    # Join query -- needs two separate facts combined; this is the
    # structural case a graph is suited for and raw-dump/vector-RAG are not.
    query("Agent_Reviewer",
          "Which component does the module owned by Agent_Implementer depend on?",
          "join", ["f_auth_owner", "f_auth_depends_ratelimiter"], "RateLimiter")

    distractor("Agent_Planner", "Wrapping up for today, good progress all around.")
    distractor("Agent_Implementer", "Agreed, see everyone tomorrow.")

    # Distant + specific value query.
    query("Agent_Planner",
          "What is the configured token expiry for the authentication module?",
          "distant", ["f_token_expiry"], "15 minutes")

    return Scenario(
        name="pipeline_review",
        description="3-agent software planning/review session with interleaved decisions and chatter.",
        turns=turns,
    )


def build_scenario_research_pipeline() -> Scenario:
    """
    A Researcher -> Writer -> Editor pipeline (mirrors a common TDS-relevant
    multi-agent pattern). Facts are research findings; queries simulate the
    Writer/Editor needing specific earlier findings well after the fact.
    """
    c = [0]
    turns: list[Turn] = []

    def fact(speaker, text, subject, predicate, obj, fact_id):
        turns.append(Turn(_t(c), TurnType.FACT, speaker, text,
                           subject=subject, predicate=predicate, object=obj,
                           fact_id=fact_id))

    def distractor(speaker, text):
        turns.append(Turn(_t(c), TurnType.DISTRACTOR, speaker, text))

    def query(speaker, text, query_type, required, ground_truth):
        turns.append(Turn(_t(c), TurnType.QUERY, speaker, text,
                           query_type=query_type,
                           required_fact_ids=tuple(required),
                           ground_truth=ground_truth))

    fact("Agent_Researcher",
         "Researcher found that Source_A reports a 23% adoption increase in 2025.",
         "Source_A", "REPORTS_METRIC", "23% adoption increase in 2025", "f_metric_a")
    distractor("Agent_Writer", "Got it, I'll keep that in mind for the intro.")
    distractor("Agent_Editor", "Let me know once a draft is ready.")
    fact("Agent_Researcher",
         "Researcher determined Source_B is a primary source, published by the original team.",
         "Source_B", "IS_TYPE", "primary source", "f_source_b_type")
    distractor("Agent_Writer", "I'll prioritize primary sources in the citations.")
    distractor("Agent_Researcher", "Still digging for a third source on this.")
    distractor("Agent_Editor", "No rush, take the time you need.")
    fact("Agent_Researcher",
         "Researcher noted Source_B's claims directly contradict Source_A's adoption numbers.",
         "Source_B", "CONTRADICTS", "Source_A", "f_b_contradicts_a")
    distractor("Agent_Writer", "Interesting, I'll flag that tension in the draft.")
    distractor("Agent_Editor", "Good, contradictions are worth surfacing.")
    distractor("Agent_Researcher", "Moving on to the methodology section now.")
    distractor("Agent_Writer", "Drafting the opening paragraph now.")
    fact("Agent_Writer",
         "Writer drafted the article with working title 'The Adoption Paradox'.",
         "Article_Draft", "HAS_TITLE", "The Adoption Paradox", "f_title")
    distractor("Agent_Editor", "Catchy title, let's see how the body holds up.")
    distractor("Agent_Researcher", "I'll have the methodology notes by tomorrow.")
    distractor("Agent_Writer", "Sounds good, I'll wait on that before the analysis section.")
    distractor("Agent_Editor", "Appreciate the coordination, team.")

    query("Agent_Editor",
          "What is the working title of the current article draft?",
          "direct", ["f_title"], "The Adoption Paradox")

    distractor("Agent_Researcher", "Found a fourth source, cross-checking now.")
    distractor("Agent_Writer", "Let me know if it changes anything material.")
    distractor("Agent_Editor", "Keep me posted either way.")
    distractor("Agent_Researcher", "Will do.")

    query("Agent_Writer",
          "What adoption metric did Source_A report?",
          "distant", ["f_metric_a"], "23% adoption increase in 2025")

    distractor("Agent_Editor", "Almost ready for a full read-through.")
    distractor("Agent_Researcher", "I'll send final notes by end of day.")

    query("Agent_Editor",
          "Which secondary source's adoption numbers are contradicted by a primary source?",
          "join", ["f_source_b_type", "f_b_contradicts_a"], "Source_A")

    return Scenario(
        name="research_pipeline",
        description="Researcher/Writer/Editor pipeline with citation facts and a contradiction join-query.",
        turns=turns,
    )


def build_scenario_incident_response() -> Scenario:
    """
    A 3-agent on-call incident response scenario: Detector, Diagnoser,
    Resolver. Longer distractor runs than the other scenarios, to stress
    distance effects more heavily, plus two join queries requiring
    different two-hop paths.
    """
    c = [0]
    turns: list[Turn] = []

    def fact(speaker, text, subject, predicate, obj, fact_id):
        turns.append(Turn(_t(c), TurnType.FACT, speaker, text,
                           subject=subject, predicate=predicate, object=obj,
                           fact_id=fact_id))

    def distractor(speaker, text):
        turns.append(Turn(_t(c), TurnType.DISTRACTOR, speaker, text))

    def query(speaker, text, query_type, required, ground_truth):
        turns.append(Turn(_t(c), TurnType.QUERY, speaker, text,
                           query_type=query_type,
                           required_fact_ids=tuple(required),
                           ground_truth=ground_truth))

    fact("Agent_Detector",
         "Detector identified that Service_Checkout is experiencing elevated error rates.",
         "Service_Checkout", "HAS_STATUS", "elevated error rate", "f_checkout_status")
    distractor("Agent_Diagnoser", "Pulling up dashboards now.")
    distractor("Agent_Resolver", "Standing by.")
    distractor("Agent_Detector", "Alert fired 2 minutes ago, severity SEV2.")
    fact("Agent_Diagnoser",
         "Diagnoser traced the errors to Service_Checkout's dependency on Service_Payments.",
         "Service_Checkout", "DEPENDS_ON", "Service_Payments", "f_checkout_depends_payments")
    distractor("Agent_Resolver", "Checking recent deploys to Service_Payments.")
    distractor("Agent_Detector", "Error rate climbing slightly, now at 4%.")
    distractor("Agent_Diagnoser", "Still narrowing down root cause.")
    fact("Agent_Diagnoser",
         "Diagnoser confirmed Service_Payments was OWNED_BY the Payments_Team.",
         "Service_Payments", "OWNED_BY", "Payments_Team", "f_payments_owner")
    distractor("Agent_Resolver", "Looping in Payments_Team on the incident channel.")
    distractor("Agent_Detector", "Error rate holding steady at 4%.")
    distractor("Agent_Diagnoser", "Cross-checking against the deploy log.")
    distractor("Agent_Resolver", "No response from Payments_Team yet.")
    distractor("Agent_Detector", "Still SEV2, no further degradation.")
    fact("Agent_Diagnoser",
         "Diagnoser found the root cause was a config change to Service_Payments deployed at 14:02 UTC.",
         "Service_Payments", "ROOT_CAUSE", "config change at 14:02 UTC", "f_root_cause")
    distractor("Agent_Resolver", "Got it, preparing a rollback plan.")
    distractor("Agent_Detector", "Error rate ticking up again, now 6%.")
    distractor("Agent_Diagnoser", "Confirmed the config change matches the error spike timing.")
    distractor("Agent_Resolver", "Rollback plan ready, awaiting go-ahead.")
    fact("Agent_Resolver",
         "Resolver executed the rollback and assigned the postmortem to Agent_Diagnoser.",
         "Postmortem", "ASSIGNED_TO", "Agent_Diagnoser", "f_postmortem_owner")
    distractor("Agent_Detector", "Error rate dropping, now 1%.")
    distractor("Agent_Resolver", "Monitoring for the next 15 minutes before declaring resolved.")
    distractor("Agent_Diagnoser", "Will start the postmortem doc shortly.")
    distractor("Agent_Detector", "Error rate back to baseline, 0.1%.")

    query("Agent_Resolver",
          "What is the current status of Service_Checkout?",
          "direct", ["f_checkout_status"], "elevated error rate")

    distractor("Agent_Diagnoser", "Drafting the incident timeline now.")
    distractor("Agent_Detector", "I'll attach the alert history to the doc.")
    distractor("Agent_Resolver", "Thanks, that'll help with the writeup.")
    distractor("Agent_Diagnoser", "Should have a first draft within the hour.")
    distractor("Agent_Detector", "No rush, take the time to get it right.")
    distractor("Agent_Resolver", "Agreed, accuracy matters more than speed here.")

    query("Agent_Detector",
          "What was the root cause identified for Service_Payments?",
          "distant", ["f_root_cause"], "config change at 14:02 UTC")

    distractor("Agent_Diagnoser", "Adding the root cause section now.")
    distractor("Agent_Resolver", "Let me know when it's ready for review.")
    distractor("Agent_Detector", "Will do a final check of the alert thresholds after this.")

    query("Agent_Resolver",
          "Which team owns the service that Service_Checkout depends on?",
          "join", ["f_checkout_depends_payments", "f_payments_owner"], "Payments_Team")

    distractor("Agent_Diagnoser", "Postmortem doc shared in the channel.")
    distractor("Agent_Detector", "Looks thorough, nice work.")

    query("Agent_Diagnoser",
          "Who was assigned the postmortem for this incident?",
          "direct", ["f_postmortem_owner"], "Agent_Diagnoser")

    return Scenario(
        name="incident_response",
        description="On-call incident response with longer distractor runs and two join-query path shapes.",
        turns=turns,
    )


def build_scenario_support_escalation() -> Scenario:
    """
    A customer-support triage scenario: Intake, Specialist, Manager agents.
    Includes a query where the ground truth fact is updated later in the
    conversation (a fact superseding an earlier one), to test whether each
    architecture surfaces the most current value rather than a stale one.
    """
    c = [0]
    turns: list[Turn] = []

    def fact(speaker, text, subject, predicate, obj, fact_id):
        turns.append(Turn(_t(c), TurnType.FACT, speaker, text,
                           subject=subject, predicate=predicate, object=obj,
                           fact_id=fact_id))

    def distractor(speaker, text):
        turns.append(Turn(_t(c), TurnType.DISTRACTOR, speaker, text))

    def query(speaker, text, query_type, required, ground_truth):
        turns.append(Turn(_t(c), TurnType.QUERY, speaker, text,
                           query_type=query_type,
                           required_fact_ids=tuple(required),
                           ground_truth=ground_truth))

    fact("Agent_Intake",
         "Intake logged that Ticket_4471 has priority level high.",
         "Ticket_4471", "HAS_PRIORITY", "high", "f_ticket_priority_v1")
    distractor("Agent_Specialist", "I'll pick this one up.")
    distractor("Agent_Manager", "Let me know if you need anything escalated.")
    fact("Agent_Intake",
         "Intake assigned Ticket_4471 to Agent_Specialist.",
         "Ticket_4471", "ASSIGNED_TO", "Agent_Specialist", "f_ticket_owner")
    distractor("Agent_Specialist", "Reviewing the customer's account history now.")
    distractor("Agent_Manager", "Thanks for the quick turnaround on intake.")
    fact("Agent_Specialist",
         "Specialist re-classified Ticket_4471's priority level as critical after reviewing impact.",
         "Ticket_4471", "HAS_PRIORITY", "critical", "f_ticket_priority_v2")
    distractor("Agent_Manager", "Understood, I'll keep an eye on this one.")
    distractor("Agent_Intake", "Updating the queue dashboard to reflect that.")
    distractor("Agent_Specialist", "Reaching out to the customer for more details.")
    distractor("Agent_Manager", "Let me know if you need a second pair of eyes.")
    fact("Agent_Specialist",
         "Specialist determined the root issue is tied to Component_Billing.",
         "Ticket_4471", "RELATED_TO", "Component_Billing", "f_ticket_component")
    distractor("Agent_Intake", "Noted in the ticket metadata.")
    distractor("Agent_Manager", "Billing issues tend to need extra care, good catch.")
    distractor("Agent_Specialist", "Drafting a response to the customer now.")
    fact("Agent_Manager",
         "Manager confirmed Component_Billing is owned by the Finance_Eng team.",
         "Component_Billing", "OWNED_BY", "Finance_Eng", "f_component_owner")
    distractor("Agent_Intake", "Good to know for future routing.")
    distractor("Agent_Specialist", "I'll loop them in if needed.")
    distractor("Agent_Manager", "Sounds good, keep me posted on resolution.")
    distractor("Agent_Intake", "Another ticket just came in, triaging now.")
    distractor("Agent_Specialist", "I'll finish this one up first.")

    query("Agent_Manager",
          "What is the current priority level of Ticket_4471?",
          "direct", ["f_ticket_priority_v2"], "critical")

    distractor("Agent_Intake", "New ticket looks like a duplicate, closing it.")
    distractor("Agent_Specialist", "Good call.")
    distractor("Agent_Manager", "Agreed, no need to track duplicates separately.")
    distractor("Agent_Intake", "On to the next one.")
    distractor("Agent_Specialist", "Almost done with my response draft.")

    query("Agent_Intake",
          "Who is currently assigned to Ticket_4471?",
          "distant", ["f_ticket_owner"], "Agent_Specialist")

    distractor("Agent_Manager", "Let's review resolution time once this closes.")
    distractor("Agent_Specialist", "Sounds fair, I'll log the timestamps.")

    query("Agent_Manager",
          "Which team owns the component related to Ticket_4471?",
          "join", ["f_ticket_component", "f_component_owner"], "Finance_Eng")

    return Scenario(
        name="support_escalation",
        description="Support triage scenario including a superseded-fact case (priority changes mid-conversation).",
        turns=turns,
    )


def build_scenario_data_pipeline() -> Scenario:
    """
    A data engineering scenario: Ingestor, Transformer, Validator agents
    managing a pipeline. Larger distractor density and a longer overall
    conversation to test degradation at higher turn-distance.
    """
    c = [0]
    turns: list[Turn] = []

    def fact(speaker, text, subject, predicate, obj, fact_id):
        turns.append(Turn(_t(c), TurnType.FACT, speaker, text,
                           subject=subject, predicate=predicate, object=obj,
                           fact_id=fact_id))

    def distractor(speaker, text):
        turns.append(Turn(_t(c), TurnType.DISTRACTOR, speaker, text))

    def query(speaker, text, query_type, required, ground_truth):
        turns.append(Turn(_t(c), TurnType.QUERY, speaker, text,
                           query_type=query_type,
                           required_fact_ids=tuple(required),
                           ground_truth=ground_truth))

    fact("Agent_Ingestor",
         "Ingestor set the pipeline's source format to Parquet.",
         "Pipeline_Daily", "HAS_SOURCE_FORMAT", "Parquet", "f_source_format")
    distractor("Agent_Transformer", "Ready to pick up once ingestion finishes.")
    distractor("Agent_Validator", "Will run schema checks after transform.")
    distractor("Agent_Ingestor", "Ingestion job started, ETA 10 minutes.")
    distractor("Agent_Transformer", "Noted.")
    fact("Agent_Ingestor",
         "Ingestor reported the pipeline depends on the Upstream_Orders dataset.",
         "Pipeline_Daily", "DEPENDS_ON", "Upstream_Orders", "f_pipeline_depends_orders")
    distractor("Agent_Validator", "Checking if Upstream_Orders had any schema changes recently.")
    distractor("Agent_Transformer", "I'll wait on that confirmation.")
    distractor("Agent_Ingestor", "Ingestion at 40% complete.")
    distractor("Agent_Validator", "No schema changes detected so far.")
    distractor("Agent_Transformer", "Good, that simplifies things.")
    fact("Agent_Validator",
         "Validator confirmed Upstream_Orders is OWNED_BY the Orders_Platform_Team.",
         "Upstream_Orders", "OWNED_BY", "Orders_Platform_Team", "f_orders_owner")
    distractor("Agent_Ingestor", "Ingestion at 75% complete.")
    distractor("Agent_Transformer", "Standing by.")
    distractor("Agent_Validator", "Will flag Orders_Platform_Team if anything looks off.")
    distractor("Agent_Ingestor", "Ingestion complete, handing off to transform stage.")
    distractor("Agent_Transformer", "Starting transform now.")
    fact("Agent_Transformer",
         "Transformer set the output partitioning strategy to daily partitions by event_date.",
         "Pipeline_Daily", "HAS_PARTITION_STRATEGY", "daily partitions by event_date", "f_partition_strategy")
    distractor("Agent_Validator", "Will validate partition boundaries once transform finishes.")
    distractor("Agent_Ingestor", "Let me know if you need anything re-pulled.")
    distractor("Agent_Transformer", "Transform running, ETA 5 minutes.")
    distractor("Agent_Validator", "Standing by for validation pass.")
    distractor("Agent_Ingestor", "All good on my end.")
    fact("Agent_Validator",
         "Validator detected a row count anomaly traced to Upstream_Orders.",
         "Upstream_Orders", "HAS_ANOMALY", "row count anomaly", "f_orders_anomaly")
    distractor("Agent_Transformer", "That would explain the lower output volume.")
    distractor("Agent_Ingestor", "I'll check if Orders_Platform_Team made any recent changes.")
    distractor("Agent_Validator", "Flagging this as blocking for today's run.")
    distractor("Agent_Transformer", "Pausing the transform stage until this clears.")
    distractor("Agent_Ingestor", "Reaching out to Orders_Platform_Team now.")
    distractor("Agent_Validator", "Appreciate the quick follow-up.")

    query("Agent_Transformer",
          "What source format does Pipeline_Daily ingest from?",
          "direct", ["f_source_format"], "Parquet")

    distractor("Agent_Ingestor", "Still waiting on a response from the upstream team.")
    distractor("Agent_Validator", "I'll re-run the row count check in the meantime.")
    distractor("Agent_Transformer", "Let me know what you find.")
    distractor("Agent_Ingestor", "Will do.")
    distractor("Agent_Validator", "Row counts still off by about 8%.")
    distractor("Agent_Transformer", "Noted, keeping the pause in place.")

    query("Agent_Ingestor",
          "What partitioning strategy did the transform stage configure?",
          "distant", ["f_partition_strategy"], "daily partitions by event_date")

    distractor("Agent_Validator", "Orders_Platform_Team just replied, looking into it.")
    distractor("Agent_Transformer", "Good, hopefully a quick fix.")

    query("Agent_Transformer",
          "Which team owns the dataset that currently has an anomaly?",
          "join", ["f_orders_anomaly", "f_orders_owner"], "Orders_Platform_Team")

    distractor("Agent_Ingestor", "Fix confirmed, re-running ingestion now.")
    distractor("Agent_Validator", "I'll re-validate once the new run lands.")

    query("Agent_Validator",
          "What dataset does Pipeline_Daily depend on?",
          "distant", ["f_pipeline_depends_orders"], "Upstream_Orders")

    return Scenario(
        name="data_pipeline",
        description="Data engineering pipeline scenario with longer distractor runs and anomaly-tracing join query.",
        turns=turns,
    )


def all_scenarios() -> list[Scenario]:
    return [
        build_scenario_pipeline_review(),
        build_scenario_research_pipeline(),
        build_scenario_incident_response(),
        build_scenario_support_escalation(),
        build_scenario_data_pipeline(),
    ]
