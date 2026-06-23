"""
intent.py
---------
Rule-based intent classifier for 5 intent types.
No ML model — deterministic, seed-independent, fully reproducible.

Intent types:
  informational  — what/why/how does X work
  troubleshooting — something is broken/failing/not working
  comparison      — comparing options, plans, tiers
  analytical      — trends, risks, metrics, patterns
  procedural      — step-by-step how to do X
"""

import re
from typing import Dict, List, Tuple

INTENT_RULES: Dict[str, List[str]] = {
    "procedural": [
        r"\bhow (do|can|should|to) (i|we|you)\b",
        r"\bsteps? (to|for)\b",
        r"\bset up\b",
        r"\bconfigure\b",
        r"\benable\b",
        r"\bcreate\b",
        r"\breset\b",
        r"\bexport\b",
        r"\badd (a |an |new )?\b",
        r"\binstall\b",
        r"\bdeploy\b",
        r"\bprovision\b",
        r"\brotate\b",
        r"\bgenerate (a |an )?\b",
    ],
    "troubleshooting": [
        r"\bcan'?t\b",
        r"\bnot working\b",
        r"\bbroken\b",
        r"\bfailing\b",
        r"\berror\b",
        r"\bissue\b",
        r"\bproblem\b",
        r"\bnot (able|connecting|loading|responding)\b",
        r"\bwon'?t\b",
        r"\bslow(er)?\b",
        r"\bdown\b",
        r"\btimeout\b",
        r"\bkicked out\b",
        r"\bunable to\b",
        r"\bwhy (is|are|does|do|did)\b.*\b(not|fail|broken|slow|wrong)\b",
    ],
    "comparison": [
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bdifference between\b",
        r"\bcompare\b",
        r"\bwhich (is|has|plan|option|tier)\b",
        r"\bbetter\b",
        r"\bstack up\b",
        r"\bpros? and cons?\b",
        r"\bor\b.{0,20}\bor\b",
        r"\beach (tier|plan|option|package)\b",
        r"\bdoes .{0,20} include\b",
    ],
    "analytical": [
        r"\bat risk\b",
        r"\btrend(s|ing)?\b",
        r"\bwhy (are|is) .{0,30} (drop|leav|churn|declin|fall)",
        r"\bwhat (drives?|causes?|affects?)\b",
        r"\bwhich (accounts?|customers?|segments?|users?)\b",
        r"\bperformance\b",
        r"\bmetrics?\b",
        r"\banalytics?\b",
        r"\binsight\b",
        r"\bpattern\b",
        r"\bforecast\b",
        r"\bpredic\b",
        r"\bchurn\b",
        r"\bretention\b",
        r"\blosing\b",
        r"\bunderused\b",
        r"\bengaged\b",
        r"\bdropping off\b",
        r"\bprioritize\b",
    ],
    "informational": [
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bhow does\b",
        r"\bhow do\b(?!.{0,5}\bi\b)",  # "how do" but not "how do I"
        r"\bwhy does\b",
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\btell me about\b",
        r"\bwhat (security|encryption|protocol|standard)\b",
        r"\bhow (is|are) .{0,20} (stored|kept|managed|handled)\b",
        r"\bwhere (is|are)\b",
        r"\bwho (can|is)\b",
    ],
}

# Priority order — more specific intents checked first
PRIORITY = ["procedural", "troubleshooting", "comparison", "analytical", "informational"]


def classify_intent(query: str) -> str:
    """
    Returns the detected intent label.
    Checks rules in priority order; returns first match.
    Falls back to 'informational' if no rules match.
    """
    q = query.lower()

    for intent in PRIORITY:
        for pattern in INTENT_RULES[intent]:
            if re.search(pattern, q):
                return intent

    return "informational"


def classify_with_confidence(query: str) -> Tuple[str, Dict[str, int]]:
    """
    Returns (intent, scores_dict) where scores_dict counts rule hits per intent.
    Useful for debugging and article illustrations.
    """
    q = query.lower()
    scores: Dict[str, int] = {intent: 0 for intent in PRIORITY}

    for intent in PRIORITY:
        for pattern in INTENT_RULES[intent]:
            if re.search(pattern, q):
                scores[intent] += 1

    # Pick intent by score, tie-break by priority order
    best_intent = max(PRIORITY, key=lambda i: (scores[i], -PRIORITY.index(i)))
    if scores[best_intent] == 0:
        best_intent = "informational"

    return best_intent, scores


if __name__ == "__main__":
    test_queries = [
        ("how do I reset my API key", "procedural"),
        ("users can't sign in after identity provider changes", "troubleshooting"),
        ("which plan is better for enterprise", "comparison"),
        ("which customers are at risk of churn", "analytical"),
        ("how does the authentication protocol work", "informational"),
        ("how do I configure SSO for my organization", "procedural"),
        ("login broken after the latest update", "troubleshooting"),
        ("what drives customer retention", "analytical"),
    ]

    print(f"{'Query':<55} {'Expected':<16} {'Predicted':<16} {'Match'}")
    print("-" * 100)
    correct = 0
    for query, expected in test_queries:
        predicted = classify_intent(query)
        match = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        print(f"{query:<55} {expected:<16} {predicted:<16} {match}")

    print()
    print(f"Accuracy: {correct}/{len(test_queries)}")
