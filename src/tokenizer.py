"""
Local, deterministic, zero-dependency token estimation.

We deliberately avoid tiktoken here: it lazily downloads its BPE rank file
from a remote blob on first use, which means "no API calls" would be true
for the LLM calls but not for the tokenizer itself -- a hidden network
dependency is exactly the kind of thing that breaks reproducibility for a
reader trying to run this offline.

Instead we use a standard, well-documented approximation for GPT-style BPE
tokenization: ~4 characters per token for English prose, with a small
correction for word density (since BPE tends to split on whitespace/
punctuation more granularly than raw character count implies). This is a
widely cited estimator (OpenAI's own documentation suggests it when a real
tokenizer isn't available) and -- critically for this benchmark -- it is
applied *identically* to all three architectures, so it cannot bias the
comparison between them. What matters here is not the absolute token count
but the *relative* token cost across architectures, and a consistent
estimator preserves that comparison faithfully.
"""


def count_tokens(text: str) -> int:
    if not text:
        return 0
    char_count = len(text)
    word_count = len(text.split())
    # Base estimate: ~4 chars/token, floor'd against word count so very
    # short/dense strings don't get an unrealistically low estimate.
    estimate = max(word_count, round(char_count / 4))
    return estimate
