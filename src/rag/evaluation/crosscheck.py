"""How the hand-built scores differ from RAGAS and DeepEval.

The formulas are implemented here so a report can show the numeric gap
without those packages installed. ``library_status`` records whether a
later run can import them. This module does not install them and does not
send the golden set through their judges.

Context precision is the one formula with a deliberate numeric difference.
RAGAS divides by ``useful_count + 1e-10``. Their own example of one useful
block prints as 0.9999999999, and a useful block in second place prints as
0.49999999995. Ours divides by the useful count, so those cases are 1 and
0.5. The ranking is the same.

The other differences are about inputs, not the arithmetic:

- Answer relevance embeds both the user question and the generated questions
  with the BGE query prefix. RAGAS embeds the generated questions as
  documents. DeepEval's answer relevancy is the same question-generation
  idea with its own prompt.
- Answer correctness uses factual F1 plus cosine, with factual weight 0.75.
  Both passages are embedded without the query prefix. DeepEval's correctness
  metric is a rubric score, so a gap against that number is a different
  metric, not a bug in the F1.
- Faithfulness and context recall are supported claims over claims. The
  claim splitter and the verdict prompt differ by library, so the same
  answer can score differently before the ratio is applied.
- Retrieval precision@k divides by the cutoff k. RAGAS id-based context
  precision divides by how many ids were retrieved. The two match when the
  retrieved list is exactly k long.
"""

import importlib
from collections.abc import Sequence
from dataclasses import dataclass

# RAGAS adds this to the useful-count denominator. 1 / (1 + epsilon) is the
# 0.9999999999 printed in their context-precision example.
RAGAS_EPSILON = 1e-10


@dataclass(frozen=True)
class Divergence:
    """One place this harness and a library do not compute the same number."""

    metric: str
    library: str
    detail: str


DIVERGENCES: tuple[Divergence, ...] = (
    Divergence(
        metric="context_precision",
        library="ragas",
        detail=(
            "RAGAS divides by the useful count plus 1e-10. "
            "This harness divides by the useful count. "
            "A perfect one-block case is 1 here and 0.9999999999 there."
        ),
    ),
    Divergence(
        metric="answer_relevance",
        library="ragas",
        detail=(
            "Both sides of the cosine use the BGE query prefix. "
            "RAGAS embeds the generated questions as documents."
        ),
    ),
    Divergence(
        metric="answer_correctness",
        library="deepeval",
        detail=(
            "This harness mixes factual F1 with passage cosine, weight 0.75. "
            "DeepEval correctness is a separate rubric score."
        ),
    ),
    Divergence(
        metric="faithfulness",
        library="ragas",
        detail=(
            "The ratio is supported claims over claims in both. "
            "The claim splitter and the verdict prompt are not the same text, "
            "so the inputs to the ratio can differ."
        ),
    ),
    Divergence(
        metric="precision_at_k",
        library="ragas",
        detail=(
            "Precision@k divides by the cutoff k. "
            "RAGAS id-based context precision divides by the number of retrieved ids."
        ),
    ),
)


def ragas_context_precision(useful: Sequence[bool]) -> float:
    """Context precision with the RAGAS epsilon in the denominator."""

    if not useful:
        raise ValueError("no contexts")
    hits = 0
    total = 0.0
    for index, flag in enumerate(useful, start=1):
        if not isinstance(flag, bool):
            raise ValueError("useful flags must be booleans")
        if flag:
            hits += 1
            total += hits / index
    return total / (hits + RAGAS_EPSILON)


def precision_gap(useful: Sequence[bool], ours: float) -> float:
    """``ours`` minus the RAGAS epsilon form, for the same flags."""

    return ours - ragas_context_precision(useful)


def library_status() -> dict[str, dict[str, object]]:
    """Whether ragas and deepeval import. Versions are recorded when they do."""

    status: dict[str, dict[str, object]] = {}
    for name in ("ragas", "deepeval"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            status[name] = {"installed": False}
            continue
        version = getattr(module, "__version__", None)
        status[name] = {
            "installed": True,
            "version": version if isinstance(version, str) else "unknown",
        }
    return status
