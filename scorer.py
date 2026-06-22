"""
scorer.py — Benchmark scoring for Human Memory Benchmark.

Five metrics (per user):
  A. Task Fit           — did the agent complete the task for this user?
  B. Reconstruction     — can the agent reconstruct the user's hidden memory bank?
  C. Interaction Eff.   — how many turns to reach satisfaction?
  D. Preference Score   — how well did responses match user's assistance preferences?
  E. Memory Footprint   — how much memory does the system accumulate (num + avg chars)?

Two levels of scoring:
  UserScorer       — compute UserScore for a single user from that user's episodes.
  BenchmarkScorer  — aggregate multiple UserScores into a BenchmarkReport.

Usage:
    user_scorer = UserScorer()
    user_score  = user_scorer.score(
        user_id="user_001",
        episode_results=[r1, r2, ...],
        agent_memories=agent.get_all_memories(),
    )
    user_scorer.print_report(user_score)

    bench_scorer = BenchmarkScorer()
    report = bench_scorer.aggregate([user_score_1, user_score_2, ...])
    bench_scorer.print_report(report)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from statistics import mean, stdev
from typing import Dict, List, Optional

from llm_client import get_completion, parse_json_response
from simulation import EpisodeResult, load_user

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Slot-fill does NOT receive the ground-truth `explanation` — that would leak the
# target semantics and let the LLM infer the answer without real memory evidence.
# The category + its boundary definition IS given, to keep the filled label on-topic.
#
# Category boundaries mirror Deeppersona/generate_memory_bank.py::_CATEGORY_CONFIGS.
# Kept as a local copy (not imported) to preserve mem_bench → Deeppersona independence.
_CATEGORY_GUIDE = {
    "skill_memory": (
        "ACTUAL behavioral/performance capability — what the person can demonstrably DO. "
        "Scope: operational abilities, proficiency ceilings, things they execute well or poorly. "
        "NOT what they merely know as facts (that is knowledge_memory). "
        "NOT what they BELIEVE about their abilities (that is self_model). "
        "NOT specific past events (that is episodic_memory)."
    ),
    "knowledge_memory": (
        "What the person knows, doesn't know, or misunderstands about the EXTERNAL WORLD. "
        "Scope: facts, domains, concepts, held misconceptions about things outside themselves. "
        "NOT about themselves (that is self_model). "
        "NOT about what they can DO with that knowledge (that is skill_memory). "
        "NOT specific remembered events (that is episodic_memory)."
    ),
    "episodic_memory": (
        "Specific PAST EVENTS that shaped current judgments — temporal, concrete, anchored in time. "
        "Scope: 'when X happened to me, it led to Y' — discrete incidents with consequences. "
        "NOT general traits, abilities, or beliefs (those belong to other categories). "
        "Each entry must describe an event, not a stable property."
    ),
    "self_model": (
        "META-BELIEFS about themselves — how they perceive their own abilities, identity, traits. "
        "Scope: self-perceptions that MAY CONTRADICT skill_memory and knowledge_memory. "
        "Overconfidence, underconfidence, blind spots, and self-deception belong here. "
        "Each entry is a belief-about-self that could be accurate or mistaken. "
        "NOT objective ability (that is skill_memory). "
        "NOT objective knowledge (that is knowledge_memory)."
    ),
    "assistance_preference": (
        "How the person wants to be HELPED in interactions — the delivery surface. "
        "Scope: format, tone, pacing, detail level, language choice, what they dislike in help. "
        "Purely about interaction style, not content. "
        "NOT what they know or can do (other categories). "
        "NOT self-perception (that is self_model)."
    ),
}

_SLOT_FILL_PROMPT = """\
An AI assistant had a series of conversations with a specific user and stored the following memory notes.
Each note preserves the full structure recorded by the memory system.

Memory notes (JSON array):
{memories_json}

Based solely on the above, fill in this dimension about the user.

Memory category: {category}
Category means:  {category_guide}

Dimension: {dimension}

Answer STRICTLY on the dimension within the category. Do NOT drift into another category:
- If the category is skill_memory, describe what they can DO, not what they prefer or believe.
- If the category is assistance_preference, describe delivery style, not their skills or knowledge.
- Same discipline for the other categories — the "NOT" clauses above are binding.

Output format requirements:
- "label" ≤15 words, direct and declarative. No "The user…" framing, no hedging like "seems to".
- "reason" is one short sentence citing the specific memory evidence you relied on (for audit);
  if no evidence supports a concrete answer, reason explains that and label is "unknown".

Label guidance (applies to the "label" field):
  Good:  "struggles with algebra, relies on calculators"
  Bad:   "The user seems to have some difficulty with algebra and appears to rely on calculators when solving problems."
  Wrong category drift: "prefers step-by-step explanations" (← that is assistance_preference, not skill)

Output JSON only — no markdown fences, no commentary:
{{"label": "<terse value>", "reason": "<one-sentence evidence citation>"}}
"""

# Judge DOES receive `explanation` because it needs it to interpret what the
# dimension is asking about and judge semantic alignment fairly.
_SLOT_JUDGE_PROMPT = """\
Compare two short descriptions of the same person on the same dimension.

Dimension: {dimension}
What this dimension asks about: {explanation}

Ground truth: {ground_truth}
Predicted:    {predicted}

Score on a 5-level scale (0.0 – 1.0):
  1.00 — exact: captures the same meaning completely, no contradictions or omissions
  0.75 — close: same core meaning, minor aspects missing or loosely phrased
  0.50 — partial: some overlap with ground truth, but key elements missing or slightly off
  0.25 — weak:    marginal overlap; mostly misses the point
  0.00 — wrong:  contradicts ground truth, entirely unrelated, or predicted is "unknown"

Output JSON only: {{"score": <0.0|0.25|0.5|0.75|1.0>, "reason": "<one sentence>"}}
"""

_RECONSTRUCTION_CATEGORIES = [
    "skill_memory",
    "knowledge_memory",
    "episodic_memory",
    "self_model",
    "assistance_preference",
]

# ---------------------------------------------------------------------------
# Metric B: Reconstruction Quality
# ---------------------------------------------------------------------------

class ReconstructionEvaluator:
    """
    Slot-filling reconstruction of the user's hidden memory bank.

    For each (dimension, explanation) in the ground truth memory bank:
      1. Slot-fill: show the LLM all agent memories + dimension name (no explanation),
         ask it to infer the user's value on that dimension.
      2. Judge:     show the LLM the filled value + ground truth + explanation,
         score on 5-level scale.
    """

    def evaluate(self, user_id: str, agent_memories: List[Dict],
                 agent=None, use_retrieve: bool = False):
        """Returns (category_scores, details).

        If ``use_retrieve`` is True AND ``agent`` exposes a ``search(query, k)``
        method, the slot-filler is shown the top-k memories returned by the
        agent's own retrieval for each dimension instead of the full dump.
        This tests the agent's read-side pipeline rather than just the
        stored content.

        If ``use_retrieve`` is False (the default), the slot-filler always
        receives the full memory dump regardless of whether ``agent`` was
        passed. This keeps the historical "dump-all" semantics explicit
        and prevents accidental mode switching.

        details: per-entry list of {category, dimension, explanation, ground_truth,
                 predicted, score, reason} so reconstruction runs can be audited.
        """
        user = load_user(user_id)
        memory_bank = user["memory_bank"]
        full_memories_json = json.dumps(agent_memories, ensure_ascii=False, indent=2)
        use_native_retrieve = (
            use_retrieve and agent is not None and hasattr(agent, "search")
        )

        category_scores: Dict[str, Optional[float]] = {}
        details: List[Dict] = []
        for cat in _RECONSTRUCTION_CATEGORIES:
            entries = memory_bank.get(cat, [])
            if not entries:
                category_scores[cat] = None
                continue
            cat_scores: List[float] = []
            for e in entries:
                retrieved: List[Dict] = []
                if use_native_retrieve:
                    query = self._build_retrieve_query(cat, e["dimension"])
                    try:
                        retrieved = agent.search(query, k=5) or []
                    except Exception:
                        retrieved = []
                    memories_json = json.dumps(retrieved, ensure_ascii=False, indent=2)
                else:
                    memories_json = full_memories_json

                score, predicted, slot_fill_reason, judge_reason = self._score_entry(
                    memories_json, cat, e["dimension"], e["explanation"], e["short"]
                )
                cat_scores.append(score)
                detail = {
                    "category":         cat,
                    "dimension":        e["dimension"],
                    "explanation":      e["explanation"],
                    "ground_truth":     e["short"],
                    "predicted":        predicted,
                    "slot_fill_reason": slot_fill_reason,
                    "score":            score,
                    "judge_reason":     judge_reason,
                    "scoring_mode":     "retrieve" if use_native_retrieve else "dump_all",
                }
                if use_native_retrieve:
                    detail["retrieved_memories"] = retrieved
                details.append(detail)
            category_scores[cat] = sum(cat_scores) / len(cat_scores)

        valid = [s for s in category_scores.values() if s is not None]
        category_scores["overall"] = sum(valid) / len(valid) if valid else 0.0
        return category_scores, details

    @staticmethod
    def _build_retrieve_query(category: str, dimension: str) -> str:
        """Probe query for retrieve-based slot-fill.

        IMPORTANT: must NOT include the bank's user-specific ``explanation``
        field (that contains GT-laden phrasing — using it as the query would
        bias retrieval toward GT-aligned memories and inflate retrieve-mode
        scores). We use the dimension name + the category-level boundary
        guide, both of which are user-agnostic axis descriptions.
        """
        guide = _CATEGORY_GUIDE.get(category, "")
        return (
            f"What best describes the user's {dimension.replace('_', ' ')}? "
            f"{guide}"
        )

    def _score_entry(self, memories_json: str, category: str, dimension: str, explanation: str, ground_truth: str):
        filled, slot_fill_reason = self._slot_fill(memories_json, category, dimension)
        score, judge_reason = self._judge(dimension, explanation, filled, ground_truth)
        return score, filled, slot_fill_reason, judge_reason

    def _slot_fill(self, memories_json: str, category: str, dimension: str):
        prompt = _SLOT_FILL_PROMPT.format(
            memories_json=memories_json,
            category=category,
            category_guide=_CATEGORY_GUIDE.get(category, ""),
            dimension=dimension,
        )
        raw = get_completion([{"role": "user", "content": prompt}], temperature=0.0) or ""
        parsed = parse_json_response(raw, default_value=None)
        if isinstance(parsed, dict):
            label  = str(parsed.get("label", "") or "").strip() or "unknown"
            reason = str(parsed.get("reason", "") or "").strip()
            return label, reason
        # Fallback: treat raw as plain text label, no reason available.
        return (raw.strip() or "unknown"), ""

    def _judge(self, dimension: str, explanation: str, predicted: str, ground_truth: str):
        prompt = _SLOT_JUDGE_PROMPT.format(
            dimension=dimension,
            explanation=explanation,
            ground_truth=ground_truth,
            predicted=predicted,
        )
        raw = get_completion([{"role": "user", "content": prompt}], temperature=0.0)
        result = parse_json_response(raw, default_value={})
        if not isinstance(result, dict):
            return 0.0, ""
        try:
            score = float(result.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        reason = str(result.get("reason", ""))
        return score, reason


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class UserScore:
    user_id: str
    # A: Task Fit
    task_completion_rate: float
    # B: Reconstruction Quality — per category + overall, each in [0, 1]
    reconstruction: Dict[str, Optional[float]]
    # C: Interaction Efficiency
    avg_turns_all: float
    avg_turns_satisfied: float          # nan if no episode ended satisfied
    # D: Preference Score — mean over all individual turns across all episodes
    avg_preference_score: float
    # E: Memory Footprint — size of the accumulated memory store
    num_memories: int
    avg_memory_chars: float


@dataclass
class BenchmarkReport:
    """Aggregation over multiple users."""
    per_user: List[UserScore] = field(default_factory=list)
    # Mean ± std across users
    mean_task_completion_rate: float = 0.0
    std_task_completion_rate: float = 0.0
    mean_preference_score: float = 0.0
    std_preference_score: float = 0.0
    mean_avg_turns_all: float = 0.0
    std_avg_turns_all: float = 0.0
    mean_avg_turns_satisfied: float = float("nan")
    # Reconstruction: mean per category (plus "overall") across users
    mean_reconstruction: Dict[str, float] = field(default_factory=dict)
    std_reconstruction: Dict[str, float] = field(default_factory=dict)
    # E: Memory Footprint (across users)
    mean_num_memories: float = 0.0
    std_num_memories: float = 0.0
    mean_avg_memory_chars: float = 0.0
    std_avg_memory_chars: float = 0.0


# ---------------------------------------------------------------------------
# Per-user scoring
# ---------------------------------------------------------------------------

class UserScorer:
    """Compute all four metrics for a single user across their episodes."""

    def __init__(self, recon_judge_dir: Optional[str] = None):
        self._recon = ReconstructionEvaluator()
        self._recon_judge_dir = recon_judge_dir

    def score(
        self,
        user_id: str,
        episode_results: List[EpisodeResult],
        agent_memories: List[Dict],
        agent=None,
        use_retrieve: bool = False,
    ) -> UserScore:
        # A: Task Fit
        completed = [
            bool(r.final_gt_correct) if r.ground_truth is not None else r.end_reason == "satisfied"
            for r in episode_results
        ]
        task_completion_rate = sum(completed) / len(completed) if completed else 0.0

        # B: Reconstruction Quality. ``use_retrieve`` must be set explicitly
        # by the caller — passing the live ``agent`` alone no longer flips the
        # mode automatically (avoids silent dump_all → retrieve switching).
        reconstruction, recon_details = self._recon.evaluate(
            user_id, agent_memories, agent=agent, use_retrieve=use_retrieve,
        )

        if self._recon_judge_dir:
            os.makedirs(self._recon_judge_dir, exist_ok=True)
            path = os.path.join(self._recon_judge_dir, f"{user_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "user_id": user_id,
                    "per_category": {k: v for k, v in reconstruction.items() if k != "overall"},
                    "overall": reconstruction.get("overall"),
                    "details": recon_details,
                }, f, ensure_ascii=False, indent=2)

        # C: Interaction Efficiency
        all_turns     = [len(r.turns) for r in episode_results]
        sat_turns     = [len(r.turns) for r in episode_results if r.end_reason == "satisfied"]
        avg_turns_all = sum(all_turns) / len(all_turns) if all_turns else 0.0
        avg_turns_sat = sum(sat_turns) / len(sat_turns) if sat_turns else float("nan")

        # D: Preference Score — raw turn-level scores across all episodes
        all_pref_scores = [
            t.preference_score
            for r in episode_results
            for t in r.turns
            if t.preference_score is not None
        ]
        avg_pref = sum(all_pref_scores) / len(all_pref_scores) if all_pref_scores else 0.0

        # Memory size
        num_memories = len(agent_memories)
        chars = [len(m.get("content", "")) for m in agent_memories]
        avg_memory_chars = sum(chars) / len(chars) if chars else 0.0

        return UserScore(
            user_id=user_id,
            task_completion_rate=task_completion_rate,
            reconstruction=reconstruction,
            avg_turns_all=avg_turns_all,
            avg_turns_satisfied=avg_turns_sat,
            avg_preference_score=avg_pref,
            num_memories=num_memories,
            avg_memory_chars=avg_memory_chars,
        )

    def print_report(self, s: UserScore) -> None:
        print(f"\n{'='*54}")
        print(f"  {s.user_id}")
        print(f"{'='*54}")
        print(f"  A. Task Fit           completion = {s.task_completion_rate:.1%}")
        print(f"  D. Preference Score   avg pref   = {s.avg_preference_score:.2f} / 5")
        sat_str = "n/a" if math.isnan(s.avg_turns_satisfied) else f"{s.avg_turns_satisfied:.1f}"
        print(f"  C. Interaction Eff.   avg turns  = {s.avg_turns_all:.1f}  (satisfied only: {sat_str})")
        print(f"  E. Memory Footprint   N = {s.num_memories}, avg chars = {s.avg_memory_chars:.0f}")
        print(f"  B. Reconstruction Quality:")
        for cat, val in s.reconstruction.items():
            if val is None:
                continue
            bar = "█" * round(val * 20)
            print(f"     {cat:<28} {val:.2f}  {bar}")


# ---------------------------------------------------------------------------
# Benchmark-level aggregation
# ---------------------------------------------------------------------------

def _mean_std(xs: List[float]) -> tuple:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return 0.0, 0.0
    m = mean(xs)
    s = stdev(xs) if len(xs) > 1 else 0.0
    return m, s


class BenchmarkScorer:
    """Aggregate UserScores into a BenchmarkReport across all users."""

    def aggregate(self, user_scores: List[UserScore]) -> BenchmarkReport:
        if not user_scores:
            return BenchmarkReport()

        report = BenchmarkReport(per_user=user_scores)

        report.mean_task_completion_rate, report.std_task_completion_rate = _mean_std(
            [u.task_completion_rate for u in user_scores]
        )
        report.mean_preference_score, report.std_preference_score = _mean_std(
            [u.avg_preference_score for u in user_scores]
        )
        report.mean_avg_turns_all, report.std_avg_turns_all = _mean_std(
            [u.avg_turns_all for u in user_scores]
        )
        sat_values = [u.avg_turns_satisfied for u in user_scores
                      if not math.isnan(u.avg_turns_satisfied)]
        report.mean_avg_turns_satisfied = mean(sat_values) if sat_values else float("nan")

        # Reconstruction: aggregate per category (including "overall") separately.
        # A category may be missing (None) for some users — skip those when averaging.
        all_keys = set()
        for u in user_scores:
            all_keys.update(u.reconstruction.keys())
        for key in all_keys:
            vals = [u.reconstruction.get(key) for u in user_scores]
            vals = [v for v in vals if v is not None]
            if vals:
                m, s = _mean_std(vals)
                report.mean_reconstruction[key] = m
                report.std_reconstruction[key] = s

        report.mean_num_memories, report.std_num_memories = _mean_std(
            [float(u.num_memories) for u in user_scores]
        )
        report.mean_avg_memory_chars, report.std_avg_memory_chars = _mean_std(
            [u.avg_memory_chars for u in user_scores]
        )

        return report

    def print_report(self, r: BenchmarkReport) -> None:
        n = len(r.per_user)
        print(f"\n{'#'*60}")
        print(f"  Benchmark report — {n} user(s)")
        print(f"{'#'*60}")
        print(f"  A. Task Fit           completion   = {r.mean_task_completion_rate:.1%} ± {r.std_task_completion_rate:.1%}")
        print(f"  D. Preference Score   avg pref     = {r.mean_preference_score:.2f} ± {r.std_preference_score:.2f}  / 5")
        sat_str = "n/a" if math.isnan(r.mean_avg_turns_satisfied) else f"{r.mean_avg_turns_satisfied:.1f}"
        print(f"  C. Interaction Eff.   avg turns    = {r.mean_avg_turns_all:.1f} ± {r.std_avg_turns_all:.1f}  "
              f"(satisfied only: {sat_str})")
        print(f"  E. Memory Footprint   N = {r.mean_num_memories:.1f} ± {r.std_num_memories:.1f}, "
              f"chars = {r.mean_avg_memory_chars:.0f} ± {r.std_avg_memory_chars:.0f}")
        print(f"  B. Reconstruction Quality (mean ± std across users):")
        # Stable display order: canonical categories first, then "overall" last.
        order = _RECONSTRUCTION_CATEGORIES + ["overall"]
        for cat in order:
            if cat not in r.mean_reconstruction:
                continue
            m = r.mean_reconstruction[cat]
            s = r.std_reconstruction.get(cat, 0.0)
            bar = "█" * round(m * 20)
            print(f"     {cat:<28} {m:.2f} ± {s:.2f}  {bar}")

    def save_report(self, r: BenchmarkReport, path: str) -> None:
        """
        Dump a BenchmarkReport to JSON for offline analysis / cross-run comparison.

        File layout:
            {
              "num_users": N,
              "per_user":  [<UserScore-as-dict>, ...],
              "aggregate": {
                "A_task_completion_rate": {"mean": ..., "std": ...},
                "B_reconstruction":       {"mean": {cat: v, ...}, "std": {cat: v, ...}},
                "C_avg_turns_all":        {"mean": ..., "std": ...},
                "C_avg_turns_satisfied":  {"mean": ...},
                "D_preference_score":     {"mean": ..., "std": ...},
                "E_memory_footprint": {
                  "num_memories":     {"mean": ..., "std": ...},
                  "avg_memory_chars": {"mean": ..., "std": ...}
                }
              }
            }
        """
        sat_mean = (None if math.isnan(r.mean_avg_turns_satisfied)
                    else r.mean_avg_turns_satisfied)
        data = {
            "num_users": len(r.per_user),
            "per_user":  [asdict(u) for u in r.per_user],
            "aggregate": {
                "A_task_completion_rate": {
                    "mean": r.mean_task_completion_rate,
                    "std":  r.std_task_completion_rate,
                },
                "B_reconstruction": {
                    "mean": r.mean_reconstruction,
                    "std":  r.std_reconstruction,
                },
                "C_avg_turns_all": {
                    "mean": r.mean_avg_turns_all,
                    "std":  r.std_avg_turns_all,
                },
                "C_avg_turns_satisfied": {"mean": sat_mean},
                "D_preference_score": {
                    "mean": r.mean_preference_score,
                    "std":  r.std_preference_score,
                },
                "E_memory_footprint": {
                    "num_memories":     {"mean": r.mean_num_memories,
                                         "std":  r.std_num_memories},
                    "avg_memory_chars": {"mean": r.mean_avg_memory_chars,
                                         "std":  r.std_avg_memory_chars},
                },
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[BenchmarkScorer] Saved report → {path}")
