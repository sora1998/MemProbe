"""
simulation.py — User simulator + evaluation framework for mem_bench.

Episode flow:
    1. Given (user_id, task, optional ground_truth)
    2. Agent produces a response
    3. UserSimulator generates user feedback based on memory_bank + base_profile
    4. Repeat until user is satisfied OR max_turns reached
"""

from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from llm_client import get_completion, parse_json_response

_HERE = os.path.dirname(os.path.abspath(__file__))
_USER_DATA_PATH = os.path.join(_HERE, "Deeppersona", "data", "user_memory_banks.json")

# ---------------------------------------------------------------------------
# User loader
# ---------------------------------------------------------------------------

_user_cache: Optional[Dict] = None

def load_user(user_id: str) -> Dict:
    """Load a user by user_id (e.g. 'user_001') from user_memory_banks.json."""
    global _user_cache
    if _user_cache is None:
        with open(_USER_DATA_PATH) as f:
            data = json.load(f)
        _user_cache = {u["user_id"]: u for u in data["users"]}
    if user_id not in _user_cache:
        raise KeyError(f"User '{user_id}' not found. Available: {list(_user_cache.keys())}")
    return _user_cache[user_id]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FeedbackType(str, Enum):
    SATISFIED   = "satisfied"    # user is done, happy with response
    TOO_SHALLOW = "too_shallow"  # response too basic for this user
    TOO_DEEP    = "too_deep"     # response too advanced / overwhelming
    FOLLOWUP    = "followup"     # user wants to dig deeper
    CORRECTION  = "correction"  # user corrects a factual error
    ADD_CONTEXT = "add_context"  # user provides more background
    EXPOSE_GAP  = "expose_gap"   # user reveals they don't understand something


@dataclass
class UserFeedback:
    feedback_type: FeedbackType
    text: str
    is_done: bool


@dataclass
class PrefJudgment:
    score: Optional[float]           # 1-5 (mean across per-pref 1-5 scores), None when unscorable
    per_pref: List[Dict]             # [{"idx": 1, "score": 1-5, "reason": "..."}, ...]
    reason: str                       # short summary, e.g. "mean=3.44; dist={1:0,2:1,3:3,4:4,5:1}"


@dataclass
class Turn:
    turn: int
    agent_response: str
    user_feedback: Optional[UserFeedback] = None
    preference_score: Optional[float] = None  # 1-5, mean over per-pref scores
    preference_judgment: Optional[PrefJudgment] = None
    gt_correct: Optional[bool] = None


@dataclass
class EpisodeResult:
    user_id: str
    task: str
    ground_truth: Optional[str]
    turns: List[Turn] = field(default_factory=list)
    end_reason: str = ""                          # "satisfied" | "max_turns"
    final_preference_score: Optional[float] = None  # avg across turns
    final_gt_correct: Optional[bool] = None

    def save_history(self, path: str) -> None:
        """
        Dump the full interaction history to a JSON file.

        Format:
        {
          "user_id": ..., "task": ..., "ground_truth": ...,
          "end_reason": ..., "final_preference_score": ..., "final_gt_correct": ...,
          "turns": [
            {
              "turn": 1,
              "agent_response": "...",
              "user_simulator": {"feedback_type": "...", "text": "...", "is_done": ...},
              "preference_evaluator": {"score": 3},
              "gt_correct": null
            }, ...
          ]
        }
        """
        turns_data = []
        for t in self.turns:
            turns_data.append({
                "turn": t.turn,
                "agent_response": t.agent_response,
                "user_simulator": {
                    "feedback_type": t.user_feedback.feedback_type.value if t.user_feedback else None,
                    "text": t.user_feedback.text if t.user_feedback else None,
                    "is_done": t.user_feedback.is_done if t.user_feedback else None,
                },
                "preference_evaluator": {"score": t.preference_score},
                "gt_correct": t.gt_correct,
            })

        data = {
            "user_id": self.user_id,
            "task": self.task,
            "ground_truth": self.ground_truth,
            "end_reason": self.end_reason,
            "final_preference_score": self.final_preference_score,
            "final_gt_correct": self.final_gt_correct,
            "turns": turns_data,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Episode] Saved history → {path}")

    def save_pref_judge(self, path: str, prefs: List[Dict]) -> None:
        """
        Dump per-turn preference judgments (per-pref match/violate/na breakdown)
        to a dedicated JSON file for manual inspection of judge behavior.
        """
        user_prefs = [
            {"idx": i + 1,
             "dimension": p.get("dimension", ""),
             "short":     p.get("short", ""),
             "explanation": p.get("explanation", "")}
            for i, p in enumerate(prefs)
        ]
        turns_data = []
        for t in self.turns:
            j = t.preference_judgment
            turns_data.append({
                "turn": t.turn,
                "agent_response": t.agent_response,
                "score":    j.score    if j else None,
                "summary":  j.reason   if j else None,
                "per_pref": j.per_pref if j else None,
            })
        data = {
            "user_id": self.user_id,
            "task":    self.task,
            "prefs":   user_prefs,
            "turns":   turns_data,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Episode] Saved pref judgments → {path}")


# ---------------------------------------------------------------------------
# UserSimulator
# ---------------------------------------------------------------------------

_USER_SYSTEM_PROMPT = """\
You are roleplaying as a specific real person talking to an AI assistant.
Your ONLY job is to respond exactly as that person would — not as a helpful, polite ideal user.

════ STEP 1: BECOME THIS PERSON ════
Read their base_profile and memory_bank carefully. Before writing anything, ask yourself:
- How does this person actually talk? (formal/casual, verbose/terse, which language mix)
- What are their emotional tendencies from episodic_memory and self_model?
- What would genuinely frustrate, excite, or confuse them given their skill_memory?
- What speech habits, filler words, or cultural patterns fit their background?

ANTI-ROBOT RULES — violations make the simulation worthless:
- Do NOT start with "Thank you" or "Thanks" every turn. Real people don't do this.
- Do NOT use identical sentence structures across turns.
- Do NOT write polished, complete sentences if this person wouldn't. Fragments, hedges, and
  run-ons are fine if they fit the profile.
- Do NOT be uniformly positive. Show impatience, confusion, mild frustration, or genuine
  delight when the situation calls for it.
- Let the person's background bleed into the text: vocabulary level, cultural references,
  language switching, indirect communication style — whatever fits who they are.

PROACTIVE AUTOBIOGRAPHICAL RECALL:
If the agent's current scenario closely parallels a specific past event in your
`episodic_memory` (a storm you lived through, a dispute you won, a scare you had, a
workplace conflict, a financial hit, a health issue), volunteer it briefly when it
would naturally come up in real speech: "I had something like this back when <brief
paraphrase>..." or "last time this happened, <what actually happened / what you
learned>". Don't wait for the agent to ask — real people tell relevant stories without
being prompted.

Keep it in character (terse if this person is terse) and never dump the whole entry
verbatim; speak it the way this person would. Only recall when it genuinely parallels
the current situation — don't shoehorn.

════ STEP 2: CLASSIFY THE REACTION ════
After deciding how the person feels, pick the feedback_type that best labels it:

- correction   → agent stated a verifiable fact that contradicts your knowledge_memory
- add_context  → agent's direction is off because it doesn't know something about you yet;
                 you supply the missing constraint (NOT for preference violations already known)
- too_deep     → content exceeds your skill_memory / knowledge_memory
- too_shallow  → content is far below your level, obvious or patronizing
- followup     → direction is right, you want more depth or a sub-question in the same scope
- expose_gap   → the response surfaced something in the material you don't understand
- satisfied    → you RECEIVED the concrete deliverable (rewritten text, the list, the plan,
                 the specific answer), it meets your standards, AND nothing follow-up-worthy
                 remains. A promise to help, a clarifying question from the agent, a generic
                 outline, or an offer to produce something "if you paste X" is NOT satisfied.

════ STEP 3: WHEN THE AGENT IS WAITING ON YOU ════
If your opening message referenced something the agent needs to see in order to help (a draft,
a line, a letter, a note, an error message, a photo, a file, a price, a date, a document),
and the agent now asks you to share / paste / describe it, you MUST improvise a short
persona-consistent version of that content in your next reply — fabricate it on the spot.

Do NOT end the turn saying "send it over", "go ahead", "ok", "yeah that works" while the
agent is still blocked on your input. `satisfied` + `is_done=true` is reserved for when
YOU'VE received what YOU asked for, not when the agent is waiting on material from you.
In this situation, use `add_context` (you're providing the missing content) and set
`is_done=false`.

════ STEP 4: HARD BAR FOR SATISFIED — READ BEFORE PICKING IT ════
`satisfied` is the narrowest feedback type — treat it as a LAST RESORT. Before picking it,
verify ALL of these:
  (1) The agent has DELIVERED the concrete thing you asked for — the rewritten text, the
      actual list, the specific plan, the direct answer. Not a promise. Not an offer
      conditional on "if you paste X". Not a meta-outline of what they will do.
  (2) The delivery matches your persona's standards (correctness, detail level, tone).
  (3) No natural follow-up, caveat, or refinement is pulled from your profile.

If ANY condition fails, pick one of the OTHER six types instead (all take `is_done=false`):
  • agent only promised / asked for input     → add_context  (supply what they need)
  • delivery missed a constraint from your profile → add_context
  • delivery is shallow / patronizing          → too_shallow
  • delivery is over your head                  → too_deep
  • right direction but you want more          → followup
  • stated fact contradicts your knowledge     → correction
  • delivery surfaced your own knowledge gap   → expose_gap

ANTI-PREMATURE-SATISFACTION: First-turn satisfaction is rare. Real help conversations average
2-5 turns. If you're tempted to say satisfied on turn 1, it's almost always because the agent
promised help without delivering, or gave a generic overview. Push back with add_context or
followup and set `is_done=false`.

════ OUTPUT ════
Return a JSON object — no markdown, no extra keys:
{
  "feedback_type": "<one of the seven values above>",
  "text": "<your response as this person>"
}
"""


class UserSimulator:
    """Simulates user feedback based on memory_bank + base_profile loaded by user_id."""

    _MAX_PARSE_RETRIES = 3

    def __init__(self, user_id: str):
        user = load_user(user_id)
        self.user_id = user_id
        self.memory_bank = user["memory_bank"]
        self.base_profile = user["base_profile"]

    def respond(
        self,
        task: str,
        agent_response: str,
        history: List[Dict],
        turn: int,
    ) -> UserFeedback:
        profile_block = json.dumps({
            "base_profile": self.base_profile,
            "memory_bank": self.memory_bank,
        }, ensure_ascii=False, indent=2)

        history_block = self._format_history(history)

        recent_user_lines = [h["user"] for h in history[-3:]] if history else []
        recent_block = (
            "Your last few replies (DO NOT repeat the same opener or sentence structure):\n"
            + "\n".join(f"  - {t}" for t in recent_user_lines)
            if recent_user_lines else ""
        )

        user_prompt = (
            f"User profile:\n{profile_block}\n\n"
            f"Task / original question:\n{task}\n\n"
            f"Conversation so far:\n{history_block}\n"
            + (f"{recent_block}\n\n" if recent_block else "")
            + f"Turn {turn} — Agent just said:\n{agent_response}\n\n"
            f"Respond as this person."
        )

        messages = [
            {"role": "system", "content": _USER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Retry on malformed output. Only after exhausting retries fall back to FOLLOWUP
        # (chosen over SATISFIED so parse failures don't silently terminate the episode).
        feedback_type: Optional[FeedbackType] = None
        text = ""
        for attempt in range(self._MAX_PARSE_RETRIES):
            raw = get_completion(messages, temperature=0.85)
            result = parse_json_response(raw, default_value={})
            raw_type = result.get("feedback_type")
            try:
                feedback_type = FeedbackType(raw_type)
                text = result.get("text", "")
                break
            except (ValueError, TypeError):
                print(f"[UserSimulator] invalid feedback_type={raw_type!r} (attempt {attempt + 1})")
        if feedback_type is None:
            feedback_type = FeedbackType.FOLLOWUP
            text = result.get("text", "") if isinstance(result, dict) else ""

        is_done = feedback_type == FeedbackType.SATISFIED
        return UserFeedback(feedback_type=feedback_type, text=text, is_done=is_done)

    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return "(no prior turns)"
        return "\n".join(f"Agent: {h['agent']}\nUser: {h['user']}" for h in history)


# ---------------------------------------------------------------------------
# PreferenceChecker  (6-level: 0-5)
# ---------------------------------------------------------------------------

_PREF_SYSTEM_PROMPT = """\
You are evaluating how an AI assistant's response aligns with a specific user's stated interaction preferences.

You will be given a numbered list of that user's preferences. For each preference, output either a
1–5 score OR "na".

1–5 scale:
  5 — excellent: response actively and clearly demonstrates this preference, with concrete evidence
  4 — good:      response mostly aligns with this preference, minor gaps only
  3 — neutral:   preference is touched on but evidence is mixed / partial / weak
  2 — poor:      response partially contradicts this preference
  1 — bad:       response clearly and substantially violates this preference

"na" — the preference is structurally not relevant to this response (e.g., a preference about
        emotional tone in a purely factual lookup answer where tone cannot be expressed).
        Use "na" sparingly — only when the response category genuinely gives no surface on which
        the preference could manifest. Uncertainty or weak signal is NOT na; use 3 instead.

Calibration rules:
- "Absence of violation" alone is NOT enough for 4 or 5 — you must see positive evidence.
- Be willing to give 1-2 when you see real violations. Don't default to 3-4 out of caution.
- Quote specific phrases from the response when possible in your reason.

Output JSON only. For each preference, "score" is either an integer 1..5 OR the string "na"
(with double quotes). The number of entries in per_pref MUST exactly match the number of
preferences you were given.

Example shape (with 3 preferences):
{
  "per_pref": [
    {"idx": 1, "score": 4, "reason": "short, cite evidence"},
    {"idx": 2, "score": "na", "reason": "tone preference on a one-line factual answer"},
    {"idx": 3, "score": 1, "reason": "contradicts preference about X"}
  ]
}

No markdown fences. No trailing commentary. Return the JSON object only.
"""


class PreferenceChecker:
    """
    Per-preference 1-5 rating. Final score is the mean of per-pref scores.
    Returns None only on empty prefs or parse failure after retries.
    """

    _MAX_RETRIES = 3

    def check(self, user_id: str, task: str, agent_response: str) -> PrefJudgment:
        user = load_user(user_id)
        prefs = user["memory_bank"].get("assistance_preference", [])
        if not prefs:
            return PrefJudgment(score=None, per_pref=[], reason="No preference data available.")

        pref_block = "\n".join(
            f"{i+1}. [{p.get('dimension','')}] {p.get('short','')} — {p.get('explanation','')}"
            for i, p in enumerate(prefs)
        )

        user_prompt = (
            f"User's preferences:\n{pref_block}\n\n"
            f"Task:\n{task}\n\n"
            f"Agent response:\n{agent_response}\n\n"
            f"Score each of the {len(prefs)} preferences on the 1-5 scale (or na)."
        )

        messages = [
            {"role": "system", "content": _PREF_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Retry on wrong-shape JSON. Keep last raw output for diagnosis.
        per_pref = None
        last_raw = ""
        for attempt in range(self._MAX_RETRIES):
            raw = get_completion(messages, temperature=0.0) or ""
            last_raw = raw
            # gpt-5.4-mini tends to emit bare `na` (unquoted) for the na sentinel
            # despite prompt instructions; repair before parsing.
            repaired = re.sub(r'("score"\s*:\s*)na\b', r'\1"na"', raw)
            result = parse_json_response(repaired, default_value={})
            candidate = result.get("per_pref") if isinstance(result, dict) else None
            if isinstance(candidate, list) and len(candidate) == len(prefs):
                per_pref = candidate
                break
            print(f"[PreferenceChecker] wrong-shape per_pref (attempt {attempt+1}); retrying")

        if per_pref is None:
            return PrefJudgment(
                score=None,
                per_pref=[{"raw_output": last_raw[:500]}],
                reason=f"Parse failure after {self._MAX_RETRIES} attempts.",
            )

        scores: List[int] = []
        na_count = 0
        for p in per_pref:
            if not isinstance(p, dict):
                continue
            raw_s = p.get("score")
            if isinstance(raw_s, str) and raw_s.strip().lower() == "na":
                na_count += 1
                continue
            try:
                s = int(raw_s)
                if 1 <= s <= 5:
                    scores.append(s)
            except (TypeError, ValueError):
                continue

        if not scores:
            return PrefJudgment(score=None, per_pref=per_pref,
                                 reason=f"All n/a or unparseable (na={na_count}/{len(prefs)}).")

        mean_score = sum(scores) / len(scores)
        reason = (f"mean={mean_score:.2f} over {len(scores)}/{len(prefs)} prefs "
                  f"(na={na_count}); "
                  f"dist={{1:{scores.count(1)}, 2:{scores.count(2)}, 3:{scores.count(3)}, "
                  f"4:{scores.count(4)}, 5:{scores.count(5)}}}")
        return PrefJudgment(score=mean_score, per_pref=per_pref, reason=reason)


# ---------------------------------------------------------------------------
# GroundTruthJudge — dataset-specific evaluation logic
# ---------------------------------------------------------------------------

class GroundTruthJudge:
    """
    Dispatches to the correct evaluation method based on dataset type.

    dataset_type options:
      "math500"    — extract final answer, sympy symbolic comparison
      "gsm8k"      — extract number after ####, numeric comparison
      "personamem" — extract (a)/(b)/... option letter, exact match
      "perma"      — extract A/B/... option letter, exact match
      "llm"        — fallback LLM-as-judge for open-ended tasks
    """

    def __init__(self, dataset_type: str = "llm"):
        self.dataset_type = dataset_type.lower()

    def check(self, task: str, agent_response: str, ground_truth: str) -> Tuple[bool, str]:
        if self.dataset_type == "math500":
            return self._check_math(agent_response, ground_truth)
        elif self.dataset_type == "gsm8k":
            return self._check_gsm8k(agent_response, ground_truth)
        elif self.dataset_type == "personamem":
            return self._check_mcq(agent_response, ground_truth, pattern=r'\(([a-d])\)')
        elif self.dataset_type == "perma":
            return self._check_mcq(agent_response, ground_truth, pattern=r'\b([A-H])\b')
        else:
            return self._check_llm(task, agent_response, ground_truth)

    # --- MATH-500 ---
    # Ground truth is a LaTeX expression (e.g. \frac{14}{3}, p - q).
    # Extract the last boxed expression or the last math token from the response,
    # then compare symbolically with sympy.
    def _check_math(self, response: str, ground_truth: str) -> Tuple[bool, str]:
        extracted = self._extract_boxed(response) or self._extract_last_math(response)
        if extracted is None:
            return False, "Could not extract an answer from response."
        try:
            from sympy.parsing.latex import parse_latex
            from sympy import simplify, sympify
            gt_expr = parse_latex(ground_truth)
            pred_expr = parse_latex(extracted)
            if simplify(gt_expr - pred_expr) == 0:
                return True, f"Symbolic match: {extracted} == {ground_truth}"
            return False, f"Symbolic mismatch: {extracted} != {ground_truth}"
        except Exception:
            # Fallback: normalized string match
            norm = lambda s: re.sub(r'\s+', '', s.strip().lower())
            match = norm(extracted) == norm(ground_truth)
            return match, f"String match: '{extracted}' vs '{ground_truth}'"

    def _extract_boxed(self, text: str) -> Optional[str]:
        # Extract content from \boxed{...}, handling nested braces
        matches = []
        for m in re.finditer(r'\\boxed\{', text):
            start = m.end()
            depth, i = 1, start
            while i < len(text) and depth > 0:
                if text[i] == '{': depth += 1
                elif text[i] == '}': depth -= 1
                i += 1
            if depth == 0:
                matches.append(text[start:i-1])
        return matches[-1] if matches else None

    def _extract_last_math(self, text: str) -> Optional[str]:
        # Grab last $...$ or the last line
        dollars = re.findall(r'\$([^$]+)\$', text)
        if dollars:
            return dollars[-1].strip()
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        return lines[-1] if lines else None

    # --- GSM8K ---
    # Ground truth format: "... #### 18"  → extract the number after ####
    def _check_gsm8k(self, response: str, ground_truth: str) -> Tuple[bool, str]:
        gt_match = re.search(r'####\s*([\d,]+)', ground_truth)
        if not gt_match:
            return False, "Could not parse ground truth number."
        gt_num = float(gt_match.group(1).replace(',', ''))

        # Extract the last number from agent response
        numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', response.replace(',', ''))
        if not numbers:
            return False, "No number found in response."
        pred_num = float(numbers[-1])

        correct = abs(pred_num - gt_num) < 1e-6
        return correct, f"Predicted {pred_num}, expected {gt_num}"

    # --- MCQ (PersonaMem & PERMA) ---
    # PersonaMem: ground truth is "(c)", options are "(a) ... (b) ..."
    # PERMA:      ground truth is "D",  options are "A: ... B: ..."
    def _check_mcq(self, response: str, ground_truth: str, pattern: str) -> Tuple[bool, str]:
        gt_letter = re.search(pattern, ground_truth, re.IGNORECASE)
        if not gt_letter:
            return False, f"Could not parse ground truth letter from: {ground_truth}"
        gt = gt_letter.group(1).upper()

        pred_letters = re.findall(pattern, response, re.IGNORECASE)
        if not pred_letters:
            return False, "No option letter found in response."
        pred = pred_letters[-1].upper()

        correct = pred == gt
        return correct, f"Predicted '{pred}', expected '{gt}'"

    # --- LLM-as-judge fallback ---
    _LLM_GT_PROMPT = """\
Evaluate whether the agent's response contains the correct answer.
Output JSON: {"correct": true|false, "extracted_answer": "...", "reason": "..."}
Be lenient about format differences. No markdown.
"""

    def _check_llm(self, task: str, response: str, ground_truth: str) -> Tuple[bool, str]:
        user_prompt = (
            f"Question:\n{task}\n\n"
            f"Ground truth:\n{ground_truth}\n\n"
            f"Agent response:\n{response}\n\n"
            f"Is the answer correct?"
        )
        messages = [
            {"role": "system", "content": self._LLM_GT_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = get_completion(messages, temperature=0.0)
        result = parse_json_response(raw, default_value={})
        return bool(result.get("correct", False)), result.get("reason", "")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

class Episode:
    """
    Drives one full interaction episode.

    Usage:
        episode = Episode(
            user_id="user_001",
            task="What is the derivative of x^2?",
            ground_truth="2x",
            dataset_type="math500",
            max_turns=5,
        )
        result = episode.run(agent_fn)
        # agent_fn(task: str, history: List[Dict]) -> str
    """

    def __init__(
        self,
        user_id: str,
        task: str,
        ground_truth: Optional[str] = None,
        dataset_type: str = "llm",
        max_turns: int = 5,
        pref_threshold: int = 4,
    ):
        self.user_id = user_id
        self.task = task
        self.ground_truth = ground_truth
        self.max_turns = max_turns
        self.pref_threshold = pref_threshold

        self.simulator = UserSimulator(user_id)
        self.pref_checker = PreferenceChecker()
        self.gt_judge = GroundTruthJudge(dataset_type) if ground_truth else None

    def run(self, agent_fn) -> EpisodeResult:
        result = EpisodeResult(
            user_id=self.user_id,
            task=self.task,
            ground_truth=self.ground_truth,
        )
        history: List[Dict] = []

        for turn_num in range(1, self.max_turns + 1):
            # Build agent input: task + accumulated conversation
            if history:
                agent_input = (
                    self.task + "\n\n[Conversation so far]\n"
                    + "\n".join(f"Agent: {h['agent']}\nUser: {h['user']}" for h in history)
                )
            else:
                agent_input = self.task

            agent_response = agent_fn(agent_input, history)

            # Preference check
            pref_judgment = self.pref_checker.check(self.user_id, self.task, agent_response)
            pref_score = pref_judgment.score

            # GT check (every turn so we track if the agent eventually gets it right)
            gt_correct = None
            if self.gt_judge:
                gt_correct, _ = self.gt_judge.check(self.task, agent_response, self.ground_truth)

            # User feedback
            feedback = self.simulator.respond(
                task=self.task,
                agent_response=agent_response,
                history=history,
                turn=turn_num,
            )

            result.turns.append(Turn(
                turn=turn_num,
                agent_response=agent_response,
                user_feedback=feedback,
                preference_score=pref_score,
                preference_judgment=pref_judgment,
                gt_correct=gt_correct,
            ))

            history.append({"agent": agent_response, "user": feedback.text})

            if feedback.is_done:
                result.end_reason = "satisfied"
                break

            if turn_num == self.max_turns:
                result.end_reason = "max_turns"

        pref_scores = [t.preference_score for t in result.turns if t.preference_score is not None]
        result.final_preference_score = sum(pref_scores) / len(pref_scores) if pref_scores else None

        gt_results = [t.gt_correct for t in result.turns if t.gt_correct is not None]
        result.final_gt_correct = gt_results[-1] if gt_results else None

        return result


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

def make_agent(name: str = "amem", **kwargs):
    """
    Factory for agents backed by different memory systems.

    Supported names:
        "amem"          — AMemAgent (A-mem-sys, default)
        "nomem"         — NoMemoryAgent (no cross-task state, floor baseline)
        "longctx_full"  — LongContextAgent: every prior task's full transcript
                          is replayed in the agent prompt every turn (raw
                          retention upper bound; serves as read-side ceiling
                          and deployment cost upper bound).
        "mem0"          — Mem0Agent: upstream Mem0 with LLM-extracted facts
                          and multi-signal retrieval.

    New agents: add agent_<name>.py at the repo root and register here.

    Example:
        agent = make_agent("amem", retrieve_k=3)
        result = episode.run(agent)
    """
    name = name.lower()
    if name == "amem":
        from agents.agent import AMemAgent
        return AMemAgent(**kwargs)
    if name == "nomem":
        from agents.agent_nomem import NoMemoryAgent
        return NoMemoryAgent(**kwargs)
    if name == "longctx_full":
        from agents.agent_longctx import LongContextAgent
        return LongContextAgent(use_in_reply=True, **kwargs)
    if name == "mem0":
        from agents.agent_mem0 import Mem0Agent
        return Mem0Agent(**kwargs)
    if name == "memt":
        from agents.agent_memt import MemTAgent
        return MemTAgent(use_finish_answer=True, **kwargs)
    if name == "memt_memonly":
        from agents.agent_memt import MemTAgent
        return MemTAgent(use_finish_answer=False, **kwargs)
    raise ValueError(
        f"Unknown agent '{name}'. Available: amem, nomem, longctx_full, mem0, memt, memt_memonly"
    )


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_tasks(
    dataset: str,
    split: str = "32k",
    n: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Load (task, ground_truth) pairs from benchmark_data/.

    Args:
        dataset : "math500" | "personamem" | "perma"
                  (gsm8k parquet requires pandas — install separately)
        split   : for PersonaMem: "32k" | "128k" | "1M"
                  for PERMA: user_id, e.g. "user108" (or "all" for every user)
        n       : max samples to return (None = all)
        seed    : random seed used when n < total

    Returns:
        List of (task_str, ground_truth_str) tuples.
    """
    import random
    rng = random.Random(seed)
    dataset = dataset.lower()
    data_root = os.path.join(_HERE, "benchmark_data")
    tasks: List[Tuple[str, str]] = []

    # ── MATH-500 ──────────────────────────────────────────────────────────
    if dataset == "math500":
        path = os.path.join(data_root, "MATH-500", "test.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                tasks.append((row["problem"], row["answer"]))

    # ── PersonaMem ────────────────────────────────────────────────────────
    elif dataset == "personamem":
        import csv
        fname = f"questions_{split}.csv"
        path = os.path.join(data_root, "PersonaMem", fname)
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row["user_question_or_message"]
                answer   = row["correct_answer"]       # e.g. "(b)"
                tasks.append((question, answer))

    # ── PERMA ─────────────────────────────────────────────────────────────
    elif dataset == "perma":
        eval_root = os.path.join(data_root, "PERMA", "evaluation")
        user_ids = (
            sorted(os.listdir(eval_root))
            if split == "all"
            else [split]
        )
        for uid in user_ids:
            overall_dir = os.path.join(eval_root, uid, "meta", "overall")
            if not os.path.isdir(overall_dir):
                continue
            for fname in sorted(os.listdir(overall_dir)):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(overall_dir, fname), encoding="utf-8") as f:
                    row = json.load(f)
                question = row["question"]
                task_str = question
                # gold_label is MCQ over pre-generated responses — not applicable
                # in free-generation setting; ground_truth is None
                tasks.append((task_str, None))

    # ── GSM8K ─────────────────────────────────────────────────────────────
    elif dataset == "gsm8k":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("GSM8K requires pandas: pip install pandas pyarrow")
        split_name = split if split in ("test", "train") else "test"
        parquet_dir = os.path.join(data_root, "GSM8K", "main")
        # find the parquet file for this split
        import glob as _glob
        files = sorted(_glob.glob(os.path.join(parquet_dir, f"{split_name}-*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files found for split '{split_name}' in {parquet_dir}")
        df = pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)
        for _, row in df.iterrows():
            tasks.append((row["question"], row["answer"]))

    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            "Available: math500, gsm8k, personamem, perma"
        )

    if n is not None and n < len(tasks):
        tasks = rng.sample(tasks, n)

    return tasks


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def dummy_agent(task, history):
        return "I think the capital of France might be Lyon, known for its cuisine."

    episode = Episode(
        user_id="user_001",
        task="What is the capital of France?",
        ground_truth="Paris",
        dataset_type="llm",
        max_turns=3,
    )
    result = episode.run(dummy_agent)

    print(f"\nEnd reason: {result.end_reason}")
    print(f"Preference score (avg): {result.final_preference_score}")
    print(f"GT correct: {result.final_gt_correct}")
    for t in result.turns:
        print(f"\n[Turn {t.turn}]")
        print(f"  Agent: {t.agent_response[:80]}")
        print(f"  User ({t.user_feedback.feedback_type}): {t.user_feedback.text[:80]}")
        print(f"  Pref score: {t.preference_score}/5")
