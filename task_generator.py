"""
task_generator.py — Generate per-user custom tasks that expose specific memory-bank entries.

Each task is a realistic first-message help request. It is NOT a direct question
about the user's skill/preference/memory. Exposure happens through the user's
reply during UserSimulator rollout: the targeted entry becomes natural to
surface given the topic/framing of the task.

Output per user → benchmark_data/CustomTasks/<user_id>.json
Loaded by load_custom_tasks(user_id) in a shape compatible with BenchmarkRunner.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

from llm_client import get_completion, parse_json_response
from simulation import load_user

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_HERE, "benchmark_data", "CustomTasks")

# Mirror of scorer.py::_CATEGORY_GUIDE. Kept local (not imported) because
# task_generator and scorer are peers — duplicating five short strings is
# cheaper than introducing a shared import just for this.
_CATEGORY_GUIDE = {
    "skill_memory": (
        "ACTUAL behavioral/performance capability — what the person can demonstrably DO. "
        "Scope: operational abilities, proficiency ceilings, things they execute well or poorly."
    ),
    "knowledge_memory": (
        "What the person knows, doesn't know, or misunderstands about the EXTERNAL WORLD. "
        "Scope: facts, domains, concepts, held misconceptions about things outside themselves."
    ),
    "episodic_memory": (
        "Specific PAST EVENTS that shaped current judgments — temporal, concrete, anchored in time. "
        "Scope: 'when X happened to me, it led to Y' — discrete incidents with consequences."
    ),
    "self_model": (
        "META-BELIEFS about themselves — how they perceive their own abilities, identity, traits. "
        "May contradict objective ability/knowledge (overconfidence, blind spots, self-deception)."
    ),
    "assistance_preference": (
        "How the person wants to be HELPED in interactions — the delivery surface. "
        "Scope: format, tone, pacing, detail level, language choice, what they dislike in help."
    ),
}

# Per-category hint for HOW to phrase an eliciting task without becoming a
# silly direct question. The generator follows these patterns to stay on-task.
_ELICITATION_HINT = {
    "skill_memory": (
        "Pose a practical task in a domain that sits at or near the user's ability ceiling, "
        "so their reply reveals WHAT THEY CAN DO / WHERE THEY STRUGGLE by accepting parts, "
        "pushing back on parts, or adding constraints from lived practice."
    ),
    "knowledge_memory": (
        "Frame a task that touches a topic where they have domain knowledge or a misconception, "
        "so their reply reveals the factual content they carry (or the misconception) through how "
        "they respond to the assistant's first answer."
    ),
    "episodic_memory": (
        "Describe a CURRENT situation that parallels the past event, so the user naturally "
        "brings up 'I went through something like this...' in a follow-up reply. "
        "Do NOT reference the past event directly in the task."
    ),
    "self_model": (
        "Create a task where a typical assistant reply would either confirm or challenge the "
        "user's self-image — the user will push back, correct, or endorse in a way that "
        "exposes the meta-belief."
    ),
    "assistance_preference": (
        "Create a task whose framing gives room for the user to push back on the assistant's "
        "default style (over-verbose, too formal, too casual, wrong pacing, wrong format), so "
        "the preference surfaces as a correction."
    ),
}


_TASK_GEN_SYSTEM = """\
You design evaluation tasks for a memory-benchmark study. Each task is the FIRST message
the user sends to an AI assistant. Its job is to create a realistic help context in which
a specific, pre-known aspect of the user will organically surface through the user's reply,
NOT through the task text itself.

HARD RULES:
1. The task MUST be a help request / planning request / recommendation request / problem-solving
   request.
2. The task MUST NOT directly probe the user ("what are you good at", "how do you like to be
   helped", "tell me about a time you...") — those are silly exposure prompts. Zero tolerance.
3. The task MUST NOT copy phrasing from the target memory entry. The entry content is what we
   hope the user will reveal in THEIR REPLY; leaking it in the task defeats the purpose.
4. The task must be plausible first-contact text: 1–3 sentences, in a voice the user would use.
5. The task must be plausibly addressable by a generalist AI assistant (no insider info,
   no physical-world actions required from the assistant).
6. The task must be plausible given the user's actual profile (values, life situation, interests).
7. The task MUST actually expose the target memory entry through the user's natural reply.
   Exposure is the only reason this task exists — if a framing looks novel but would not
   draw out the target entry, it is useless. Reject it and redo.
8. Scene diversity: prefer a scene/domain that the prior tasks for this user have NOT already
   used (typical scenes: household, health, finance, travel, shopping, social, tech, hobbies,
   food, work/volunteering, transportation). BUT rule 7 wins: if genuine exposure of THIS
   entry demands a scene that was already used, reuse it rather than picking a mismatched
   scene that weakens exposure.
9. ANTI-FISHING (situation vs trait): SITUATION-level framing is fine — describe a neutral
   circumstance the user plausibly faces. TRAIT-level framing is fishing — do NOT echo the
   dimension name, short, explanation, or a paraphrase of the trait, and do NOT directly
   solicit self-description. A third party reading ONLY the task text must not be able to
   guess which trait is being probed; exposure must come from the USER'S REPLY given their
   profile, not from the task's wording.
   - BAD (echoes dimension phrasing): "keep better track of service calls" when dimension
     is `recordkeeping_precision` — "tracking" is the trait itself.
   - BAD (paraphrases the trait): "a fallback plan that doesn't get messy" when dimension
     is `low_disruption_adaptation` — "doesn't get messy" restates "low disruption".
   - BAD (solicits self-description): "help me write a personal statement that sounds like
     me" — directly asks the user to describe themselves.
   - BAD (names the trait): "how much detail is actually needed" when dimension is
     `privacy_orientation` — names the privacy decision explicitly.
   - GOOD (situation setup, not trait echo): "I've got a problem bigger than a simple fix,
     how do I handle it right?" for `problem_escalation` — the situation is neutral; the
     user's handling style (escalate / DIY / call pros / document) emerges in THEIR reply.
   - GOOD (open-ended): "a couple of repairs lined up this month — what should I think
     about before I call anyone?" — records-habits, escalation, voice, preferences all
     surface in the reply, nothing in the task telegraphs any single trait.

OUTPUT: strict JSON, no markdown fence, no commentary.
"""


def _compact_base_profile(base_profile: Dict) -> Dict:
    """Drop the long narrative `personal_story`; keep the short structured fields."""
    return {k: v for k, v in base_profile.items() if k != "personal_story"}


def _format_prior_tasks(prior: List[Dict]) -> str:
    """Render already-generated tasks for the same user, for scene-diversity awareness."""
    if not prior:
        return "(none yet — this is the first task for this user)"
    lines = []
    for t in prior:
        lines.append(
            f"- [{t.get('target_category','?')} / {t.get('target_dimension','?')}] "
            f"{t.get('task','').strip()}"
        )
    return "\n".join(lines)


def _build_user_prompt(
    base_profile: Dict,
    category: str,
    entry: Dict,
    n_tasks: int,
    prior_tasks: List[Dict],
) -> str:
    category_guide = _CATEGORY_GUIDE.get(category, "")
    elicitation_hint = _ELICITATION_HINT.get(category, "")

    return f"""\
USER BASE PROFILE (for voice + plausibility reference — do not echo):
{json.dumps(_compact_base_profile(base_profile), ensure_ascii=False, indent=2)}

ALREADY-GENERATED TASKS FOR THIS SAME USER (for scene-diversity; rule 8):
{_format_prior_tasks(prior_tasks)}

TARGET MEMORY ENTRY (this is what the user's reply should expose — NEVER appears in the task text):
- category: {category}
- category means: {category_guide}
- dimension: {entry.get('dimension', '')}
- short: {entry.get('short', '')}
- full explanation: {entry.get('explanation', '')}

ELICITATION STRATEGY FOR THIS CATEGORY:
{elicitation_hint}

Produce {n_tasks} task(s). If more than one, each must take a DIFFERENT angle that could
plausibly surface the same target entry via the user's reply. Obey all HARD RULES —
rule 7 (exposure must actually happen) outranks rule 8 (scene diversity) when they conflict.

OUTPUT JSON (exactly this shape):
{{
  "tasks": [
    {{"task": "<first message text>", "rationale": "<one sentence: why this elicits the target entry in the user's reply, and what scene it uses>"}}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Critic + Editor pipeline (anti-fishing)
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM_PROMPT = """\
You are a BLIND anti-fishing auditor for evaluation tasks. You see ONLY the task text and
the memory category it claims to probe — NOT the user, NOT the target entry.

Your job: given only the task, guess the most likely trait / stance / habit the task
telegraphs about its sender. Then classify the task:

- "neutral"  — task describes a situation generic enough that users with very different
               traits would plausibly send it; exposure would have to come from the user's
               own reply, not from the task's wording.
- "fishing"  — the task echoes or paraphrases a specific trait the sender is expected to
               have; reading the task alone lets you predict the sender's stance. Includes:
               (a) naming the trait ("how much detail is needed" → privacy),
               (b) paraphrasing the trait ("fallback plan that doesn't get messy" → low
               disruption), (c) soliciting self-description ("write a statement that sounds
               like me"), (d) encoding the trait's content (a note-taking template that
               pre-lists dates/numbers/times), (e) adjectives matching the expected stance
               ("make it sound calm and sensible" when target is "steady and practical").

Output JSON:
{
  "inferred_trait":  "<one-sentence guess at the trait / stance the task telegraphs, from the task alone>",
  "verdict":         "neutral" | "fishing",
  "why":             "<short reason>"
}
No commentary outside the JSON.
"""

_EDITOR_SYSTEM_PROMPT = """\
You are a task editor. Rewrite a task that was flagged as FISHING so it becomes NEUTRAL
while still creating a situation where the target memory entry will naturally emerge in
the user's reply.

Preserve:
- The general scene / activity domain (unless the scene ITSELF is what encodes the trait).
- The situational trigger that would invite the trait to surface.

Remove:
- Any vocabulary that names, paraphrases, or presupposes the target trait's stance.
- Any phrasing that solicits self-description ("sound like me", "fit my style", etc.).
- Any list / template / structure that pre-encodes the target's content.

The rewritten task must describe a SITUATION; the user's reply (informed by THIS user's
profile) does the exposing. A reader seeing only the task text must not be able to guess
the target trait.

Output JSON:
{
  "task":      "<rewritten first-message text; 1-3 sentences>",
  "rationale": "<one sentence: why this neutral situation still elicits the target in the user's reply>"
}
No commentary outside the JSON.
"""


def _critic_task(task_text: str, category: str) -> Optional[Dict]:
    """Blind fishing check. Returns {inferred_trait, verdict, why} or None on failure."""
    messages = [
        {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
        {"role": "user",
         "content": (
             f"Memory category being probed: {category}\n\n"
             f"Task text:\n{task_text.strip()}\n\n"
             f"Classify."
         )},
    ]
    raw = get_completion(messages, temperature=0.0) or ""
    parsed = parse_json_response(raw, default_value=None)
    if isinstance(parsed, dict) and parsed.get("verdict") in ("neutral", "fishing"):
        return parsed
    return None


def _edit_task(
    original_task: str,
    critic: Dict,
    base_profile: Dict,
    category: str,
    entry: Dict,
    prior_tasks: List[Dict],
) -> Optional[Dict]:
    """Rewrite a fishing task. Returns {task, rationale} or None on failure."""
    category_guide = _CATEGORY_GUIDE.get(category, "")
    user_prompt = (
        f"USER BASE PROFILE (for voice + plausibility; do not echo):\n"
        f"{json.dumps(_compact_base_profile(base_profile), ensure_ascii=False, indent=2)}\n\n"
        f"ALREADY-GENERATED TASKS FOR THIS USER (keep scene diverse from these):\n"
        f"{_format_prior_tasks(prior_tasks)}\n\n"
        f"TARGET MEMORY ENTRY (what the user's REPLY should expose — never in task text):\n"
        f"- category: {category}\n"
        f"- category means: {category_guide}\n"
        f"- dimension: {entry.get('dimension', '')}\n"
        f"- short: {entry.get('short', '')}\n"
        f"- full explanation: {entry.get('explanation', '')}\n\n"
        f"ORIGINAL TASK (flagged as fishing):\n{original_task.strip()}\n\n"
        f"CRITIC'S INFERENCE (what the original task gave away):\n"
        f"  inferred_trait: {critic.get('inferred_trait', '')}\n"
        f"  why:            {critic.get('why', '')}\n\n"
        f"Rewrite the task per the system prompt."
    )
    messages = [
        {"role": "system", "content": _EDITOR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    raw = get_completion(messages, temperature=0.5) or ""
    parsed = parse_json_response(raw, default_value=None)
    if isinstance(parsed, dict) and parsed.get("task"):
        return {
            "task":      parsed["task"].strip(),
            "rationale": (parsed.get("rationale") or "").strip(),
        }
    return None


def _generate_for_entry(
    base_profile: Dict,
    category: str,
    entry: Dict,
    n_tasks: int,
    temperature: float,
    prior_tasks: List[Dict],
) -> List[Dict]:
    """Generate → critic → (if fishing) editor, per task."""
    messages = [
        {"role": "system", "content": _TASK_GEN_SYSTEM},
        {"role": "user",   "content": _build_user_prompt(base_profile, category, entry, n_tasks, prior_tasks)},
    ]
    raw = get_completion(messages, temperature=temperature) or ""
    parsed = parse_json_response(raw, default_value={})
    tasks = parsed.get("tasks", []) if isinstance(parsed, dict) else []

    out: List[Dict] = []
    for t in tasks:
        if not isinstance(t, dict) or not t.get("task"):
            continue
        task_text = t["task"].strip()
        rationale = (t.get("rationale") or "").strip()

        critic = _critic_task(task_text, category)
        edited_note = ""
        if critic and critic.get("verdict") == "fishing":
            print(f"      critic: fishing → editing ({critic.get('inferred_trait','')[:70]}...)")
            edited = _edit_task(task_text, critic, base_profile, category, entry, prior_tasks)
            if edited:
                task_text = edited["task"]
                rationale = edited["rationale"] or rationale
                edited_note = "edited_after_fishing_flag"

        out.append({
            "task":             task_text,
            "rationale":        rationale,
            "target_category":  category,
            "target_dimension": entry.get("dimension", ""),
            "target_short":     entry.get("short", ""),
            "pipeline_note":    edited_note,
        })
    return out


def generate_tasks_for_user(
    user_id: str,
    n_per_entry: int = 1,
    temperature: float = 0.7,
) -> List[Dict]:
    """
    Generate custom tasks for a user. For every entry across all 5 categories,
    produce `n_per_entry` task(s). Total ≈ n_per_entry × sum(len(cat)) (usually ~40).
    """
    user = load_user(user_id)
    base_profile = user["base_profile"]
    memory_bank = user["memory_bank"]

    all_tasks: List[Dict] = []
    for category in ("skill_memory", "knowledge_memory", "episodic_memory",
                     "self_model", "assistance_preference"):
        entries = memory_bank.get(category, [])
        for i, entry in enumerate(entries, 1):
            dim = entry.get("dimension", f"entry_{i}")
            print(f"  [{user_id}] {category}/{dim} ({i}/{len(entries)})")
            out = _generate_for_entry(
                base_profile=base_profile,
                category=category,
                entry=entry,
                n_tasks=n_per_entry,
                temperature=temperature,
                prior_tasks=all_tasks,
            )
            if not out:
                print(f"    (!) no tasks produced for {category}/{dim}")
            all_tasks.extend(out)
    return all_tasks


def save_tasks(user_id: str, tasks: List[Dict], path: Optional[str] = None) -> str:
    """Save generated tasks to JSON. Returns the path written."""
    path = path or os.path.join(_OUT_DIR, f"{user_id}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "user_id":      user_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_tasks":      len(tasks),
        "tasks":        tasks,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[task_generator] Saved {len(tasks)} tasks → {path}")
    return path


def load_custom_tasks(user_id: str, path: Optional[str] = None) -> List[Tuple[str, None]]:
    """
    Load custom tasks in runner-compatible shape: [(task_str, None), ...].

    No ground truth — Task Fit falls back to `end_reason == "satisfied"`.
    """
    path = path or os.path.join(_OUT_DIR, f"{user_id}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [(t["task"], None) for t in data.get("tasks", []) if t.get("task")]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("user_ids", nargs="+", help="e.g. user_001 user_002 user_003")
    p.add_argument("--n-per-entry", type=int, default=1,
                   help="tasks per memory-bank entry (1 or 2)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output-dir", type=str, default=None,
                   help="override default benchmark_data/CustomTasks/ output dir")
    p.add_argument("--bank", type=str, default=None,
                   help="override path to the user-memory bank (default: Deeppersona/data/user_memory_banks_pooled_final.json)")
    args = p.parse_args()

    if args.bank:
        import simulation
        simulation._USER_DATA_PATH = os.path.abspath(args.bank)
        simulation._user_cache = None
        print(f"[task_generator] bank path overridden → {simulation._USER_DATA_PATH}")

    for uid in args.user_ids:
        print(f"\n=== Generating tasks for {uid} ===")
        tasks = generate_tasks_for_user(
            uid, n_per_entry=args.n_per_entry, temperature=args.temperature
        )
        out_path = None
        if args.output_dir:
            out_path = os.path.join(args.output_dir, f"{uid}.json")
        save_tasks(uid, tasks, path=out_path)
