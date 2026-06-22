"""
failure_attribution.py — 2-stage attribution pipeline for low B scores.

For each dim with score < 0.75, decide where the loss came from. We do
NOT investigate the read side (slot-fill) — it is assumed to be fine.
The pipeline therefore has two LLM-driven stages:

    Stage 1 — DISCLOSURE
        Find the episode whose target_dimension matches this dim.
        Read its dialogue. Did the user simulator disclose enough about
        the GT that a downstream memory pipeline could plausibly recover
        it?
        → if YES: classify as `memory_failure` (the dialogue carried
          the signal but the memory pipeline didn't surface it; we
          aggregate write + read into one bucket).
        → if NO: continue to Stage 1B.

    Stage 1B — DISCLOSURE-FAILURE SUB-CLASSIFICATION
        The dialogue did not disclose the GT. Decide which of the three
        upstream components is most responsible:
            task_design_failure   — task itself can't naturally invite
                                    this dim, regardless of agent / sim
            agent_elicitation_failure
                                  — task is relevant but the agent never
                                    asked questions in the dim's
                                    direction
            simulator_too_strict  — agent did probe; the simulator gave
                                    clipped or evasive responses /
                                    declared satisfied early

Output categories per dim:
    ok | memory_failure | task_design_failure | agent_elicitation_failure |
    simulator_too_strict | no_targeted_task

Inputs:
    output/<run>/recon_judge/<user_id>.json
    history/<run>/<user_id>/episode_*.json
    benchmark_data/<tasks_dir>/<user_id>.json   (for target_dimension → episode mapping)

Output:
    output/<run>/attribution/<user_id>.json

Usage:
    python failure_attribution.py --run mem0_pooled --user user_001
    python failure_attribution.py --run mem0_pooled
    python failure_attribution.py --all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List, Optional

from llm_client import get_completion, parse_json_response

_HERE = os.path.dirname(os.path.abspath(__file__))


# ── tasks-dir inference ──────────────────────────────────────────────────────

def _infer_tasks_dir(run_id: str) -> str:
    rid = run_id.lower()
    # 50-user runs use the regenerated final task pool.
    if rid.endswith("_50") or rid.endswith("_50_retrieve") or "_50_" in rid:
        return "CustomTasksPooledFinal" if "pooled" in rid else "CustomTasksFinal"
    return "CustomTasksPooled" if "pooled" in rid else "CustomTasks"


# ── data loaders ─────────────────────────────────────────────────────────────

def _load_tasks(user_id: str, tasks_dir: str) -> List[Dict]:
    path = os.path.join(_HERE, "benchmark_data", tasks_dir, f"{user_id}.json")
    if not os.path.exists(path):
        return []
    return json.load(open(path, encoding="utf-8")).get("tasks", [])


def _load_episode(run: str, user_id: str, ep_index: int) -> Optional[Dict]:
    path = os.path.join(_HERE, "history", run, user_id, f"episode_{ep_index}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def _load_recon_judge(run: str, user_id: str) -> Optional[Dict]:
    path = os.path.join(_HERE, "output", run, "recon_judge", f"{user_id}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


# ── prompts ──────────────────────────────────────────────────────────────────

_DISCLOSURE_PROMPT = """\
You are checking whether one episode of dialogue contains enough information
about a target dimension that a competent memory system could later infer
the user's value on this dimension.

Target dimension: {dimension}
Ground truth (short):   {gt_short}
Ground truth (full):    {gt_explanation}

The episode is a multi-turn assistance dialogue between an Agent and a User
simulator playing a fixed persona. Read it carefully and decide:

  Did the User simulator (or the Agent quoting the user) reveal enough
  about the target dimension that a competent memory system, after
  reading the dialogue, could plausibly arrive at the user's value on
  this dimension?

Important guidance:
- You are NOT asking whether the literal ground-truth phrasing was uttered.
- Indirect signals count: behavioural hints, contextual mentions, clear
  preferences expressed in passing, implications of what the user said.
- If a reasonable downstream reader would, given the dialogue, conclude
  something close to the ground truth (even if loosely worded), answer YES.
- Only answer NO if the dialogue offers essentially no signal a reader
  could anchor on.

Episode transcript:
================================
TASK: {task_text}

{transcript}
================================

Output JSON only (no markdown):
{{"disclosed": true|false, "evidence": "<short quote or 'none'>", "reason": "<one sentence>"}}\
"""


_ORACLE_PROMPT = """\
You are deciding whether a single task can ever naturally invite a user to
reveal information about a target dimension. Assume a perfectly probing
assistant AND a perfectly cooperative user simulator. You are NOT looking
at any actual conversation — judge purely from the task statement and
the dimension being asked about.

Target dimension: {dimension}
Ground truth (short):   {gt_short}
Ground truth (full):    {gt_explanation}

Task:
{task_text}

Could this task plausibly lead — through reasonable follow-up questions
and natural elaboration — to a moment where the user reveals their value
on the target dimension?

Output JSON only (no markdown):
{{"can_invite": true|false, "reason": "<one sentence>"}}\
"""


_SUBCLASS_PROMPT = """\
The dialogue below failed to disclose the target dimension's ground truth.
We have already verified that the task COULD invite this dimension under
perfect conditions, so the failure must come from how the conversation
actually unfolded. Decide which side is responsible.

Target dimension: {dimension}
Ground truth (short):   {gt_short}
Ground truth (full):    {gt_explanation}

Pick exactly one:
  A. agent_elicitation_failure
     The agent never went anywhere near this dimension. The conversation
     stayed on unrelated territory; the agent did not roughly steer toward
     the topic this dim is about.

  B. simulator_too_strict
     The agent did roughly steer the conversation toward this dim's topic
     area (does not need to be a literal pinpoint question), but the user
     simulator did not actually disclose enough — gave shallow / non-
     committal answers, changed direction, declared satisfied early, or
     simply did not elaborate when the topic was on the table.

A rough rule of thumb: if reading the transcript you feel "the topic came
up, the user just didn't say much about it" → B. If you feel "the agent
never even brought this kind of thing up" → A.

Episode transcript:
================================
TASK: {task_text}

{transcript}
================================

Output JSON only (no markdown):
{{"category": "A|B", "reason": "<one sentence>", "evidence": "<short quote>"}}\
"""


def _disclosure_check(task_text: str, transcript: str, dim_record: Dict) -> Dict:
    prompt = _DISCLOSURE_PROMPT.format(
        dimension=dim_record.get("dimension", ""),
        gt_short=dim_record.get("ground_truth", ""),
        gt_explanation=dim_record.get("explanation", ""),
        task_text=task_text or "(unknown task)",
        transcript=transcript or "(empty)",
    )
    raw = get_completion([{"role": "user", "content": prompt}], temperature=0.0)
    parsed = parse_json_response(raw, default_value={})
    if not isinstance(parsed, dict):
        return {"disclosed": False, "evidence": "", "reason": "parse failed"}
    return {
        "disclosed": bool(parsed.get("disclosed")),
        "evidence":  str(parsed.get("evidence", ""))[:300],
        "reason":    str(parsed.get("reason", ""))[:300],
    }


def _subclassify(task_text: str, transcript: str, dim_record: Dict) -> Dict:
    prompt = _SUBCLASS_PROMPT.format(
        dimension=dim_record.get("dimension", ""),
        gt_short=dim_record.get("ground_truth", ""),
        gt_explanation=dim_record.get("explanation", ""),
        task_text=task_text or "(unknown task)",
        transcript=transcript or "(empty)",
    )
    raw = get_completion([{"role": "user", "content": prompt}], temperature=0.0)
    parsed = parse_json_response(raw, default_value={})
    if not isinstance(parsed, dict):
        return {"category": "?", "reason": "parse failed", "evidence": ""}
    cat = str(parsed.get("category", "?")).strip().upper()
    if cat not in ("A", "B"):
        cat = "?"
    return {
        "category": cat,
        "reason":   str(parsed.get("reason", ""))[:300],
        "evidence": str(parsed.get("evidence", ""))[:300],
    }


_SUBCLASS_LABEL = {
    "A": "agent_elicitation_failure",
    "B": "simulator_too_strict",
}


# ── task-design oracle (run-invariant per (user_id, dimension)) ──────────────

_ORACLE_DIR = os.path.join(_HERE, "output", "_task_design_oracle")


def _oracle_path(user_id: str) -> str:
    return os.path.join(_ORACLE_DIR, f"{user_id}.json")


def _load_user_oracle(user_id: str) -> Dict:
    p = _oracle_path(user_id)
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}


def _save_user_oracle(user_id: str, oracle: Dict) -> None:
    os.makedirs(_ORACLE_DIR, exist_ok=True)
    p = _oracle_path(user_id)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(oracle, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def _oracle_check(task_text: str, dim_record: Dict) -> Dict:
    """LLM call: can this task naturally invite this dim under ideal agent + sim?"""
    prompt = _ORACLE_PROMPT.format(
        dimension=dim_record.get("dimension", ""),
        gt_short=dim_record.get("ground_truth", ""),
        gt_explanation=dim_record.get("explanation", ""),
        task_text=task_text or "(unknown task)",
    )
    raw = get_completion([{"role": "user", "content": prompt}], temperature=0.0)
    parsed = parse_json_response(raw, default_value={})
    if not isinstance(parsed, dict):
        return {"can_invite": True, "reason": "parse failed (default to True)"}
    return {
        "can_invite": bool(parsed.get("can_invite", True)),
        "reason":     str(parsed.get("reason", ""))[:300],
    }


# ── per-dim driver ───────────────────────────────────────────────────────────

def _build_episode_transcript(ep: Dict) -> str:
    out = []
    for t in ep.get("turns", []):
        agent = t.get("agent_response") or ""
        sim = t.get("user_simulator", {}).get("text") or ""
        if agent:
            out.append(f"Agent: {agent}")
        if sim:
            out.append(f"User: {sim}")
    return "\n".join(out)


def _find_target_episode(dim_name: str, tasks: List[Dict]) -> Optional[int]:
    for i, t in enumerate(tasks, 1):
        if t.get("target_dimension") == dim_name:
            return i
    return None


def attribute_dim(dim_record: Dict, run: str, user_id: str,
                  tasks: List[Dict], oracle: Dict) -> Dict:
    score = dim_record.get("score") or 0.0
    if score >= 0.75:
        return {"category": "ok", "score": score, "stages": []}

    dim_name = dim_record.get("dimension", "")

    ep_index = _find_target_episode(dim_name, tasks)
    if ep_index is None:
        return {
            "category": "no_targeted_task",
            "score": score,
            "stages": [],
            "note": "No task in this user's task list targets this dimension.",
        }
    ep = _load_episode(run, user_id, ep_index)
    task_text = (tasks[ep_index - 1] or {}).get("task", "")
    transcript = _build_episode_transcript(ep) if ep else ""

    stages: List[Dict] = []

    # Stage 0: task-design oracle (run-invariant, cached per user_id+dim)
    cached = oracle.get(dim_name)
    if cached is None:
        cached = _oracle_check(task_text, dim_record)
        oracle[dim_name] = cached
    stages.append({"stage": "oracle", **cached})
    if not cached.get("can_invite", True):
        return {
            "category": "task_design_failure",
            "score":    score,
            "stages":   stages,
            "ep_index": ep_index,
        }

    # Stage 1: disclosure check (per-run dialogue)
    s1 = _disclosure_check(task_text, transcript, dim_record)
    stages.append({"stage": "disclosure", **s1})
    if s1["disclosed"]:
        return {
            "category": "memory_failure",
            "score":    score,
            "stages":   stages,
            "ep_index": ep_index,
        }

    # Stage 1B: agent vs simulator (task_design already excluded by oracle)
    s2 = _subclassify(task_text, transcript, dim_record)
    stages.append({"stage": "disclosure_subclass", **s2})
    return {
        "category": _SUBCLASS_LABEL.get(s2["category"], "unclassified"),
        "score":    score,
        "stages":   stages,
        "ep_index": ep_index,
    }


# ── batch driver ─────────────────────────────────────────────────────────────

def attribute_user(run: str, user_id: str, tasks_dir: Optional[str] = None) -> Dict:
    if tasks_dir is None:
        tasks_dir = _infer_tasks_dir(run)
    recon = _load_recon_judge(run, user_id)
    if recon is None:
        return {"user_id": user_id, "error": "recon_judge file missing"}
    tasks = _load_tasks(user_id, tasks_dir)
    oracle = _load_user_oracle(user_id)
    oracle_size_in = len(oracle)

    details_out = []
    for i, d in enumerate(recon.get("details", []), 1):
        attrib = attribute_dim(d, run, user_id, tasks, oracle)
        details_out.append({**d, "attribution": attrib})
        cat = attrib.get("category", "?")
        print(f"  [{i:>2}] {d.get('category'):<22} {d.get('dimension'):<35} "
              f"score={d.get('score'):.2f}  → {cat}")

    if len(oracle) > oracle_size_in:
        _save_user_oracle(user_id, oracle)

    counts = Counter(it["attribution"]["category"] for it in details_out)
    out = {
        "user_id":   user_id,
        "tasks_dir": tasks_dir,
        "counts":    dict(counts),
        "details":   details_out,
    }
    out_dir = os.path.join(_HERE, "output", run, "attribution")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{user_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  → saved {out_path}")
    return out


def attribute_run(run: str) -> None:
    user_pat = os.path.join(_HERE, "output", run, "recon_judge", "*.json")
    users = sorted(os.path.basename(f).replace(".json", "") for f in glob.glob(user_pat))
    if not users:
        print(f"[{run}] no users under output/{run}/recon_judge/")
        return
    print(f"\n=== run: {run} ({len(users)} users) ===")
    agg = Counter()
    for uid in users:
        res = attribute_user(run, uid)
        agg.update(res.get("counts", {}))
    print(f"  aggregate: {dict(agg)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None,
                   help="run-id (e.g. mem0_pooled). Omit + --all to do all runs.")
    p.add_argument("--user", default=None, help="user_id (omit for all in run)")
    p.add_argument("--all", action="store_true",
                   help="run on every output/<run>/recon_judge/ found")
    args = p.parse_args()

    if args.all:
        runs = sorted(
            os.path.basename(d) for d in glob.glob(os.path.join(_HERE, "output", "*"))
            if os.path.isdir(os.path.join(d, "recon_judge"))
        )
        for r in runs:
            attribute_run(r)
        return

    if not args.run:
        p.error("provide --run <run-id> or --all")

    if args.user:
        attribute_user(args.run, args.user)
    else:
        attribute_run(args.run)


if __name__ == "__main__":
    main()
