#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_memory_bank_pooled.py

Alternative memory-bank generator. NO core dimensions, NO LLM-proposed dims —
every dim must come from `data/dimension_pool.json`. For each (user, category),
one LLM call picks N dims from the pool and fills each with {short, explanation}.
Hallucinated names are rejected via local validation (retries up to 3 times).

Output: data/user_memory_banks_pooled.json (separate file — does not overwrite
the canonical user_memory_banks.json).
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "generate_user_profile"))

from config import get_completion, parse_json_response, get_token_usage  # noqa: E402
from based_data import (                                                   # noqa: E402
    generate_age_info,
    generate_gender,
    generate_career_info,
    generate_location,
    generate_personal_values,
    generate_life_attitude,
    generate_personal_story,
    generate_interests_and_hobbies,
)

_POOL_PATH = os.path.join(_HERE, "data", "dimension_pool.json")


# ---------------------------------------------------------------------------
# Category configs — no core dimensions, total count only.
# ---------------------------------------------------------------------------

_CATEGORY_CONFIGS = [
    {
        "key": "skill_memory",
        "description": (
            "ACTUAL behavioral/performance capability — what the person can demonstrably DO. "
            "Scope: operational abilities, proficiency ceilings, things they execute well or poorly. "
            "NOT what they merely know as facts (that is knowledge_memory). "
            "NOT what they BELIEVE about their abilities (that is self_model). "
            "NOT specific past events (that is episodic_memory)."
        ),
        "target_n": 7,
    },
    {
        "key": "knowledge_memory",
        "description": (
            "What the person knows, doesn't know, or misunderstands about the EXTERNAL WORLD. "
            "Scope: facts, domains, concepts, held misconceptions about things outside themselves. "
            "NOT about themselves (that is self_model). "
            "NOT about what they can DO with that knowledge (that is skill_memory). "
            "NOT specific remembered events (that is episodic_memory)."
        ),
        "target_n": 7,
    },
    {
        "key": "episodic_memory",
        "description": (
            "Specific PAST EVENTS that shaped current judgments — temporal, concrete, anchored in time. "
            "Scope: 'when X happened to me, it led to Y' — discrete incidents with consequences. "
            "NOT general traits, abilities, or beliefs (those belong to other categories). "
            "Each entry must describe an event, not a stable property."
        ),
        "target_n": 7,
    },
    {
        "key": "self_model",
        "description": (
            "META-BELIEFS about themselves — how they perceive their own abilities, identity, traits. "
            "Scope: self-perceptions that MAY CONTRADICT skill_memory and knowledge_memory. "
            "NOT objective ability (that is skill_memory). "
            "NOT objective knowledge (that is knowledge_memory)."
        ),
        "target_n": 5,
    },
    {
        "key": "assistance_preference",
        "description": (
            "How the person wants to be HELPED in interactions — the delivery surface. "
            "Scope: format, tone, pacing, detail level, language choice, what they dislike in help. "
            "Purely about interaction style, not content. "
            "NOT what they know or can do (other categories). "
            "NOT self-perception (that is self_model)."
        ),
        "target_n": 5,
    },
]


# ---------------------------------------------------------------------------
# Pool loader
# ---------------------------------------------------------------------------

def _load_pool() -> Dict[str, List[Dict]]:
    """Return {category_key: [{name, axis, answer_space, source}, ...]} (meta stripped)."""
    with open(_POOL_PATH, "r", encoding="utf-8") as f:
        pool = json.load(f)
    return {k: v for k, v in pool.items() if k != "meta"}


# ---------------------------------------------------------------------------
# Persona block
# ---------------------------------------------------------------------------

def build_base_profile() -> Dict:
    age_info = generate_age_info()
    gender = generate_gender()
    location = generate_location(country_code="US")
    career_info = generate_career_info(age_info["age"])
    values = generate_personal_values(
        age=age_info["age"],
        gender=gender,
        occupation=career_info["status"],
        location=location,
    )
    life_attitude = generate_life_attitude(
        age=age_info["age"],
        gender=gender,
        occupation=career_info["status"],
        location=location,
        values_orientation=values.get("values_orientation", ""),
    )
    personal_story = generate_personal_story(
        age=age_info["age"],
        gender=gender,
        occupation=career_info["status"],
        location=location,
        values_orientation=values.get("values_orientation", ""),
        life_attitude=life_attitude,
    )
    interests = generate_interests_and_hobbies(personal_story)
    return {
        "age_info": age_info,
        "gender": gender,
        "location": location,
        "career_info": career_info,
        "personal_values": values,
        "life_attitude": life_attitude,
        "personal_story": personal_story,
        "interests": interests,
    }


def _persona_block(base_profile: Dict) -> str:
    age = base_profile["age_info"]["age"]
    gender = base_profile["gender"]
    occupation = base_profile["career_info"].get("status", "unknown")
    location = base_profile["location"]
    values = base_profile["personal_values"].get("values_orientation", "")
    attitude = base_profile["life_attitude"].get("attitude", "")
    story = base_profile["personal_story"].get("personal_story", "")
    interests_raw = base_profile["interests"]
    interests = (
        interests_raw.get("interests", interests_raw)
        if isinstance(interests_raw, dict)
        else interests_raw
    )
    return (
        f"Age: {age}, Gender: {gender}\n"
        f"Location: {location.get('city', '')}, {location.get('country', '')}\n"
        f"Occupation: {occupation}\n"
        f"Core values: {values}\n"
        f"Life attitude: {attitude}\n"
        f"Personal story: {story}\n"
        f"Interests/hobbies: {json.dumps(interests, ensure_ascii=False)}"
    )


# ---------------------------------------------------------------------------
# Pool-gated select + fill (one LLM call per category, with local name validation)
# ---------------------------------------------------------------------------

_SELECT_AND_FILL_SYSTEM = """\
You are generating one category of a user's hidden memory bank by PICKING dims
from a fixed pool and filling each with a user-specific short + explanation.

HARD RULES:
- Pick EXACTLY the requested count of dim names from the provided pool.
  You MUST NOT invent, rename, misspell, or merge pool entries. Names must be
  copied verbatim from the pool.
- Only select pool dims where THIS user has a meaningful, non-generic stance
  on that axis. If the user has no clear stance, skip it and pick a better fit
  from the pool.
- Prefer LATERAL DIVERSITY — avoid picking multiple dims that all point at the
  same underlying trait.
- "short": ≤15 words, user-specific label. Concrete, directly expressing the
  user's stance on the axis.
- "explanation": 2-4 sentences. Grounded in the persona, concrete, specific to
  this user. Gaps, misconceptions, and weaknesses matter as much as strengths.
- Stay strictly within the stated category boundary.

Output JSON only — no markdown fences, no commentary:
{
  "dims": [
    {"name": "<pool dim name, verbatim>", "short": "...", "explanation": "..."},
    ...
  ]
}
"""


def _format_pool_for_prompt(pool_entries: List[Dict]) -> str:
    lines = []
    for e in pool_entries:
        name = e["name"]
        axis = e.get("axis", "")
        answer_space = e.get("answer_space", [])
        answers_str = " / ".join(answer_space) if answer_space else ""
        lines.append(f"- {name}: {axis}  [e.g. {answers_str}]")
    return "\n".join(lines)


def _select_and_fill(
    persona_block: str,
    category: Dict,
    pool_entries: List[Dict],
    so_far: Dict,
) -> List[Dict]:
    """One LLM call per category: pick target_n dims from pool + fill each.
    Local validation: all chosen names must exist in the pool. Retries on failure."""
    target_n = category.get("target_n", 5)
    pool_names = {e["name"] for e in pool_entries}

    context = ""
    if so_far:
        context = (
            "\n\nALREADY-GENERATED CATEGORIES (for cross-category consistency; "
            "do not restate content from these):\n"
            + json.dumps(so_far, ensure_ascii=False, indent=2)
        )

    user_prompt = (
        f"Persona:\n{persona_block}{context}\n\n"
        f"Category: {category['key']} — {category['description']}\n\n"
        f"Dimension pool (pick EXACTLY {target_n}; names MUST come from here, verbatim):\n"
        f"{_format_pool_for_prompt(pool_entries)}\n\n"
        f"Emit JSON per the system prompt."
    )

    messages = [
        {"role": "system", "content": _SELECT_AND_FILL_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    parsed: Optional[Dict] = None
    for attempt in range(3):
        response = get_completion(messages, temperature=0.8)
        if not response:
            continue
        maybe = parse_json_response(response, default_value=None)
        if not isinstance(maybe, dict):
            print(f"      Warning: parse failed (attempt {attempt+1}/3)")
            continue
        dims = maybe.get("dims", [])
        if not isinstance(dims, list) or not dims:
            print(f"      Warning: dims missing/empty (attempt {attempt+1}/3)")
            continue
        # Local validation: every name must be in the pool.
        bad = [d for d in dims if not (isinstance(d, dict) and d.get("name") in pool_names)]
        if bad:
            bad_names = [d.get("name") for d in bad if isinstance(d, dict)]
            print(f"      Warning: off-pool names (attempt {attempt+1}/3): {bad_names}")
            continue
        # Check for duplicate names (LLM can double-pick).
        names_seen = [d["name"] for d in dims]
        if len(set(names_seen)) != len(names_seen):
            print(f"      Warning: duplicate picks (attempt {attempt+1}/3): {names_seen}")
            continue
        parsed = maybe
        break

    if parsed is None:
        print(f"      ERROR: gave up on {category['key']} — returning empty")
        return []

    items: List[Dict] = []
    for d in parsed["dims"][:target_n]:
        items.append({
            "dimension":   d["name"],
            "short":       (d.get("short") or "").strip(),
            "explanation": (d.get("explanation") or "").strip(),
        })
    return items


# ---------------------------------------------------------------------------
# Per-user generation
# ---------------------------------------------------------------------------

def generate_memory_bank(base_profile: Dict, pool: Dict[str, List[Dict]]) -> Dict:
    persona_block = _persona_block(base_profile)
    memory_bank: Dict = {}
    for category in _CATEGORY_CONFIGS:
        key = category["key"]
        pool_entries = pool.get(key, [])
        print(f"    Generating {key} (pool={len(pool_entries)}, target={category['target_n']}) ...")
        items = _select_and_fill(persona_block, category, pool_entries, memory_bank)
        print(f"      chosen: {[i['dimension'] for i in items]}")
        memory_bank[key] = items
    return memory_bank


def generate_all_users(num_users: int, pool: Dict[str, List[Dict]]) -> List[Dict]:
    users = []
    attempt = 0
    while len(users) < num_users:
        attempt += 1
        user_id = f"user_{len(users)+1:03d}"
        print(f"\n[{len(users)+1}/{num_users}] Building persona for {user_id} (attempt {attempt}) ...")
        try:
            base_profile = build_base_profile()
        except Exception as e:
            print(f"  Error building base profile: {e}; retrying ...")
            continue
        try:
            memory_bank = generate_memory_bank(base_profile, pool)
        except Exception as e:
            print(f"  Error generating memory bank: {e}; retrying ...")
            continue
        users.append({
            "user_id":      user_id,
            "base_profile": base_profile,
            "memory_bank":  memory_bank,
        })
        print(f"  Done: {user_id}")
    return users


def save_users(users: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":       "pooled (dimension_pool.json) — no core dims, all from pool",
        "num_users":    len(users),
        "users":        users,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(users)} users to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate memory banks by picking dims from a fixed pool (no core, no LLM-proposed dims)."
    )
    parser.add_argument("--num_users", type=int, default=3)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(_HERE, "data", "user_memory_banks_pooled.json"),
    )
    args = parser.parse_args()

    pool = _load_pool()
    print("Loaded pool: " + ", ".join(f"{k}={len(v)}" for k, v in pool.items()))

    print(f"\nGenerating {args.num_users} user memory banks (pool-gated) ...")
    users = generate_all_users(args.num_users, pool)
    save_users(users, args.output)

    usage = get_token_usage()
    print(
        f"\nToken totals: prompt={usage['prompt_tokens']}"
        f"  completion={usage['completion_tokens']}  total={usage['total_tokens']}"
    )


if __name__ == "__main__":
    main()
