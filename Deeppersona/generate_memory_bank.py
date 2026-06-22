#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_memory_bank.py

Minimal addition to Deeppersona: generate 10 users, each with a stable hidden
memory bank consisting of five memory types.

Memory types (each item has "short" and "explanation"):
  1. skill_memory       - actual capability boundary
  2. knowledge_memory   - knowledge state, including misconceptions
  3. episodic_memory    - key past experiences that shape current behavior
  4. self_model         - self-perception, possibly biased
  5. assistance_preference - preferred interaction style

Usage:
    cd Deeppersona
    python generate_memory_bank.py [--num_users 10] [--output data/user_memory_banks.json]
"""

import json
import os
import sys
import argparse
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

# Make generate_user_profile package importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "generate_user_profile"))

from config import get_completion, parse_json_response, get_token_usage
from based_data import (
    generate_age_info,
    generate_gender,
    generate_career_info,
    generate_location,
    generate_personal_values,
    generate_life_attitude,
    generate_personal_story,
    generate_interests_and_hobbies,
)


# ---------------------------------------------------------------------------
# Step 1: Build base persona (reuses Deeppersona's existing based_data logic)
# ---------------------------------------------------------------------------

def build_base_profile() -> Dict:
    """Generate a base persona using existing Deeppersona based_data functions."""
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


# ---------------------------------------------------------------------------
# Step 2: Generate the five-part memory bank from the base profile
# ---------------------------------------------------------------------------

_SELECT_SYSTEM_PROMPT = """\
You are a cognitive profile designer. Select dimension slots for one memory category for a specific person.

Rules:
- You MUST include all core dimensions provided.
- Add the requested number of additional dimensions tailored to THIS specific person.
- Extra dimensions should feel natural and grounded in the persona — not generic filler.
- CRITICAL: Dimensions must NOT overlap with other memory categories.
  Respect the category boundaries stated in the task. If a candidate dimension could
  plausibly fit another category better, it does NOT belong here.
- CRITICAL (within-category diversity): Dimensions within this category MUST be laterally
  diverse. Each added dimension must slot a semantically distinct aspect of the category —
  not a rewording / close sibling / emphasis-shift of another. If two candidate names could
  describe the same trait (e.g. `independent_identity` vs `pride_in_handling_things_himself`
  vs `doesnt_like_being_a_burden` all point at "I am self-reliant"), pick ONE and drop the
  others. Prefer breadth over depth within a category.
- CRITICAL (axis, not answer): Dimension names describe the AXIS of inquiry, never the
  user's stance on it. Two users with OPPOSITE stances on this axis must both plausibly
  have this exact dimension name. If the name presupposes a particular stance (a trait
  value), it is WRONG — even if abstracted into snake_case.
    Good (axis, no stance):   pacing_preference (answer: stepwise/continuous/adaptive),
                              disclosure_posture (answer: private/open/conditional),
                              structural_preference_self_view (answer: order/flexibility),
                              evidence_channel_preference (answer: paper/verbal/digital),
                              learning_channel (answer: video/text/hands-on),
                              autonomy_self_view (answer: self-reliant/interdependent/mixed)
    Bad (answer encoded):     privacy_orientation (presupposes 'private' — use disclosure_posture),
                              orderliness_self_concept (presupposes 'orderly' — use structural_preference_self_view),
                              self_reliance_identity (presupposes 'self-reliant' — use autonomy_self_view),
                              paper_trail_support (presupposes 'paper trail' — use evidence_channel_preference),
                              stepwise_pacing (presupposes 'stepwise' — use pacing_preference),
                              video_based_repair_learning_knowledge (presupposes 'video' — use learning_channel),
                              pride_in_not_being_pushed_around (full stance — use assertiveness_self_view),
                              doesnt_like_being_a_burden (full stance — use burden_aversion_self_view)
  The concrete stance goes in short/explanation, NEVER in the dimension name.
- Output a JSON array of dimension name strings only (no markdown).
"""

_FILL_SYSTEM_PROMPT = """\
You are a cognitive profile generator. Fill memory items for a specific person, one per dimension slot.

Rules:
- Output a JSON object: {"dimension_name": {"short": "...", "explanation": "..."}, ...}
- "short": a concise label (≤15 words), directly expressing the content of this dimension.
- "explanation": concrete, specific to this person (2-4 sentences).
- Gaps, misconceptions, and weaknesses matter as much as strengths.
- Stay strictly within the stated category boundary — do not drift into self-perception,
  preferences, or other categories unless this is the correct category.
- Output valid JSON only (no markdown).
"""

# Category boundaries are defined STRICTLY to minimize overlap. Each description
# contains both what belongs (positive scope) and what does NOT belong (negative
# scope referring to other categories).
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
        "core_dimensions": ["core_skill", "skill_gap"],
        "extra_dimensions": 5,
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
        "core_dimensions": ["domain_knowledge", "misconception"],
        "extra_dimensions": 5,
    },
    {
        "key": "episodic_memory",
        "description": (
            "Specific PAST EVENTS that shaped current judgments — temporal, concrete, anchored in time. "
            "Scope: 'when X happened to me, it led to Y' — discrete incidents with consequences. "
            "NOT general traits, abilities, or beliefs (those belong to other categories). "
            "Each entry must describe an event, not a stable property."
        ),
        "core_dimensions": ["formative_experience", "failure_or_setback"],
        "extra_dimensions": 5,
    },
    {
        "key": "self_model",
        "description": (
            "META-BELIEFS about themselves — how they perceive their own abilities, identity, traits. "
            "Scope: self-perceptions that MAY CONTRADICT skill_memory and knowledge_memory. "
            "Overconfidence, underconfidence, blind spots, and self-deception belong here. "
            "Each entry is a belief-about-self that could be accurate or mistaken. "
            "NOT objective ability (that is skill_memory). "
            "NOT objective knowledge (that is knowledge_memory)."
        ),
        "core_dimensions": ["self_perception", "blind_spot"],
        "extra_dimensions": 3,
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
        "core_dimensions": ["preferred_style", "what_they_dislike"],
        "extra_dimensions": 3,
    },
]


def _build_persona_block(base_profile: Dict) -> str:
    age = base_profile["age_info"]["age"]
    gender = base_profile["gender"]
    occupation = base_profile["career_info"].get("status", "unknown")
    location = base_profile["location"]
    values = base_profile["personal_values"].get("values_orientation", "")
    attitude = base_profile["life_attitude"].get("attitude", "")
    story = base_profile["personal_story"].get("personal_story", "")
    interests_raw = base_profile["interests"]
    interests = interests_raw.get("interests", interests_raw) if isinstance(interests_raw, dict) else interests_raw
    return (
        f"Age: {age}, Gender: {gender}\n"
        f"Location: {location.get('city', '')}, {location.get('country', '')}\n"
        f"Occupation: {occupation}\n"
        f"Core values: {values}\n"
        f"Life attitude: {attitude}\n"
        f"Personal story: {story}\n"
        f"Interests/hobbies: {json.dumps(interests, ensure_ascii=False)}"
    )


def _select_dimensions(persona_block: str, category: Dict, so_far: Dict) -> List[str]:
    """Step 1 (select): choose core + N persona-specific dimension slots (N from category config)."""
    core = category["core_dimensions"]
    extra_n = category.get("extra_dimensions", 5)
    total = len(core) + extra_n
    context = ""
    if so_far:
        context = "\n\nAlready generated memory categories (use for consistency):\n"
        context += json.dumps(so_far, ensure_ascii=False, indent=2)

    user_prompt = (
        f"Persona:\n{persona_block}{context}\n\n"
        f"Category: {category['key']} — {category['description']}\n\n"
        f"Core dimensions you MUST include: {json.dumps(core)}\n"
        f"Add exactly {extra_n} more dimensions specific to this person. Total: {total} dimensions.\n"
        f"Do NOT pad with near-duplicates to hit the count — if you cannot find {extra_n} "
        f"laterally distinct aspects, return fewer rather than rewording the same trait.\n\n"
        f"Output a JSON array of dimension name strings only."
    )

    messages = [
        {"role": "system", "content": _SELECT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Retry on parse failure: API can emit malformed JSON mid-response under
    # temperature=0.9. Same prompt, fresh sampling each attempt.
    result = None
    for attempt in range(3):
        response = get_completion(messages, temperature=0.9)
        if not response:
            continue
        parsed = parse_json_response(response, default_value=None)
        if isinstance(parsed, list):
            result = parsed
            break
        print(f"  Warning: select parse failed (attempt {attempt+1}/3) for {category['key']}")

    if result is None:
        return core[:]

    # Ensure core dimensions are always present
    for d in core:
        if d not in result:
            result.insert(0, d)

    return result


def _fill_dimensions(persona_block: str, category: Dict, dimensions: List[str], so_far: Dict) -> List:
    """Step 2 (fill): fill each slot with {short, explanation}."""
    context = ""
    if so_far:
        context = "\n\nAlready generated memory categories (use for consistency):\n"
        context += json.dumps(so_far, ensure_ascii=False, indent=2)

    user_prompt = (
        f"Persona:\n{persona_block}{context}\n\n"
        f"Category: {category['key']} — {category['description']}\n\n"
        f"Fill each of these dimension slots:\n{json.dumps(dimensions)}\n\n"
        f"Output a JSON object: {{\"dimension_name\": {{\"short\": \"...\", \"explanation\": \"...\"}}, ...}}"
    )

    messages = [
        {"role": "system", "content": _FILL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Retry on parse failure: the LLM can emit malformed JSON mid-response
    # (unescaped quote in explanation, etc.). Same prompt, fresh sampling.
    result = None
    for attempt in range(3):
        response = get_completion(messages, temperature=0.85)
        if not response:
            continue
        parsed = parse_json_response(response, default_value=None)
        if isinstance(parsed, dict):
            result = parsed
            break
        print(f"  Warning: fill parse failed (attempt {attempt+1}/3) for {category['key']}")

    if result is None:
        print(f"  Warning: all fill attempts failed for {category['key']}")
        return []

    items = []
    for dim in dimensions:
        if dim in result and isinstance(result[dim], dict):
            items.append({
                "dimension": dim,
                "short": result[dim].get("short", ""),
                "explanation": result[dim].get("explanation", ""),
            })
    return items


# ---------------------------------------------------------------------------
# Audit + Repair pipeline (dim-name leakage, within-category redundancy)
# ---------------------------------------------------------------------------

_AUDIT_SYSTEM_PROMPT = """\
You are a strict auditor of dimension-slot NAMES for a memory bank. Your review is BLIND —
you see only the category and the proposed dimension names, NOT the user's persona or the
filled content. Apply two checks:

(1) NAME LEAKAGE — a dimension name must describe the AXIS of inquiry, not the user's stance
    on it. Flag any name that presupposes a particular trait value.
      Leaky: `privacy_orientation` (presupposes privacy), `orderliness_self_concept`
             (presupposes orderly), `paper_trail_support` (presupposes paper trail),
             `stepwise_pacing` (presupposes stepwise).
      Clean: `disclosure_posture`, `structural_preference_self_view`,
             `evidence_channel_preference`, `pacing_preference`.
(2) LATERAL REDUNDANCY — flag any pair/group of names that point at the same underlying
    axis with only surface rewording. Two dims that would always move together in the same
    direction for any user are redundant. **Include the CORE dimensions** in this check —
    an extra dim that restates a core dim (e.g. `interaction_dislikes` alongside the core
    `what_they_dislike`) is just as redundant as two extras overlapping.

Output JSON:
{
  "leaky":     [{"name": "...", "why": "...", "replacement": "..."}, ...],
  "redundant": [{"group": ["...", "..."], "why": "...", "keep": "...", "drop": ["..."]}, ...]
}
When a redundant group contains a CORE dimension, ALWAYS keep the core and drop the extras.
If nothing is wrong, return {"leaky": [], "redundant": []}.
"""

_REPAIR_SYSTEM_PROMPT = """\
You are a dimension-slot editor. Given a proposed dim list and the auditor's flags, produce
a corrected list of the SAME length, preserving core dimensions verbatim. For each leaky
name, use the provided `replacement` (or propose an equally clean axis label). For each
redundant group, keep one and replace the rest with DIFFERENT axes of this category that
do NOT overlap with any remaining name.

Rules:
- Core dimensions listed as immutable MUST appear unchanged.
- Total count must match the original.
- Every new name must follow the AXIS, NOT ANSWER principle (no stance baked in).
- Names must stay within the stated category boundary (no cross-category drift).

Output JSON: {"dimensions": ["...", "...", ...]} — the corrected list only, no commentary.
"""


def _audit_dimensions(category: Dict, dimensions: List[str]) -> Optional[Dict]:
    """Blind audit of dim names. Returns flags dict or None on parse failure."""
    core = category["core_dimensions"]
    user_prompt = (
        f"Category: {category['key']} — {category['description']}\n\n"
        f"Core dimensions (MUST be kept in any redundant group): {json.dumps(core)}\n"
        f"Proposed dimensions:\n{json.dumps(dimensions)}\n\n"
        f"Apply NAME LEAKAGE and LATERAL REDUNDANCY checks. Remember: check each extra dim "
        f"against the core dims too — an extra that restates a core is redundant.\n"
        f"Output JSON per the system prompt."
    )
    messages = [
        {"role": "system", "content": _AUDIT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    for attempt in range(3):
        response = get_completion(messages, temperature=0.3)
        if not response:
            continue
        parsed = parse_json_response(response, default_value=None)
        if isinstance(parsed, dict) and "leaky" in parsed and "redundant" in parsed:
            return parsed
        print(f"      Warning: audit parse failed (attempt {attempt+1}/3)")
    return None


def _repair_dimensions(
    category: Dict, dimensions: List[str], flags: Dict
) -> Optional[List[str]]:
    """Apply flags to produce corrected dim list. Returns None on failure."""
    core = category["core_dimensions"]
    user_prompt = (
        f"Category: {category['key']} — {category['description']}\n\n"
        f"Core dimensions (IMMUTABLE): {json.dumps(core)}\n"
        f"Original dimensions: {json.dumps(dimensions)}\n"
        f"Auditor flags: {json.dumps(flags, ensure_ascii=False)}\n\n"
        f"Produce a corrected list of exactly {len(dimensions)} names."
    )
    messages = [
        {"role": "system", "content": _REPAIR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    for attempt in range(3):
        response = get_completion(messages, temperature=0.3)
        if not response:
            continue
        parsed = parse_json_response(response, default_value=None)
        if isinstance(parsed, dict):
            new = parsed.get("dimensions")
            if isinstance(new, list) and len(new) == len(dimensions):
                # defensively restore any missing core dim
                for d in core:
                    if d not in new:
                        new.insert(0, d)
                        if len(new) > len(dimensions):
                            new.pop()
                return new
        print(f"      Warning: repair parse failed (attempt {attempt+1}/3)")
    return None


def _generate_one_category(persona_block: str, category: Dict, so_far: Dict) -> List:
    """Select → audit → repair → fill pipeline (one round; further audit LLM judgments
    proved unstable and drove repair toward abstraction, so we accept round 1)."""
    dimensions = _select_dimensions(persona_block, category, so_far)
    print(f"      dimensions: {dimensions}")

    flags = _audit_dimensions(category, dimensions)
    if flags and (flags.get("leaky") or flags.get("redundant")):
        n_leak = len(flags.get("leaky", []))
        n_red  = len(flags.get("redundant", []))
        print(f"      audit: {n_leak} leaky, {n_red} redundant → repairing")
        repaired = _repair_dimensions(category, dimensions, flags)
        if repaired:
            dimensions = repaired
            print(f"      after repair: {dimensions}")
        else:
            print(f"      repair failed; keeping original dimensions")

    return _fill_dimensions(persona_block, category, dimensions, so_far)


def generate_memory_bank(base_profile: Dict) -> Dict:
    """Generate memory bank category by category, each with context from prior categories."""
    persona_block = _build_persona_block(base_profile)
    memory_bank: Dict = {}

    for category in _CATEGORY_CONFIGS:
        key = category["key"]
        print(f"    Generating {key} ...")
        items = _generate_one_category(persona_block, category, memory_bank)
        memory_bank[key] = items

    return memory_bank


def _empty_memory_bank() -> Dict:
    return {
        "skill_memory": [],
        "knowledge_memory": [],
        "episodic_memory": [],
        "self_model": [],
        "assistance_preference": [],
    }


# ---------------------------------------------------------------------------
# Step 3: Generate N users and save
# ---------------------------------------------------------------------------

def generate_all_users(num_users: int = 10, max_retries: int = 3) -> List[Dict]:
    users = []
    i = 0
    while len(users) < num_users:
        i += 1
        user_id = f"user_{len(users)+1:03d}"
        print(f"\n[{len(users)+1}/{num_users}] Generating base profile for {user_id} (attempt {i}) ...")

        try:
            base_profile = build_base_profile()
        except Exception as e:
            print(f"  Error building base profile: {e}, retrying ...")
            continue

        print(f"  Generating memory bank for {user_id} ...")
        try:
            memory_bank = generate_memory_bank(base_profile)
        except Exception as e:
            print(f"  Error generating memory bank: {e}, retrying ...")
            continue

        users.append({
            "user_id": user_id,
            "base_profile": base_profile,
            "memory_bank": memory_bank,
        })
        print(f"  Done: {user_id}")

    return users


def save_users(users: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_users": len(users),
        "users": users,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(users)} users to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate user memory banks using Deeppersona")
    parser.add_argument("--num_users", type=int, default=10, help="Number of users to generate")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(_HERE, "data", "user_memory_banks.json"),
        help="Output JSON file path",
    )
    args = parser.parse_args()

    print(f"Generating {args.num_users} user memory banks ...")
    users = generate_all_users(args.num_users)
    save_users(users, args.output)

    usage = get_token_usage()
    from config import _get_session_log
    usage_log = _get_session_log()
    with open(usage_log, "a", encoding="utf-8") as f:
        f.write(
            f"\n=== RUN TOTAL [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===\n"
            f"  prompt_tokens:     {usage['prompt_tokens']}\n"
            f"  completion_tokens: {usage['completion_tokens']}\n"
            f"  total_tokens:      {usage['total_tokens']}\n"
        )
    print(f"Token usage summary written to {usage_log}")


if __name__ == "__main__":
    main()
