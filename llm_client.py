"""
llm_client.py — mem_bench's own OpenAI completion helper.

Owns: API key, model selection, retry, reasoning-model max-token handling,
token accounting (via token_tracker), and JSON parsing of LLM output.

Why this exists separately from Deeppersona/generate_user_profile/config.py:
benchmark code should not depend on the user-profile generation pipeline's
internals. That config.py is kept as-is for Deeppersona's own use.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

import token_tracker

# ── Configuration ─────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL = "gpt-5.4-mini"

_client = OpenAI(api_key=OPENAI_API_KEY)

# Reasoning models (gpt-5, o1, o3 series) consume part of the output budget on
# hidden reasoning tokens. Without a generous ceiling, structured JSON outputs
# frequently get truncated mid-field.
_DEFAULT_MAX_COMPLETION_TOKENS = 8192


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(tag in m for tag in ("gpt-5", "o1", "o3"))


# ── Completion ────────────────────────────────────────────────────────────────

def get_completion(
    messages: List[Dict[str, str]],
    model: str = GPT_MODEL,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> Optional[str]:
    """Call OpenAI chat completions with retry + token logging."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if _is_reasoning_model(model):
        kwargs["max_completion_tokens"] = _DEFAULT_MAX_COMPLETION_TOKENS

    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(**kwargs)
            if getattr(resp, "usage", None):
                token_tracker.log_usage(
                    model,
                    resp.usage.prompt_tokens,
                    resp.usage.completion_tokens,
                    resp.usage.total_tokens,
                )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[llm_client] API attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
    return None


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _strip_markdown_fence(text: str) -> str:
    """Remove ```json ... ``` fences the model sometimes adds."""
    if text and text.strip().startswith("```") and "```" in text:
        inner = text.split("```", 2)[1]
        if inner.startswith("json"):
            inner = inner[4:].strip()
        return inner.strip()
    return text


def parse_json_response(text: str, default_value: Any = None) -> Any:
    if not text:
        return default_value
    text = _strip_markdown_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[llm_client] JSON parse failed: {e}")
        preview = text[:200] + ("..." if len(text) > 200 else "")
        print(f"[llm_client] Response was: {preview}")
        return default_value
