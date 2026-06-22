"""
agent_nomem.py — No-memory baseline agent.

This agent has *no* persistent state across tasks. Each turn within an episode
sees only the current task and the in-episode dialogue history that the runner
already concatenates into the `task` argument; nothing is carried over between
tasks for the same user. For reconstruction scoring, get_all_memories() returns
an empty list (the slot-filler will produce "unknown" for every dimension,
which is the correct floor behaviour).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from openai import OpenAI

import token_tracker
from base_agent import MemoryAgent
from llm_client import GPT_MODEL, OPENAI_API_KEY


_SYSTEM_PROMPT = """\
You are a personalized AI assistant having an ongoing conversation with a specific user.
Respond directly and concisely to the user's message.
"""


class NoMemoryAgent(MemoryAgent):
    """
    A baseline that does not store anything across tasks.
    Useful as a floor in cross-system comparison.
    """

    def __init__(
        self,
        llm_model: str = GPT_MODEL,
        api_key: Optional[str] = None,
    ):
        self.llm_model = llm_model
        self._client = OpenAI(api_key=api_key or OPENAI_API_KEY)

    # ── agent_fn interface ────────────────────────────────────────────────────

    def __call__(self, task: str, history: List[Dict]) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": task},
        ]
        response = self._call_llm(messages)
        return response or "I'm sorry, I couldn't generate a response."

    # ── helpers ───────────────────────────────────────────────────────────────

    def _call_llm(self, messages: List[Dict], temperature: float = 0.7,
                  max_retries: int = 3) -> Optional[str]:
        import time
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=temperature,
                )
                if resp.usage:
                    token_tracker.log_usage(
                        self.llm_model,
                        resp.usage.prompt_tokens,
                        resp.usage.completion_tokens,
                        resp.usage.total_tokens,
                    )
                return resp.choices[0].message.content
            except Exception as e:
                print(f"[NoMemoryAgent] LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    return None

    # ── memory interface (all empty) ──────────────────────────────────────────

    def get_all_memories(self) -> List[Dict]:
        return []

    def save_memories(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"num_memories": 0, "memories": []}, f, ensure_ascii=False, indent=2)
        print(f"[NoMemoryAgent] Saved 0 memories → {path}")

    def reset(self) -> None:
        # nothing to clear
        pass
