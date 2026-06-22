"""
agent_longctx.py — Long-context dump baseline.

Stores every turn (task statement, agent reply, user reply) verbatim with no
abstraction or evolution. Two variants control whether the accumulated history
is fed back into the agent at reply time:

    use_in_reply=False  →  agent acts like NoMemoryAgent during the dialogue,
                           but the full transcript is preserved for the
                           reconstruction probe (isolates the write-side test
                           — read-side conditions match the no-memory baseline).
    use_in_reply=True   →  every prior task's transcript is replayed as
                           assistant/user messages in the agent prompt before
                           the current turn (raw long-context retention upper
                           bound under the read-side condition).

In both variants get_all_memories() returns a flat list of memory dicts with
a `content` field, so the existing slot-fill / judge / E-metric pipeline works
unchanged. A long-context agent will naturally show a very large
`num_memories` and the corresponding `avg_memory_chars` of raw turns, in
contrast with the much smaller, abstracted footprints of A-Mem or Mem0.
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


class LongContextAgent(MemoryAgent):
    """
    No-abstraction baseline that stores every turn verbatim.

    Memory representation (per turn):
        {"content": "[Task k] <task>",  "category": "task",        "task_index": k}
        {"content": "Agent: <reply>",   "category": "agent_reply", "task_index": k, "turn": t}
        {"content": "User: <message>",  "category": "user_reply",  "task_index": k, "turn": t}
    """

    def __init__(
        self,
        llm_model: str = GPT_MODEL,
        api_key: Optional[str] = None,
        use_in_reply: bool = False,
    ):
        self.llm_model = llm_model
        self.use_in_reply = use_in_reply
        self._client = OpenAI(api_key=api_key or OPENAI_API_KEY)

        # Completed tasks (each task is a dict {"task": str, "turns": [...]} ).
        self.completed_tasks: List[Dict] = []
        # In-progress task buffer (current task being processed).
        self._current: Optional[Dict] = None

    # ── agent_fn interface ────────────────────────────────────────────────────

    def __call__(self, task: str, history: List[Dict]) -> str:
        # Empty history signals the start of a new task. Finalize any
        # in-progress task buffer first.
        if not history:
            if self._current is not None:
                self.completed_tasks.append(self._current)
            self._current = {"task": task, "turns": []}

        # Sync current task's turns with the history reported by the runner.
        if self._current is not None:
            self._current["turns"] = [dict(h) for h in history]

        # Build LLM input
        if self.use_in_reply:
            messages = self._build_long_context_messages(task)
        else:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ]

        response = self._call_llm(messages)
        return response or "I'm sorry, I couldn't generate a response."

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_long_context_messages(self, current_task_text: str) -> List[Dict]:
        msgs: List[Dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for t in self.completed_tasks:
            msgs.append({"role": "user", "content": t["task"]})
            for turn in t["turns"]:
                if turn.get("agent"):
                    msgs.append({"role": "assistant", "content": turn["agent"]})
                if turn.get("user"):
                    msgs.append({"role": "user", "content": turn["user"]})
        # current task — already contains [Conversation so far] when applicable.
        msgs.append({"role": "user", "content": current_task_text})
        return msgs

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
                print(f"[LongContextAgent] LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    return None

    # ── memory interface ──────────────────────────────────────────────────────

    def add_external_memory(self, content: str, **kwargs) -> None:
        """Capture the final user message after the agent's last reply."""
        if self._current is not None:
            self._current["turns"].append({"agent": None, "user": content})

    def _flatten_to_notes(self) -> List[Dict]:
        all_tasks = list(self.completed_tasks)
        if self._current is not None:
            all_tasks.append(self._current)
        notes: List[Dict] = []
        for ti, t in enumerate(all_tasks, 1):
            notes.append({
                "content": f"[Task {ti}] {t['task']}",
                "category": "task",
                "task_index": ti,
            })
            for turn_i, turn in enumerate(t["turns"], 1):
                if turn.get("agent"):
                    notes.append({
                        "content": f"Agent: {turn['agent']}",
                        "category": "agent_reply",
                        "task_index": ti,
                        "turn": turn_i,
                    })
                if turn.get("user"):
                    notes.append({
                        "content": f"User: {turn['user']}",
                        "category": "user_reply",
                        "task_index": ti,
                        "turn": turn_i,
                    })
        return notes

    def get_all_memories(self) -> List[Dict]:
        return self._flatten_to_notes()

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Native top-k retrieval over the flat list of stored turns.

        LongContext has no learned retriever; we score with a uniform
        sentence-transformer embedder over each note's `content` field.
        This still tests the LongContext premise (raw turns are searchable
        out of the box) without requiring it to ship its own retrieval.
        """
        notes = self._flatten_to_notes()
        if not notes:
            return []
        try:
            encoder = self._get_encoder()
            import numpy as np
            note_texts = [n["content"] for n in notes]
            note_emb = encoder.encode(note_texts, convert_to_numpy=True, normalize_embeddings=True)
            q_emb = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            scores = note_emb @ q_emb
            top_idx = np.argsort(-scores)[:k]
            return [{**notes[i], "score": float(scores[i])} for i in top_idx]
        except Exception as e:
            print(f"[LongContextAgent] search failed: {e}")
            # fallback: return first k notes
            return notes[:k]

    def _get_encoder(self):
        """Lazily build a sentence-transformer encoder; cached on the class."""
        if getattr(LongContextAgent, "_encoder", None) is None:
            from sentence_transformers import SentenceTransformer
            LongContextAgent._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return LongContextAgent._encoder

    def save_memories(self, path: str) -> None:
        notes = self._flatten_to_notes()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"num_memories": len(notes), "memories": notes},
                f, ensure_ascii=False, indent=2,
            )
        print(f"[LongContextAgent] Saved {len(notes)} memories → {path}")

    def reset(self) -> None:
        self.completed_tasks = []
        self._current = None
