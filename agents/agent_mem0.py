"""
agent_mem0.py — Mem0 baseline agent.

Wraps the upstream `mem0ai` package (https://mem0.ai). Mem0 performs LLM-based
fact extraction at write time and exposes a simple add/search interface; the
extracted facts are short, structured strings (``"User prefers concise
checklists"``) rather than verbose notes.

Lifecycle (one Mem0Agent per user, as enforced by BenchmarkRunner):
    add_note     ←  m.add(messages, user_id=USER)
    retrieve     ←  m.search(query, filters={"user_id": USER}, top_k=K)
    save / dump  ←  m.get_all(filters={"user_id": USER})

The instance-level ``USER`` is a constant within one agent (the cross-user
isolation is provided by the runner constructing a fresh agent and we call
``m.reset()`` in ``__init__`` and ``reset()`` to drop any state from prior
instances that shared the same vector store).
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
Your goal is to give responses that are tailored to this particular person.

{memory_block}\
Respond directly and concisely to the user's message.
"""

_MEMORY_BLOCK = """\
--- Retrieved memories about this user ---
{memories}
------------------------------------------

"""

# Process-wide singleton: mem0 holds two qdrant stores that exclusive-lock
# their dirs (main vector store + migrations store under MEM0_DIR). Two live
# Memory instances in one process therefore collide. Sharing a single
# Memory and routing each agent's writes/reads through its own per-instance
# user_id namespace sidesteps the lock collision while still keeping every
# agent's memory cleanly isolated.
_SINGLETON_MEMORY = None
_SINGLETON_LOCK = __import__("threading").Lock()


def _get_or_create_memory(config: Dict, mem0_cls):
    global _SINGLETON_MEMORY
    with _SINGLETON_LOCK:
        if _SINGLETON_MEMORY is None:
            _SINGLETON_MEMORY = mem0_cls.from_config(config)
        return _SINGLETON_MEMORY


class Mem0Agent(MemoryAgent):
    """Mem0-backed agent."""

    def __init__(
        self,
        llm_model: str = GPT_MODEL,
        retrieve_k: int = 5,
        api_key: Optional[str] = None,
    ):
        api_key = api_key or OPENAI_API_KEY
        os.environ.setdefault("OPENAI_API_KEY", api_key)

        self.retrieve_k = retrieve_k
        self.llm_model = llm_model
        self._client = OpenAI(api_key=api_key)

        # Per-instance user_id keeps each agent's memory in its own
        # namespace within the shared singleton Memory instance.
        import uuid
        self._user_id = f"agent_{uuid.uuid4().hex[:8]}"

        # The shared Memory's qdrant + MEM0_DIR are configured exactly
        # once per process (see `_get_or_create_memory`). Paths sit
        # under TMPDIR (caller sets `TMPDIR` to a writable scratch dir),
        # unique per process via tempfile.mkdtemp().
        import tempfile
        self._qdrant_path = tempfile.mkdtemp(prefix="qdrant_mem0_")
        self._mem0_dir = tempfile.mkdtemp(prefix="mem0_dir_")
        os.environ.setdefault("MEM0_DIR", self._mem0_dir)

        from mem0 import Memory
        # Mem0's internal fact-extraction LLM is pinned to gpt-4o-mini.
        # mem0's out-of-the-box default (gpt-5-mini at the time of writing)
        # is a reasoning-class model that rejects `max_tokens`, and the
        # mem0 library does not adapt to the reasoning-model API. We pin a
        # non-reasoning sibling here so extraction actually runs. The
        # *agent reply* LLM (self._client / self.llm_model) is unchanged
        # and stays aligned with the other baselines for fair comparison.
        _MEM0_INTERNAL_MODEL = "gpt-4o-mini"
        self._mem0_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": _MEM0_INTERNAL_MODEL,
                    "temperature": 0.0,
                    "max_tokens": 2000,
                    "api_key": api_key,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": api_key,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0",
                    "path": self._qdrant_path,
                },
            },
        }
        self._mem0_cls = Memory
        self.memory = _get_or_create_memory(self._mem0_config, Memory)

    # ── agent_fn interface ────────────────────────────────────────────────────

    def __call__(self, task: str, history: List[Dict]) -> str:
        latest_user_msg = history[-1]["user"] if history else task

        memory_context = self._retrieve(latest_user_msg)
        messages = self._build_messages(task, memory_context)
        response = self._call_llm(messages)
        if not response:
            response = "I'm sorry, I couldn't generate a response."

        # Store the user's most recent utterance (matches the A-Mem contract).
        self._add(latest_user_msg)
        return response

    # ── helpers ───────────────────────────────────────────────────────────────

    def _retrieve(self, query: str) -> str:
        try:
            res = self.memory.search(
                query, top_k=self.retrieve_k,
                filters={"user_id": self._user_id},
            )
        except Exception as e:
            print(f"[Mem0Agent] search failed: {e}")
            return ""
        items = res.get("results") if isinstance(res, dict) else res
        if not items:
            return ""
        lines = [f"• {it.get('memory') or it.get('content', '')}" for it in items]
        return "\n".join(lines)

    def _add(self, content: str) -> None:
        try:
            self.memory.add(
                [{"role": "user", "content": content}],
                user_id=self._user_id,
            )
        except Exception as e:
            print(f"[Mem0Agent] add failed: {e}")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Native top-k retrieval used by the retrieve-based scorer."""
        try:
            res = self.memory.search(
                query, top_k=k,
                filters={"user_id": self._user_id},
            )
        except Exception:
            return []
        items = res.get("results") if isinstance(res, dict) else res
        if not items:
            return []
        out: List[Dict] = []
        for it in items:
            content = it.get("memory") or it.get("content") or ""
            out.append({
                "id":        it.get("id"),
                "content":   content,
                "category":  "fact",
                "metadata":  it.get("metadata"),
                "timestamp": it.get("created_at") or it.get("updated_at"),
            })
        return out

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
                print(f"[Mem0Agent] LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    return None

    def _build_messages(self, task: str, memory_context: str) -> List[Dict]:
        memory_block = _MEMORY_BLOCK.format(memories=memory_context) if memory_context else ""
        system = _SYSTEM_PROMPT.format(memory_block=memory_block)
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": task},
        ]

    # ── memory inspection ────────────────────────────────────────────────────

    def _list_all_items(self) -> List[Dict]:
        try:
            res = self.memory.get_all(filters={"user_id": self._user_id}, top_k=10000)
        except Exception as e:
            print(f"[Mem0Agent] get_all failed: {e}")
            return []
        items = res.get("results") if isinstance(res, dict) else res
        return list(items or [])

    def get_all_memories(self) -> List[Dict]:
        out: List[Dict] = []
        for it in self._list_all_items():
            content = it.get("memory") or it.get("content") or ""
            out.append({
                "id":        it.get("id"),
                "content":   content,
                "category":  it.get("metadata", {}).get("category", "fact") if isinstance(it.get("metadata"), dict) else "fact",
                "metadata":  it.get("metadata"),
                "timestamp": it.get("created_at") or it.get("updated_at"),
            })
        return out

    def save_memories(self, path: str) -> None:
        memories = self.get_all_memories()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"num_memories": len(memories), "memories": memories},
                      f, ensure_ascii=False, indent=2)
        print(f"[Mem0Agent] Saved {len(memories)} memories → {path}")

    # Intentionally NO `save_native_state` override.
    #
    # Mem-0 holds a process-wide singleton ``Memory`` instance whose qdrant
    # path is fixed by the FIRST agent created in the process (see
    # ``_get_or_create_memory`` above). Subsequent agents' ``self._qdrant_path``
    # / ``self._mem0_dir`` are fresh empty mkdtemp directories that the
    # singleton never touches, so per-agent on-disk snapshots cannot be
    # taken cleanly. Rebuild for re-scoring is supported via
    # ``memories.json`` (the textual list returned by ``get_all_memories``,
    # already filtered to this agent's ``user_id`` namespace) plus
    # re-embedding into a fresh Mem0 instance.

    def reset(self) -> None:
        # Clear only this agent's namespace from the shared singleton
        # Memory; other agents created in the same process keep their data.
        try:
            self.memory.delete_all(user_id=self._user_id)
        except Exception:
            pass

    def add_external_memory(self, content: str, **kwargs) -> None:
        self._add(content)
