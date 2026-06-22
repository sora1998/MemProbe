"""
agent.py — A-Mem based personalized agent for mem_bench.

Thin agent wrapper around AgenticMemorySystem (A-mem-sys):
  1. Retrieve top-k relevant memories for the current query
  2. Prepend memories to the system prompt
  3. Call LLM to generate a response
  4. Store the user's latest message as a new memory note

Compatible with simulation.py's Episode.run(agent_fn) interface:
    agent_fn(task: str, history: List[Dict]) -> str
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Optional

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "A-mem-sys"))

from agentic_memory.memory_system import AgenticMemorySystem
from llm_client import OPENAI_API_KEY, GPT_MODEL
from openai import OpenAI
import token_tracker
from base_agent import MemoryAgent

# ── prompts ───────────────────────────────────────────────────────────────────

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


class AMemAgent(MemoryAgent):
    """
    Personalized agent backed by A-mem-sys AgenticMemorySystem.

    Usage as agent_fn in simulation.py:
        agent = AMemAgent()
        result = episode.run(agent)   # AMemAgent.__call__ is the agent_fn

    After the episode, inspect what the agent learned:
        agent.get_all_memories()
    """

    def __init__(
        self,
        llm_model: str = GPT_MODEL,
        retrieve_k: int = 5,
        api_key: Optional[str] = None,
    ):
        api_key = api_key or OPENAI_API_KEY
        # A-mem-sys reads OPENAI_API_KEY from env
        os.environ.setdefault("OPENAI_API_KEY", api_key)

        self.retrieve_k = retrieve_k
        self.llm_model = llm_model
        self._client = OpenAI(api_key=api_key)

        # Response-generation LLM (this class) and A-mem's internal LLM both use OpenAI.
        self.memory = AgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend="openai",
            llm_model=llm_model,
            api_key=api_key,
        )

    # ── agent_fn interface ────────────────────────────────────────────────────

    def __call__(self, task: str, history: List[Dict]) -> str:
        """
        Called by Episode.run() each turn.

        task    — original task on turn 1;
                  "original task\n\n[Conversation so far]\n..." on later turns.
        history — list of {"agent": ..., "user": ...} for prior turns.
        """
        # What the user most recently said
        latest_user_msg = history[-1]["user"] if history else task

        # 1. Retrieve relevant memories
        memory_context = self._retrieve(latest_user_msg)

        # 2. Build messages
        messages = self._build_messages(task, memory_context)

        # 3. LLM call
        response = self._call_llm(messages)
        if not response:
            response = "I'm sorry, I couldn't generate a response."

        # 4. Store task and user message with separate categories
        if not history:
            # first turn: store the original task only (latest_user_msg == task here)
            self.memory.add_note(task, category="task")
        else:
            self.memory.add_note(latest_user_msg, category="user")

        return response

    # ── helpers ───────────────────────────────────────────────────────────────

    def _call_llm(self, messages: List[Dict], temperature: float = 0.7, max_retries: int = 3) -> Optional[str]:
        """Call OpenAI and log token usage to token_tracker (mem_bench/usage/)."""
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
                print(f"[AMemAgent] LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    return None

    def _retrieve(self, query: str) -> str:
        """Return a formatted string of top-k retrieved memories, or empty string."""
        results = self.memory.search(query, k=self.retrieve_k)
        if not results:
            return ""
        lines = []
        for r in results:
            tags = ", ".join(r.get("tags", [])) or "—"
            lines.append(f"• {r['content']}  [tags: {tags}]")
        return "\n".join(lines)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Native top-k retrieval used by the retrieve-based reconstruction scorer.

        Returns memory dicts shaped like get_all_memories() entries (have a
        ``content`` field plus original A-Mem metadata).
        """
        try:
            results = self.memory.search(query, k=k) or []
        except Exception:
            return []
        out: List[Dict] = []
        for r in results:
            out.append({
                "id":        r.get("id"),
                "content":   r.get("content", ""),
                "keywords":  r.get("keywords", []),
                "context":   r.get("context", ""),
                "tags":      r.get("tags", []),
                "category":  r.get("category", ""),
                "score":     r.get("score"),
            })
        return out

    def _build_messages(self, task: str, memory_context: str) -> List[Dict]:
        memory_block = (
            _MEMORY_BLOCK.format(memories=memory_context)
            if memory_context
            else ""
        )
        system = _SYSTEM_PROMPT.format(memory_block=memory_block)
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": task},
        ]

    # ── inspection ────────────────────────────────────────────────────────────

    def get_all_memories(self) -> List[Dict]:
        """
        Return all memory notes accumulated during the episode.
        Used for downstream evaluation (memory bank reconstruction).
        """
        return [
            {
                "id":        note.id,
                "content":   note.content,
                "keywords":  note.keywords,
                "context":   note.context,
                "tags":      note.tags,
                "category":  note.category,
                "timestamp": note.timestamp,
            }
            for note in self.memory.memories.values()
        ]

    def save_memories(self, path: str) -> None:
        """
        Dump all accumulated memories to a JSON file for offline inspection.

        Output format:
        {
            "num_memories": N,
            "memories": [ {id, content, keywords, context, tags, category, timestamp}, ... ]
        }
        """
        memories = self.get_all_memories()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"num_memories": len(memories), "memories": memories},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[AMemAgent] Saved {len(memories)} memories → {path}")

    def reset(self) -> None:
        """Clear all memories (reinitialise for a new user)."""
        # Explicitly drop the old system before building the new one so
        # the ChromaDB collection holding the previous user's embeddings
        # is released rather than lingering until GC.
        old = self.memory
        self.memory = None
        try:
            if hasattr(old, "memories"):
                old.memories.clear()
        except Exception:
            pass
        del old

        self.memory = AgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend="openai",
            llm_model=self.llm_model,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def add_external_memory(self, content: str, **kwargs) -> None:
        """Inject a memory observed outside a normal turn (e.g., final user feedback)."""
        self.memory.add_note(content, **kwargs)


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, _PROJECT_ROOT)
    from simulation import Episode

    agent = AMemAgent()

    episode = Episode(
        user_id="user_001",
        task="Can you recommend a good book for me?",
        ground_truth=None,
        dataset_type="llm",
        max_turns=25,
    )
    result = episode.run(agent)

    # Store the last user message (not captured inside __call__ because the
    # episode ends after the agent's final response, before another call)
    if result.turns:
        agent.add_external_memory(result.turns[-1].user_feedback.text, category="user")

    print(f"\nEnd reason : {result.end_reason}")
    print(f"Pref score : {result.final_preference_score}")
    for t in result.turns:
        print(f"\n[Turn {t.turn}]")
        print(f"  Agent : {t.agent_response[:120]}")
        print(f"  User  : {t.user_feedback.text[:120]}")
        print(f"  Pref  : {t.preference_score}/5")

    print("\n--- Accumulated memories ---")
    for m in agent.get_all_memories():
        tags = ", ".join(m["tags"]) or "—"
        print(f"  [{tags}] {m['content'][:100]}")

    # Save memories for offline inspection:
    #   memory/<user_id>/memories.json
    agent.save_memories(
        os.path.join(_PROJECT_ROOT, "memory", episode.user_id, "memories.json")
    )

    # Save full interaction history:
    #   history/<user_id>/episode_1.json
    result.save_history(
        os.path.join(_PROJECT_ROOT, "history", episode.user_id, "episode_1.json")
    )
