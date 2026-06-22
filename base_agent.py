"""
base_agent.py — Abstract interface for memory-backed agents in mem_bench.

Any agent evaluated by the benchmark must implement this interface.
See agent.py (AMemAgent) for a reference implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class MemoryAgent(ABC):
    """
    Minimum contract a memory-backed agent must satisfy.

    Lifecycle during BenchmarkRunner.run():
        for each user_id:
            agent = MyAgent()            # fresh instance per user
            for each task in tasks:
                response = agent(task, history)       # __call__ per turn
                ...
            agent.add_external_memory(...)            # optional: inject missed signals
            agent.save_memories(path)                 # dump memories to disk

    Cross-user isolation is achieved by constructing a new agent per user.
    `reset()` is provided for alternate lifecycles that reuse one agent
    across users (not used by the current runner).
    """

    # ── Required: conversation interface ─────────────────────────────────────

    @abstractmethod
    def __call__(self, task: str, history: List[Dict]) -> str:
        """
        Produce one response given the task and prior turns.

        Args:
            task    : original task on turn 1; task + conversation-so-far on later turns.
            history : list of {"agent": str, "user": str} for prior turns (may be empty).

        Returns:
            The agent's response text.
        """
        ...

    # ── Required: memory access ──────────────────────────────────────────────

    @abstractmethod
    def get_all_memories(self) -> List[Dict]:
        """
        Return all accumulated memories as JSON-serializable dicts.

        Each memory dict SHOULD contain at minimum a readable "content" field;
        additional fields (tags, keywords, category, timestamp, ...) are allowed
        and preserved for downstream reconstruction evaluation.
        """
        ...

    @abstractmethod
    def save_memories(self, path: str) -> None:
        """Persist all accumulated memories to a JSON file."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all memory state. Called between users."""
        ...

    # ── Optional: out-of-band memory injection ───────────────────────────────

    def add_external_memory(self, content: str, **kwargs) -> None:
        """
        Inject a memory observed outside a normal turn (e.g., the final user
        feedback that occurs after the agent's last response).

        Default is a no-op; override if the underlying memory system supports it.
        """
        return None

    # ── Optional: native (retriever-side) state snapshot ─────────────────────

    def save_native_state(self, target_dir: str) -> None:
        """
        Persist the agent's underlying retriever state to ``target_dir``.

        ``save_memories`` writes the textual memory list (for slot-fill /
        reconstruction). ``save_native_state`` writes the raw on-disk
        structures used at retrieve time (chromadb, qdrant, etc.) so that
        a future tool can rebuild the live agent and re-invoke
        ``agent.search(...)`` without re-running the episode loop.

        Default is a no-op (for stateless agents like ``nomem``); each
        memory-backed agent should override.
        """
        return None
