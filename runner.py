"""
runner.py — Benchmark runner for Human Memory Benchmark.

Manages m × n episode execution with cross-episode memory:
  - Same user's n tasks: agent memory persists (no reset)
  - Between users: agent.reset()
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

from simulation import Episode, load_user, make_agent
from scorer import UserScorer, BenchmarkScorer, UserScore, BenchmarkReport

_HERE = os.path.dirname(os.path.abspath(__file__))


class BenchmarkRunner:
    """
    Runs m × n episodes with correct cross-episode memory management.

    Memory rule:
      - Same user, task 1 → ... → task n : memory persists
      - After all tasks for user_i done  : agent.reset()
      - Next user starts with a clean memory
    """

    def __init__(
        self,
        user_ids: List[str],
        tasks: Optional[List[Tuple[str, Optional[str]]]] = None,
        tasks_per_user: Optional[Dict[str, List[Tuple[str, Optional[str]]]]] = None,
        agent_name: str = "amem",
        dataset_type: str = "llm",
        max_turns: int = 25,
        pref_threshold: int = 4,
        memory_dir: str = "memory",
        report_dir: str = "output",
        run_id: Optional[str] = None,
        agent_kwargs: Optional[Dict] = None,
        scoring_modes: Optional[List[str]] = None,
    ):
        if tasks is None and tasks_per_user is None:
            raise ValueError("Must provide either `tasks` (shared) or `tasks_per_user` (per-user).")
        scoring_modes = scoring_modes or ["dump_all"]
        for m in scoring_modes:
            if m not in ("dump_all", "retrieve"):
                raise ValueError(f"scoring_mode must be 'dump_all' or 'retrieve', got {m!r}")
        self.user_ids        = user_ids
        self.tasks           = tasks
        self.tasks_per_user  = tasks_per_user
        self.agent_name      = agent_name
        self.dataset_type    = dataset_type
        self.max_turns       = max_turns
        self.pref_threshold  = pref_threshold
        self.scoring_modes   = scoring_modes
        # Each output type lives under its own parent directory; run_id (if
        # provided) is a sub-directory under each, so all artefacts of one
        # run are clearly grouped:
        #   memory/<run_id>/<user>/memories.json
        #   history/<run_id>/<user>/episode_*.json
        #   pref_judge/<run_id>/<user>/episode_*.json
        #   output/<run_id>/<agent>_<ts>.json (dump_all)
        #   output/<run_id>_retrieve/<agent>_<ts>.json (retrieve, if requested)
        # Episode artefacts (memory/history/pref_judge) are mode-independent
        # and therefore always live under the dump_all run_id; only the
        # scoring outputs split per mode.
        rid = run_id or "default"
        self._rid            = rid
        self.memory_dir      = os.path.join(memory_dir, rid)
        self.history_dir     = os.path.join("history", rid)
        self.pref_judge_dir  = os.path.join("pref_judge", rid)
        self._report_root    = report_dir
        self.agent_kwargs    = agent_kwargs or {}

    def _tasks_for(self, user_id: str) -> List[Tuple[str, Optional[str]]]:
        """Return the task list for a given user (per-user if set, else shared)."""
        if self.tasks_per_user is not None:
            return self.tasks_per_user[user_id]
        return self.tasks

    def _output_dir_for_mode(self, mode: str) -> str:
        # dump_all writes to output/<rid>/, retrieve writes to output/<rid>_retrieve/.
        suffix = "" if mode == "dump_all" else "_retrieve"
        return os.path.join(self._report_root, self._rid + suffix)

    def run(self) -> Dict[str, BenchmarkReport]:
        """
        Run all m × n episodes once, then score each user under every requested
        scoring mode using the same in-memory agent state. Returns a dict
        {scoring_mode: BenchmarkReport}.
        """
        # One scorer per requested mode so each mode writes its recon_judge
        # files into its own output subtree.
        user_scorers = {
            mode: UserScorer(
                recon_judge_dir=os.path.join(_HERE, self._output_dir_for_mode(mode), "recon_judge"),
            )
            for mode in self.scoring_modes
        }
        bench_scorer = BenchmarkScorer()
        all_user_scores: Dict[str, List[UserScore]] = {m: [] for m in self.scoring_modes}

        for user_id in self.user_ids:
            tasks = self._tasks_for(user_id)
            print(f"\n{'='*50}")
            print(f"User: {user_id}  ({len(tasks)} tasks)")
            print(f"{'='*50}")

            agent = make_agent(self.agent_name, **self.agent_kwargs)
            user_episode_results = []

            for i, (task, gt) in enumerate(tasks, 1):
                print(f"\n  [Task {i}/{len(tasks)}] {task[:80]}...")

                episode = Episode(
                    user_id=user_id,
                    task=task,
                    ground_truth=gt,
                    dataset_type=self.dataset_type,
                    max_turns=self.max_turns,
                    pref_threshold=self.pref_threshold,
                )
                result = episode.run(agent)

                # Store the last user message — the episode ends after the agent's
                # final response, so this feedback isn't observed by the agent in-turn.
                if result.turns:
                    agent.add_external_memory(
                        result.turns[-1].user_feedback.text,
                        category="user",
                    )

                hist_path = os.path.join(_HERE, self.history_dir, user_id, f"episode_{i}.json")
                result.save_history(hist_path)

                judge_path = os.path.join(_HERE, self.pref_judge_dir, user_id, f"episode_{i}.json")
                user_prefs = load_user(user_id)["memory_bank"].get("assistance_preference", [])
                result.save_pref_judge(judge_path, user_prefs)

                user_episode_results.append(result)

                print(f"    end_reason : {result.end_reason}")
                print(f"    turns      : {len(result.turns)}")

            # save memories after all tasks for this user
            mem_path = os.path.join(_HERE, self.memory_dir, user_id, "memories.json")
            agent.save_memories(mem_path)
            # Also snapshot the underlying retriever state (chromadb / qdrant
            # / tmpdir) so a downstream tool can reattach and re-invoke
            # search() without rerunning the episode loop.
            try:
                agent.save_native_state(
                    os.path.join(_HERE, self.memory_dir, user_id, "raw")
                )
            except Exception as e:
                print(f"  [{user_id}] save_native_state warning: {e}")

            # Score the same agent state under every requested mode.
            agent_memories = agent.get_all_memories()
            for mode in self.scoring_modes:
                use_retrieve = (mode == "retrieve")
                if use_retrieve and not hasattr(agent, "search"):
                    print(f"  [skip {mode}] agent has no search(); skipping retrieve scoring")
                    continue
                user_score = user_scorers[mode].score(
                    user_id=user_id,
                    episode_results=user_episode_results,
                    agent_memories=agent_memories,
                    agent=agent,
                    use_retrieve=use_retrieve,
                )
                print(f"\n--- scoring_mode={mode} ---")
                user_scorers[mode].print_report(user_score)
                all_user_scores[mode].append(user_score)

        # Benchmark-level aggregation per mode.
        reports: Dict[str, BenchmarkReport] = {}
        ts = time.strftime("%Y%m%d_%H%M%S")
        for mode in self.scoring_modes:
            scores = all_user_scores[mode]
            if not scores:
                continue
            report = bench_scorer.aggregate(scores)
            print(f"\n========== aggregate ({mode}) ==========")
            bench_scorer.print_report(report)
            report_path = os.path.join(
                _HERE, self._output_dir_for_mode(mode), f"{self.agent_name}_{ts}.json"
            )
            bench_scorer.save_report(report, report_path)
            reports[mode] = report
        return reports


if __name__ == "__main__":
    import argparse
    import simulation  # noqa: E402  (we mutate its data path before first load_user call)

    p = argparse.ArgumentParser()
    p.add_argument("--tasks-dir", default="CustomTasks",
                   help="folder under benchmark_data/ holding <user_id>.json task files")
    p.add_argument("--run-id", default=None,
                   help="suffix for all output dirs (memory/history/pref_judge/output) to avoid collisions")
    p.add_argument("--bank", default=None,
                   help="override path to user_memory_banks.json (default: Deeppersona/data/user_memory_banks.json)")
    p.add_argument("--agent", default="amem",
                   help="memory system to evaluate (e.g. amem, nomem)")
    p.add_argument("--scoring-modes", nargs="+", default=["dump_all"],
                   choices=["dump_all", "retrieve"],
                   help="reconstruction probe input: full dump (dump_all, default) "
                        "and/or top-k from agent.search() (retrieve). Multiple modes "
                        "share one episode loop and write separate output subdirs "
                        "(retrieve goes to <run-id>_retrieve).")
    p.add_argument("--users", nargs="+", default=None,
                   help="user_ids to run (e.g. user_001 user_002 ...); "
                        "default user_001 user_002 user_003")
    args = p.parse_args()

    # Tag the usage log with this run's name so /usage/<run_name>_<ts>.txt
    # is identifiable per run.
    import token_tracker
    token_tracker.set_run_name(args.run_id or args.agent)

    if args.bank:
        simulation._USER_DATA_PATH = args.bank
        simulation._user_cache = None
        print(f"[runner] bank path overridden → {args.bank}")

    from task_generator import load_custom_tasks

    user_ids = args.users or ["user_001", "user_002", "user_003"]
    tasks_per_user = {
        uid: load_custom_tasks(
            uid,
            path=os.path.join(_HERE, "benchmark_data", args.tasks_dir, f"{uid}.json"),
        )
        for uid in user_ids
    }

    runner = BenchmarkRunner(
        user_ids=user_ids,
        tasks_per_user=tasks_per_user,
        agent_name=args.agent,
        dataset_type="personamem",   # no-GT dataset; Task Fit falls back to `satisfied`
        max_turns=25,
        pref_threshold=4,
        run_id=args.run_id,
        scoring_modes=args.scoring_modes,
    )
    runner.run()
