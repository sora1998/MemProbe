"""
agent_memt.py — Mem-T baseline agent.

Wraps the upstream Mem-T pipeline (https://github.com/yanweiyue/Mem-T) and
exposes two variants via the make_agent registry:

    "memt"          — Full Mem-T deployment shape: write-side fact /
                      experience / persona / summary extraction + ReAct
                      retrieval where the trained Mem-T-4B model produces
                      the final answer through its FinishTool. The reply
                      returned to the runner is Mem-T-4B's answer.
    "memt_memonly"  — Same Mem-T memory mechanism (write + retrieval), but
                      the FinishTool answer is ignored. Instead, the raw
                      `<observation>` blocks accumulated during the ReAct
                      loop are extracted and passed as context to the
                      shared backbone reply LLM (the same model the other
                      baselines use). Isolates the memory mechanism from
                      Mem-T-4B's reply ability.

Infra prerequisites:
    - vLLM serving Mem-T-4B at http://127.0.0.1:8765/v1 (model id
      "Mem-T-4B"). Caller starts this once per process pool.
    - chromadb for the vector store, in `persistent` mode under TMPDIR
      (caller exports `TMPDIR` to a writable scratch dir).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import uuid
from typing import Dict, List, Optional

from openai import OpenAI

import token_tracker
from base_agent import MemoryAgent
from llm_client import GPT_MODEL, OPENAI_API_KEY

# Make Mem-T's modules importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MEMT_DIR = os.path.join(_PROJECT_ROOT, "Mem-T")
if _MEMT_DIR not in sys.path:
    sys.path.insert(0, _MEMT_DIR)


_REPLY_SYSTEM_PROMPT = """\
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

# vLLM server hosting Mem-T-4B (OpenAI-compatible endpoint).
_MEMT_BASE_URL = os.environ.get("MEMT_BASE_URL", "http://127.0.0.1:8765/v1")
_MEMT_MODEL_ID = os.environ.get("MEMT_MODEL_ID", "Mem-T-4B")


# ── Mem-T LLM adapter ────────────────────────────────────────────────────────
# Mem-T's LLMAPIClientBase expects a get_completion(messages, stop=, json_mode=)
# interface. We give it one backed by our vLLM server.

def _make_memt_llm_client():
    from llm_api import OpenAIAPIClient

    client = OpenAIAPIClient.__new__(OpenAIAPIClient)
    # Bypass the constructor's env var requirements; set the fields we need.
    client.client = OpenAI(base_url=_MEMT_BASE_URL, api_key="EMPTY")
    client.model = _MEMT_MODEL_ID
    # Mem-T's get_completion uses temperature/max_tokens off self.config.
    # Build a stand-in.
    class _Cfg:
        temperature = 0.0
        max_tokens = 4096
    client.config = _Cfg()
    return client


# ── Per-instance Mem-T config ────────────────────────────────────────────────

def _build_memt_config(traj_dir: str, db_path: str):
    from config import SystemConfig, VectorDBConfig
    cfg = SystemConfig()
    cfg.vector_db = VectorDBConfig(
        backend="chroma",
        db_type="persistent",
        path=db_path,
        from_scratch=True,
    )
    cfg.data_name = "memprobe"
    cfg.traj_dir = traj_dir
    cfg.log_path = os.path.join(traj_dir, "memt.log")
    cfg.USE_LOCAL_LLM = True
    cfg.USE_PARALLEL = False
    return cfg


# ── Memory wrapper ───────────────────────────────────────────────────────────

class _MemTMemory:
    """Per-instance Mem-T memory pipeline (formation+update+retrieval)."""

    def __init__(self):
        self._tmp_root = tempfile.mkdtemp(prefix="memt_")
        os.makedirs(os.path.join(self._tmp_root, "traj"), exist_ok=True)
        os.makedirs(os.path.join(self._tmp_root, "db"), exist_ok=True)
        self.cfg = _build_memt_config(
            traj_dir=os.path.join(self._tmp_root, "traj"),
            db_path=os.path.join(self._tmp_root, "db"),
        )

        from vector_db import VectorDBFactory
        from memory_formation import MemoryFormation
        from memory_update import MemoryUpdate
        from memory_retrieval import MemoryRetriever

        self.llm = _make_memt_llm_client()
        self.db = VectorDBFactory.create_db(self.cfg.vector_db)
        self.formation = MemoryFormation(llm_executor=self.llm)
        self.update = MemoryUpdate(llm_executor=self.llm, vector_db=self.db)
        self.retriever = MemoryRetriever(
            vector_db=self.db, llm_executor=self.llm, config=self.cfg,
        )

        # Each agent instance gets a unique sample_id namespace.
        self.sample_id = f"hmb_{uuid.uuid4().hex[:8]}"
        # Initialise the per-sample collections (turns/facts/experiences/personas/summary).
        for base in ("turns", "facts", "experiences", "personas", "summary"):
            self.db.create_collection(f"{self.sample_id}_{base}", get_or_create=True)

        # Local index of all stored memories so we can dump for slot-fill.
        self._collections = [
            f"{self.sample_id}_turns",
            f"{self.sample_id}_facts",
            f"{self.sample_id}_experiences",
            f"{self.sample_id}_personas",
            f"{self.sample_id}_summary",
        ]

        # Cross-turn state mirroring Mem-T's session-level continuity.
        # The original MemoryBuilder feeds prev_summary and prev_personas
        # into formation each turn so the LLM can decide whether to extend
        # them. We maintain the same state here, just streamed per turn.
        self._session_id = "hmb_session"
        self._prev_summary: str = ""
        self._prev_summary_turns: List[str] = []
        self._prev_summary_turn_ids: List[List[str]] = []
        self._prev_personas: Dict[str, str] = {}
        self._prev_personas_turns: Dict[str, List[str]] = {}
        self._prev_personas_turn_ids: Dict[str, List[List[str]]] = {}

    # ── write side ──────────────────────────────────────────────────────────

    def add_turn(self, turn_text: str, turn_id: str):
        """Per-turn write: formation → update.

        Mirrors the original Mem-T MemoryBuilder._process_session loop body:
          - feed formation the running summary + personas (cross-turn state)
          - upsert summary / persona outputs (Mem-T's hierarchical layers)
          - for fact/experience candidates, retrieve top-k similar items from
            the corresponding collection and let the update LLM decide
            add/update/delete/ignore.
        """
        import datetime, time, uuid as _uuid
        from utils import format_memory_content, parse_memory_content

        # Use a real wall-clock turn_time. start_time / end_time are left
        # empty: Mem-T-4B otherwise hallucinates "15 July, 2023" from its
        # training prompt template; HMBench's synthetic personas have no
        # event timeline, so the right answer is "unknown".
        turn_time = datetime.datetime.now().isoformat(timespec="seconds")
        c_turns    = f"{self.sample_id}_turns"
        c_facts    = f"{self.sample_id}_facts"
        c_exp      = f"{self.sample_id}_experiences"
        c_personas = f"{self.sample_id}_personas"
        c_summary  = f"{self.sample_id}_summary"

        # 1. raw turn
        self.db.add(
            c_turns, ids=[turn_id],
            documents=[turn_text],
            metadatas=[{
                "id": turn_id, "col_name": c_turns,
                "session_id": self.sample_id,
                "turn_time": turn_time,
                "source_turn_ids": [[turn_id]],
            }],
        )

        # 2. formation: feed the running summary + persona context so the LLM
        #    can decide whether to extend them (faithful to the original).
        prev_personas_text = "\n".join(
            f"Name: {n}, Profile: {p}" for n, p in self._prev_personas.items()
        )
        formation_messages = self.formation.construct_prompt(
            turn_text, self._prev_summary, prev_personas_text,
        )
        formation_response = self.formation.llm_executor.get_completion(formation_messages) or ""
        formation_op_id = str(_uuid.uuid4())
        tool_calls = _parse_tool_calls(formation_response)

        extracted_for_update: List[Dict] = []
        for tc in tool_calls:
            try:
                result = self.formation.execute_tool(tc["name"], tc["arguments"])
            except Exception:
                continue
            if not result:
                continue
            r_type = result.get("type")

            if r_type == "summary":
                # Upsert the running summary; track the source turns it
                # accumulates so SearchSummaryTool can attribute it.
                self._prev_summary_turns.append(turn_text)
                self._prev_summary_turn_ids.append([turn_id])
                new_source = "\n".join(self._prev_summary_turns)
                full_doc = format_memory_content(result["document"], new_source)
                self.db.upsert(
                    c_summary, ids=[self._session_id],
                    documents=[full_doc],
                    metadatas=[{
                        "id": self._session_id, "col_name": c_summary,
                        "session_id": self._session_id,
                        "turn_time": turn_time,
                        "source_turn_ids": list(self._prev_summary_turn_ids),
                        "op_ids": [{"formation": formation_op_id}],
                        "updated_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "original_turns": list(self._prev_summary_turns),
                        "memory_content": result["document"],
                    }],
                )
                self._prev_summary = result["document"]

            elif r_type == "persona":
                name = result.get("name") or "user"
                self._prev_personas_turns.setdefault(name, []).append(turn_text)
                self._prev_personas_turn_ids.setdefault(name, []).append([turn_id])
                new_source = "\n".join(self._prev_personas_turns[name])
                full_doc = format_memory_content(result["document"], new_source)
                self.db.upsert(
                    c_personas, ids=[name],
                    documents=[full_doc],
                    metadatas=[{
                        "id": name, "col_name": c_personas,
                        "name": name, "turn_time": turn_time,
                        "source_turn_ids": list(self._prev_personas_turn_ids[name]),
                        "op_ids": [{"formation": formation_op_id}],
                        "updated_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "original_turns": list(self._prev_personas_turns[name]),
                        "memory_content": result["document"],
                    }],
                )
                self._prev_personas[name] = result["document"]

            else:  # fact / experience: candidate for update phase
                result["turn_time"]  = turn_time
                result["start_time"] = ""
                result["end_time"]   = ""
                result["original_text"] = turn_text
                result["original_turns"] = [turn_text]
                extracted_for_update.append(result)

        # 3. update: for each fact/experience candidate, retrieve top-k
        #    similar existing items so the update LLM can decide
        #    add/update/delete/ignore. This is Mem-T's dedup + refinement step.
        topk_neighbours = self.cfg.memory.update_retrieval_topk
        for item in extracted_for_update:
            col_name = c_facts if item["type"] == "fact" else c_exp

            related_items: List[Dict] = []
            try:
                search_res = self.db.search(
                    col_name, query_texts=[item["document"]],
                    top_k=topk_neighbours, include=["documents", "metadatas"],
                )
                rel_ids   = (search_res.get("ids")        or [[]])[0]
                rel_docs  = (search_res.get("documents")  or [[]])[0]
                rel_metas = (search_res.get("metadatas")  or [[]])[0]
                for rid, rdoc, rmeta in zip(rel_ids, rel_docs, rel_metas):
                    if rmeta is None:
                        rmeta = {}
                    r_content, _ = parse_memory_content(rdoc) if rdoc else ("", "")
                    related_items.append({
                        "id":         rid,
                        "document":   r_content,
                        "turn_time":  rmeta.get("turn_time"),
                        "start_time": rmeta.get("start_time"),
                        "end_time":   rmeta.get("end_time"),
                    })
            except Exception:
                related_items = []

            try:
                schemas = self.update.get_tool_schemas(col_name)
                potential = {
                    "type":       item["type"],
                    "document":   item["document"],
                    "turn_time":  item["turn_time"],
                    "start_time": item["start_time"],
                    "end_time":   item["end_time"],
                }
                update_messages = self.update.construct_prompt(potential, related_items, schemas)
                update_response = self.update.llm_executor.get_completion(update_messages) or ""
                update_calls = _parse_tool_calls(update_response)
            except Exception:
                update_calls = []

            update_op_id = str(_uuid.uuid4())
            for tc in update_calls:
                tc["arguments"].update({
                    "source_turn_ids": [turn_id],
                    "op_ids": {"formation": formation_op_id, "update": update_op_id},
                    "original_text": turn_text,
                    "original_turns": [turn_text],
                    "turn_time":  turn_time,
                    "start_time": "",
                    "end_time":   "",
                })
                try:
                    self.update.execute_tool(col_name, tc["name"], tc["arguments"])
                except Exception:
                    continue

    # ── read side ────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> Dict:
        """Run Mem-T's ReAct retrieval. Returns both the LLM-authored finish
        answer AND the actual memory items surfaced during the loop.

        Mem-T's upstream ``retrieve_and_answer`` does not expose the
        memories it observed — it only returns the finish answer and a
        thought trace. We therefore re-implement the loop here so we can
        accumulate ``mem_metadatas`` from each search step, exactly as the
        upstream loop does, but return them alongside the answer.
        """
        retr = self.retriever
        max_steps = self.cfg.memory.max_tool_steps

        messages = retr.construct_prompt(query, category="")
        seen_turns: set = set()
        accumulated: List[Dict] = []
        traces: List[Dict] = []
        answer = ""

        for step in range(max_steps):
            try:
                response_text = retr.llm.get_completion(
                    messages, stop=["</tool_call>"], json_mode=False,
                ) or ""
            except Exception as e:
                return {"answer": "", "traces": traces, "observations": accumulated,
                        "error": f"LLM call failed: {e}"}
            if "<tool_call>" in response_text and "</tool_call>" not in response_text:
                response_text += "</tool_call>"

            try:
                tool_name, tool_args = retr._parse_xml_tool(response_text)
            except Exception:
                break
            if not tool_name:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user",
                                 "content": "No tool call detected. Use proper format."})
                continue
            tool_args["sample_id"] = self.sample_id

            if tool_name == "finish":
                answer = (tool_args.get("answer") or "").strip()
                traces.append({"step": step, "tool_call": {"name": "finish",
                                                            "args": tool_args}})
                break

            try:
                observation, mem_metadatas = retr.execute_tool(
                    tool_name, tool_args, seen_turns=seen_turns,
                )
            except Exception:
                continue

            traces.append({"step": step,
                           "tool_call": {"name": tool_name, "args": tool_args}})
            for m in (mem_metadatas or []):
                if not m:
                    continue
                content = m.get("memory_content") or m.get("content") or ""
                if not content or "dummy note for initializing" in content.lower():
                    continue
                col = m.get("col_name", "")
                cat = col.rsplit("_", 1)[-1] if "_" in col else col or "memt"
                accumulated.append({
                    "id":       m.get("id"),
                    "content":  content,
                    "category": cat,
                    "score":    m.get("score") or m.get("distance"),
                    "timestamp": m.get("turn_time"),
                })

            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": (
                    f"The retrieved memories are: <observation>{observation}</observation>\n"
                    f"If you have enough information, call finish; otherwise refine your search."
                ),
            })

        # max_steps fallback (mirrors the upstream): if no finish was called,
        # synthesise an answer from the accumulated observations using
        # AnswerWithMemoriesTool, so memt with use_finish_answer=True still
        # gets a usable reply when the ReAct loop didn't converge in time.
        if not answer:
            try:
                from memory_retrieval import AnswerWithMemoriesTool
                full_memory_text = "\n\n".join(
                    item["content"] for item in accumulated if item.get("content")
                )
                if full_memory_text:
                    answer_tool = AnswerWithMemoriesTool(
                        retr.llm, retr.benchmark_name, "",
                    )
                    answer, _ = answer_tool(query, full_memory_text)
                    answer = (answer or "").strip()
            except Exception:
                pass

        return {"answer": answer, "traces": traces, "observations": accumulated}

    # ── inspection ───────────────────────────────────────────────────────────

    def dump_all(self) -> List[Dict]:
        """Return all memories across collections for slot-fill."""
        out: List[Dict] = []
        for col in self._collections:
            try:
                data = self.db.get(col, include=["documents", "metadatas"])
            except Exception:
                continue
            ids = data.get("ids", [])
            docs = data.get("documents", [])
            metas = data.get("metadatas", []) or [None] * len(ids)
            # category = the suffix after the sample_id prefix.
            cat = col[len(self.sample_id) + 1:] if col.startswith(self.sample_id + "_") else col
            for _id, doc, meta in zip(ids, docs, metas):
                if not doc:
                    continue
                # Skip chromadb's auto-inserted dummy initialiser notes.
                if "dummy note for initializing" in doc.lower():
                    continue
                out.append({
                    "id": _id,
                    "content": doc,
                    "category": cat,
                    "metadata": meta,
                    "timestamp": (meta or {}).get("turn_time"),
                })
        return out


def _parse_tool_calls(llm_output: str) -> List[Dict]:
    """Mem-T-style <tool_call>JSON</tool_call> parser."""
    if not llm_output:
        return []
    if llm_output.startswith("<tool_call>") and llm_output.endswith("}"):
        llm_output = f"{llm_output}</tool_call>"
    out: List[Dict] = []
    for content in re.findall(r"<tool_call>(.*?)</tool_call>", llm_output, re.DOTALL):
        try:
            data = json.loads(content.strip())
        except Exception:
            continue
        if isinstance(data, list):
            for d in data:
                if isinstance(d, dict) and "name" in d and "arguments" in d:
                    out.append(d)
        elif isinstance(data, dict) and "name" in data and "arguments" in data:
            out.append(data)
    return out


# ── HMBench agent ────────────────────────────────────────────────────────────

class MemTAgent(MemoryAgent):
    """
    Wraps Mem-T as a HMBench MemoryAgent. Two variants via use_finish_answer:
      - True   ("memt"):         Mem-T-4B's finish answer is the reply.
      - False  ("memt_memonly"): retrieved memories go to the shared reply LLM.
    """

    def __init__(
        self,
        llm_model: str = GPT_MODEL,
        api_key: Optional[str] = None,
        use_finish_answer: bool = False,
    ):
        api_key = api_key or OPENAI_API_KEY
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        self.use_finish_answer = use_finish_answer
        self.llm_model = llm_model
        self._reply_client = OpenAI(api_key=api_key)
        self.memory = _MemTMemory()
        self._turn_counter = 0

    # ── agent_fn interface ────────────────────────────────────────────────────

    def __call__(self, task: str, history: List[Dict]) -> str:
        latest_user = history[-1]["user"] if history else task

        # 1. Retrieve via Mem-T's ReAct (returns answer + actual memories).
        res = self.memory.retrieve(latest_user)
        answer = (res.get("answer") or "").strip()
        retrieved_items: List[Dict] = res.get("observations") or []

        # 2. Decide reply.
        if self.use_finish_answer and answer:
            response = answer
        else:
            # Use the actual ReAct-retrieved memories as context for the
            # shared reply LLM.
            mem_lines = [m["content"] for m in retrieved_items if m.get("content")]
            if not mem_lines:
                # Last-resort fallback: dump-all bounded to avoid context blow-up.
                all_mem = self.memory.dump_all()
                mem_lines = [m["content"][:400] for m in all_mem[:20]]
            # Cap context size to ~20 retrieved items (Mem-T may surface
            # many across iterative steps).
            memory_context = "\n".join(f"• {ln}" for ln in mem_lines[:20] if ln)
            response = self._reply_with_context(task, memory_context)
            if not response:
                response = answer or "I'm sorry, I couldn't generate a response."

        # 3. Write the user's most recent utterance through Mem-T's pipeline.
        self._turn_counter += 1
        self.memory.add_turn(
            turn_text=f"user said: {latest_user}",
            turn_id=f"t{self._turn_counter:04d}",
        )
        return response

    # ── helpers ───────────────────────────────────────────────────────────────

    def _reply_with_context(self, task: str, memory_context: str) -> Optional[str]:
        memory_block = (
            _MEMORY_BLOCK.format(memories=memory_context) if memory_context else ""
        )
        system = _REPLY_SYSTEM_PROMPT.format(memory_block=memory_block)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        return self._call_reply_llm(messages)

    def _call_reply_llm(self, messages: List[Dict], temperature: float = 0.7,
                       max_retries: int = 3) -> Optional[str]:
        import time
        for attempt in range(max_retries):
            try:
                resp = self._reply_client.chat.completions.create(
                    model=self.llm_model, messages=messages, temperature=temperature,
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
                print(f"[MemTAgent] reply LLM attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
        return None

    # ── memory inspection ────────────────────────────────────────────────────

    def get_all_memories(self) -> List[Dict]:
        return self.memory.dump_all()

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Native top-k retrieval through Mem-T's ReAct loop.

        Faithful to the paper: the agent's iterative-reasoning retriever
        (Mem-T-4B) decides which tool to call each step (any of
        search_facts / search_experiences / search_turns / search_summary /
        search_personas) and observes the result before deciding the next
        step. We accumulate all memory items surfaced across steps; the
        LLM's final "finish" answer is discarded (we want the memories,
        not the LLM-authored answer). Returns top-k by retrieval score.
        """
        retr = self.memory.retriever
        sample_id = self.memory.sample_id
        max_steps = self.memory.cfg.memory.max_tool_steps

        # Build the same prompt the retriever uses, then run its loop body.
        messages = retr.construct_prompt(query, category="")
        seen_turns: set = set()
        accumulated: List[Dict] = []

        for step in range(max_steps):
            try:
                response_text = retr.llm.get_completion(
                    messages, stop=["</tool_call>"], json_mode=False,
                ) or ""
            except Exception:
                break
            if "<tool_call>" in response_text and "</tool_call>" not in response_text:
                response_text += "</tool_call>"

            try:
                tool_name, tool_args = retr._parse_xml_tool(response_text)
            except Exception:
                break
            if not tool_name:
                # Encourage the LLM to retry with a tool call.
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": "No tool call detected. Use the proper tool-call format."})
                continue
            tool_args["sample_id"] = sample_id

            # The finish tool would emit an answer; we don't want it. Just
            # exit the loop and rely on memories collected so far.
            if tool_name == "finish":
                break

            try:
                observation, mem_metadatas = retr.execute_tool(
                    tool_name, tool_args, seen_turns=seen_turns,
                )
            except Exception:
                continue

            for m in (mem_metadatas or []):
                if not m:
                    continue
                content = m.get("memory_content") or m.get("content") or ""
                if not content or "dummy note for initializing" in content.lower():
                    continue
                col = m.get("col_name", "")
                cat = col.rsplit("_", 1)[-1] if "_" in col else col or "memt"
                accumulated.append({
                    "id":       m.get("id"),
                    "content":  content,
                    "category": cat,
                    "score":    m.get("score") or m.get("distance"),
                    "timestamp": m.get("turn_time"),
                })

            # Update conversation so the LLM can reflect on this step.
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": (
                    f"The retrieved memories are: <observation>{observation}</observation>\n"
                    f"If you have enough information, call finish; otherwise refine your search."
                ),
            })

        # De-dup by id, keep best score, return top-k.
        by_id: Dict[str, Dict] = {}
        for c in accumulated:
            cid = c.get("id") or c["content"][:40]
            if cid not in by_id or (c.get("score") or 0) > (by_id[cid].get("score") or 0):
                by_id[cid] = c
        ranked = sorted(by_id.values(), key=lambda x: -(x.get("score") or 0))
        return ranked[:k]

    def save_memories(self, path: str) -> None:
        memories = self.get_all_memories()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"num_memories": len(memories), "memories": memories},
                      f, ensure_ascii=False, indent=2)
        print(f"[MemTAgent] Saved {len(memories)} memories → {path}")

    def save_native_state(self, target_dir: str) -> None:
        """Snapshot the per-instance Mem-T workspace + reattach manifest.

        ``_MemTMemory`` keeps everything under ``self.memory._tmp_root``: the
        chromadb persistent store under ``db/``, the trajectory log under
        ``traj/``, and the per-instance ``sample_id`` collection prefix.
        Copying ``_tmp_root`` plus a manifest with the cfg values needed to
        rebuild the live agent (model id, vLLM URL, retrieval / ReAct knobs,
        the use_finish_answer flag) lets a future tool reattach a chromadb
        client and reissue ``agent.search(...)`` with the EXACT same
        runtime config that produced this snapshot, even if defaults in
        the codebase later drift.
        """
        import shutil
        import time
        import json as _json
        os.makedirs(target_dir, exist_ok=True)
        src = getattr(self.memory, "_tmp_root", None)
        if src and os.path.isdir(src):
            dst = os.path.join(target_dir, "memt_workspace")
            try:
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, symlinks=False)
            except Exception as e:
                print(f"[MemTAgent] save_native_state copy failed: {e}")

        # Reattach manifest. Pull every value that affects scoring behaviour.
        try:
            cfg = getattr(self.memory, "cfg", None)
            mem_cfg = getattr(cfg, "memory", None) if cfg else None
            vdb_cfg = getattr(cfg, "vector_db", None) if cfg else None
            llm = getattr(self.memory, "llm", None)
            llm_client = getattr(llm, "client", None) if llm else None
            base_url = str(getattr(llm_client, "base_url", "")) if llm_client else ""
            manifest = {
                "agent_class": "MemTAgent",
                "use_finish_answer": bool(self.use_finish_answer),
                "sample_id": getattr(self.memory, "sample_id", ""),
                "session_id": getattr(self.memory, "_session_id", ""),
                "memt": {
                    "base_url":   base_url,
                    "model_id":   getattr(llm, "model", _MEMT_MODEL_ID),
                    "embedder":   "BAAI/bge-m3",
                },
                "cfg": {
                    "data_name":             getattr(cfg, "data_name", None),
                    "USE_LOCAL_LLM":         getattr(cfg, "USE_LOCAL_LLM", None),
                    "USE_PARALLEL":          getattr(cfg, "USE_PARALLEL", None),
                    "memory": {
                        "max_tool_steps":        getattr(mem_cfg, "max_tool_steps", None),
                        "update_retrieval_topk": getattr(mem_cfg, "update_retrieval_topk", None),
                    },
                    "vector_db": {
                        "backend": getattr(vdb_cfg, "backend", None),
                        "db_type": getattr(vdb_cfg, "db_type", None),
                    },
                },
                "saved_at_unix": time.time(),
            }
            with open(os.path.join(target_dir, "manifest.json"), "w") as f:
                _json.dump(manifest, f, indent=2, default=str)
        except Exception as e:
            print(f"[MemTAgent] save_native_state manifest failed: {e}")

    def reset(self) -> None:
        # Old _MemTMemory will be GC-ed; recreate fresh.
        try:
            del self.memory
        except Exception:
            pass
        self.memory = _MemTMemory()
        self._turn_counter = 0

    def add_external_memory(self, content: str, **kwargs) -> None:
        self._turn_counter += 1
        try:
            self.memory.add_turn(
                turn_text=f"user said: {content}",
                turn_id=f"t{self._turn_counter:04d}",
            )
        except Exception as e:
            print(f"[MemTAgent] add_external_memory failed: {e}")
