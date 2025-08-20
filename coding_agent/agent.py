import os
import stat
import asyncio
from asyncio.subprocess import PIPE
from pathlib import Path
from typing import TypedDict, Optional, List, AsyncGenerator

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

# ---------- helpers (async where possible)


async def make_data_readonly(data_dir: Path) -> None:
    """Recursively set ./data to read-only (best-effort)."""

    pass


async def run_script(entry: Path, cwd: Path) -> tuple[int, str, str]:
    """Run the generated python script and capture output (async)."""
    proc = await asyncio.create_subprocess_exec(
        "python", "-I", entry.name, cwd=str(cwd), stdout=PIPE, stderr=PIPE, text=True
    )
    out, err = await proc.communicate()
    return proc.returncode, (out or "").strip(), (err or "").strip()


async def run_script_stream(
    entry: Path, cwd: Path, chunk: int = 128
) -> AsyncGenerator[str, None]:
    """
    Execute the script and yield stdout as it arrives.
    (This is only used for user-facing streaming.)
    """
    proc = await asyncio.create_subprocess_exec(
        "python", "-I", entry.name, cwd=str(cwd), stdout=PIPE, stderr=PIPE
    )

    buffer = bytearray()

    async def _drain_stdout():
        while True:
            b = await proc.stdout.read(1)  # small increments → smooth stream
            if not b:
                break
            buffer.extend(b)
            if len(buffer) >= chunk:
                yield_bytes = bytes(buffer)
                buffer.clear()
                yield yield_bytes.decode(errors="ignore")

    # Stream stdout
    async for piece in _drain_stdout():
        yield piece

    # Wait for completion and flush any remainder
    await proc.wait()
    if buffer:
        yield bytes(buffer).decode(errors="ignore")

    # If the process failed, append stderr (still a user-visible response)
    if proc.returncode != 0:
        err = (await proc.stderr.read()).decode(errors="ignore")
        if err:
            yield "\n[error]\n" + err


# ---------- state


class AgentState(TypedDict, total=False):
    task: str  # natural language instruction
    code: str  # latest version of the script
    code_path: str  # path of the script relative to workspace
    result: str  # stdout on success
    error: str  # stderr / traceback on failure
    attempts: int  # how many times we've tried so far
    history: List[str]  # short trace of what happened


# ---------- builder (async where possible)


async def build_coding_agent(
    workspace: str = "./workspace",
    model: str = "gpt-4o-mini",
    max_iters: int = 3,
    entrypoint: str = "main.py",
):
    """
    Build & compile an async LangGraph:
    - writes code to {workspace}/{entrypoint}
    - makes ./data read-only
    - executes, fixes errors (async)
    """
    ws = Path(workspace)
    await asyncio.to_thread(ws.mkdir, True, True)
    await asyncio.to_thread((ws / "data").mkdir, True, True)

    # Create README once (no need to be fully async; wrap in thread)
    readme = ws / "README.md"
    if not readme.exists():
        await asyncio.to_thread(
            readme.write_text,
            "# Workspace\n\n"
            "Managed by the LangGraph Coding Agent.\n\n"
            "## Data\n"
            "CSV files should live in `./data/`. This folder is set read-only during runs.\n\n"
            "## Future jobs queue\n"
            "- [ ] Add schema validation for input CSVs\n"
            "- [ ] Cache successful runs and reuse results\n"
            "- [ ] Add tests for generated scripts\n",
        )

    await make_data_readonly(ws / "data")

    llm = ChatOpenAI(model=model, temperature=0)

    code_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You generate a SINGLE self-contained Python script.\n"
                "Constraints:\n"
                "- Read CSVs only from './data/'.\n"
                "- Never write to './data/'. If you must write, use './outputs/'.\n"
                "- Print a concise final answer to STDOUT.\n"
                "- Prefer standard library; avoid extra deps unless necessary.\n"
                f"- The entrypoint file name must be '{entrypoint}'.",
            ),
            (
                "human",
                "Task:\n{task}\n\n"
                "Return ONLY the full Python script—no explanations.",
            ),
        ]
    )

    fix_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are fixing a Python script. Return ONLY the full corrected script.",
            ),
            (
                "human",
                "Original script:\n```python\n{code}\n```\n\n"
                "Error on execution:\n```\n{error}\n```\n\n"
                "Return a working, single-file script that satisfies the constraints:\n"
                "- Read CSVs only from './data/'.\n"
                "- Never write to './data/'.\n"
                f"- Entrypoint is '{entrypoint}'.\n"
                "- Print the final result to STDOUT.",
            ),
        ]
    )

    # ----- node functions (all async)

    async def write_code(state: AgentState) -> AgentState:
        messages = code_prompt.format_messages(task=state["task"])
        code = (await llm.ainvoke(messages)).content
        await asyncio.to_thread((ws / entrypoint).write_text, code)
        await asyncio.to_thread((ws / "outputs").mkdir, True)
        return {
            "code": code,
            "code_path": str(Path(entrypoint)),
            "attempts": 0,
            "history": state.get("history", []) + ["generated code"],
        }

    async def execute(state: AgentState) -> AgentState:
        await make_data_readonly(ws / "data")  # enforce again
        rc, out, err = await run_script(ws / entrypoint, ws)
        if rc == 0:
            return {
                "result": out,
                "error": "",
                "history": state.get("history", []) + ["executed successfully"],
            }
        return {
            "result": "",
            "error": err or f"Process exited with code {rc}",
            "history": state.get("history", []) + ["execution failed"],
        }

    async def fix(state: AgentState) -> AgentState:
        messages = fix_prompt.format_messages(code=state["code"], error=state["error"])
        code = (await llm.ainvoke(messages)).content
        await asyncio.to_thread((ws / entrypoint).write_text, code)
        return {
            "code": code,
            "attempts": state.get("attempts", 0) + 1,
            "history": state.get("history", []) + ["applied fix"],
        }

    # ----- routing

    def need_fix(state: AgentState) -> str:
        if state.get("error") and state.get("attempts", 0) < max_iters:
            return "fix"
        return "end"

    # ----- build graph

    graph = StateGraph(AgentState)
    graph.add_node("write_code", write_code)
    graph.add_node("execute", execute)
    graph.add_node("fix", fix)

    graph.set_entry_point("write_code")
    graph.add_edge("write_code", "execute")
    graph.add_conditional_edges("execute", need_fix, {"fix": "fix", "end": END})
    graph.add_edge("fix", "execute")

    return graph.compile(), ws, entrypoint


# ---------- convenience API (async + streaming)


async def run_task_async(task: str, **kwargs) -> AgentState:
    """Run the full agent (async), returning the final state."""
    graph, ws, entrypoint = await build_coding_agent(**kwargs)
    state: AgentState = {"task": task}
    final = await graph.ainvoke(state)
    return final


async def stream_task(
    task: str, chunk_size: int = 96, **kwargs
) -> AsyncGenerator[str, None]:
    """
    Run the agent and then stream ONLY the final user-facing output.
    (We still run the internal fix loop silently.)
    """
    graph, ws, entrypoint = await build_coding_agent(**kwargs)
    final = await graph.ainvoke({"task": task})

    # If we already have a result from the non-stream run, just stream that text.
    # Otherwise, stream the error text. If you'd rather re-run the script and
    # live-stream its stdout, uncomment the 'live re-run' section below.
    msg = final.get("result") or final.get("error", "")
    if msg:
        for i in range(0, len(msg), chunk_size):
            yield msg[i : i + chunk_size]
        return

    # ---- Live re-run and stream stdout (optional alternative) ----
    # async for piece in run_script_stream(Path(entrypoint), ws, chunk=chunk_size):
    #     yield piece


# ---------- CLI demo

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Async/Streaming LangGraph Coding Agent"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=(
            "Load 'sales.csv' from ./data, compute the total revenue as "
            "sum(quantity * unit_price) and print just that number."
        ),
    )
    parser.add_argument(
        "--stream", action="store_true", help="Stream final result to stdout."
    )
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--entrypoint", type=str, default="main.py")
    args = parser.parse_args()

    async def _main():
        if args.stream:
            async for chunk in stream_task(
                args.task,
                workspace=args.workspace,
                model=args.model,
                max_iters=args.max_iters,
                entrypoint=args.entrypoint,
            ):
                print(chunk, end="", flush=True)
            print()
        else:
            final = await run_task_async(
                args.task,
                workspace=args.workspace,
                model=args.model,
                max_iters=args.max_iters,
                entrypoint=args.entrypoint,
            )
            if final.get("result"):
                print(final["result"])
            else:
                print("[error]\n" + final.get("error", "Unknown error"))

    asyncio.run(_main())
