from pathlib import Path
import mlflow
import sys

mlflow.set_experiment("EDA-Agent")
mlflow.autolog()

from dataset_understanding.agent import get_compiled_graph

import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from asyncio.subprocess import PIPE


async def run_chat():
    # Open the async checkpointer for the lifetime of the app
    async with get_compiled_graph() as app:
        thread_id = "test_thread"

        print("Chat Agent Started! Type 'q' or 'quit' to exit.")

        while True:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in {"q", "quit"}:
                break

            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "system_message": [SystemMessage("You are a helpful assistant.")],
                }
            }

            # Stream a single assistant turn; the graph will END after replying
            async for event in app.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            ):
                for value in event.values():
                    if "messages" in value:
                        print(f"Assistant: {value['messages'][-1].content}")


async def run_script(entry: Path, cwd: Path) -> tuple[int, str, str]:
    """Run the generated python script and capture output (async)."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-I", entry.name, cwd=str(cwd), stdout=PIPE, stderr=PIPE
    )
    out_b, err_b = await proc.communicate()
    out = (out_b or b"").decode(errors="ignore").strip()
    err = (err_b or b"").decode(errors="ignore").strip()
    print(out, err, proc.returncode)
    print(out_b, err_b, proc.returncode)

    return proc.returncode, out, err


asyncio.run(
    run_script(Path("./coding_agent_space/hello.py"), Path("./coding_agent_space"))
)
