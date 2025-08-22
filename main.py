from pathlib import Path
import mlflow
import sys

mlflow.set_experiment("EDA-Agent")
mlflow.autolog()

# from dataset_understanding.agent import get_compiled_graph
from coding_agent.agent import get_compiled_graph

import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from asyncio.subprocess import PIPE


async def run_chat():
    # Open the async checkpointer for the lifetime of the app
    async with get_compiled_graph() as app:
        thread_id = "test_thread_123"

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


asyncio.run(run_chat())
