from pathlib import Path
import mlflow
import sys

mlflow.set_experiment("EDA-Agent")
mlflow.autolog()

# from dataset_understanding.agent import get_compiled_graph
# from dataset_understanding.agent import (
#     get_compiled_graph,
#     print_graph_ascii,
#     save_graph_png,
# )

from main_agent.agent import get_compiled_graph

import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from asyncio.subprocess import PIPE
import uuid


async def run_chat():
    # Open the async checkpointer for the lifetime of the app
    async with get_compiled_graph() as app:
        thread_id = uuid.uuid4()

        print("Chat Agent Started! Type 'q' or 'quit' to exit.")

        config = {
            "configurable": {
                "thread_id": thread_id,
                "system_message": [SystemMessage("You are a helpful assistant.")],
            }
        }

        result = await app.ainvoke({}, config=config)  # single run, no loop
        print(result)


asyncio.run(run_chat())
