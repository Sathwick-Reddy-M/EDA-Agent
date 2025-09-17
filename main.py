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
            }
        }

        # result = await app.ainvoke(
        #     {
        #         "task": "Read the csv file at /Users/sathwick/SDrive/projects/EDA-Agent/coding_agent_space/data/sample.csv and give me the column names and their data types.",
        #     },
        #     config=config,
        # )

        result = await app.ainvoke(
            {
                "query": "Give me the survival rates of the passengers based on their gender."
            },
            config=config,
        )

        print(result)


asyncio.run(run_chat())
