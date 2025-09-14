from typing import TypedDict, Annotated, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .helpers import get_data_file_paths

from coding_agent.agent import get_compiled_graph as get_coding_agent_graph


import os
from dotenv import load_dotenv

load_dotenv()

from utils.llm import get_llm
from utils.db import DB_PATH


def merge_dicts(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    return {**left, **right}


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    is_data_folder_empty: bool
    uploaded_files: dict[str, str]
    filename: str  # transient, set by Send for each worker
    file_info_from_user: Annotated[dict[str, str], merge_dicts]
    file_info_from_llm: Annotated[dict[str, str], merge_dicts]


llm = get_llm("gemini-2.5-flash", "google_genai")

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="system_message"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


async def check_for_the_files(state: State) -> State:
    """
    Check if the data folder is empty and update the state accordingly.
    """
    data_folder_path = Path(os.getenv("DATA_FOLDER_PATH"))

    if not data_folder_path:
        raise ValueError("DATA_FOLDER_PATH environment variable is not set.")
    uploaded_files = await get_data_file_paths(data_folder_path, ignore_hidden=True)

    return {
        "is_data_folder_empty": (len(uploaded_files) == 0),
        "uploaded_files": uploaded_files,
    }


def continue_from_check_for_the_files(state: State):
    if state["is_data_folder_empty"]:
        return END
    return [
        Send(
            "get_file_info_from_user",
            {"filename": f, "uploaded_files": state["uploaded_files"]},
        )
        for f in state["uploaded_files"]
    ] + [
        Send(
            "get_file_info_from_llm",
            {"filename": f, "uploaded_files": state["uploaded_files"]},
        )
        for f in state["uploaded_files"]
    ]


def get_file_info_from_user(state: State) -> str:
    filename = state["filename"]
    file_info_from_user = input(
        f"Provide a brief description of the file '{filename}': "
    )

    return {"file_info_from_user": {filename: file_info_from_user}}


async def answer_with_coding_agent(task: str, config: RunnableConfig) -> str:
    async with get_coding_agent_graph() as app:
        final_state = await app.ainvoke(
            {"task": task},
            config=config,
        )

        if final_state.get("error_text"):
            return final_state["error_text"]

        return final_state["task_output"]


async def get_file_info_from_llm(state: State, config: RunnableConfig) -> str:
    filename = state["filename"]

    task = f"I want to get a brief description of the file named '{filename}' which is stored at {state['uploaded_files'][filename]}. The description provided should include the basic details about the file such as number of rows, columns, data types of columns, unique values in each column, null value proportions. Finally, your understanding of what the file represents."

    file_info_from_llm = await answer_with_coding_agent(task, config)
    return {"file_info_from_llm": {filename: file_info_from_llm}}


def create_eda_agent(checkpointer: AsyncSqliteSaver):
    """
    Create and return the compiled LangGraph agent.
    """
    graph = StateGraph(State)

    # First Require to upload atleast a single file or multiple files.

    graph.add_node("check_for_the_files", check_for_the_files)

    graph.add_edge(START, "check_for_the_files")

    graph.add_node("get_file_info_from_user", get_file_info_from_user)

    graph.add_conditional_edges(
        "check_for_the_files", continue_from_check_for_the_files
    )
    graph.add_edge("get_file_info_from_user", END)
    graph.add_node("get_file_info_from_llm", get_file_info_from_llm)
    graph.add_edge("get_file_info_from_llm", END)

    # graph.add_node(
    #     "get_file_info_from_llm", get_file_info_from_llm
    # )  # Agent that uses LLM and returns a dictionary of file descriptions.
    # graph.add_node(
    #     "query"
    # )  # Agent that translates the User Query to Python code and/or visualizations and returns the results

    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def get_compiled_graph():
    """
    Async context manager that yields a compiled graph with an AsyncSqliteSaver.

    Usage:
        async with get_compiled_graph() as app:
            ...
    """
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as memory:
        app = create_eda_agent(memory)
        yield app
