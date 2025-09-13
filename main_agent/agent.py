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

from .helpers import is_folder_empty, get_data_filenames

from utils.paths import get_data_folder_path


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
    uploaded_files: List[str]
    filename: str  # transient, set by Send for each worker
    file_info: Annotated[dict[str, str], merge_dicts]


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
    uploaded_files = await get_data_filenames(data_folder_path, ignore_hidden=True)

    return {
        "is_data_folder_empty": (len(uploaded_files) == 0),
        "uploaded_files": uploaded_files,
    }


def continue_from_check_for_the_files(state: State):
    if state["is_data_folder_empty"]:
        return END
    return [Send("get_file_info", {"filename": f}) for f in state["uploaded_files"]]


def get_file_info(state: State) -> str:
    filename = state["filename"]
    file_info = input(f"Provide a brief description of the file '{filename}': ")

    return {"file_info": {filename: file_info}}


def create_eda_agent(checkpointer: AsyncSqliteSaver):
    """
    Create and return the compiled LangGraph agent.
    """
    graph = StateGraph(State)

    # First Require to upload atleast a single file or multiple files.

    graph.add_node("check_for_the_files", check_for_the_files)

    graph.add_edge(START, "check_for_the_files")

    graph.add_node("get_file_info", get_file_info)

    graph.add_conditional_edges(
        "check_for_the_files", continue_from_check_for_the_files
    )
    graph.add_edge("get_file_info", END)

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
