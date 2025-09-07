import os
import asyncio
import base64
from pathlib import Path
from typing import Annotated, TypedDict, Optional, List, Union

from contextlib import asynccontextmanager

from utils.llm import get_llm  # your existing factory

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from utils.db import DB_PATH
from .helpers import build_png_messages


# ---------- env: plots folder
plots_folder_path_env = os.getenv("PLOTS_FOLDER_PATH")
if not plots_folder_path_env:
    raise ValueError("PLOTS_FOLDER_PATH environment variable is not set.")
PLOTS_FOLDER_PATH = plots_folder_path_env  # keep this exact binding as requested


# ---------- llm
llm = get_llm("gemini-2.5-flash", "google_genai")


# ---------- state
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ---------- graph node
async def plot_understanding(state: State, config: RunnableConfig) -> State:
    """
    Build a single HumanMessage containing:
      - user_input_text
      - all PNG plots from PLOTS_FOLDER_PATH
    Send it to the LLM and return the AI's response.
    """
    system_message = (
        config.get("configurable", {}).get("system_message", None)
        or "You are a helpful data assistant that analyzes plots and provides insights."
    )

    # Resolve user instruction text
    user_text = state.get("messages")[-1]
    if not user_text:
        user_text = "Analyze these plots and describe key insights."
    else:
        user_text = user_text.content

    # Compose a single multimodal HumanMessage: [text, *images]
    human = await build_png_messages(
        folder=PLOTS_FOLDER_PATH, prompt_text=user_text, recursive=False
    )

    # Invoke LLM with the combined message
    ai_msg: AIMessage = await llm.ainvoke([system_message, human])

    # Return the AI's response as the node's emitted message
    return {"messages": [ai_msg]}


# ---------- graph wiring
def create_plots_agent(checkpointer: AsyncSqliteSaver):
    """
    Create and return the compiled LangGraph with a single PLOT_UNDERSTANDING step.
    Topology: START -> PLOT_UNDERSTANDING -> END
    """
    graph = StateGraph(State)

    # Node name kept lowercase in code; conceptually it's PLOT_UNDERSTANDING
    graph.add_node("plot_understanding", plot_understanding)

    graph.add_edge(START, "plot_understanding")
    graph.add_edge("plot_understanding", END)

    return graph.compile(checkpointer=checkpointer)


# ---------- context manager
@asynccontextmanager
async def get_compiled_graph():
    """
    Async context manager that yields the compiled plots-understanding agent.

    Usage:
        async with get_compiled_graph() as app:
            final_state = await app.ainvoke(
                {"user_input_text": "Summarize each plot and suggest actions."}
            )
    """
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as memory:
        app = create_plots_agent(memory)
        yield app
