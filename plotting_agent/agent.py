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
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from coding_agent.agent import get_compiled_graph as get_coding_agent_graph
from .prompts import PLOT_AGENT_SYSTEM_PROMPT
from utils.db import DB_PATH
from .helpers import build_png_messages
from utils.paths import get_plots_folder_path


# ---------- env: plots folder
PLOTS_FOLDER_PATH = get_plots_folder_path()


# ---------- llm
llm = get_llm("gemini-2.5-flash", "google_genai")


# ---------- state
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    task: str
    task_output: str
    is_plot_gen_success: bool
    plot_gen_error: Optional[str]


async def answer_with_coding_agent(task: str, config: RunnableConfig) -> str:
    async with get_coding_agent_graph() as app:
        final_state = await app.ainvoke(
            {"task": task},
            config=config,
        )

        if final_state.get("error_text"):
            return final_state["error_text"], False

        return final_state["task_output"], True


async def generate_plots(state: State, config: RunnableConfig) -> State:
    new_config = dict(config or {})
    configurable = dict(new_config.get("configurable", {}))
    configurable["system_message"] = SystemMessage(PLOT_AGENT_SYSTEM_PROMPT)
    new_config["configurable"] = configurable

    user_msg = f"""
    Here is the task: {state["task"]}.

    Generate the Python code required to create the plots needed to answer the task. Only use matplotlib, seaborn, or plotly for plotting. The generated code should save each plot as a PNG file in the folder at {PLOTS_FOLDER_PATH}. Do not display or show the plots in the code. Only generate the code to create and save the plots.

    At max generate 2 plots to answer the task.
    """

    coding_agent_result, is_success = await answer_with_coding_agent(
        task=user_msg, config=config
    )

    return {
        "is_plot_gen_success": is_success,
        "plot_gen_error": None if is_success else coding_agent_result,
    }


# ---------- graph node
async def plot_understanding(state: State, config: RunnableConfig) -> State:
    """
    Build a single HumanMessage containing:
      - user_input_text
      - all PNG plots from PLOTS_FOLDER_PATH
    Send it to the LLM and return the AI's response.
    """

    system_message = SystemMessage(PLOT_AGENT_SYSTEM_PROMPT)

    if state["is_plot_gen_success"]:
        # Resolve user instruction text
        user_text = f"Analyze these plots and describe key insights required by the {state['task']}."

        # Compose a single multimodal HumanMessage: [text, *images]
        human = await build_png_messages(
            folder=PLOTS_FOLDER_PATH, prompt_text=user_text, recursive=False
        )

        # Invoke LLM with the combined message
        ai_msg: AIMessage = await llm.ainvoke([system_message, human])

    else:
        # If plot generation failed, inform the user
        error_text = state.get("plot_gen_error", "Unknown error in plot generation.")
        human = HumanMessage(
            content=f"Here is the task: {state['task']}.\n\n. However, I was unable to generate the required plots due to the following error: {error_text}. Explain what went wrong."
        )
        ai_msg = await llm.ainvoke([system_message, human])

    # Return the AI's response as the node's emitted message
    return {"messages": [system_message, human, ai_msg], "task_output": ai_msg.content}


# ---------- graph wiring
def create_plots_agent(checkpointer: AsyncSqliteSaver):
    """
    Create and return the compiled LangGraph
    """
    graph = StateGraph(State)

    graph.add_node("generate_plots", generate_plots)
    graph.add_node("plot_understanding", plot_understanding)

    graph.add_edge(START, "generate_plots")
    graph.add_edge("generate_plots", "plot_understanding")
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
