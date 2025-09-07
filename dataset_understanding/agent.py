from typing import TypedDict, Annotated, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import os
from dotenv import load_dotenv

load_dotenv()

from utils.llm import get_llm
from utils.db import DB_PATH
from coding_agent.agent import get_compiled_graph as get_coding_agent_graph
from plot_understanding_agent.agent import (
    get_compiled_graph as get_plot_understanding_agent_graph,
)

csv_file_path_env = os.getenv("CSV_FILE_PATH")
if not csv_file_path_env:
    raise ValueError("CSV_FILE_PATH environment variable is not set.")
CSV_FILE_PATH = csv_file_path_env

plots_folder_path_env = os.getenv("PLOTS_FOLDER_PATH")
if not plots_folder_path_env:
    raise ValueError("PLOTS_FOLDER_PATH environment variable is not set.")
PLOTS_FOLDER_PATH = plots_folder_path_env


from .prompts import (
    SYSTEM_PROMPT,
    DATA_DICTIONARY_INITIAL_PROMPT,
)


def add_list(prev: List[str], new: List[str]) -> List[str]:
    prev = prev or []
    if new is None:
        return prev
    # Allow either a single string or a list of strings
    if isinstance(new, list):
        return prev + new
    return prev + [new]


async def run_null_value_analysis(
    thread_id: str = uuid.uuid4().hex,
    data_dictionary_info: Optional[str] = None,
) -> tuple[str, bool]:
    """
    Runs a comprehensive NULL value analysis for the CSV file.
    Returns: (analysis_result, success_flag)
    """
    system_message = "You are an coding agent that helps to understand the NULL values if present in the dataset."

    user_input = f"""

    {f"Context from data dictionary: {data_dictionary_info}" if data_dictionary_info else ""}

    Read the CSV file at '{Path(CSV_FILE_PATH).absolute()}' using pandas and perform the below NULL value analysis:

    Load the dataset and get the columns info:
        - use the .info() method to get a summary of the DataFrame including non-null counts and data types.
    
    check for NULL values:
       - df.isnull().sum() (count of NULL values per column)
       - (df.isnull().sum() / len(df)) * 100 (percentage of NULL values per column)

    Get the unique values for each column:
        - Use the unique() method on the columns

    Make sure you print those above outputs in a clear manner. 
    
    Handle any potential errors with try-except blocks and provide clear explanations.
    """

    config = {
        "configurable": {
            "thread_id": thread_id,
            "system_message": SystemMessage(content=system_message),
        }
    }

    async with get_coding_agent_graph() as app:

        final_state = await app.ainvoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        execution_successful = final_state.get("error_text") is None
        analysis_result = final_state.get("output")

        return (
            analysis_result,
            execution_successful,
        )


async def run_plot_generation(
    user_input: str,
    data_dict_info: Optional[str] = "",
    thread_id: str = uuid.uuid4().hex,
) -> tuple[str, bool]:
    """
    Runs a plot generation analysis based on the NULL value analysis.
    Returns: (plot_info, success_flag)
    """
    system_message = "You are an coding agent that helps to generate plots."

    plotgen_user_prompt = f"""
    {user_input}

    Generate the plots that are required to answer the above user query and save them to '{Path(PLOTS_FOLDER_PATH).absolute()}'.

    {f"Context from data dictionary: {data_dict_info}" if data_dict_info else ""}

    Here is the CSV file path: '{Path(CSV_FILE_PATH).absolute()}' which you can read using pandas to generate the plots.

    You can use either the matplotlib, seaborn and plolty for generating the plots. Make sure to put all of the necessary plots required in two .png files.
    
    Handle any potential errors with try-except blocks and provide clear explanations.
    """

    config = {
        "configurable": {
            "thread_id": thread_id,
            "system_message": SystemMessage(content=system_message),
        }
    }

    async with get_coding_agent_graph() as app:

        final_state = await app.ainvoke(
            {"messages": [HumanMessage(content=plotgen_user_prompt)]}, config=config
        )
        execution_successful = final_state.get("error_text", None) is None
        plot_info = final_state.get("output")

    if not execution_successful:
        return "There was an error during plot generation. Please check the coding agent for details."

    async with get_plot_understanding_agent_graph() as app:

        final_state = await app.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Analyze these plots and describe key insights regarding each plot and answer this user query {user_input}."
                    )
                ]
            },
            config=config,
        )

        plot_info = final_state["messages"][-1].content
        return plot_info


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    data_dictionary_info: Annotated[List[str], add_list]
    collect_data_dictionary: bool
    null_analysis_info: Optional[str]
    plot_info: Optional[str]


llm = get_llm("gemini-2.5-flash", "google_genai")

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="system_message"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# This part should be done before calling the graph start
async def ask_for_data_dictionary_node(state: State, config: RunnableConfig) -> State:
    """Initial node to ask user about data dictionary availability."""

    print(DATA_DICTIONARY_INITIAL_PROMPT)
    user_response = input("Your response (YES/NO): ").strip().upper()

    if user_response == "YES":
        return {
            "messages": [
                AIMessage(content=DATA_DICTIONARY_INITIAL_PROMPT),
                HumanMessage(content="YES"),
            ],
            "collect_data_dictionary": True,
        }

    return {
        "messages": [
            AIMessage(content=DATA_DICTIONARY_INITIAL_PROMPT),
            HumanMessage(content="NO"),
        ],
        "collect_data_dictionary": False,
    }


async def collect_data_dictionary_node(state: State, config: RunnableConfig) -> State:
    """Collect optional data dictionary information from the user."""

    user_input = input("Please provide the data dictionary information you have:\n")

    messages = state["messages"]
    messages.append(
        AIMessage(content="Please provide the data dictionary information you have:")
    )
    messages.append(HumanMessage(content=user_input))

    data_dictionary_info = state.get("data_dictionary_info", [])
    data_dictionary_info.append(user_input)

    user_input = (
        input("Do you have additional data dictionary information to share?")
        .strip()
        .upper()
    )

    messages.append(
        AIMessage(
            content="Do you have additional data dictionary information to share?"
        )
    )
    messages.append(HumanMessage(content=user_input))

    while user_input != "NO":
        additional_info = input(
            "Please provide the additional data dictionary information:\n"
        )

        messages.append(
            AIMessage(
                content="Please provide the additional data dictionary information:"
            )
        )
        messages.append(HumanMessage(content=additional_info))

        data_dictionary_info.append(additional_info)

        user_input = (
            input("Do you have additional data dictionary information to share?")
            .strip()
            .upper()
        )

        messages.append(
            AIMessage(
                content="Do you have additional data dictionary information to share?"
            )
        )
        messages.append(HumanMessage(content=user_input))

    return {
        "messages": messages,
        "data_dictionary_info": data_dictionary_info,
        "collect_data_dictionary": False,
    }


async def null_value_analyzer_node(state: State, config: RunnableConfig) -> State:
    """Analyze NULL values in the dataset with reasoning based on any provided data dictionary."""
    _ = config  # Unused parameter

    # Get data dictionary information if available
    data_dict_info = "\n".join(state.get("data_dictionary_info", []))

    # Run the NULL value analysis
    null_analysis_result, execution_successful = await run_null_value_analysis(
        thread_id=uuid.uuid4(),
        data_dictionary_info=data_dict_info if data_dict_info else None,
    )

    if not execution_successful:
        explanation_response = AIMessage(
            content="There was an error during the NULL value analysis. Please check the coding agent for details."
        )
    else:
        explanation_response = llm.invoke(
            [
                SystemMessage(
                    content="You are a helpful assistant that explains the results of a NULL value analysis clearly and concisely."
                ),
                HumanMessage(
                    content=f"Here are the results of the NULL value analysis:\n\n{null_analysis_result}\n\n.{"Here is the data dictionary info " + data_dict_info if data_dict_info else ""} Please provide a clear explanation of these results."
                ),
            ]
        )

    return {
        "messages": [explanation_response],
        "null_analysis_info": null_analysis_result if execution_successful else None,
    }


async def plot_generation_node(state: State, config: RunnableConfig) -> State:

    user_input_prompt = f"Can you tell me what kind of distribution for each of the numerical columns and the dominant categories for the categorical column? You can plot histograms with different bins or whatever plots that might make sense"

    plot_generation_response = await run_plot_generation(
        user_input_prompt, "\n\n".join(state.get("data_dictionary_info", []))
    )

    return {
        "messages": [AIMessage(plot_generation_response)],
        "plot_info": plot_generation_response,
    }


async def null_value_analysis_node(state: State, config: RunnableConfig) -> State:
    """
    Delegate immediately to the NULL value analyzer and return its results.
    This stage no longer performs a separate DataFrame-info step.
    """
    return await null_value_analyzer_node(state, config)


def has_data_dictionary(state: State) -> str:
    """Check if user has a data dictionary."""
    if state["collect_data_dictionary"]:
        return "collect_data_dictionary"
    return "null_value_analysis"


def create_dataset_understanding_agent(checkpointer: AsyncSqliteSaver):
    """
    Create and return the compiled LangGraph dataset understanding agent.
    """
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("ask_for_data_dictionary", ask_for_data_dictionary_node)
    graph.add_node("collect_data_dictionary", collect_data_dictionary_node)
    graph.add_node("null_value_analysis", null_value_analysis_node)
    graph.add_node("plot_generation", plot_generation_node)

    # Add edges
    graph.add_edge(START, "ask_for_data_dictionary")

    graph.add_conditional_edges(
        "ask_for_data_dictionary",
        has_data_dictionary,
        {
            "collect_data_dictionary": "collect_data_dictionary",
            "null_value_analysis": "null_value_analysis",
        },
    )

    graph.add_edge("collect_data_dictionary", "null_value_analysis")
    graph.add_edge("null_value_analysis", "plot_generation")
    graph.add_edge("plot_generation", END)

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
        app = create_dataset_understanding_agent(memory)
        yield app
