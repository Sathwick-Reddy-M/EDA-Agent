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

csv_file_path_env = os.getenv("CSV_FILE_PATH")
if not csv_file_path_env:
    raise ValueError("CSV_FILE_PATH environment variable is not set.")
CSV_FILE_PATH = csv_file_path_env

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


async def csv_info_tool(
    thread_id: str = uuid.uuid4().hex,
    system_prompt: str = "You are a data analysis assistant. Read the CSV file and provide DataFrame info.",
) -> tuple[str, bool]:
    """
    Uses the coding agent to read a CSV file and return DataFrame info.
    Similar to chat_tool but specifically for CSV analysis.
    Uses CSV_FILE_PATH from environment variables.

    Returns:
        tuple[str, bool]: (analysis_result, success_flag)
    """
    user_input = f"""
    Please read the CSV file at '{Path(CSV_FILE_PATH).absolute()}' using pandas and print the following information:
    1. df.info()
    2. df.shape
    3. df.columns.tolist()
    4. df.head()
    5. df.describe()
    
    Make sure to import pandas as pd and handle any potential errors with try-except blocks.
    If there are any errors, clearly explain what went wrong and provide suggestions for fixing the issue.
    """

    async with get_coding_agent_graph() as app:
        config = {
            "configurable": {
                "thread_id": thread_id,
                "system_message": [SystemMessage(system_prompt)],
            }
        }
        analysis_result = ""
        execution_successful = True

        async for event in app.astream(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        ):
            for value in event.values():
                if "messages" in value:
                    message_content = value["messages"][-1].content
                    analysis_result = message_content

                    # Check if the execution failed based on common error indicators
                    if any(
                        error_phrase in message_content.lower()
                        for error_phrase in ["error", "exception", "traceback"]
                    ):
                        execution_successful = False

        return (
            analysis_result or "No analysis result received from coding agent.",
            execution_successful,
        )


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    data_dictionary_info: Annotated[List[str], add_list]
    collect_data_dictionary: bool
    dataframe_info: Optional[str]


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
    """Ask user about data dictionary availability."""

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


async def code_generation_node(state: State, config: RunnableConfig) -> State:
    """Code generation node that uses coding agent to analyze CSV file and incorporates data dictionary."""
    _ = config  # Unused parameter

    # Get data dictionary information if available
    data_dict_info = "\n".join(state.get("data_dictionary_info", []))

    # Use the coding agent to analyze the CSV file from environment
    dataframe_analysis_result, execution_successful = await csv_info_tool(
        thread_id=uuid.uuid4(),
    )

    # Use AI to explain the results or errors
    system_message = [
        "You are an assistant the helps me understand my dataset information."
    ]

    if execution_successful:
        explanation_prompt = f"""
        The coding agent successfully analyzed the CSV file at {CSV_FILE_PATH}. 
        Here's the output:
        
        {dataframe_analysis_result}
        
        {f"Data dictionary information: {data_dict_info}" if data_dict_info else ""}
        
        Please explain what this dataset information tells us about the data structure, quality, and characteristics.
        Keep it concise and practical.
        """
    else:
        explanation_prompt = f"""
        The coding agent encountered an error while analyzing the CSV file at {CSV_FILE_PATH}.
        Here's the error output:
        
        {dataframe_analysis_result}
        
        Please explain what went wrong and suggest how to fix it. Keep it simple and actionable.
        """

    # Get AI explanation
    explanation_response = await (prompt | llm).ainvoke(
        {
            "messages": [HumanMessage(content=explanation_prompt)],
            "system_message": system_message,
        }
    )

    return {
        "messages": [AIMessage(content=explanation_response.content)],
        "dataframe_info": dataframe_analysis_result if execution_successful else None,
    }


def has_data_dictionary(state: State) -> str:
    """Check if user has a data dictionary."""
    if state["collect_data_dictionary"]:
        return "collect_data_dictionary"
    return "code_generation"


def should_continue_from_code_gen(state: State) -> str:
    """Determine next step from code generation node."""
    # Check if dataframe_info was successfully generated
    if state.get("dataframe_info"):
        # Analysis complete - end the flow
        return END
    else:
        # If analysis failed, we could add error handling here
        return END


def create_dataset_understanding_agent(checkpointer: AsyncSqliteSaver):
    """
    Create and return the compiled LangGraph dataset understanding agent.
    """
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("ask_for_data_dictionary", ask_for_data_dictionary_node)
    graph.add_node("collect_data_dictionary", collect_data_dictionary_node)
    graph.add_node("code_generation", code_generation_node)

    # Add edges
    graph.add_edge(START, "ask_for_data_dictionary")

    graph.add_conditional_edges(
        "ask_for_data_dictionary",
        has_data_dictionary,
        {
            "collect_data_dictionary": "collect_data_dictionary",
            "code_generation": "code_generation",
        },
    )

    graph.add_edge("collect_data_dictionary", "code_generation")

    graph.add_conditional_edges("code_generation", should_continue_from_code_gen)

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
