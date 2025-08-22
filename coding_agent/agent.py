from contextlib import asynccontextmanager
from typing import Annotated, TypedDict, Optional, List

from utils.llm import get_llm, get_llm_with_structured_output

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from utils.db import DB_PATH, CODING_AGENT_DIR_PATH

from .data_models import CodeGenResponse, ErrorResolutionResponse
from .prompts import (
    CODE_GEN_SYSTEM_PROMPT,
    ERROR_RESOLUTION_SYSTEM_PROMPT,
    ERROR_RESOLUTION_PROMPT,
)
from .helpers import write_to_file, run_script

# ---------- helpers


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    code: Optional[str]
    error_text: Optional[str]
    output: Optional[str]
    execution_count: int


llm = get_llm("gemini-2.5-flash", "google_genai")


prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="system_message"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


async def generate_code(state: State, config: RunnableConfig) -> State:
    system_message = [HumanMessage(content=CODE_GEN_SYSTEM_PROMPT)]

    code_gen_response = await (
        prompt | get_llm_with_structured_output(llm, CodeGenResponse)
    ).ainvoke({"messages": state["messages"], "system_message": system_message})

    response = AIMessage(
        content=f"Generated code:\n```python\n{code_gen_response.code}\n```"
    )
    return {
        "messages": [response],
        "code": code_gen_response.code,
        "execution_count": 0,
    }


async def execute_code(state: State) -> State:
    code = state["code"]
    if not code:
        return {
            "error_text": "No code to execute.",
            "execution_count": state.get("execution_count", 0),
        }

    await write_to_file(CODING_AGENT_DIR_PATH / "placeholder.py", code)
    return_code, output, err = await run_script(
        CODING_AGENT_DIR_PATH / "placeholder.py", CODING_AGENT_DIR_PATH
    )

    execution_count = state.get("execution_count", 0) + 1

    if err or return_code != 0:
        error_msg = err or f"Script failed with return code {return_code}"
        return {"error_text": error_msg, "execution_count": execution_count}

    response = AIMessage(content=f"Execution successful. Output:\n{output}")
    return {
        "output": output,
        "error_text": None,
        "execution_count": execution_count,
        "messages": [response],
    }


async def resolve_error(state: State, config: RunnableConfig) -> State:
    """Resolve code errors by generating corrected code."""
    system_message = [HumanMessage(content=ERROR_RESOLUTION_SYSTEM_PROMPT)]

    error_context = ERROR_RESOLUTION_PROMPT.format(
        code=state["code"], error_context=state["error_text"]
    )

    messages = state["messages"] + [HumanMessage(content=error_context)]

    error_resolution_response = await (
        prompt | get_llm_with_structured_output(llm, ErrorResolutionResponse)
    ).ainvoke({"messages": messages, "system_message": system_message})

    response = AIMessage(
        content=f"Fixed code (attempt {state.get('execution_count', 0) + 1}):\n```python\n{error_resolution_response.code}\n```"
    )
    return {"messages": [response], "code": error_resolution_response.code}


def should_continue_after_execution(state: State) -> str:
    """Decide next step after code execution."""
    if state.get("error_text"):
        execution_count = state.get("execution_count", 0)
        if execution_count >= 3:  # Max 3 attempts
            return "end_with_failure"
        return "resolve_error"
    return END


def create_coding_agent(checkpointer: AsyncSqliteSaver):
    """
    Create and return the compiled LangGraph coding agent with execution and error resolution.
    """
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("generate_code", generate_code)
    graph.add_node("execute", execute_code)
    graph.add_node("resolve_error", resolve_error)
    graph.add_node(
        "end_with_failure",
        lambda state: {
            "messages": [AIMessage(content="Failed to execute code after 3 attempts.")]
        },
    )

    # Add edges
    graph.add_edge(START, "generate_code")

    graph.add_edge("generate_code", "execute")

    graph.add_conditional_edges(
        "execute",
        should_continue_after_execution,
        {
            "resolve_error": "resolve_error",
            "end_with_failure": "end_with_failure",
            END: END,
        },
    )

    graph.add_edge("resolve_error", "execute")

    graph.add_edge("end_with_failure", END)

    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def get_compiled_graph():
    """
    Async context manager that yields a compiled coding agent with an AsyncSqliteSaver.

    Usage:
        async with get_compiled_graph() as app:
            ...
    """
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as memory:
        app = create_coding_agent(memory)
        yield app
