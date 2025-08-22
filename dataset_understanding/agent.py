from typing import TypedDict, Annotated, List
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

# Async checkpointer (requires: pip install langgraph-checkpoint-sqlite)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from utils.llm import get_llm
from utils.db import DB_PATH


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


llm = get_llm("gemini-2.5-flash", "google_genai")

SYSTEM_PROMPT = "<SYSTEM PROMPT HERE>"

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="system_message"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


async def chat_node(state: State, config: RunnableConfig) -> State:
    """Process messages through LLM and return response."""
    system_message = config.get("configurable", {}).get("system_message", SYSTEM_PROMPT)

    response = await (prompt | llm).ainvoke(
        {"messages": state["messages"], "system_message": system_message}
    )
    return {"messages": [response]}


def should_continue(state: State) -> str:
    """
    Run node once per user turn and stop after assistant replies.

    - If the last message is from the human:
        * end if it's "q"/"quit"
        * otherwise, run "chat" node
    - If the last message is from the assistant (or anything else):
        * end and wait for next user input
    """
    last = state["messages"][-1] if state["messages"] else None

    if isinstance(last, HumanMessage):
        content = (last.content or "").strip().lower()
        if content in {"q", "quit"}:
            return END
        return "chat"

    # Assistant just replied (or no message) → stop and return to caller
    return END


def create_chat_agent(checkpointer: AsyncSqliteSaver):
    """
    Create and return the compiled LangGraph chat agent using the provided async checkpointer.
    The caller manages the checkpointer's async lifetime.
    """
    graph = StateGraph(State)

    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")

    # One assistant turn per user input
    graph.add_conditional_edges("chat", should_continue, {"chat": "chat", END: END})

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
        app = create_chat_agent(memory)
        yield app
