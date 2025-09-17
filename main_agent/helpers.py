import asyncio
from pathlib import Path
from typing import List, Optional, Sequence, Union

from langchain_core.runnables import RunnableConfig

from coding_agent.agent import get_compiled_graph as get_coding_agent_graph
from plotting_agent.agent import get_compiled_graph as get_plotting_agent_graph


async def is_folder_empty(path: Union[str, Path], ignore_hidden: bool = True) -> bool:
    """
    Return True if the folder has no entries (optionally ignoring hidden files), else False.
    Raises FileNotFoundError if the path doesn't exist; NotADirectoryError if not a directory.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {p}")

    entries = await asyncio.to_thread(lambda: list(p.iterdir()))
    for entry in entries:
        if ignore_hidden and entry.name.startswith("."):
            continue
        return False
    return True


async def get_data_filenames(
    path: Union[str, Path], ignore_hidden: bool = True
) -> List[str]:
    data_exts = {
        ".csv",
        ".tsv",
        ".txt",
        ".json",
        ".xls",
        ".xlsx",
        ".xlsm",
        ".xlsb",
        ".ods",
        ".parquet",
    }
    base = Path(path)
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")

    def scan() -> List[str]:
        names: List[str] = []
        for p in base.iterdir():
            if not p.is_file():
                continue
            if ignore_hidden and p.name.startswith("."):
                continue
            if p.suffix.lower() in data_exts:
                names.append(p.name)
        return names

    return await asyncio.to_thread(scan)


async def get_data_file_paths(
    path: Union[str, Path], ignore_hidden: bool = True
) -> dict[str, str]:
    data_exts = {
        ".csv",
        ".tsv",
        ".txt",
        ".json",
        ".xls",
        ".xlsx",
        ".xlsm",
        ".xlsb",
        ".ods",
        ".parquet",
    }
    base = Path(path)
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")

    def scan() -> dict[str, str]:
        out: dict[str, str] = {}
        for p in base.iterdir():
            if not p.is_file():
                continue
            if ignore_hidden and p.name.startswith("."):
                continue
            if p.suffix.lower() in data_exts:
                out[p.name] = str(p.resolve())
        return out

    return await asyncio.to_thread(scan)


async def answer_with_coding_agent(task: str, config: RunnableConfig) -> str:
    async with get_coding_agent_graph() as app:
        final_state = await app.ainvoke(
            {"task": task},
            config=config,
        )

        if final_state.get("error_text"):
            return final_state["error_text"], False

        return final_state["task_output"], True


async def answer_with_plotting_agent(task: str, config: RunnableConfig) -> str:
    async with get_plotting_agent_graph() as app:
        final_state = await app.ainvoke(
            {"task": task},
            config=config,
        )

        if final_state.get("plot_gen_error"):
            return final_state["task_output"], False

        return final_state["task_output"], True
