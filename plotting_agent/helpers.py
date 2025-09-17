import asyncio
import base64
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.messages import HumanMessage


async def _read_bytes(path: Path) -> bytes:
    """Read file bytes asynchronously using a thread."""
    return await asyncio.to_thread(path.read_bytes)


def _b64_data_url_png(b: bytes) -> str:
    """Return a data URL for PNG bytes."""
    return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")


async def build_png_messages(
    folder: Union[str, Path],
    prompt_text: str,
    *,
    per_image: bool = False,
    recursive: bool = True,
    limit: Optional[int] = None,
    max_concurrency: int = 16,
) -> List[HumanMessage]:
    """
    Scan a folder for .png files and build HumanMessage objects you can pass to the LLM.

    Args:
        folder: Folder to search.
        prompt_text: The user instruction that tells the model what to do with the images.
        per_image: If False (default), returns a single HumanMessage that contains
                   the prompt_text and all images as content blocks. If True, returns
                   one HumanMessage per image (each includes prompt_text + that image).
        recursive: Search subfolders if True (default).
        limit: Optional max number of images to include (useful for huge dirs).
        max_concurrency: Max parallel file reads.

    Returns:
        List[HumanMessage]: messages ready to drop into `state["messages"]`.

    Raises:
        FileNotFoundError: If the folder doesn't exist or no .png files are found.
    """
    root = Path(folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")

    pattern = "**/*.png" if recursive else "*.png"
    files = sorted(root.glob(pattern), key=lambda p: p.as_posix())
    if limit is not None:
        files = files[: max(0, int(limit))]

    if not files:
        raise FileNotFoundError(f"No .png files found in: {root}")

    # Read images concurrently (bounded)
    sem = asyncio.Semaphore(max_concurrency)

    async def _load_block(p: Path) -> dict:
        async with sem:
            data = await _read_bytes(p)
        return {"type": "image_url", "image_url": {"url": _b64_data_url_png(data)}}

    if per_image:
        # One HumanMessage per image (prompt + single image block)
        blocks_list = await asyncio.gather(*(_load_block(p) for p in files))
        messages = [
            HumanMessage(content=[{"type": "text", "text": prompt_text}, block])
            for block in blocks_list
        ]
        return messages
    else:
        # Single HumanMessage with prompt + all image blocks
        blocks = await asyncio.gather(*(_load_block(p) for p in files))
        message = HumanMessage(content=[{"type": "text", "text": prompt_text}, *blocks])
        return message
