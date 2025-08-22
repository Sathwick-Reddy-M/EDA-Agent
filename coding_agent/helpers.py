import asyncio
from asyncio.subprocess import PIPE
from pathlib import Path
import sys


async def write_to_file(path: Path, content: str) -> None:
    """Write content to a file asynchronously."""
    await asyncio.to_thread(path.write_text, content)


async def run_script(entry: Path, cwd: Path) -> tuple[int, str, str]:
    """Run the generated python script and capture output (async)."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-I", entry.name, cwd=str(cwd), stdout=PIPE, stderr=PIPE
    )
    out_b, err_b = await proc.communicate()
    out = (out_b or b"").decode(errors="ignore").strip()
    err = (err_b or b"").decode(errors="ignore").strip()

    return proc.returncode, out, err
