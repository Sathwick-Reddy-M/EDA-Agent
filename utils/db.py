import sqlite3
from pathlib import Path

# ---- For Development
from dotenv import load_dotenv
import os

load_dotenv()
# ----

db_path_env = os.getenv("DB_PATH")
coding_agent_dir_path_env = os.getenv("CODING_AGENT_DIR_PATH")

if not db_path_env:
    raise ValueError("DB_PATH environment variable is not set.")

DB_PATH = Path(db_path_env).absolute()
CODING_AGENT_DIR_PATH = Path(coding_agent_dir_path_env).absolute()


def get_conn():
    conn = sqlite3.connect(DB_PATH)  # autocommit
    # cur = conn.cursor()
    # cur.execute("PRAGMA journal_mode=WAL;")  # better r/w concurrency
    # cur.execute("PRAGMA synchronous=NORMAL;")  # durability balance
    # cur.execute("PRAGMA foreign_keys=ON;")
    # cur.execute("PRAGMA busy_timeout=5000;")  # wait up to 5s if locked
    return conn
