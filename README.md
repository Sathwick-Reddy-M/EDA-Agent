# EDA-Agent

An interactive Exploratory Data Analysis assistant that:

- Inspects your uploaded CSV datasets
- Plans how to answer a natural‑language query (coding / plotting / both / none)
- Delegates work to specialized agents (Coding Agent, Plotting Agent)
- Merges results into a concise final answer
- Persists state (LangGraph + SQLite checkpoint) and logs runs (MLflow)

## Environment Setup

1. Install `uv` (https://github.com/astral-sh/uv)
2. Sync dependencies
   ```
   uv sync
   ```
3. Create `.env`

   ```
   GOOGLE_API_KEY=<YOUR_GOOGLE_API_KEY>
   DB_PATH=db/agents.db
   CODING_AGENT_DIR_PATH=/absolute/path/EDA-Agent/coding_agent_space
   DATA_FOLDER_PATH=/absolute/path/EDA-Agent/coding_agent_space/data
   PLOTS_FOLDER_PATH=/absolute/path/EDA-Agent/coding_agent_space/plots
   ```

## Project Layout (Key Parts)

```
coding_agent_space/
  data/          # Place CSV files here before starting
  plots/         # Generated plot artifacts
  placeholder.py # place for Coding agent generated code
coding_agent/    # Coding agent graph + execution helpers
main_agent/      # Orchestrator (LangGraph state machine)
utils/           # LLM + DB helpers
```

## Running the Agent

In repo root:

```
uv run python main.py
```

You’ll see:

```
Chat Agent Started! Type 'q' or 'quit' to exit.
Enter your query:
```

Flow:

1. It scans `DATA_FOLDER_PATH` for CSV files.
2. For each file you’re asked for a short description (you can skip with Enter).
3. An internal coding sub‑agent may generate file summaries (LLM‑derived).
4. Your query is planned into:
   - coding task
   - plotting task
   - merge instructions (or direct answer if neither needed)
5. Results are combined and printed.

Quit with `q` or `quit`.
