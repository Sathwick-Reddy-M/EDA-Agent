from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    is_coding_task_required: bool = Field(
        description="Whether a coding task is required to answer the user's query."
    )
    coding_task: str = Field(
        description="If coding is required, detailed instructions for the coding agent; otherwise an empty string."
    )
    is_plot_task_required: bool = Field(
        description="Whether a plotting/visualization task is required to answer the user's query."
    )
    plot_task: str = Field(
        description="If plotting is required, detailed instructions for the plotting agent; otherwise an empty string."
    )
    instructions_for_merging_task_results: str = Field(
        description="How to combine the outputs (and/or a direct answer if neither agent is required)."
    )
