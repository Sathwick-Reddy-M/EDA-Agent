from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class DatasetUnderstanding(BaseModel):
    """
    Model representing the dataset understanding process.
    """

    column_data_type_map: dict[str, str] = Field(
        ..., description="Mapping of column names to their data types."
    )
    column_info_map: dict[str, str] = Field(
        ...,
        description="Mapping of column names to their descriptions (of what the column represent) in a couple of sentences.",
    )
