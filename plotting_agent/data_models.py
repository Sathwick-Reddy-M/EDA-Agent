from pydantic import BaseModel, Field


class CodeGenResponse(BaseModel):
    code: str = Field(..., description="The generated Python code as a string.")


class ErrorResolutionResponse(BaseModel):
    code: str = Field(..., description="The corrected Python code.")
