import os
from dotenv import load_dotenv

load_dotenv()


async def get_env_variable(var_name: str) -> str:
    # Environment lookups are fast; kept simple (no thread offload needed)
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} environment variable is not set.")
    return value


async def get_csv_file_path() -> str:
    return await get_env_variable("CSV_FILE_PATH")


async def get_plots_folder_path() -> str:
    return await get_env_variable("PLOTS_FOLDER_PATH")


async def get_data_folder_path() -> str:
    return await get_env_variable("DATA_FOLDER_PATH")
