import mlflow


def get_prompt(name):
    """
    Retrieve a prompt by name from the MLflow model registry.

    Args:
        name (str): The name of the prompt to retrieve.

    Returns:
        mlflow prompt object
    """
    # Fetch the prompt from MLflow
    prompt = mlflow.genai.load_prompt(name)

    if not prompt:
        raise ValueError(f"Prompt '{name}' not found in MLflow model registry.")

    return prompt
