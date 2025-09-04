SYSTEM_PROMPT = """
You are a helpful assistant that helps users understand their datasets by utilizing the information in data dictionarys and insights from their data.
Be friendly and clear in your communication.
"""

DATA_DICTIONARY_INITIAL_PROMPT = """
Hi! I'm here to understand your dataset better.

Do you have a data dictionary available for your dataset? A data dictionary typically contains information about:
- Column names and their meanings
- Data types for each column
- Value ranges or possible values
- Descriptions of what each field represents

Please respond with either "YES" or "NO".
"""

DATA_DICTIONARY_INPUT_PROMPT = """
Great! Please provide the data dictionary information you have. You can include:
- Column descriptions
- Data types
- Value ranges
- Any other relevant information about your dataset

Please share this information below:
"""

DATA_DICTIONARY_MORE_INFO_PROMPT = """
Thank you for the information! 

Do you have any additional data dictionary information to add, or is this all the information you have available?

Please respond with "YES" if you have more information to add, or "NO" if this is complete.
"""
