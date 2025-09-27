from langchain_core.tools import tool
from datetime import datetime, timezone

@tool
def date_time() -> datetime:
  """
  Returns the current datetime in python.
  Parameters: None
  """

  return datetime.now(timezone.utc)



tools = {"date_time": date_time}