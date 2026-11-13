from datetime import timezone
from datetime import datetime
from langchain_core.tools import tool
from enum import Enum
from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.graph import StateGraph, MessagesState, START, END

from src.intro_to_langchain.llms import llm

@tool
def date_time() -> datetime:
    """
    Returns the current datetime in python.
    Parameters: None
    """

    return datetime.now(timezone.utc)




class DateTimeChoice(str, Enum):
    DATE = "date"
    TIME = "time"
    MONTH = "month"
    DATE_TIME = "date and time"
    DATE_TIME_ISO = "date and time, in ISO format?"


# Build the agent once, at import time.
date_time_agent = create_agent(
    model=llm,
    tools=[date_time],
    system_prompt=(
        "You are a helpful assistant. Using the date_time tool, answer the "
        "question below, providing the answer in the format requested."
    ),
)


class DateTimeInput(TypedDict):
    date_time_choice: DateTimeChoice


class DateTimeState(MessagesState):
    date_time_choice: DateTimeChoice


def build_question(state: DateTimeState):
    # Input arrives as JSON, so convert the plain string back to the enum.
    choice = DateTimeChoice(state["date_time_choice"])
    return {"messages": [HumanMessage(f"What is the current {choice.value}?")]}


builder = StateGraph(DateTimeState, input_schema=DateTimeInput)
builder.add_node("build_question", build_question)
builder.add_node("agent", date_time_agent)  # the compiled agent runs as a subgraph
builder.add_edge(START, "build_question")
builder.add_edge("build_question", "agent")
builder.add_edge("agent", END)

graph = builder.compile()