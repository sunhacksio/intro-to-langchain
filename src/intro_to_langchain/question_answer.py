from langgraph.graph import END, START, MessagesState, StateGraph

from src.intro_to_langchain.llms import llm


def call_model(state: MessagesState):
    # Send the whole conversation to the chat model and append its reply.
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()
