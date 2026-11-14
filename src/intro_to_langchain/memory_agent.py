from langchain.agents import create_agent
from langchain_core.stores import InMemoryStore
from langchain_core.tools import BaseTool, tool

from src.intro_to_langchain.llms import llm


class MemoryTools:
    def __init__(self, store: InMemoryStore | None = None):
        self.store = store if store is not None else InMemoryStore()

    def create_memory(self, memories: dict[str, str]) -> list[str]:
        """Save memories mapped by short descriptive keys."""
        self.store.mset(list(memories.items()))
        return list(self.store.yield_keys())

    def read_memory(self, keys: list[str]) -> list[str | None]:
        """Read memories by keys, run list_memories first to see all memories."""
        return self.store.mget(keys)

    def list_memories(self) -> list[str]:
        """List all stored memories, run this first to see all memories."""
        return list(self.store.yield_keys())

    def delete_memory(self, keys: list[str]) -> list[str]:
        """Delete memories by keys."""
        self.store.mdelete(keys)
        return keys

    def get(self) -> list[BaseTool]:
        """Return the class-scoped tools, bound to this instance's store.

        `self.create_memory` is a bound method, so `self` is already
        bound and `tool(...)` only sees the real tool arguments.
        """
        return [
            tool(self.create_memory),
            tool(self.read_memory),
            tool(self.list_memories),
            tool(self.delete_memory),
        ]


memory_agent = create_agent(
    model=llm,
    tools=MemoryTools().get(),
    system_prompt=(
        "You are a helpful assistant. Use the tools to remember important dates and information about users."
    ),
)

# create_agent returns a compiled graph, which langgraph dev can load directly.
graph = memory_agent
