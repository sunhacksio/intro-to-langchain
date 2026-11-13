from langchain.agents import create_agent
from langchain_core.tools import BaseTool, tool
from langchain_core.stores import InMemoryStore
from src.intro_to_langchain.llms import llm


class MemoryTools:
    def __init__(self):
        self.store = InMemoryStore()

    def get(self) -> list[BaseTool]:
        """Return the tools, each bound to this instance's store."""
        store = self.store

        @tool
        def create_memory(memories: dict[str, str]) -> list[str]:
            """Save memories. `memories` maps a short descriptive key
            (for example "user_birthday") to the value to remember.
            Returns every key currently stored."""
            store.mset(list(memories.items()))
            return list(store.yield_keys())

        @tool
        def read_memory(keys: list[str]) -> list[str | None]:
            """Read the memories stored under `keys`. Returns one value per
            key, or null for a key that has no memory."""
            return store.mget(keys)

        @tool
        def delete_memory(keys: list[str]) -> list[str]:
            """Delete the memories stored under `keys`. Returns the keys
            that were deleted."""
            store.mdelete(keys)
            return keys

        return [create_memory, read_memory, delete_memory]


memory_agent = create_agent(
    model=llm,
    tools=MemoryTools().get(),
    system_prompt=(
        "You are a helpful assistant. Use the tools to remember important dates and information about users."
    ),
)

# create_agent returns a compiled graph, which langgraph dev can load directly.
graph = memory_agent
