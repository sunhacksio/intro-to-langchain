from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from os import getenv

llm = ChatOpenAI(
    openai_api_base="https://models.github.ai/inference",
    model="openai/gpt-4.1-mini",
    api_key=getenv("GITHUB_TOKEN")
)

embedding_model = OpenAIEmbeddings(
    openai_api_base="https://models.github.ai/inference",
    model="openai/text-embedding-3-small",
    api_key=getenv("GITHUB_TOKEN")
)


