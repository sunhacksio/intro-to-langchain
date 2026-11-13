from langchain_openai import ChatOpenAI
from os import getenv
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    openai_api_base="https://api.nova.amazon.com/v1",
    model="nova-2-pro-v1",
    api_key=getenv("API_TOKEN"),
    use_responses_api=False
)