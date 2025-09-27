from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from app.runnables import date_time_agent, embedding, q_and_a, date_time

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add

add_routes(app, RunnableLambda(q_and_a), path="/llm")
add_routes(app, RunnableLambda(date_time), path="/date_time")
add_routes(app, RunnableLambda(date_time_agent), path="/date_time_agent")
add_routes(app, RunnableLambda(embedding), path="/embedding")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
