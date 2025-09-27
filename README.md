# intro-to-langchain

## Installation

Install the `uv` Python package and environment manager:
MacOS:
```bash
brew install uv
```
Everyone Else:
```bash
pip install uv
```

Use `uv` to install required packages:
```bash
uv sync
```

Feel free to use your package manager of choice (pipenv, poetry) as well! Necessary dependencies are in `pyproject.toml`.

## Launch API Server
```bash
langchain serve
```