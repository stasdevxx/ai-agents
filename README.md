# LLM + tool-calling (Python)

This script demonstrates how to let an LLM call a custom tool (a weather lookup function), 
execute it in Python, and return the result back to the model so it can provide the final answer.

## Requirements
- Python 3.9+
- API key stored in the environment variable `OPENAI_API_KEY`
  - (optional: create a `.env` file with `OPENAI_API_KEY=...`)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python main.py "What's the weather in London?"
```
