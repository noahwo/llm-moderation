# LLM Moderation

> Note this project can now only run server on Turso nodes, and run client inside the network of the university. 


A content moderation system backed by LLMs, exposing a FastAPI server and a Python client with swappable strategies.

## Structure
The project is split by Client-Server architecture, client side only needs the files under `client/`.

**`server/`** — FastAPI application that loads and serves two moderation models:
- LlamaGuard-4 (`/lg4/moderate`) — supports text and images
- ToxicChat-T5 (`/t5/moderate`) — text only

**`client/`** — Python client with three moderation strategies sharing a common interface:
- **`demo.py` - current entry point for client to click&run simple demos.**
- Three language model strategies:
  - `LlamaGuard4Strategy` — calls the local LlamaGuard-4 server
  - `ToxicChatT5Strategy` — calls the local ToxicChat-T5 server
  - `OpenAIModerationStrategy` — calls the OpenAI Moderation API
  - Much duplicated code between `backends.py` and `moderation.py`. 

**`datasets/`** — Evaluation datasets (UnsafeBench train/test splits). Not used yet. 

**`scripts/`** — Shell scripts for launching the server. **Current entry point for server to click&host the models. 

**`logs/`** — Server log output.

**`build_with_llama4.py`** — Exploratory notebook for the Llama API.

**`playground.ipynb`** — Exploratory notebook for testing moderation strategies.

## Quickstart

```bash
# On both sides
pip install -r requirements.txt
# On server side
bash scripts/serve.sh        # start the model server
# On client side
python -m client.demo        # run the demo client
```