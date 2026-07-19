# Local LLM Delegation MCP

An optimized Model Context Protocol (MCP) server designed to dramatically reduce premium token costs by intelligently delegating low-complexity tasks to high-performance local LLMs via **LiteLLM**.

This server allows you to leverage massive local context windows for "surgical" development tasks (refactoring, unit tests, documentation) while keeping your primary LLM focused on complex architecture and reasoning.

## 🚀 Key Features

- **LiteLLM Integration:** Seamlessly connect to Ollama, vLLM, LM Studio, Anthropic, or any OpenAI-compatible provider.
- **Externalized Configuration:** Fine-tune model parameters (temperature, top_p, max_tokens) via `config.yaml`.
- **Customizable Prompts:** Define your own task templates and system messages in `prompts.yaml`.
- **Token Optimization:** Automatically offload simple tasks to local models like Qwen 2.5/3.5 Coder.
- **Usage Tracking:** Monitor your savings with built-in usage logging and statistics.

## 🛠️ Requirements

- **Local LLM Runner** (e.g., Ollama, vLLM, LM Studio)
- **Python 3.13+**

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mrrodriguez/local-llm-delegation-mcp.git
    cd local-llm-delegation-mcp
    ```

2.  **Set up environment variables:**
    ```bash
    cp .env.example .env
    # Edit .env to match your local setup
    ```

3.  **Choose your installation method:**

    **Option A: Global CLI (Recommended for simplicity)**
    Install as a global tool. This adds the `local-llm-delegator` command to your PATH.
    ```bash
    pip install .
    # Or using pipx (preferred)
    pipx install .
    ```

    **Option B: Local Environment (Recommended for development)**
    Use a virtual environment and point your MCP client directly to a startup script.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install .
    ```

## ⚙️ Configuration

The server uses a flexible configuration system. Settings are loaded from `config.yaml` and `prompts.yaml` located in the project root, with key overrides available via Environment Variables (or a `.env` file).

### Configuration Hierarchy
1.  **Base Defaults:** Hardcoded in `config.py`.
2.  **YAML Files:** `config.yaml` (model settings) and `prompts.yaml` (task templates).
3.  **Environment Variables:** Overrides specific settings like `LOCAL_MODEL_NAME` and `OPENAI_BASE_URL`.

### 1. Quick Start: Environment Variables
The fastest way to switch models without editing files:
```bash
# In your .env file or shell environment
LOCAL_MODEL_NAME=ollama/llama3:8b  # Overrides config.yaml
OPENAI_BASE_URL=http://localhost:11434/v1
```

### 2. Model Tuning (`config.yaml`)
Update the `name` to match your local provider's format (e.g., `ollama/qwen2.5-coder`). The provided template is optimized for high-end hardware (e.g., Apple M-series Max).

```yaml
model:
  name: "openai/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit" # Example: OMLX/MLX
  temperature: 0.1
  max_tokens: 32768
  extra_params:
    num_ctx: 65536
    num_predict: 32768
    top_p: 0.80
```
*Note: The `name` must follow the [LiteLLM Provider Format](https://docs.litellm.ai/docs/providers).*

### 3. Custom Tasks (`prompts.yaml`)
Define your own tools by adding entries to `prompts.yaml`. Each top-level key becomes a sub-task for the `query_local_llm_with_context` tool.

## 🔌 Integration

You can add this server to your preferred client using either the global command or a direct path to the project.

### 1. Gemini Code CLI

**If installed globally (Option A):**
```bash
gemini mcp add local-llm-delegator local-llm-delegator --trust
```

**If using a direct script or path (Option B):**
```bash
gemini mcp add local-llm-delegator /path/to/your/start-script.sh --trust
```

### 2. Claude Code
Add to your `~/.claude.json`:

```json
"mcpServers": {
  "local-llm-delegator": {
    "command": "local-llm-delegator" // If installed globally; otherwise any wrapper script you have to start it
  }
}
```
*Note: If using a custom startup script (like for OMLX), set the `command` to the absolute path of your script.*

## 🧑‍⚖️ Quorum Code Review

Three reviewer models (llama-swap instances `delegation-reviewer-1/2/3` by default;
override via `QUORUM_REVIEWERS`) review code in parallel and the verdicts are
reconciled into a consensus/majority/split report.

- `quorum_code_review(code, context, require_unanimous, timeout)` — single call.
  Emits MCP progress notifications every 10s so clients that reset their request
  timeout on progress survive cold model loads (2-3 min).
- `start_quorum_review(...)` → job id, returns immediately;
  `get_quorum_result(job_id)` — poll for the report. Use this pair when the MCP
  client enforces a hard per-request timeout (**Claude Desktop cancels tool calls
  at ~60s** — the server-side timeout cannot fix that). The review runs in the
  background on the server and survives client-side cancellation; results are
  kept for 1 hour.

## 📊 Usage Tracking

The server logs usage to `~/.local/share/local-llm-delegation-mcp/mcp_usage.jsonl`
(outside the install directory, so reinstalls/upgrades keep your stats). View stats via:
- `get_local_llm_usage_stats` (MCP tool)
- `show-stats` (CLI command - coming soon) or `python show_stats.py`

## 📜 License
MIT
