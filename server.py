#!/usr/bin/env python3
"""
FastMCP Server for Local LLM Integration
Exposes a local LLM as an MCP tool for Claude Code to delegate tasks.
Refactored for LiteLLM and externalized configuration.
"""

import os
import time
import json
import signal
import threading
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastmcp import FastMCP
import litellm
from config import settings

# Initialize MCP server
mcp = FastMCP("Local LLM Server")

def _watchdog():
    """Simple watchdog that exits if the parent process disappears."""
    parent_pid = os.getppid()
    while True:
        time.sleep(10)
        # If the parent PID changes, it usually means the original parent died
        # and we were reparented (usually to PID 1 on Unix).
        if os.getppid() != parent_pid:
            os._exit(0)

# Start watchdog in a daemon thread so it doesn't block exit
threading.Thread(target=_watchdog, daemon=True).start()

# Configure LiteLLM
litellm.api_key = settings.openai_api_key
litellm.api_base = settings.openai_base_url

def log_mcp_usage(tool_name: str, task_type: str, duration: float, prompt_tokens: int = 0, completion_tokens: int = 0, status: str = "success", error: str = None):
    """Log MCP tool usage to a JSONL file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "task_type": task_type,
        "duration_seconds": round(duration, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "status": status,
        "error": error
    }
    try:
        with open(settings.usage_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # Fallback to printing if logging fails
        print(f"Error writing to log file: {e}")

def resolve_model(passed_model: Optional[str] = None) -> str:
    """Resolve the model name, applying auto-prefixing if needed."""
    model_name = passed_model or settings.model.name

    # If it was explicitly passed as ollama/, convert it to openai/ to avoid LiteLLM's native ollama provider
    # which has issues with the /v1 path.
    if model_name.startswith("ollama/"):
        return f"openai/{model_name[7:]}"

    # Auto-prefix with openai/ provider if missing (LiteLLM requirement).
    # Applies to ANY OpenAI-compatible endpoint (Ollama /v1, LM Studio, llama-swap,
    # vLLM, etc.) — detected via the /v1 path in the base URL or known ports.
    if "/" not in model_name:
        base = settings.openai_base_url.lower()
        if "/v1" in base or "11434" in base or "1234" in base or "ollama" in base:
            return f"openai/{model_name}"

    return model_name

@mcp.tool()
def query_local_llm(
    prompt: str,
    system_message: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None
) -> str:
    """
    Query the local LLM for simple, well-defined subtasks that have already been broken down.
    IMPORTANT: Always try this tool FIRST for any simple code generation to save costs!
    """
    start_time = time.time()
    prompt_tokens = 0
    completion_tokens = 0
    
    try:
        # Use settings defaults or overrides
        sys_msg = system_message or settings.prompts["general"].system_message
        temp = temperature if temperature is not None else settings.model.temperature
        max_tok = max_tokens if max_tokens is not None else settings.model.max_tokens
        target_model = resolve_model(model)
        
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt}
        ]
        
        # Prepare LiteLLM completion arguments
        completion_args = {
            "model": target_model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
            **settings.model.extra_params
        }
        
        response = litellm.completion(**completion_args)
        
        # Extract usage
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            
        duration = time.time() - start_time
        log_mcp_usage("query_local_llm", "general", duration, prompt_tokens, completion_tokens)
        
        return response.choices[0].message.content
        
    except Exception as e:
        duration = time.time() - start_time
        log_mcp_usage("query_local_llm", "general", duration, status="error", error=str(e))
        return f"Error querying local LLM: {str(e)}"

@mcp.tool()
def query_local_llm_with_context(
    prompt: str,
    context: str,
    task_type: str = "general",
    system_message: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    Query the local LLM for simple subtasks that require additional context.
    Use this for code reviews, documentation, or refactoring.
    """
    start_time = time.time()
    prompt_tokens = 0
    completion_tokens = 0
    
    try:
        # Load template based on task_type
        template_cfg = settings.prompts.get(task_type, settings.prompts["general"])
        
        sys_msg = system_message or template_cfg.system_message
        full_user_prompt = template_cfg.user_template.format(context=context, prompt=prompt)
        target_model = resolve_model(model)
        
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": full_user_prompt}
        ]
        
        completion_args = {
            "model": target_model,
            "messages": messages,
            "temperature": settings.model.temperature,
            "max_tokens": settings.model.max_tokens,
            **settings.model.extra_params
        }
        
        response = litellm.completion(**completion_args)
        
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            
        duration = time.time() - start_time
        log_mcp_usage("query_local_llm_with_context", task_type, duration, prompt_tokens, completion_tokens)
        
        return response.choices[0].message.content
        
    except Exception as e:
        duration = time.time() - start_time
        log_mcp_usage("query_local_llm_with_context", task_type, duration, status="error", error=str(e))
        return f"Error querying local LLM with context: {str(e)}"


# ---------------------------------------------------------------------------
# Quorum code review: fan out to N parallel independent reviewers, reconcile.
# Each reviewer (a dedicated llama-swap instance) reasons independently.
# ---------------------------------------------------------------------------

# Default reviewer pool — override via env QUORUM_REVIEWERS (comma-separated).
def _get_quorum_reviewers() -> List[str]:
    env_val = os.environ.get("QUORUM_REVIEWERS", "").strip()
    if env_val:
        return [r.strip() for r in env_val.split(",") if r.strip()]
    return ["delegation-reviewer-1", "delegation-reviewer-2", "delegation-reviewer-3"]


QUORUM_SYSTEM_PROMPT = (
    "You are an independent senior code reviewer. Review the code for correctness, "
    "security, maintainability, and edge cases. Be specific and cite line numbers or "
    "function names. Conclude with a single line in exactly this format:\n"
    "VERDICT: APPROVE\n"
    "or\n"
    "VERDICT: REJECT\n"
    "followed by a one-sentence reason. A REVIEWER is one of several voting; only "
    "flag genuine issues, not style preferences."
)


async def _single_review(
    reviewer_model: str, code: str, context: str, semaphore: asyncio.Semaphore
) -> Tuple[str, str, str, bool, Optional[Exception]]:
    """Run one independent review.
    Returns (model_name, verdict_text, reasoning_text, succeeded, error).
    reasoning_text captures the thinking model's analysis (reasoning_content)
    which is where most of the actual review lives for thinking-capable models.
    """
    async with semaphore:
        try:
            target = resolve_model(reviewer_model)
            user_content = f"Context:\n{context}\n\nCode to review:\n```{code}```" if context else f"Code to review:\n```{code}```"
            messages = [
                {"role": "system", "content": QUORUM_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            response = await litellm.acompletion(
                model=target,
                messages=messages,
                temperature=settings.model.temperature,
                max_tokens=settings.model.max_tokens,
                **settings.model.extra_params,
            )
            msg = response.choices[0].message
            content = getattr(msg, "content", "") or ""
            # Thinking models (e.g. Gemma-DECKARD) put analysis in reasoning_content.
            reasoning = getattr(msg, "reasoning_content", "") or getattr(msg, "reasoning", "") or ""
            return (reviewer_model, content, reasoning, True, None)
        except Exception as e:
            return (reviewer_model, "", "", False, e)


def _parse_verdict(text: str) -> Tuple[Optional[bool], str]:
    """Extract APPROVE/REJECT from a reviewer's output. Returns (decision_bool_or_None, reason)."""
    if not text:
        return (None, "no output")
    # Find the VERDICT: line (case-insensitive).
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("VERDICT:"):
            v = stripped.split(":", 1)[1].strip().upper()
            if "REJECT" in v:
                reason = v.replace("REJECT", "").strip(" :-")
                return (False, reason or "rejected")
            if "APPROVE" in v or "ACCEPT" in v:
                reason = v.replace("APPROVE", "").replace("ACCEPT", "").strip(" :-")
                return (True, reason or "approved")
    # No explicit verdict line found — treat as None (no decision).
    return (None, "no explicit verdict")


@mcp.tool()
async def quorum_code_review(
    code: str,
    context: str = "",
    require_unanimous: bool = False,
    timeout: float = 300.0,
) -> str:
    """
    Run 3 parallel independent code reviews and return a reconciled verdict.

    Each reviewer (a dedicated llama-swap model instance) reasons independently
    and returns APPROVE or REJECT. Results are reconciled:
      - All agree    -> strong verdict (consensus)
      - 2 of 3 agree -> majority verdict with dissent noted
      - Full split   -> all opinions returned for you to adjudicate
    Set require_unanimous=True to require all reviewers to APPROVE.

    Args:
        code: The source code to review.
        context: Optional surrounding context (what the code is for, constraints).
        require_unanimous: If True, any REJECT or failure means the overall
            verdict is REJECT / no consensus.
        timeout: Per-review timeout in seconds (default 300 — model swaps can be slow).
    """
    reviewers = _get_quorum_reviewers()
    start_time = time.time()

    # Semaphore isn't strictly needed (we want them parallel) but caps at N.
    # Kept as an extension point in case the pool grows.
    semaphore = asyncio.Semaphore(len(reviewers))

    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *[_single_review(r, code, context, semaphore) for r in reviewers]
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        duration = time.time() - start_time
        log_mcp_usage("quorum_code_review", "review", duration, status="error",
                      error=f"timed out after {timeout}s")
        return f"Quorum review timed out after {timeout}s. Model swaps on the backend can be slow — retry."

    duration = time.time() - start_time

    # Tally verdicts.
    parsed = []
    total_prompt = 0
    total_completion = 0
    decisions = {"approve": 0, "reject": 0, "no_decision": 0}

    for model_name, text, reasoning, succeeded, err in results:
        if not succeeded:
            parsed.append((model_name, None, f"ERROR: {err}", "", ""))
            decisions["no_decision"] += 1
            continue
        # Thinking models may put the verdict in reasoning_content or content.
        combined = f"{reasoning}\n{text}".strip() if reasoning else text
        decision, reason = _parse_verdict(combined)
        if decision is None and text:
            # Fall back to parsing content alone (some models put verdict there).
            decision, reason = _parse_verdict(text)
        parsed.append((model_name, decision, reason, text, reasoning))
        if decision is True:
            decisions["approve"] += 1
        elif decision is False:
            decisions["reject"] += 1
        else:
            decisions["no_decision"] += 1

    log_mcp_usage("quorum_code_review", "review", duration,
                  total_prompt, total_completion)

    # Reconcile.
    n = len(parsed)
    approves = decisions["approve"]
    rejects = decisions["reject"]
    no_decisions = decisions["no_decision"]

    if require_unanimous:
        if approves == n:
            overall = "APPROVE (unanimous)"
        else:
            overall = "NO CONSENSUS (unanimous required)"
    elif approves > rejects and approves >= (n / 2):
        overall = f"APPROVE ({approves}/{n} majority)"
    elif rejects >= (n / 2):
        overall = f"REJECT ({rejects}/{n} majority)"
    else:
        overall = "SPLIT — no majority, adjudicate manually"

    # Build the report.
    lines = [
        f"### Quorum Code Review — {overall}",
        f"Tally: {approves} approve, {rejects} reject, {no_decisions} no-decision "
        f"(of {n} reviewers, {duration:.1f}s)",
        "",
    ]
    for model_name, decision, reason, content_text, reasoning_text in parsed:
        verdict_str = {True: "APPROVE", False: "REJECT", None: "NO DECISION"}.get(decision, "NO DECISION")
        lines.append(f"#### {model_name}: {verdict_str}")
        if reason and reason not in ("ERROR: ", "rejected", "approved", ""):
            lines.append(f"  Reason: {reason}")
        # Include the reviewer's actual analysis. Prefer content (the final answer);
        # fall back to reasoning_content if content is empty (thinking models).
        analysis = content_text.strip() if content_text.strip() else reasoning_text.strip()
        if analysis:
            lines.append("  Analysis:")
            for al in analysis.splitlines():
                if al.strip():
                    lines.append(f"    {al.rstrip()}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def get_local_llm_usage_stats() -> str:
    """Retrieve and summarize usage statistics for the local LLM MCP server."""
    if not os.path.exists(settings.usage_log_path):
        return "No usage data found yet."
    
    stats = {
        "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
        "total_prompt_tokens": 0, "total_completion_tokens": 0, "total_duration": 0.0,
        "tool_usage": {}, "task_usage": {}
    }
    
    try:
        with open(settings.usage_log_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    stats["total_requests"] += 1
                    if entry.get("status") == "success":
                        stats["successful_requests"] += 1
                        stats["total_prompt_tokens"] += entry.get("prompt_tokens", 0)
                        stats["total_completion_tokens"] += entry.get("completion_tokens", 0)
                        stats["total_duration"] += entry.get("duration_seconds", 0.0)
                        
                        tool = entry.get("tool", "unknown")
                        stats["tool_usage"][tool] = stats["tool_usage"].get(tool, 0) + 1
                        task = entry.get("task_type", "unknown")
                        stats["task_usage"][task] = stats["task_usage"].get(task, 0) + 1
                    else:
                        stats["failed_requests"] += 1
                except json.JSONDecodeError:
                    continue
        
        if stats["successful_requests"] == 0:
            return f"Usage Log Summary:\nTotal Requests: {stats['total_requests']}\nSuccessful: 0\nFailed: {stats['failed_requests']}"
        
        avg_duration = stats["total_duration"] / stats["successful_requests"]
        total_tokens = stats["total_prompt_tokens"] + stats["total_completion_tokens"]
        
        summary = [
            "### Local LLM Usage Summary",
            f"- **Total Requests**: {stats['total_requests']}",
            f"- **Success Rate**: {(stats['successful_requests'] / stats['total_requests'] * 100):.1f}%",
            f"- **Total Tokens**: {total_tokens:,} ({stats['total_prompt_tokens']:,} prompt, {stats['total_completion_tokens']:,} completion)",
            f"- **Avg Duration**: {avg_duration:.2f}s",
            "\n#### Usage by Tool:",
        ]
        for tool, count in stats["tool_usage"].items():
            summary.append(f"- {tool}: {count}")
        summary.append("\n#### Usage by Task Type:")
        for task, count in stats["task_usage"].items():
            summary.append(f"- {task}: {count}")
        return "\n".join(summary)
    except Exception as e:
        return f"Error reading usage stats: {str(e)}"

if __name__ == "__main__":
    mcp.run()
