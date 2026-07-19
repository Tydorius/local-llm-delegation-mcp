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
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastmcp import FastMCP, Context
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
        os.makedirs(os.path.dirname(settings.usage_log_path), exist_ok=True)
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
#
# Two entry points cover different MCP client timeout behaviors:
#   - quorum_code_review: single call. Emits a progress notification every
#     QUORUM_HEARTBEAT_SECONDS so clients that reset their request timeout on
#     progress survive cold model loads (~2-3 min).
#   - start_quorum_review + get_quorum_result: async job pattern for clients
#     with a hard per-request timeout (Claude Desktop cancels tool calls at
#     60s). Each call returns immediately; the review runs in the background
#     on the server's event loop and is unaffected by client cancellation.
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
) -> Dict[str, Any]:
    """Run one independent review.
    Returns a dict with model, content, reasoning, ok, error, and token counts.
    reasoning captures the thinking model's analysis (reasoning_content)
    which is where most of the actual review lives for thinking-capable models.
    """
    result: Dict[str, Any] = {
        "model": reviewer_model, "content": "", "reasoning": "",
        "ok": False, "error": None, "prompt_tokens": 0, "completion_tokens": 0,
    }
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
                timeout=600,  # 10 min — thinking models (~7 t/s, ~1000 tokens) need >60s; also covers cold llama-swap loads.
                **settings.model.extra_params,
            )
            msg = response.choices[0].message
            result["content"] = getattr(msg, "content", "") or ""
            # Thinking models put analysis in reasoning_content.
            result["reasoning"] = getattr(msg, "reasoning_content", "") or getattr(msg, "reasoning", "") or ""
            usage = getattr(response, "usage", None)
            if usage:
                result["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
                result["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
            result["ok"] = True
        except Exception as e:
            err_str = str(e)
            # Surface timeout errors explicitly so they're not confused with model failures.
            if "timed out" in err_str.lower() or "timeout" in err_str.lower():
                result["error"] = f"reviewer timed out — {err_str}"
            else:
                result["error"] = err_str
        return result


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


def _reconcile_reviews(
    results: List[Dict[str, Any]], duration: float, require_unanimous: bool
) -> str:
    """Tally reviewer verdicts, log usage (with real token totals), build the report."""
    parsed = []
    total_prompt = 0
    total_completion = 0
    decisions = {"approve": 0, "reject": 0, "no_decision": 0}

    for r in results:
        total_prompt += r["prompt_tokens"]
        total_completion += r["completion_tokens"]
        if not r["ok"]:
            parsed.append((r["model"], None, f"ERROR: {r['error']}", "", ""))
            decisions["no_decision"] += 1
            continue
        text, reasoning = r["content"], r["reasoning"]
        # Thinking models may put the verdict in reasoning_content or content.
        combined = f"{reasoning}\n{text}".strip() if reasoning else text
        decision, reason = _parse_verdict(combined)
        if decision is None and text:
            # Fall back to parsing content alone (some models put verdict there).
            decision, reason = _parse_verdict(text)
        parsed.append((r["model"], decision, reason, text, reasoning))
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


QUORUM_HEARTBEAT_SECONDS = 10.0


async def _run_quorum(
    code: str,
    context: str,
    require_unanimous: bool,
    timeout: float,
    ctx: Optional[Context] = None,
) -> str:
    """Fan out to all reviewers, reconcile, and return the report.

    Heartbeats progress via ctx (if provided) so MCP clients that reset their
    request timeout on progress notifications don't cancel the call during cold
    model loads. If the client cancels anyway, log it before propagating —
    silent CancelledError is indistinguishable from a hang in the usage log.
    """
    reviewers = _get_quorum_reviewers()
    start_time = time.time()

    # Semaphore isn't strictly needed (we want them parallel) but caps at N.
    # Kept as an extension point in case the pool grows.
    semaphore = asyncio.Semaphore(len(reviewers))
    gather = asyncio.gather(
        *[_single_review(r, code, context, semaphore) for r in reviewers]
    )

    try:
        while True:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                gather.cancel()
                log_mcp_usage("quorum_code_review", "review", elapsed, status="error",
                              error=f"timed out after {timeout}s")
                return (f"Quorum review timed out after {timeout}s. Model swaps on the "
                        f"backend can be slow — retry, or raise the timeout parameter.")
            try:
                # shield: a heartbeat interval elapsing must not cancel the reviews.
                results = await asyncio.wait_for(
                    asyncio.shield(gather),
                    timeout=min(QUORUM_HEARTBEAT_SECONDS, remaining),
                )
                break
            except asyncio.TimeoutError:
                if ctx is not None:
                    try:
                        await ctx.report_progress(progress=elapsed, total=timeout)
                    except Exception:
                        pass  # progress is best-effort; never fail the review over it
    except asyncio.CancelledError:
        gather.cancel()
        log_mcp_usage("quorum_code_review", "review", time.time() - start_time,
                      status="error",
                      error="cancelled by MCP client — likely the client's own request "
                            "timeout (Claude Desktop: 60s); use start_quorum_review instead")
        raise

    return _reconcile_reviews(results, time.time() - start_time, require_unanimous)


@mcp.tool()
async def quorum_code_review(
    code: str,
    context: str = "",
    require_unanimous: bool = False,
    timeout: float = 300.0,
    ctx: Optional[Context] = None,
) -> str:
    """
    Run 3 parallel independent code reviews and return a reconciled verdict.

    NOTE: cold model loads take 2-3 minutes. If your MCP client enforces a hard
    per-request timeout (Claude Desktop cancels at ~60s), use start_quorum_review
    + get_quorum_result instead — this call emits progress heartbeats, but not
    all clients honor them.

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
    return await _run_quorum(code, context, require_unanimous, timeout, ctx)


# Background quorum jobs, keyed by job id. Kept in-process: jobs die with the
# server, which is acceptable — the client that started a job polls it within
# the same session.
_quorum_jobs: Dict[str, Dict[str, Any]] = {}
_QUORUM_JOB_TTL_SECONDS = 3600.0


def _prune_quorum_jobs() -> None:
    now = time.time()
    stale = [jid for jid, rec in _quorum_jobs.items()
             if rec["done"] and now - rec["finished"] > _QUORUM_JOB_TTL_SECONDS]
    for jid in stale:
        del _quorum_jobs[jid]


@mcp.tool()
async def start_quorum_review(
    code: str,
    context: str = "",
    require_unanimous: bool = False,
    timeout: float = 300.0,
) -> str:
    """
    Start a quorum code review as a background job and return immediately.

    Use this (with get_quorum_result) instead of quorum_code_review when your
    MCP client enforces a hard per-request timeout: the review keeps running
    server-side no matter what the client does. Same semantics as
    quorum_code_review otherwise — see its docstring for args and verdicts.

    Returns a job id. Poll get_quorum_result(job_id) after ~60s (warm models)
    or 2-3 minutes (cold load).
    """
    _prune_quorum_jobs()
    job_id = uuid.uuid4().hex[:12]
    rec: Dict[str, Any] = {"done": False, "started": time.time(),
                           "finished": None, "result": None, "task": None}

    async def _runner():
        try:
            rec["result"] = await _run_quorum(code, context, require_unanimous, timeout)
        except Exception as e:
            rec["result"] = f"Quorum job {job_id} failed: {e}"
        finally:
            rec["done"] = True
            rec["finished"] = time.time()

    rec["task"] = asyncio.create_task(_runner())
    _quorum_jobs[job_id] = rec
    n = len(_get_quorum_reviewers())
    return (f"Quorum review started: job_id={job_id} ({n} reviewers, timeout {timeout:.0f}s). "
            f"Poll get_quorum_result(job_id=\"{job_id}\") in ~60s (warm) or 2-3 min (cold load). "
            f"The job runs server-side and survives client-side call timeouts.")


@mcp.tool()
async def get_quorum_result(job_id: str) -> str:
    """
    Fetch the result of a quorum review started with start_quorum_review.

    Returns the full reconciled review report if the job is done, or a status
    line with elapsed time if it is still running (poll again in ~30s).
    Results are kept for 1 hour after completion.
    """
    rec = _quorum_jobs.get(job_id)
    if rec is None:
        return (f"Unknown job_id '{job_id}'. Either it expired (results are kept 1 hour), "
                f"the server restarted, or the id is wrong.")
    if not rec["done"]:
        elapsed = time.time() - rec["started"]
        return (f"Job {job_id} still running ({elapsed:.0f}s elapsed). Poll again in ~30s. "
                f"Cold model loads can take 2-3 minutes.")
    return rec["result"]


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
