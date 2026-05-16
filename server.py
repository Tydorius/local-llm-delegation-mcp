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
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    
    # Auto-prefix with provider if missing (LiteLLM requirement)
    if "/" not in model_name:
        if "11434" in settings.openai_base_url or "ollama" in settings.openai_base_url.lower():
            # Use openai/ prefix even for Ollama to leverage the /v1 OpenAI-compatible endpoint
            return f"openai/{model_name}"
        elif "1234" in settings.openai_base_url: # LM Studio default
            return f"openai/{model_name}"
    
    # If it was explicitly passed as ollama/, convert it to openai/ to avoid LiteLLM's native ollama provider
    # which has issues with the /v1 path.
    if model_name.startswith("ollama/"):
        return f"openai/{model_name[7:]}"
        
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
