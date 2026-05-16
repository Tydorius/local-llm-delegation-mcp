#!/usr/bin/env python3
import os
import sys

# Ensure we can import from server.py in the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from server import get_local_llm_usage_stats
except ImportError as e:
    print(f"Error: Could not import stats logic from server.py: {e}")
    sys.exit(1)

def main():
    """Display the Local LLM usage statistics."""
    print("\nChecking Local LLM Usage Statistics...")
    print("=" * 40)
    
    stats = get_local_llm_usage_stats()
    print(stats)
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
