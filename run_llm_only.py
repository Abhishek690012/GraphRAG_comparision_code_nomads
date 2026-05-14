#!/usr/bin/env python3
"""
LLM-Only API Inference — Standalone Entry Point

Runs the LLM-only baseline inference pipeline against a configured API endpoint.

Usage:
    python run_llm_only.py --config config/llm_only_config.yaml --query "What is diabetes?"
    python run_llm_only.py --config config/llm_only_config.yaml --query "What is diabetes?" --dry-run
    python run_llm_only.py --config config/llm_only_config.yaml --query "What is diabetes?" --system-prompt "Answer briefly."
"""

import argparse
import asyncio
import json
import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llm_only import LLMOnlyPipeline, ConfigurationError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-Only API Inference Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run (no API call):
  python run_llm_only.py --config config/llm_only_config.yaml --query "What is diabetes?" --dry-run

  # Live API call:
  export MISTRAL_API_KEY=your_key_here
  python run_llm_only.py --config config/llm_only_config.yaml --query "What is diabetes?"

  # Custom system prompt:
  python run_llm_only.py --config config/llm_only_config.yaml --query "Explain X" --system-prompt "Be concise."
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/llm_only_config.yaml",
        help="Path to the YAML configuration file (default: config/llm_only_config.yaml)",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The query string to send to the LLM",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt override (default: uses config value)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Enable dry-run mode (simulates API call, no network request)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: true)",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    try:
        dry_run = args.dry_run or None  # None = use config value, True = override
        async with LLMOnlyPipeline(args.config, dry_run_override=dry_run) as pipeline:
            # Execute the query
            result = await pipeline.run(
                query=args.query,
                system_prompt=args.system_prompt,
            )

            # Print result
            indent = 2 if args.pretty else None
            print("\n" + "=" * 60)
            print("INFERENCE RESULT")
            print("=" * 60)
            print(json.dumps(result, indent=indent, default=str))

            # Print manifest
            manifest = pipeline.generate_manifest()
            print("\n" + "=" * 60)
            print("EXECUTION MANIFEST")
            print("=" * 60)
            print(json.dumps(manifest, indent=indent, default=str))

            # Return non-zero if there was an error
            if result.get("error_type"):
                return 1
            return 0

    except ConfigurationError as e:
        print(f"\nCONFIGURATION ERROR:\n{e}", file=sys.stderr)
        return 2

    except FileNotFoundError as e:
        print(f"\nFILE NOT FOUND: {e}", file=sys.stderr)
        return 3

    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
