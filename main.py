import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rich.console import Console
from rich.table import Table
from rich.align import Align

from llm_only import LLMOnlyPipeline, ConfigurationError
from llm_only.config_validator import ConfigValidator
from basic_rag import BasicRAGPipeline
from basic_rag.rag_config_validator import RAGConfigValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline comparison orchestrator for LLM-Only and Basic RAG pipelines."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to run against both pipelines. If not provided and not in batch mode, interactive prompt is shown."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batch test mode with predefined questions. Runs the first N questions."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Enable dry-run mode (simulates LLM API calls, no network request)."
    )
    return parser.parse_args()


async def run_pipeline_safe(
    pipeline_name: str,
    pipeline: Optional[Any],
    query: str,
    is_rag: bool
) -> Dict[str, Any]:
    if pipeline is None:
        return {
            "status": "SKIPPED",
            "response": "ERROR: Skipped (API Key Unset)",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0,
            "cost_estimate": 0.0,
            "cache_hit": False,
        }

    start = time.perf_counter()
    try:
        # Execute the query
        result = await pipeline.run(query)
        # End-to-end latency measured independently using perf_counter
        latency = (time.perf_counter() - start) * 1000
        
        # Check if pipeline internally returned an error
        error_type = result.get("error_type")
        if error_type is not None:
            print(f"Pipeline {pipeline_name} failed with error type '{error_type}': {result.get('response')}", file=sys.stderr)
            return {
                "status": "FAILURE",
                "response": f"ERROR: {result.get('response', 'Unknown error')}",
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0),
                "latency_ms": result.get("latency_ms", latency),
                "cost_estimate": result.get("cost_estimate", 0.0),
                "cache_hit": result.get("retrieval_metadata", {}).get("cache_hit", False) if is_rag else False,
            }
            
        return {
            "status": "SUCCESS",
            "response": result.get("response", ""),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "latency_ms": latency,
            "cost_estimate": result.get("cost_estimate", 0.0),
            "cache_hit": result.get("retrieval_metadata", {}).get("cache_hit", False) if is_rag else False,
        }
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        import traceback
        traceback.print_exc()
        print(f"Pipeline {pipeline_name} threw exception: {e}", file=sys.stderr)
        return {
            "status": "FAILURE",
            "response": f"ERROR: {str(e)}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": latency,
            "cost_estimate": 0.0,
            "cache_hit": False,
        }


def truncate_response(text: str, max_len: int = 150) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."


def make_comparison_table(query: str, p1_res: Dict[str, Any], p2_res: Dict[str, Any]) -> Table:
    table = Table(
        title=f"[bold white]PIPELINE COMPARISON RESULTS[/bold white]\n[dim]Query: {query}[/dim]",
        title_justify="center",
        show_header=True,
        header_style="bold magenta",
    )
    
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Pipeline 1 (LLM-Only)", justify="left")
    table.add_column("Pipeline 2 (Basic RAG)", justify="left")
    
    # 1. Response
    p1_resp = truncate_response(p1_res["response"])
    p2_resp = truncate_response(p2_res["response"])
    table.add_row("Response", p1_resp, p2_resp)
    
    # 2. Prompt Tokens
    p1_pt = str(p1_res["prompt_tokens"]) if p1_res["status"] != "SKIPPED" else "-"
    p2_pt = str(p2_res["prompt_tokens"]) if p2_res["status"] != "SKIPPED" else "-"
    table.add_row("Prompt Tokens", Align.right(p1_pt), Align.right(p2_pt))
    
    # 3. Completion Tokens
    p1_ct = str(p1_res["completion_tokens"]) if p1_res["status"] != "SKIPPED" else "-"
    p2_ct = str(p2_res["completion_tokens"]) if p2_res["status"] != "SKIPPED" else "-"
    table.add_row("Completion Tokens", Align.right(p1_ct), Align.right(p2_ct))
    
    # 4. Total Tokens
    p1_tt = str(p1_res["total_tokens"]) if p1_res["status"] != "SKIPPED" else "-"
    p2_tt = str(p2_res["total_tokens"]) if p2_res["status"] != "SKIPPED" else "-"
    table.add_row("Total Tokens", Align.right(p1_tt), Align.right(p2_tt))
    
    # 5. Latency (ms)
    p1_lat = f"{int(p1_res['latency_ms'])}" if p1_res["status"] != "SKIPPED" else "-"
    p2_lat = f"{int(p2_res['latency_ms'])}" if p2_res["status"] != "SKIPPED" else "-"
    table.add_row("Latency (ms)", Align.right(p1_lat), Align.right(p2_lat))
    
    # 6. Est. Cost ($)
    p1_cost = f"{p1_res['cost_estimate']:.6f}" if p1_res["status"] != "SKIPPED" else "-"
    p2_cost = f"{p2_res['cost_estimate']:.6f}" if p2_res["status"] != "SKIPPED" else "-"
    table.add_row("Est. Cost ($)", Align.right(p1_cost), Align.right(p2_cost))
    
    # 7. Cache Hit
    p1_ch = "No" if p1_res["status"] != "SKIPPED" else "-"
    p2_ch = "Yes" if p2_res["cache_hit"] else "No" if p2_res["status"] != "SKIPPED" else "-"
    table.add_row("Cache Hit", Align.right(p1_ch), Align.right(p2_ch))
    
    # 8. Status
    def get_status_styled(status):
        if status == "SUCCESS":
            return "[green]SUCCESS[/green]"
        elif status == "FAILURE":
            return "[red]FAILURE[/red]"
        elif status == "SKIPPED":
            return "[yellow]SKIPPED[/yellow]"
        return status
        
    table.add_row("Status", Align.right(get_status_styled(p1_res["status"])), Align.right(get_status_styled(p2_res["status"])))
    
    # Summary Row
    table.add_section()
    table.add_row("[bold white]SUMMARY[/bold white]", "", "")
    
    # Calculate Differences only if both succeeded
    if p1_res["status"] == "SUCCESS" and p2_res["status"] == "SUCCESS":
        token_diff = p2_res["total_tokens"] - p1_res["total_tokens"]
        latency_diff = p2_res["latency_ms"] - p1_res["latency_ms"]
        cost_diff = p2_res["cost_estimate"] - p1_res["cost_estimate"]
        
        token_diff_str = f"{token_diff:+d} tokens" if token_diff != 0 else "0 tokens"
        latency_diff_str = f"{latency_diff:+.0f} ms"
        cost_diff_str = f"+${cost_diff:.6f}" if cost_diff >= 0 else f"-${abs(cost_diff):.6f}"
        
        table.add_row("Token Difference", Align.right("-"), Align.right(token_diff_str))
        table.add_row("Latency Difference", Align.right("-"), Align.right(latency_diff_str))
        table.add_row("Cost Difference", Align.right("-"), Align.right(cost_diff_str))
    else:
        table.add_row("Token Difference", Align.right("-"), Align.right("-"))
        table.add_row("Latency Difference", Align.right("-"), Align.right("-"))
        table.add_row("Cost Difference", Align.right("-"), Align.right("-"))
        
    return table


async def main():
    args = parse_args()
    
    llm_config_path = "config/llm_only_config.yaml"
    rag_config_path = "config/basic_rag_config.yaml"
    
    if not os.path.exists(llm_config_path):
        print(f"Error: Config file not found at {llm_config_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(rag_config_path):
        print(f"Error: Config file not found at {rag_config_path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(llm_config_path, "r") as f:
            llm_config = yaml.safe_load(f)
        with open(rag_config_path, "r") as f:
            rag_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading configuration files: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Determine dry-run status
    llm_dry_run = args.dry_run or llm_config.get("dry_run", False)
    rag_dry_run = args.dry_run or rag_config.get("dry_run", False)
    
    # Check env vars
    llm_key_var = llm_config.get("api", {}).get("api_key_env_var", "")
    rag_key_var = rag_config.get("api", {}).get("api_key_env_var", "")
    
    llm_key_set = bool(os.environ.get(llm_key_var)) if llm_key_var else False
    rag_key_set = bool(os.environ.get(rag_key_var)) if rag_key_var else False
    
    llm_skip = not llm_key_set and not llm_dry_run
    rag_skip = not rag_key_set and not rag_dry_run
    
    # Run validation
    try:
        llm_validator = ConfigValidator(llm_config)
        llm_validator.validate(dry_run_override=True if llm_skip else llm_dry_run)
    except ConfigurationError as e:
        print(f"Configuration validation failed for LLM-Only pipeline:\n{e}", file=sys.stderr)
        sys.exit(1)
        
    try:
        rag_validator = RAGConfigValidator(rag_config)
        rag_validator.validate(dry_run_override=True if rag_skip else rag_dry_run)
    except ConfigurationError as e:
        print(f"Configuration validation failed for Basic RAG pipeline:\n{e}", file=sys.stderr)
        sys.exit(1)
        
    # Print warnings for skipped pipelines
    if llm_skip:
        print(f"Warning: API key environment variable '{llm_key_var}' is not set. Skipping Pipeline 1 (LLM-Only).", file=sys.stderr)
    if rag_skip:
        print(f"Warning: API key environment variable '{rag_key_var}' is not set. Skipping Pipeline 2 (Basic RAG).", file=sys.stderr)
        
    # Instantiate pipelines
    llm_pipeline = None
    if not llm_skip:
        try:
            llm_pipeline = LLMOnlyPipeline(llm_config_path, dry_run_override=llm_dry_run)
        except Exception as e:
            print(f"Failed to initialize LLM-Only pipeline: {e}", file=sys.stderr)
            sys.exit(1)
            
    rag_pipeline = None
    if not rag_skip:
        try:
            rag_pipeline = BasicRAGPipeline(rag_config_path, dry_run_override=rag_dry_run)
        except Exception as e:
            print(f"Failed to initialize Basic RAG pipeline: {e}", file=sys.stderr)
            if llm_pipeline:
                await llm_pipeline.close()
            sys.exit(1)
            
    try:
        console = Console()
        
        # Batch Mode
        if args.limit is not None:
            if args.limit <= 0:
                print("Error: --limit must be a positive integer.", file=sys.stderr)
                sys.exit(1)
                
            questions_path = "data/medical_questions.json"
            if not os.path.exists(questions_path):
                questions_path = "data/input/medical_questions.json"
                
            if not os.path.exists(questions_path):
                print(f"Error: Medical questions file not found at {questions_path}", file=sys.stderr)
                sys.exit(1)
                
            try:
                with open(questions_path, "r") as f:
                    questions_data = json.load(f)
            except Exception as e:
                print(f"Error reading medical questions: {e}", file=sys.stderr)
                sys.exit(1)
                
            questions = [q["question"] for q in questions_data[:args.limit] if "question" in q]
            
            if not questions:
                print("Error: No questions found in the JSON file.", file=sys.stderr)
                sys.exit(1)
                
            p1_results = []
            p2_results = []
            
            console.print(f"[bold cyan]Starting batch mode on {len(questions)} questions...[/bold cyan]")
            
            for idx, q in enumerate(questions, 1):
                console.print(f"\n[bold yellow][{idx}/{len(questions)}] Query: {q}[/bold yellow]")
                p1_task = run_pipeline_safe("LLM-Only", llm_pipeline, q, is_rag=False)
                p2_task = run_pipeline_safe("Basic RAG", rag_pipeline, q, is_rag=True)
                p1_res, p2_res = await asyncio.gather(p1_task, p2_task)
                
                p1_results.append(p1_res)
                p2_results.append(p2_res)
                
                table = make_comparison_table(q, p1_res, p2_res)
                console.print(table)
                
            # Print Aggregated Summary Table
            summary_table = Table(
                title=f"[bold white]BATCH COMPARISON SUMMARY (N={len(questions)})[/bold white]",
                title_justify="center",
                show_header=True,
                header_style="bold magenta",
            )
            
            summary_table.add_column("Metric", style="cyan", justify="left")
            summary_table.add_column("Pipeline 1 (LLM-Only)", justify="left")
            summary_table.add_column("Pipeline 2 (Basic RAG)", justify="left")
            
            p1_success = [r for r in p1_results if r["status"] == "SUCCESS"]
            p1_success_count = len(p1_success)
            p1_skipped_count = sum(1 for r in p1_results if r["status"] == "SKIPPED")
            p1_fail_count = sum(1 for r in p1_results if r["status"] == "FAILURE")
            p1_total = len(p1_results)
            p1_success_rate = (p1_success_count / (p1_total - p1_skipped_count) * 100.0) if (p1_total - p1_skipped_count) > 0 else 0.0
            
            p1_avg_pt = sum(r["prompt_tokens"] for r in p1_success) / p1_success_count if p1_success_count > 0 else 0.0
            p1_avg_ct = sum(r["completion_tokens"] for r in p1_success) / p1_success_count if p1_success_count > 0 else 0.0
            p1_avg_tt = sum(r["total_tokens"] for r in p1_success) / p1_success_count if p1_success_count > 0 else 0.0
            p1_avg_lat = sum(r["latency_ms"] for r in p1_success) / p1_success_count if p1_success_count > 0 else 0.0
            p1_tot_cost = sum(r["cost_estimate"] for r in p1_results)
            
            p2_success = [r for r in p2_results if r["status"] == "SUCCESS"]
            p2_success_count = len(p2_success)
            p2_skipped_count = sum(1 for r in p2_results if r["status"] == "SKIPPED")
            p2_fail_count = sum(1 for r in p2_results if r["status"] == "FAILURE")
            p2_total = len(p2_results)
            p2_success_rate = (p2_success_count / (p2_total - p2_skipped_count) * 100.0) if (p2_total - p2_skipped_count) > 0 else 0.0
            
            p2_avg_pt = sum(r["prompt_tokens"] for r in p2_success) / p2_success_count if p2_success_count > 0 else 0.0
            p2_avg_ct = sum(r["completion_tokens"] for r in p2_success) / p2_success_count if p2_success_count > 0 else 0.0
            p2_avg_tt = sum(r["total_tokens"] for r in p2_success) / p2_success_count if p2_success_count > 0 else 0.0
            p2_avg_lat = sum(r["latency_ms"] for r in p2_success) / p2_success_count if p2_success_count > 0 else 0.0
            p2_tot_cost = sum(r["cost_estimate"] for r in p2_results)
            
            p1_all_skipped = p1_skipped_count == p1_total
            p2_all_skipped = p2_skipped_count == p2_total
            
            def fmt_val(val, format_spec, skipped):
                if skipped:
                    return "-"
                return format_spec(val)
                
            summary_table.add_row("Total Queries", Align.right(str(p1_total)), Align.right(str(p2_total)))
            summary_table.add_row("Successful Queries", Align.right(str(p1_success_count)), Align.right(str(p2_success_count)))
            summary_table.add_row("Failed Queries", Align.right(str(p1_fail_count)), Align.right(str(p2_fail_count)))
            summary_table.add_row("Success Rate", 
                                  Align.right(fmt_val(p1_success_rate, lambda x: f"{x:.1f}%", p1_all_skipped)), 
                                  Align.right(fmt_val(p2_success_rate, lambda x: f"{x:.1f}%", p2_all_skipped)))
            summary_table.add_row("Avg Prompt Tokens", 
                                  Align.right(fmt_val(p1_avg_pt, lambda x: f"{x:.1f}", p1_all_skipped)), 
                                  Align.right(fmt_val(p2_avg_pt, lambda x: f"{x:.1f}", p2_all_skipped)))
            summary_table.add_row("Avg Completion Tokens", 
                                  Align.right(fmt_val(p1_avg_ct, lambda x: f"{x:.1f}", p1_all_skipped)), 
                                  Align.right(fmt_val(p2_avg_ct, lambda x: f"{x:.1f}", p2_all_skipped)))
            summary_table.add_row("Avg Total Tokens", 
                                  Align.right(fmt_val(p1_avg_tt, lambda x: f"{x:.1f}", p1_all_skipped)), 
                                  Align.right(fmt_val(p2_avg_tt, lambda x: f"{x:.1f}", p2_all_skipped)))
            summary_table.add_row("Avg Latency (ms)", 
                                  Align.right(fmt_val(p1_avg_lat, lambda x: f"{x:.1f}", p1_all_skipped)), 
                                  Align.right(fmt_val(p2_avg_lat, lambda x: f"{x:.1f}", p2_all_skipped)))
            summary_table.add_row("Total Cost ($)", 
                                  Align.right(fmt_val(p1_tot_cost, lambda x: f"${x:.6f}", p1_all_skipped)), 
                                  Align.right(fmt_val(p2_tot_cost, lambda x: f"${x:.6f}", p2_all_skipped)))
            
            console.print("\n" + "=" * 60)
            console.print(summary_table)
            console.print("=" * 60 + "\n")
            
            # Check if both pipelines completely failed / were skipped
            all_p1_failed = all(r["status"] in ("FAILURE", "SKIPPED") for r in p1_results)
            all_p2_failed = all(r["status"] in ("FAILURE", "SKIPPED") for r in p2_results)
            if all_p1_failed and all_p2_failed:
                print("Error: Both pipelines failed to execute successfully for all queries.", file=sys.stderr)
                sys.exit(2)
                
        # Single Query Mode
        else:
            query = args.query
            if not query:
                try:
                    query = input("Enter your query: ").strip()
                except KeyboardInterrupt:
                    print("\nInteractive input cancelled by user.", file=sys.stderr)
                    sys.exit(0)
                    
            if not query:
                print("Error: Empty query provided.", file=sys.stderr)
                sys.exit(1)
                
            p1_task = run_pipeline_safe("LLM-Only", llm_pipeline, query, is_rag=False)
            p2_task = run_pipeline_safe("Basic RAG", rag_pipeline, query, is_rag=True)
            p1_res, p2_res = await asyncio.gather(p1_task, p2_task)
            
            table = make_comparison_table(query, p1_res, p2_res)
            console.print(table)
            
            if p1_res["status"] in ("FAILURE", "SKIPPED") and p2_res["status"] in ("FAILURE", "SKIPPED"):
                print("Error: Both pipelines failed or were skipped.", file=sys.stderr)
                sys.exit(2)
                
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Cleaning up...", file=sys.stderr)
        sys.exit(2)
    finally:
        if llm_pipeline:
            await llm_pipeline.close()
        if rag_pipeline:
            await rag_pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())