#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""
AI-powered performance analysis for GPU traces.

This module analyzes rocpd database files and provides human-readable insights,
bottleneck identification, and optimization recommendations.

Pure analysis logic lives in the ``analysis/`` sub-package; this file is the
thin orchestration and CLI layer.
"""

import argparse
import os
import re
import shlex
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from importlib.metadata import version as _pkg_version

    _ROCINSIGHT_VERSION = _pkg_version("rocinsight")
except Exception:
    _ROCINSIGHT_VERSION = "0.1.0"  # fallback if metadata not available (common in dev / ROCm system installs)

from .connection import RocinsightConnection as RocpdImportData, execute_statement
from .tracelens_port import (
    compute_interval_timeline,
    analyze_kernels_by_category,
    analyze_short_kernels,
)
from . import output_config

# ---------------------------------------------------------------------------
# Re-export analysis functions from the analysis/ sub-package so that
# ``from rocinsight.analyze import compute_time_breakdown`` (etc.) keeps
# working for all existing callers.
# ---------------------------------------------------------------------------
from .analysis import (  # noqa: F401 -- re-exports for backward compat
    identify_hotspots,
    analyze_memory_copies,
    analyze_hardware_counters,
    analyze_thread_trace,
    generate_recommendations,
    _split_pmc_into_passes,
    _detect_already_collected,
    _filter_rec_commands,
    _is_code_change_rec,
    _ATT_STALL_CATEGORY_MAP,
    _ATT_MIN_HITCOUNT,
    _att_stall_category,
    _SYS_TRACE_IMPLIED,
    _OUTPUT_ONLY_ARGS,
    _PMC_BLOCK_LIMIT_DEFAULT,
    _PMC_BLOCK_LIMITS,
    _TCC_DERIVED_COUNTERS,
    _pmc_block,
    _pmc_block_limit,
    _INIT_OVERHEAD_MAX_KERNEL_PCT,
    _INIT_OVERHEAD_MAX_RUNTIME_NS,
)
from .analysis import core as _analysis_core


def compute_time_breakdown(connection: RocpdImportData) -> Dict[str, Any]:
    """Backward-compat shim — delegates to ``analysis.core`` but uses
    this module's ``execute_statement`` so that
    ``mock.patch("rocinsight.analyze.execute_statement")`` keeps working."""
    import rocinsight.analysis.core as _m

    _saved = _m.execute_statement
    try:
        _m.execute_statement = execute_statement  # pick up any mock on this module
        return _analysis_core.compute_time_breakdown(connection)
    finally:
        _m.execute_statement = _saved

__all__ = [
    "compute_time_breakdown",
    "identify_hotspots",
    "analyze_memory_copies",
    "analyze_hardware_counters",
    "generate_recommendations",
    "format_analysis_output",
    "analyze_performance",
    "add_args",
    "execute",
    "main",
]


# ---------------------------------------------------------------------------
# Output formatting functions (extracted to formatters.py)
# ---------------------------------------------------------------------------
from .formatters import (  # noqa: F401 -- re-exports for backward compat
    _format_as_json,
    _build_summary,
    _build_hw_counters_json,
    _build_recommendations_json,
    _build_warnings_json,
    _format_as_markdown,
    _format_as_webview,
    _tier0_recommendations_text,
    _format_tier0_text,
    _tier0_to_dict,
    _format_tier0_json,
    _format_tier0_markdown,
    _format_tier0_webview,
    format_analysis_output,
    _CATEGORY_IDS,
)


def analyze_source_code(
    source_dir: str,
    prompt: Optional[str] = None,
    llm: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    verbose: bool = False,
) -> Any:
    """
    Run Tier 0 static source code analysis.

    Args:
        source_dir: Path to source directory
        prompt: Optional user question to guide analysis
        llm: LLM provider ("anthropic", "openai")
        llm_api_key: API key for LLM provider
        llm_model: Override LLM model name
        verbose: Enable verbose logging

    Returns:
        SourceAnalysisResult from ai_analysis.api
    """
    from pathlib import Path as _Path
    from .ai_analysis.source_analyzer import SourceAnalyzer
    from .ai_analysis.api import _plan_to_source_result

    _src_path = _Path(source_dir)
    if not _src_path.exists() or not _src_path.is_dir():
        from .ai_analysis.exceptions import SourceDirectoryNotFoundError

        raise SourceDirectoryNotFoundError(
            f"Source directory not found or not a directory: {source_dir}"
        )

    if verbose:
        print(f"[Tier0] Scanning source directory: {source_dir}")

    scanner = SourceAnalyzer(_src_path, verbose=verbose)
    plan = scanner.analyze()

    if verbose:
        print(
            f"[Tier0] Scanned {plan.files_scanned} files, "
            f"{plan.kernel_count} kernels, model: {plan.programming_model}"
        )

    # Convert ProfilingPlan -> SourceAnalysisResult dataclass
    result = _plan_to_source_result(plan)

    if llm:
        _prev = os.environ.get("ROCINSIGHT_LLM_MODEL")
        try:
            from .ai_analysis.llm_analyzer import LLMAnalyzer

            if llm_model:
                os.environ["ROCINSIGHT_LLM_MODEL"] = llm_model
            try:
                analyzer = LLMAnalyzer(provider=llm, api_key=llm_api_key, verbose=verbose)
                from .ai_analysis.llm_analyzer import AnalysisContext as _AnalysisContext

                _llm_ctx = _AnalysisContext(tier=0, custom_prompt=prompt)
                _mdl = llm_model or os.environ.get("ROCINSIGHT_LLM_MODEL", "")
                _mdl_str = f" ({_mdl})" if _mdl else ""
                print(
                    f"  Contacting {llm}{_mdl_str} for source analysis — please wait...",
                    file=sys.stderr,
                    flush=True,
                )
                result.llm_explanation = analyzer.analyze_source_with_llm(
                    result, custom_prompt=prompt, context=_llm_ctx
                )
            finally:
                if llm_model:
                    if _prev is None:
                        os.environ.pop("ROCINSIGHT_LLM_MODEL", None)
                    else:
                        os.environ["ROCINSIGHT_LLM_MODEL"] = _prev
        except Exception as e:
            print(f"⚠️  Tier 0 LLM enhancement failed: {e}", file=sys.stderr)

    return result


def analyze_performance(
    connection: Optional[RocpdImportData],
    prompt: Optional[str] = None,
    top_kernels: int = 10,
    min_duration: float = 0.0,
    output_format: str = "text",
    database_path: str = "",
    llm: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_thinking: Optional[int] = None,
    verbose: bool = False,
    source_dir: Optional[str] = None,
    att_dir: Optional[str] = None,
    _collect_result: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> str:
    """
    Main analysis orchestrator that runs all analyses and formats output.

    Args:
        connection: RocpdImportData database connection
        prompt: Optional custom analysis prompt
        top_kernels: Number of top kernels to analyze
        min_duration: Minimum kernel duration threshold
        output_format: Output format (text, json, markdown)
        database_path: Path to database file
        llm: LLM provider (anthropic or openai)
        llm_api_key: API key for LLM provider
        verbose: Enable verbose logging
        **kwargs: Additional arguments

    Returns:
        Formatted analysis output string
    """
    # ------------------------------------------------------------------
    # Tier 0 — static source code analysis (optional)
    # ------------------------------------------------------------------
    tier0_result = None
    if source_dir:
        tier0_result = analyze_source_code(
            source_dir=source_dir,
            prompt=prompt,
            llm=llm,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Tier 1/2 — database analysis (only when a connection is provided)
    # ------------------------------------------------------------------
    source_only = connection is None
    if not source_only:
        time_breakdown = compute_time_breakdown(connection)
        hotspots = identify_hotspots(
            connection, top_n=top_kernels, min_duration=min_duration
        )
        memory_analysis = analyze_memory_copies(connection)
        hardware_counters = analyze_hardware_counters(connection)  # Tier 2
        already_collected = _detect_already_collected(connection)
        # Tier 3: ATT thread trace (optional — only when --att-dir is provided)
        att_analysis: Dict[str, Any] = {}
        if att_dir:
            att_analysis = analyze_thread_trace(att_dir)
            if verbose and not att_analysis.get("has_att_data"):
                print(
                    f"[ATT] {att_analysis.get('reason', 'No ATT data')}",
                    file=sys.stderr,
                )
        # TraceLens-derived analysis (Phase 1)
        interval_timeline = compute_interval_timeline(connection)
        kernel_categories = analyze_kernels_by_category(
            connection, interval_timeline["total_wall_ns"]
        )
        short_kernels_data = analyze_short_kernels(connection)
        # Generate recommendations (redundant re-collection commands are filtered out)
        recommendations = generate_recommendations(
            time_breakdown,
            hotspots,
            memory_analysis,
            hardware_counters,
            already_collected=already_collected,
            short_kernels=short_kernels_data,
            interval_timeline=interval_timeline,
            att_analysis=att_analysis if att_dir else None,
        )
    else:
        time_breakdown = {}
        hotspots = []
        memory_analysis = {}
        hardware_counters = {}
        already_collected = frozenset()
        att_analysis = {}
        interval_timeline = {}
        kernel_categories = []
        short_kernels_data = {}
        recommendations = tier0_result.recommendations if tier0_result else []

    # Format output
    output = format_analysis_output(
        time_breakdown=time_breakdown,
        hotspots=hotspots,
        memory_analysis=memory_analysis,
        recommendations=recommendations,
        hardware_counters=hardware_counters,
        database_path=database_path,
        output_format=output_format,
        tier0_result=tier0_result,
        source_only=source_only,
        interval_timeline=interval_timeline,
        kernel_categories=kernel_categories,
        short_kernels=short_kernels_data,
        att_analysis=att_analysis if att_analysis else None,
        custom_prompt=prompt,
    )

    # Expose structured results to caller (used by interactive mode)
    if _collect_result is not None:
        _collect_result["recommendations"] = recommendations
        _collect_result["tier0_result"] = tier0_result
        _collect_result["database_path"] = database_path
        _collect_result["att_analysis"] = att_analysis

    # LLM enhancement (if enabled) — only for Tier 1/2; Tier 0 LLM runs in analyze_source_code()
    if llm and not source_only:
        # Initialize before try so the finally block can always reference these names safely.
        _prev_model_env = os.environ.get("ROCINSIGHT_LLM_MODEL")
        try:
            if verbose:
                print(f"[LLM] Enabling {llm} enhancement...")

            from .ai_analysis.llm_analyzer import LLMAnalyzer

            # If caller provided --llm-model, set it in the environment so
            # LLMAnalyzer._call_anthropic/_call_openai can pick it up.
            # We restore the original value afterwards.
            if llm_model:
                os.environ["ROCINSIGHT_LLM_MODEL"] = llm_model

            _mdl = llm_model or os.environ.get("ROCINSIGHT_LLM_MODEL", "")
            _mdl_str = f" ({_mdl})" if _mdl else ""
            print(
                f"  Contacting {llm}{_mdl_str} for trace analysis — please wait...",
                file=sys.stderr,
                flush=True,
            )

            # Initialize LLM analyzer
            analyzer = LLMAnalyzer(
                provider=llm,
                api_key=llm_api_key,
                verbose=verbose,
                thinking_budget_tokens=llm_thinking,
            )

            # Prepare data for LLM
            analysis_data = {
                "gpu": {"name": "AMD GPU", "arch": "unknown"},  # TODO: Extract from DB
                "execution_breakdown": {
                    "kernel_time_pct": time_breakdown.get("kernel_percent", 0),
                    "memcpy_time_pct": time_breakdown.get("memcpy_percent", 0),
                    "api_overhead_pct": time_breakdown.get("overhead_percent", 0),
                },
                "kernels": [
                    {
                        "name": h.get("name", "unknown"),
                        "dispatch_count": h.get("calls", 0),
                        "pct_total_time": h.get("percent_of_total", 0),
                        "avg_duration_ns": h.get("avg_duration", 0),
                    }
                    for h in hotspots[:5]  # Top 5 kernels
                ],
                "memory_ops": {
                    direction: {
                        "count": data.get("count", 0),
                        "total_bytes": data.get("total_bytes", 0),
                        "bandwidth_gbps": data.get("bandwidth_bytes_per_sec", 0) / 1e9,
                    }
                    for direction, data in memory_analysis.items()
                },
                "has_counters": bool(hardware_counters),
                "has_pc_sampling": False,
            }

            # Build analysis context for guide filtering
            from .ai_analysis.llm_analyzer import AnalysisContext as _AnalysisContext

            _has_ctr = bool(hardware_counters and hardware_counters.get("has_counters"))
            _summary = _build_summary(time_breakdown, hotspots, _has_ctr)
            _llm_ctx = _AnalysisContext(
                tier=2 if _has_ctr else 1,
                has_counters=_has_ctr,
                bottleneck_type=_summary.get("primary_bottleneck"),
                gpu_arch=None,  # reserved for future per-GPU filtering
                custom_prompt=prompt,
            )

            # Get LLM enhancement
            llm_explanation = analyzer.analyze_with_llm(
                analysis_data=analysis_data,
                custom_prompt=prompt,
                context=_llm_ctx,
            )

            # Append LLM explanation to output
            if output_format == "text":
                output += "\n\n" + "=" * 80 + "\n"
                output += (
                    "AI-ENHANCED EXPLANATION (powered by {})".format(llm.upper()).center(
                        80
                    )
                    + "\n"
                )
                output += "=" * 80 + "\n\n"
                output += llm_explanation
                output += "\n\n" + "=" * 80 + "\n"
            elif output_format == "json":
                # Parse JSON and add LLM explanation
                import json

                try:
                    output_dict = json.loads(output)
                    output_dict["llm_enhanced_explanation"] = llm_explanation
                    output = json.dumps(output_dict, indent=2)
                except (json.JSONDecodeError, ValueError, KeyError) as _je:
                    print(
                        f"Warning: Could not embed LLM explanation in JSON output: {_je}",
                        file=sys.stderr,
                    )

            if verbose:
                print("[LLM] Enhancement complete")

        except Exception as e:
            # Always show LLM failures on console (even without --verbose)
            error_msg = f"⚠️  LLM enhancement failed: {e}"
            print(error_msg, file=sys.stderr)

            # Also add to output file
            warning_msg = (
                f"\n\n{error_msg}\n(Analysis completed with local results only)\n"
            )
            if output_format == "text":
                output += warning_msg

            # Show full traceback only in verbose mode
            if verbose:
                import traceback

                traceback.print_exc()

        finally:
            # Restore the ROCINSIGHT_LLM_MODEL env var to its previous state
            if llm_model:
                if _prev_model_env is None:
                    os.environ.pop("ROCINSIGHT_LLM_MODEL", None)
                else:
                    os.environ["ROCINSIGHT_LLM_MODEL"] = _prev_model_env

    return output


def _resolve_api_key(provider: Optional[str], explicit_key: Optional[str]) -> Optional[str]:
    """Return the right API key for *provider*, preferring the explicit CLI value.

    When the user passes ``--llm openai`` we must read ``OPENAI_API_KEY`` and
    ignore ``ANTHROPIC_API_KEY`` even if both are exported.  Passing a raw
    explicit key always wins over any environment variable.
    """
    if explicit_key:
        return explicit_key
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    # private / local providers: caller handles their own key lookup
    return None


def _call_llm_for_code(
    provider: str,
    api_key: Optional[str],
    model: Optional[str],
    prompt: str,
) -> str:
    """Call Anthropic or OpenAI to generate code-change suggestions."""
    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No Anthropic API key. Set ANTHROPIC_API_KEY or pass --llm-api-key."
            )
        use_model = model or os.environ.get("ROCINSIGHT_LLM_MODEL", "claude-sonnet-4-20250514")
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=use_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    elif provider in ("openai", "gpt"):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No OpenAI API key. Set OPENAI_API_KEY or pass --llm-api-key."
            )
        use_model = model or os.environ.get("ROCINSIGHT_LLM_MODEL", "gpt-4-turbo-preview")
        client = openai.OpenAI(api_key=key)
        try:
            resp = client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4096,
            )
        except Exception:
            resp = client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")


def _apply_code_change_interactive(
    rec: Dict[str, Any],
    source_dir: str,
    llm_provider: Optional[str],
    llm_api_key: Optional[str],
    llm_model: Optional[str],
    colors: Dict[str, str],
) -> None:
    """Walk the user through applying a code-change recommendation."""
    _os = os  # alias to keep existing _os.path.* calls working
    import glob as _glob
    import difflib
    import shutil

    C = colors["C"]
    G = colors["G"]
    Y = colors["Y"]
    R = colors["R"]
    DIM = colors["DIM"]
    N = colors["N"]

    cat = rec.get("category", "")
    issue = rec.get("issue", "")
    suggestion = rec.get("suggestion", "")
    actions = rec.get("actions", [])
    impact = rec.get("estimated_impact", "")

    # -- Show recommendation details ------------------------------------------
    print(f"\n{C}{'─' * 80}{N}")
    print(f"{C}  Code Change Recommendation: {cat}{N}")
    print(f"{C}{'─' * 80}{N}")
    print(f"\n  {Y}Issue:{N}      {issue}")
    print(f"  {Y}Suggestion:{N} {suggestion}")
    if actions:
        print(f"\n  {Y}Required Changes:{N}")
        for i, action in enumerate(actions, 1):
            print(f"    {i}. {action}")
    if impact:
        print(f"\n  {Y}Estimated Impact:{N} {impact}")
    print()

    if not source_dir:
        print(f"  {DIM}Tip: run with --source-dir <path> to enable AI code editing.{N}\n")
        return

    # -- Find GPU source files ------------------------------------------------
    source_files: List[str] = []
    for ext in ("*.hip", "*.cpp", "*.cu", "*.cuh", "*.h"):
        source_files.extend(
            _glob.glob(_os.path.join(source_dir, "**", ext), recursive=True)
        )
    source_files = [f for f in source_files if _os.path.isfile(f)]

    if not source_files:
        print(f"  {DIM}No GPU source files found in {source_dir}/{N}\n")
        return

    # -- Auto-detect LLM provider from environment if not explicitly set ------
    if not llm_provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            llm_provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            llm_provider = "openai"

    # -- No LLM configured: show manual steps and offer $EDITOR ---------------
    if not llm_provider:
        print(
            f"  {DIM}To enable AI code editing, set ANTHROPIC_API_KEY (or OPENAI_API_KEY) in your"
            f" environment, or pass --llm anthropic to ROCInsight analyze.{N}"
        )
        print(f"\n  {Y}Manual steps:{N}")
        for i, action in enumerate(actions, 1):
            print(f"    {i}. {action}")
        editor = _os.environ.get("EDITOR", "")
        if editor and source_files:
            try:
                ans = input(f"\n  Open source files in {editor}? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "n"
            if ans in ("y", "yes"):
                import subprocess

                if not shutil.which(editor):
                    print(f"  {Y}Editor '{editor}' not found on PATH.{N}")
                else:
                    subprocess.run([editor] + source_files[:3])
        print()
        return

    # -- Ask user before invoking LLM ----------------------------------------
    try:
        ans = (
            input(
                f"  {Y}Would you like the AI to apply this change to your source code? [y/N]: {N}"
            )
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if ans not in ("y", "yes"):
        print()
        return

    # -- Read source files ----------------------------------------------------
    MAX_FILES = 5
    MAX_FILE_SIZE = 50_000  # bytes per file

    print(f"\n  {DIM}Reading source files...{N}")
    file_contents: Dict[str, str] = {}
    for fpath in source_files[:MAX_FILES]:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                file_contents[fpath] = fh.read(MAX_FILE_SIZE)
        except OSError:
            pass

    if not file_contents:
        print(f"  {R}Could not read source files.{N}\n")
        return

    # -- Build LLM prompt -----------------------------------------------------
    files_text = "\n\n".join(
        f"=== {_os.path.relpath(fp, source_dir)} ===\n{content}"
        for fp, content in file_contents.items()
    )
    changes_text = "\n".join(f"- {a}" for a in actions)

    llm_prompt = (
        "You are a GPU performance optimization expert. The following GPU source files "
        "have a performance issue that needs to be fixed.\n\n"
        f"ISSUE: {issue}\n"
        f"SUGGESTION: {suggestion}\n"
        f"REQUIRED CHANGES:\n{changes_text}\n\n"
        f"SOURCE FILES:\n{files_text}\n\n"
        "OUTPUT INSTRUCTIONS:\n"
        "For each file that needs modification, output EXACTLY this format:\n"
        "MODIFY_FILE: <relative_filename>\n"
        "<<<ORIGINAL\n"
        "<exact original code section to replace — copy verbatim from the source>\n"
        "ORIGINAL\n"
        "<<<REPLACEMENT\n"
        "<new replacement code>\n"
        "REPLACEMENT\n\n"
        "Only output sections that need to change. Be precise — the ORIGINAL block must "
        "match exactly what appears in the file (used for find-and-replace). "
        "If no changes are needed, output: NO_CHANGES_NEEDED"
    )

    print(f"  {DIM}Calling {llm_provider} for code change suggestions...{N}")

    try:
        llm_response = _call_llm_for_code(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model,
            prompt=llm_prompt,
        )
    except Exception as exc:
        print(f"  {R}LLM error: {exc}{N}\n")
        return

    if "NO_CHANGES_NEEDED" in llm_response:
        print(f"  {G}AI analysis: no code changes are needed for this issue.{N}\n")
        return

    # -- Parse MODIFY_FILE blocks ---------------------------------------------
    patches: List[tuple] = []
    pattern = re.compile(
        r"MODIFY_FILE:\s*(\S+)\s*<<<ORIGINAL\n(.*?)ORIGINAL\s*<<<REPLACEMENT\n(.*?)REPLACEMENT",
        re.DOTALL,
    )
    for m in pattern.finditer(llm_response):
        rel_path = m.group(1).strip()
        original = m.group(2).strip()
        replacement = m.group(3).strip()
        abs_path = _os.path.join(source_dir, rel_path)
        # Guard against path traversal (e.g. rel_path = "../../etc/passwd")
        _resolved = _os.path.realpath(abs_path)
        _src_resolved = _os.path.realpath(source_dir)
        if (
            not _resolved.startswith(_src_resolved + _os.sep)
            and _resolved != _src_resolved
        ):
            continue  # reject: path escapes source_dir
        if _os.path.isfile(abs_path) and abs_path in file_contents:
            patches.append((abs_path, rel_path, original, replacement))

    if not patches:
        print(f"  {Y}AI did not produce actionable code changes.{N}")
        print(f"  {DIM}Raw AI response (first 20 lines):{N}")
        for line in llm_response.splitlines()[:20]:
            print(f"    {DIM}{line}{N}")
        print()
        return

    # -- Show unified diff ----------------------------------------------------
    print(f"\n{C}{'─' * 80}{N}")
    print(f"{C}  Proposed changes:{N}")
    print(f"{C}{'─' * 80}{N}")

    valid_patches: List[tuple] = []
    for abs_path, rel_path, original, replacement in patches:
        orig_content = file_contents[abs_path]
        if original not in orig_content:
            print(f"\n  {R}✗ Could not locate original code in {rel_path} — skipping.{N}")
            continue
        new_content = orig_content.replace(original, replacement, 1)
        diff = list(
            difflib.unified_diff(
                orig_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
                n=3,
            )
        )
        print(f"\n  File: {rel_path}")
        for line in diff[:80]:
            if line.startswith("+") and not line.startswith("+++"):
                print(f"  {G}{line.rstrip()}{N}")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"  {R}{line.rstrip()}{N}")
            elif line.startswith("@@"):
                print(f"  {C}{line.rstrip()}{N}")
            else:
                print(f"  {DIM}{line.rstrip()}{N}")
        if len(diff) > 80:
            print(f"  {DIM}  ... ({len(diff) - 80} more lines){N}")
        valid_patches.append((abs_path, rel_path, orig_content, new_content))

    if not valid_patches:
        print()
        return

    print()
    try:
        ans = input(f"  {Y}Apply these changes? [y/N]: {N}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if ans not in ("y", "yes"):
        print(f"  {DIM}Changes not applied.{N}\n")
        return

    # -- Apply with backup ----------------------------------------------------
    applied = 0
    for abs_path, rel_path, orig_content, new_content in valid_patches:
        backup_path = abs_path + ".rocinsight.bak"
        try:
            import shutil
            shutil.copy2(abs_path, backup_path)
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(new_content)
            print(
                f"  {G}✓ Applied: {rel_path}  (backup: {_os.path.basename(backup_path)}){N}"
            )
            applied += 1
        except OSError as exc:
            print(f"  {R}✗ Failed to write {rel_path}: {exc}{N}")

    if applied:
        print(
            f"\n  {G}✓ {applied} file(s) modified. Rebuild your application to test.{N}\n"
        )
        return True
    else:
        print(f"  {Y}No files were modified.{N}\n")
        return False


def _get_app_path_from_db(database_path: str) -> str:
    """
    Extract the profiled application's executable path from a rocpd database.

    rocprofv3 writes the process command into rocpd_info_process_<uuid>.command.
    Returns the path string, or "" if the database cannot be read or has no entry.
    """
    if not database_path:
        return ""
    try:
        import sqlite3 as _sqlite3

        with _sqlite3.connect(database_path) as con:
            # Find all rocpd_info_process_* tables
            tables = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_info_process_%'"
            ).fetchall()
            for (tname,) in tables:
                row = con.execute(
                    f'SELECT command FROM "{tname}" WHERE command IS NOT NULL LIMIT 1'
                ).fetchone()
                if row and row[0]:
                    return row[0].strip()
    except Exception:
        pass
    return ""


def _run_interactive_session(
    recommendations: List[Dict[str, Any]],
    tier0_result: Optional[Any] = None,
    database_path: str = "",
    source_dir: str = "",
    llm_provider: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_local: Optional[str] = None,
    llm_local_model: Optional[str] = None,
    resume_session: Optional[str] = None,
) -> None:
    """Thin shim: delegates to InteractiveSession in ai_analysis/interactive.py."""
    from rocinsight.ai_analysis.interactive import InteractiveSession, SessionStore

    InteractiveSession(
        source_dir=source_dir,
        tier0_result=tier0_result,
        recommendations=recommendations,
        database_path=database_path,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_local=llm_local,
        llm_local_model=llm_local_model,
        session_store=SessionStore(),
        resume_session_id=resume_session,
    ).run()


def add_args(parser: argparse.ArgumentParser):
    """
    Add command-line arguments for AI analysis.

    Args:
        parser: Argument parser to add arguments to

    Returns:
        Function to process parsed arguments
    """
    analysis_options = parser.add_argument_group("Analysis options")

    analysis_options.add_argument(
        "--source-dir",
        type=str,
        default=None,
        dest="source_dir",
        help=(
            "Path to GPU application source directory for Tier 0 static analysis. "
            "Scans .hip/.cpp/.cu files and generates a profiling plan. "
            "Can be used alone (no -i required) or alongside -i for combined analysis."
        ),
    )

    analysis_options.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom analysis prompt/question to guide analysis (e.g., 'Why is my matmul kernel slow?')",
    )

    analysis_options.add_argument(
        "--top-kernels",
        type=int,
        default=10,
        help="Number of top kernels to analyze (default: 10)",
    )

    analysis_options.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "markdown", "webview"],
        default="text",
        help="Output format: text, json, markdown, or webview (default: text). "
        "File extension is set automatically: .txt, .json, .md, .html",
    )

    analysis_options.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Minimum kernel duration threshold in microseconds (filter out short kernels)",
    )

    # LLM Enhancement Options
    llm_options = parser.add_argument_group(
        "LLM enhancement options (optional)",
        "Enable natural language explanations via Anthropic Claude or OpenAI GPT. "
        "Requires API key - see https://console.anthropic.com/ or https://platform.openai.com/api-keys",
    )

    llm_options.add_argument(
        "--llm",
        type=str,
        choices=["anthropic", "openai", "claude-code"],
        default=None,
        help=(
            "Enable LLM-powered analysis enhancement. "
            "'anthropic' uses the Anthropic API (requires ANTHROPIC_API_KEY). "
            "'openai' uses the OpenAI API (requires OPENAI_API_KEY). "
            "'claude-code' uses the Claude Code CLI installed on this machine — "
            "no API key needed, uses existing Claude Code credentials. "
            "Local analysis always runs first; LLM provides additional natural language insights."
        ),
    )

    llm_options.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key for LLM provider. Alternatively, set environment variable: "
        "ANTHROPIC_API_KEY for Anthropic Claude, or OPENAI_API_KEY for OpenAI GPT. "
        "Example: --llm anthropic --llm-api-key sk-ant-... "
        "Or: export ANTHROPIC_API_KEY='sk-ant-...' && rocinsight analyze --llm anthropic",
    )

    llm_options.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override the LLM model name. Defaults to claude-sonnet-4-20250514 for Anthropic "
        "and gpt-4-turbo-preview for OpenAI. Can also be set via ROCINSIGHT_LLM_MODEL environment "
        "variable (--llm-model takes precedence). "
        "Examples: --llm-model claude-opus-4-6, --llm-model gpt-4o",
    )

    llm_options.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging (shows LLM API calls, reference guide loading, etc.)",
    )

    analysis_options.add_argument(
        "--att-dir",
        type=str,
        default=None,
        dest="att_dir",
        help=(
            "Path to directory containing ATT stats_*.csv files from rocprofv3 --att. "
            "Enables Tier 3 Advanced Thread Trace analysis: per-instruction stall ratios "
            "and bottleneck classification (VMEM latency, LDS bank conflict, dependency chains, "
            "branch divergence). Requires rocprof-trace-decoder to be installed. "
            "Example: --att-dir ./att_output"
        ),
    )

    analysis_options.add_argument(
        "--interactive",
        "-I",
        metavar="RUN_COMMAND",
        type=str,
        default=None,
        dest="interactive",
        help=(
            "Launch the 7-phase interactive profiling + optimization workflow. "
            "RUN_COMMAND is the full command used to run your GPU application. "
            'Example: --interactive "./my_gpu_app --batch-size 64". '
            "The workflow automatically wraps your command with rocprofv3, collects "
            "a trace, analyzes bottlenecks with AI, and offers to apply optimizations."
        ),
    )

    analysis_options.add_argument(
        "--resume-session",
        nargs="?",
        const="latest",
        default=None,
        dest="resume_session",
        help=(
            "Resume a previous workflow session. Accepts a session ID, an absolute "
            "path to a workflow_*.json file, or 'latest' (default when flag has no argument). "
            "Example: --resume-session  OR  --resume-session workflow_2026-03-10_14-23_myapp"
        ),
    )

    llm_options.add_argument(
        "--llm-thinking",
        metavar="TOKENS",
        type=int,
        default=None,
        dest="llm_thinking",
        help=(
            "Enable extended thinking for deeper LLM analysis. Specify the thinking "
            "budget in tokens (e.g. --llm-thinking 8000). Only available with the "
            "Anthropic provider and compatible models (claude-opus-4, "
            "claude-sonnet-4-5, claude-3-7-sonnet). Adds latency but improves "
            "analysis quality for complex traces with multiple interacting "
            "bottlenecks. Requires --llm anthropic. Also configurable via the "
            "ROCINSIGHT_LLM_THINKING environment variable (set to token count)."
        ),
    )

    llm_options.add_argument(
        "--llm-local",
        type=str,
        choices=["ollama"],
        default=None,
        dest="llm_local",
        help=(
            "Local LLM provider for Stage 1 source summarization (before online LLM). "
            "Choices: 'ollama'. Requires Ollama running at localhost:11434. "
            "Set ROCINSIGHT_LLM_LOCAL_URL to override endpoint."
        ),
    )

    llm_options.add_argument(
        "--llm-local-model",
        type=str,
        default=None,
        dest="llm_local_model",
        help=(
            "Model name for local LLM (default: codellama:13b). "
            "Can also be set via ROCINSIGHT_LLM_LOCAL_MODEL environment variable."
        ),
    )

    def process_args(input: RocpdImportData, args: argparse.Namespace):
        """Process and return valid arguments as dictionary."""
        valid_args = [
            "source_dir",
            "att_dir",
            "prompt",
            "top_kernels",
            "format",
            "min_duration",
            "llm",
            "llm_api_key",
            "llm_model",
            "llm_thinking",
            "verbose",
            "interactive",
            "resume_session",
            "llm_local",
            "llm_local_model",
        ]
        ret = {}
        for itr in valid_args:
            if hasattr(args, itr):
                val = getattr(args, itr)
                if val is not None:
                    ret[itr] = val
        # Convert min_duration from microseconds to nanoseconds
        if "min_duration" in ret:
            ret["min_duration"] = ret["min_duration"] * 1000
        return ret

    return process_args


def execute(
    input: Optional[RocpdImportData],
    config: Optional[output_config.output_config] = None,
    **kwargs: Any,
) -> Optional[RocpdImportData]:
    """
    Execute AI analysis on rocpd database and/or source directory.

    Args:
        input: RocpdImportData object with database connection, or None for source-only mode
        config: Optional output configuration
        **kwargs: Analysis parameters (may include source_dir for Tier 0)

    Returns:
        The input RocpdImportData object (for chaining), or None in source-only mode
    """
    # Update config if provided
    if config is not None:
        config = config.update(**kwargs)
    else:
        config = output_config.output_config(**kwargs)

    # Get database path for display
    database_path = ""
    if input is not None and hasattr(input, "_paths") and input._paths:
        database_path = str(
            input._paths[0] if isinstance(input._paths, list) else input._paths
        )

    # Pop interactive before passing to analyze_performance (it doesn't accept it)
    interactive = kwargs.pop("interactive", None)

    # 7-phase workflow mode: triggered when --interactive is provided with a RUN_COMMAND
    if interactive and isinstance(interactive, str):
        from rocinsight.ai_analysis.interactive import WorkflowSession  # type: ignore[import]

        source_paths: list = []
        source_dir = kwargs.get("source_dir")
        if source_dir:
            source_paths.append(source_dir)
        ws = WorkflowSession(
            app_command=interactive,
            source_paths=source_paths,
            llm_provider=kwargs.get("llm"),
            llm_api_key=_resolve_api_key(kwargs.get("llm"), kwargs.get("llm_api_key")),
            llm_model=kwargs.get("llm_model"),
            resume_session=kwargs.get("resume_session"),
        )
        ws.run()
        return input

    # Map 'format' CLI key -> 'output_format' parameter expected by analyze_performance
    if "format" in kwargs:
        kwargs["output_format"] = kwargs.pop("format")

    # In interactive mode: skip the upfront LLM call entirely — the user will
    # trigger LLM requests explicitly via [p] and [o] inside the session.
    # Save credentials first so _run_interactive_session can still use them.
    _interactive_llm_provider = kwargs.get("llm")
    _interactive_llm_api_key = _resolve_api_key(
        _interactive_llm_provider, kwargs.get("llm_api_key")
    )
    _interactive_llm_model = kwargs.get("llm_model")
    if interactive:
        kwargs.pop("llm", None)
        kwargs.pop("llm_model", None)
        kwargs.pop("llm_api_key", None)
        kwargs.pop("llm_thinking", None)

    # Collect structured results so interactive mode can build its command menu
    result_store: Dict[str, Any] = {}

    # Run analysis
    output = analyze_performance(
        connection=input,
        database_path=database_path,
        _collect_result=result_store,
        **kwargs,
    )

    # Determine file extension based on output format
    _ext_map = {"json": ".json", "markdown": ".md", "webview": ".html", "text": ".txt"}
    _fmt = kwargs.get("output_format", "text")
    _ext = _ext_map.get(_fmt, ".txt")

    # Handle output
    # When -d is given without -o, auto-generate a default filename from the db name.
    if config and config.output_path and not config.output_file:
        if database_path:
            config.output_file = os.path.splitext(os.path.basename(database_path))[0]
        else:
            config.output_file = "analysis"

    if config and config.output_file and config.output_path:
        base = config.output_file
        # Append the format extension if the base name doesn't already have it
        if not base.endswith(_ext):
            base = base + _ext
        output_file = os.path.join(config.output_path, base)
        os.makedirs(config.output_path, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(output)
        print(f"Analysis written to: {output_file}")
        if _fmt == "text":
            print(
                "Tip: use --format webview for an interactive HTML report, "
                "--format json for machine-readable output, "
                "or --format markdown for Markdown."
            )
    else:
        print(output)

    # -- Interactive mode -----------------------------------------------------
    if interactive:
        _run_interactive_session(
            recommendations=result_store.get("recommendations", []),
            tier0_result=result_store.get("tier0_result"),
            database_path=result_store.get("database_path", database_path),
            source_dir=kwargs.get("source_dir", ""),
            llm_provider=_interactive_llm_provider,
            llm_api_key=_interactive_llm_api_key,
            llm_model=_interactive_llm_model,
            llm_local=kwargs.get("llm_local"),
            llm_local_model=kwargs.get("llm_local_model"),
            resume_session=kwargs.get("resume_session"),
        )

    return input


def main(argv=None) -> int:
    """
    Main entry point for standalone execution.

    Args:
        argv: Command-line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        prog="rocpd.analyze",
        description="AI-powered performance analysis for GPU traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=str,
        required=True,
        help="Input rocpd database file(s)",
    )

    # Add output config args
    output_config.add_args(parser)

    # Add analysis args
    process_args = add_args(parser)

    # Parse arguments
    args = parser.parse_args(argv)

    try:
        # Create database connection
        input_data = RocpdImportData(args.input)

        # Process arguments
        analysis_args = process_args(input_data, args)

        # Execute analysis
        execute(input_data, **analysis_args)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
