"""Analysis decision-maker (Layer 1) — classifies bottleneck from trace facts.

Responsibilities:
- Call analysis tools (time_breakdown, hotspots) to collect metrics.
- Classify bottleneck via pure-rule tool (bottleneck.classify_from_metrics).
- Let the LLM (if enabled) refine or flag edge cases; never override the
  rule-verdict in air-gap mode.
- Return AnalysisOutput (frozen) to Root.

Tool allowlist (exactly 5 per spec §2 cap):
  analysis.time_breakdown, analysis.hotspots,
  bottleneck.classify_from_metrics, roofline.classify,
  counters.validate_for_gpu
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from perfxpert.agents import schemas

_log = logging.getLogger(__name__)
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.tools import arch as arch_tools
from perfxpert.tools import bottleneck, counters, roofline

# trace_analysis — delegates to perfxpert.analyze helper functions
try:
    from perfxpert.tools import trace_analysis  # type: ignore
except ImportError:
    # fallback: adapter around perfxpert.analyze helper functions
    class _TraceAnalysisAdapter:
        @staticmethod
        def time_breakdown(db_path: str) -> Dict[str, Any]:
            """Compute time breakdown from database."""
            from perfxpert.connection import PerfxpertConnection
            from perfxpert.analyze import compute_time_breakdown

            conn = PerfxpertConnection(db_path)
            breakdown = compute_time_breakdown(conn)

            # Map analyze.py keys to agentic schema
            return {
                "kernel_pct": breakdown.get("kernel_percent", 0.0),
                "memcpy_pct": breakdown.get("memcpy_percent", 0.0),
                "api_pct": breakdown.get("overhead_percent", 0.0),
                "idle_pct": 0.0,  # computed as remainder if needed
                "counter_data_available": _check_counters_available(db_path),
            }

        @staticmethod
        def hotspots(db_path: str, top_n: int = 10, min_duration: float = 0.0) -> list:
            """Identify top kernels by execution time."""
            from perfxpert.connection import PerfxpertConnection
            from perfxpert.analyze import identify_hotspots

            conn = PerfxpertConnection(db_path)
            try:
                hotspots = identify_hotspots(
                    conn, top_n=top_n, min_duration=min_duration
                )
            except TypeError:
                # Backward-compat for tests and older shims that still expose
                # ``identify_hotspots(conn, top_n=...)`` without the
                # ``min_duration`` kwarg.
                hotspots = identify_hotspots(conn, top_n=top_n)

            # Convert to agentic schema if needed
            return hotspots or []

    def _check_counters_available(db_path: str) -> bool:
        """Check if hardware counters (pmc_events) are available in the database.

        pmc_events is a VIEW in rocpd databases (not a TABLE), so we check
        sqlite_master for both 'table' and 'view' types.
        """
        try:
            from perfxpert.connection import PerfxpertConnection, execute_statement

            conn = PerfxpertConnection(db_path)
            # pmc_events is a VIEW in rocpd databases — check both table and view
            tables_query = (
                "SELECT name FROM sqlite_master "
                "WHERE (type='table' OR type='view') AND name='pmc_events'"
            )
            result = execute_statement(conn, tables_query).fetchone()
            has_pmc = result is not None

            if has_pmc:
                # Check if pmc_events has any rows with actual counter data
                count_query = "SELECT COUNT(*) FROM pmc_events LIMIT 1"
                count_result = execute_statement(conn, count_query).fetchone()
                return bool(count_result and count_result[0] > 0)
            return False
        except Exception:
            return False

    trace_analysis = _TraceAnalysisAdapter()  # type: ignore[misc,assignment]


_FENCE_PATH = Path(__file__).parent / "fence" / "analysis.md"
_VALID_BOTTLENECKS = {
    "compute",
    "memory_transfer",
    "latency",
    "api_overhead",
    "mixed",
    "data_insufficient",
}


def build_analysis_agent() -> Agent:
    tools = [
        ToolBinding(name="analysis.time_breakdown", fn=trace_analysis.time_breakdown),
        ToolBinding(name="analysis.hotspots", fn=trace_analysis.hotspots),
        ToolBinding(name="bottleneck.classify_from_metrics", fn=bottleneck.classify_from_metrics),
        ToolBinding(name="roofline.classify", fn=roofline.classify),
        ToolBinding(name="counters.validate_for_gpu", fn=counters.validate_for_gpu),
    ]
    return Agent(
        name="Analysis",
        layer=1,
        fence_path=str(_FENCE_PATH) if _FENCE_PATH.exists() else None,
        input_schema=schemas.AnalysisInput,
        output_schema=schemas.AnalysisOutput,
        tools=tools,
        allowed_handoffs=[],   # Layer-1 returns to Root
        token_budget=4096,
    )


def _extract_hw_metrics(db: str) -> Dict[str, Any]:
    """Extract hardware counter-derived metrics from the database.

    Returns a dict with keys matching YAML signature metric names.
    All values are None when counter data is unavailable — the classifier
    must treat None as "unknown", never as 0.0.
    """
    try:
        from perfxpert.connection import PerfxpertConnection, execute_statement

        conn = PerfxpertConnection(db)

        # Check if pmc_events has any rows
        check = execute_statement(conn, "SELECT COUNT(*) FROM pmc_events LIMIT 1").fetchone()
        if not check or check[0] == 0:
            return {}  # Empty dict — no counter data at all

        # Pull aggregate counter values
        counter_query = """
        SELECT counter_name, AVG(counter_value) as avg_val, SUM(counter_value) as total_val
        FROM pmc_events
        GROUP BY counter_name
        """
        rows = execute_statement(conn, counter_query).fetchall()
        counters = {r[0]: {"avg": r[1], "total": r[2]} for r in rows}
        gfx_id = _detect_gfx_id_from_connection(conn, execute_statement)
        arch_specs = None
        if gfx_id:
            try:
                arch_specs = arch_tools.lookup_peaks(gfx_id)
            except KeyError:
                arch_specs = None

        hw: Dict[str, Any] = {}

        # --- valu_util_pct: SQ_INSTS_VALU / (SQ_WAVES * wavefront_size * cycles_per_instr)
        # Approximation: SQ_INSTS_VALU / GRBM_GUI_ACTIVE as fraction of peak VALU issue slots.
        if "SQ_INSTS_VALU" in counters and "GRBM_GUI_ACTIVE" in counters:
            sq_valu = counters["SQ_INSTS_VALU"]["avg"]
            grbm_active = counters["GRBM_GUI_ACTIVE"]["avg"]
            if grbm_active and grbm_active > 0:
                # Each active cycle can issue up to 4 VALU instructions (4 SIMDs/CU)
                hw["valu_util_pct"] = min(1.0, sq_valu / (grbm_active * 4.0))
            else:
                hw["valu_util_pct"] = None
        else:
            hw["valu_util_pct"] = None

        # --- mfma_util_pct: SQ_INSTS_VALU_MFMA / GRBM_GUI_ACTIVE
        if "SQ_INSTS_VALU_MFMA" in counters and "GRBM_GUI_ACTIVE" in counters:
            sq_mfma = counters["SQ_INSTS_VALU_MFMA"]["avg"]
            grbm_active = counters["GRBM_GUI_ACTIVE"]["avg"]
            if grbm_active and grbm_active > 0:
                hw["mfma_util_pct"] = min(1.0, sq_mfma / grbm_active)
            else:
                hw["mfma_util_pct"] = None
        else:
            hw["mfma_util_pct"] = None

        # --- gpu_util_pct: GRBM_GUI_ACTIVE / GRBM_COUNT
        if "GRBM_GUI_ACTIVE" in counters and "GRBM_COUNT" in counters:
            grbm_active = counters["GRBM_GUI_ACTIVE"]["avg"]
            grbm_count = counters["GRBM_COUNT"]["avg"]
            if grbm_count and grbm_count > 0:
                hw["gpu_util_pct"] = min(1.0, grbm_active / grbm_count)
            else:
                hw["gpu_util_pct"] = None
        else:
            hw["gpu_util_pct"] = None

        # --- occupancy_pct + avg_waves_per_cu
        # SQ_WAVES is the total wavefronts dispatched per kernel invocation, not
        # the in-flight concurrency. Accurate occupancy requires SQ_BUSY_CYCLES or
        # per-cycle active-wave sampling which is not available from basic --pmc runs.
        #
        # Without a dedicated occupancy counter, we set both to None so the
        # latency signature (avg_waves_per_cu < 16) does not fire incorrectly on
        # compute-bound kernels that simply have low wavefront count.
        hw["avg_waves_per_cu"] = None
        hw["occupancy_pct"] = None

        # --- hbm_bw_utilization: (FETCH_SIZE + WRITE_SIZE) / (peak_BW * duration)
        # Arch-sensitive. If the DB does not identify a supported GPU, leave the
        # metric unknown instead of silently assuming MI300X.
        peak_bw_bytes_per_ns = None
        if arch_specs:
            peak_bw_tbs = float(arch_specs.get("memory_bandwidth_tbs") or 0.0)
            if peak_bw_tbs > 0:
                peak_bw_bytes_per_ns = peak_bw_tbs * 1e12 / 1e9
        if "FETCH_SIZE" in counters and "WRITE_SIZE" in counters and peak_bw_bytes_per_ns:
            fetch = counters["FETCH_SIZE"]["total"]
            write = counters["WRITE_SIZE"]["total"]
            # Get total trace duration from kernels table
            try:
                dur_row = execute_statement(
                    conn, "SELECT MAX(end) - MIN(start) FROM kernels"
                ).fetchone()
                trace_dur_ns = dur_row[0] if dur_row and dur_row[0] else 1
                total_bytes = (fetch or 0) + (write or 0)
                peak_bytes = peak_bw_bytes_per_ns * trace_dur_ns
                hw["hbm_bw_utilization"] = min(1.0, total_bytes / peak_bytes) if peak_bytes > 0 else None
            except Exception:
                hw["hbm_bw_utilization"] = None
        else:
            hw["hbm_bw_utilization"] = None

        # --- arithmetic_intensity_above_ridge / arithmetic_intensity_below_ridge
        ridge_point = None
        if gfx_id:
            try:
                ridge_point = arch_tools.lookup_ridge_point(gfx_id, dtype="fp32")
            except KeyError:
                ridge_point = None
        if ridge_point and "FETCH_SIZE" in counters and "WRITE_SIZE" in counters and (
            "SQ_INSTS_VALU" in counters or "SQ_INSTS_VALU_MFMA" in counters
        ):
            fetch_t = counters["FETCH_SIZE"]["total"] or 0
            write_t = counters["WRITE_SIZE"]["total"] or 0
            total_bytes_ai = fetch_t + write_t
            # Rough FLOPs: SQ_INSTS_VALU * wave_size * 2 (FMA) + MFMA ops
            wave_size = int((arch_specs or {}).get("wave_size") or 64)
            valu_t = counters.get("SQ_INSTS_VALU", {}).get("total", 0) or 0
            mfma_t = counters.get("SQ_INSTS_VALU_MFMA", {}).get("total", 0) or 0
            total_flops = valu_t * wave_size * 2 + mfma_t * 512  # MFMA legacy counter uses a coarse fp32-equivalent weight.
            if total_bytes_ai > 0:
                ai = total_flops / total_bytes_ai
                hw["arithmetic_intensity_above_ridge"] = 1 if ai > ridge_point else 0
                hw["arithmetic_intensity_below_ridge"] = 1 if ai <= ridge_point else 0
            else:
                hw["arithmetic_intensity_above_ridge"] = None
                hw["arithmetic_intensity_below_ridge"] = None
        else:
            hw["arithmetic_intensity_above_ridge"] = None
            hw["arithmetic_intensity_below_ridge"] = None

        return hw

    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        _log.exception(
            "_extract_hw_metrics: DB error reading pmc_events — user will see "
            "'data_insufficient'; real cause: %s", e
        )
        return {"_error": str(e)}  # marker so callers can surface the real reason
    except Exception:
        # Non-DB exceptions (e.g. import errors, attribute errors) are programmer
        # bugs — let them propagate so they are not silently swallowed.
        raise


def _detect_gfx_id_from_connection(conn: Any, execute_statement: Any) -> Optional[str]:
    try:
        row = execute_statement(
            conn,
            "SELECT name FROM rocpd_info_agent WHERE type='GPU' LIMIT 1",
        ).fetchone()
    except Exception:
        return None

    name = row[0] if row and row[0] else None
    if not name:
        return None

    match = re.search(r"gfx[0-9a-f]+", str(name))
    if not match:
        return None

    gfx_id = match.group(0)
    try:
        arch_tools.lookup_peaks(gfx_id)
    except KeyError:
        return None
    return gfx_id


def _extract_dispatch_metrics(db: str, hotspots: list) -> Dict[str, Any]:
    """Extract kernel dispatch count + average duration metrics.

    Returns keys: total_kernel_calls, avg_kernel_duration_us.
    Values are None when data is unavailable.
    On DB error the dict also includes ``_error`` so the caller can propagate
    the cause to the user instead of silently reporting data_insufficient.
    """
    try:
        from perfxpert.connection import PerfxpertConnection, execute_statement

        conn = PerfxpertConnection(db)
        row = execute_statement(
            conn, "SELECT COUNT(*), AVG(duration) FROM kernels"
        ).fetchone()
        if row:
            total_calls = row[0] or 0
            avg_dur_ns = row[1] or 0
            return {
                "total_kernel_calls": total_calls,
                "avg_kernel_duration_us": avg_dur_ns / 1000.0 if avg_dur_ns else None,
            }
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        _log.exception(
            "_extract_dispatch_metrics: DB error reading kernels table — "
            "dispatch metrics unavailable; real cause: %s", e
        )
        return {
            "total_kernel_calls": None,
            "avg_kernel_duration_us": None,
            "_error": str(e),
        }
    except FileNotFoundError:
        # DB file missing — tolerated (e.g. unit tests using fake.db path).
        return {"total_kernel_calls": None, "avg_kernel_duration_us": None}
    except Exception:
        raise
    return {"total_kernel_calls": None, "avg_kernel_duration_us": None}


def _collect_deterministic_metrics(
    db: str,
    top_n: int = 10,
    min_duration: float = 0.0,
) -> Dict[str, Any]:
    """Collect the rule-based metrics needed by the classifier.

    Exposed at module level for test injection. Tests can monkeypatch this function
    to stub out database access.

    The returned ``metrics_for_classifier`` dict uses None (not 0.0) for every key
    that cannot be computed due to missing counter data.  The downstream classifier
    treats None as "unknown / neutral" — it will NOT silently score a missing metric
    as 0.0 and return a misleading verdict.
    """
    try:
        breakdown = trace_analysis.time_breakdown(db)
        hotspots = trace_analysis.hotspots(db, top_n=top_n, min_duration=min_duration)
    except FileNotFoundError:
        # Database not found (common in unit tests with fake.db).
        # Return empty defaults so tests can mock this function.
        breakdown = {
            "kernel_pct": 0.0,
            "memcpy_pct": 0.0,
            "api_pct": 0.0,
            "idle_pct": 0.0,
            "counter_data_available": False,
        }
        hotspots = []

    counter_data_available = breakdown.get("counter_data_available", False)

    # Always populate the trace-derived metrics (available from any --sys-trace DB).
    # time_breakdown values are in 0–100 (percentage) scale; YAML thresholds use
    # 0.0–1.0 (fraction) scale.  Divide by 100 to normalize.
    raw_memcpy_pct = breakdown.get("memcpy_pct", 0.0) or 0.0
    raw_api_pct = breakdown.get("api_pct", 0.0) or 0.0
    m: Dict[str, Any] = {
        "memcpy_pct": raw_memcpy_pct / 100.0,
        "api_overhead_pct": raw_api_pct / 100.0,
    }

    # Dispatch-level metrics (from kernels table — available without --pmc)
    dispatch_metrics = _extract_dispatch_metrics(db, hotspots)
    # Surface any DB error from dispatch extraction — keep the _error key so
    # _collect_deterministic_metrics can propagate it to the caller.
    dispatch_error = dispatch_metrics.pop("_error", None)
    m.update(dispatch_metrics)

    if counter_data_available:
        # Populate hardware-counter derived metrics; each may still be None
        # if the specific counter was not collected in this run.
        hw = _extract_hw_metrics(db)
        hw_error = hw.pop("_error", None)
        m["valu_util_pct"] = hw.get("valu_util_pct")
        m["mfma_util_pct"] = hw.get("mfma_util_pct")
        m["arithmetic_intensity_above_ridge"] = hw.get("arithmetic_intensity_above_ridge")
        m["arithmetic_intensity_below_ridge"] = hw.get("arithmetic_intensity_below_ridge")
        m["occupancy_pct"] = hw.get("occupancy_pct")
        m["avg_waves_per_cu"] = hw.get("avg_waves_per_cu")
        m["gpu_util_pct"] = hw.get("gpu_util_pct")
        m["hbm_bw_utilization"] = hw.get("hbm_bw_utilization")
        m["no_dominant_bottleneck"] = None  # only set if classifier decides "mixed"
    else:
        hw_error = None
        # No PMC data: set all counter-derived keys to None so classifier
        # can distinguish "not measured" from "measured zero".
        m["valu_util_pct"] = None
        m["mfma_util_pct"] = None
        m["arithmetic_intensity_above_ridge"] = None
        m["arithmetic_intensity_below_ridge"] = None
        m["occupancy_pct"] = None
        m["avg_waves_per_cu"] = None
        m["gpu_util_pct"] = None
        m["hbm_bw_utilization"] = None
        m["no_dominant_bottleneck"] = None

    # Aggregate any DB errors so run_analysis can surface them to the user.
    db_error: Optional[str] = hw_error or dispatch_error or None

    return {
        "time_breakdown": breakdown,
        "hot_kernels": hotspots,
        "metrics_for_classifier": m,
        "counter_data_available": counter_data_available,
        "db_error": db_error,
    }


def _validated_bottleneck_type(value: Any) -> str:
    if value not in _VALID_BOTTLENECKS:
        raise ValueError(f"Unknown bottleneck type {value!r}")
    return value


def _trace_only_fallback_verdict(
    facts: Dict[str, Any],
    *,
    db_error: Optional[str],
) -> Dict[str, Any]:
    """Classify the small subset of trace-only cases that are unambiguous.

    We stay conservative: if the trace does not clearly show either a memcpy-
    dominated runtime split or classic launch-overhead symptoms, the result
    remains ``data_insufficient``.
    """
    if db_error:
        return {
            "type": "data_insufficient",
            "confidence": 0.0,
            "reasoning": (
                f"DB error prevented counter data extraction: {db_error}. "
                "PerfXpert cannot classify the bottleneck because the database "
                "could not be read reliably."
            ),
            "all_scores": {},
        }

    metrics = facts.get("metrics_for_classifier", {})
    memcpy_pct = float(metrics.get("memcpy_pct") or 0.0)
    api_pct = float(metrics.get("api_overhead_pct") or 0.0)
    avg_kernel_duration_us = metrics.get("avg_kernel_duration_us")
    total_kernel_calls = int(metrics.get("total_kernel_calls") or 0)

    if (
        api_pct > 0.15
        and avg_kernel_duration_us is not None
        and float(avg_kernel_duration_us) < 10.0
        and total_kernel_calls > 1000
    ):
        return {
            "type": "latency",
            "confidence": 0.75,
            "reasoning": (
                "Trace-only fallback: API overhead dominates and the trace shows "
                "many tiny kernels, which is a strong launch-overhead / latency signal."
            ),
            "all_scores": {"latency_trace_only": 0.75},
        }

    if memcpy_pct >= 0.50 and memcpy_pct >= api_pct + 0.20:
        confidence = 0.85 if memcpy_pct > 0.30 else 0.70
        return {
            "type": "memory_transfer",
            "confidence": confidence,
            "reasoning": (
                "Trace-only fallback: memcpy traffic clearly dominates runtime, "
                "so host-device transfer overhead is unambiguous even without "
                "PMC counters."
            ),
            "all_scores": {"memory_transfer_trace_only": confidence},
        }

    return {
        "type": "data_insufficient",
        "confidence": 0.0,
        "reasoning": (
            "No hardware counter data in this trace (profiled without --pmc). "
            "The available trace-only metrics do not point to an unambiguous "
            "memory-transfer or launch-overhead bottleneck, so PerfXpert will "
            "not guess."
        ),
        "all_scores": {},
    }


def _promote_sparse_compute_verdict(
    facts: Dict[str, Any],
    rule_verdict: Dict[str, Any],
) -> Dict[str, Any]:
    """Promote clear compute-bound cases when only a sparse PMC subset exists."""
    if rule_verdict.get("type") != "mixed" or not facts.get("counter_data_available"):
        return rule_verdict

    metrics = facts.get("metrics_for_classifier", {})
    time_breakdown = facts.get("time_breakdown", {})
    kernel_pct = float(time_breakdown.get("kernel_pct", 0.0) or 0.0) / 100.0
    memcpy_pct = float(metrics.get("memcpy_pct") or 0.0)
    api_pct = float(metrics.get("api_overhead_pct") or 0.0)
    gpu_util_pct = metrics.get("gpu_util_pct")

    if (
        kernel_pct > 0.70
        and memcpy_pct < 0.20
        and api_pct < 0.15
        and gpu_util_pct is not None
        and float(gpu_util_pct) > 0.80
    ):
        return {
            "type": "compute",
            "confidence": max(float(rule_verdict.get("confidence", 0.0)), 0.65),
            "reasoning": (
                "Sparse-counter fallback: the trace is kernel-dominant and the GPU "
                "is highly utilized even though detailed VALU/MFMA counters were "
                "not collected."
            ),
            "all_scores": {
                **(rule_verdict.get("all_scores") or {}),
                "compute_sparse_counter_fallback": 0.65,
            },
        }

    return rule_verdict


def _override_runtime_dominant_verdict(
    facts: Dict[str, Any],
    rule_verdict: Dict[str, Any],
) -> Dict[str, Any]:
    """Prefer the whole-workload bottleneck when runtime overhead dominates.

    Counter-derived compute signatures describe the hot kernel. They should not
    override the end-to-end classification when runtime/API overhead is the vast
    majority of wall-clock time. In that situation the user-visible bottleneck
    is launch/runtime latency, not kernel throughput.
    """
    metrics = facts.get("metrics_for_classifier", {})
    time_breakdown = facts.get("time_breakdown", {})
    api_pct = float(metrics.get("api_overhead_pct") or 0.0)
    memcpy_pct = float(metrics.get("memcpy_pct") or 0.0)
    kernel_pct = float(time_breakdown.get("kernel_pct", 0.0) or 0.0) / 100.0

    if api_pct > 0.50 and api_pct > kernel_pct and api_pct > memcpy_pct:
        return {
            "type": "latency",
            "confidence": max(float(rule_verdict.get("confidence", 0.0)), 0.75),
            "reasoning": (
                "Runtime/API overhead dominates end-to-end time, so the workload "
                "is latency-bound even if the hottest kernel itself is compute-efficient."
            ),
            "all_scores": {
                **(rule_verdict.get("all_scores") or {}),
                "runtime_overhead_dominance": 0.75,
            },
        }

    return rule_verdict


def run_analysis(
    payload: schemas.AnalysisInput,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
) -> schemas.AnalysisOutput:
    """Run Analysis for one trace database."""
    # Step 1: deterministic metric collection (always).
    try:
        facts = _collect_deterministic_metrics(
            payload.database_path,
            top_n=payload.top_kernels,
            min_duration=payload.min_duration,
        )
    except TypeError:
        # Backward-compat for older tests / shims that still monkeypatch
        # _collect_deterministic_metrics(db_path, top_n=...) without the
        # min_duration kwarg.
        facts = _collect_deterministic_metrics(
            payload.database_path,
            top_n=payload.top_kernels,
        )

    # Step 2: deterministic classifier verdict (always).
    #
    # If a DB error was recorded during metric extraction, surface it here so the
    # user sees WHY data is insufficient instead of just being told to re-profile.
    db_error = facts.get("db_error")
    if db_error:
        import sys
        print(
            f"\nERROR: Database query failed during metric extraction:\n  {db_error}\n"
            "PerfXpert cannot classify the bottleneck because the database could not be read.\n"
            "Check for schema mismatches, corrupt databases, or renamed tables.\n",
            file=sys.stderr,
            flush=True,
        )
    if not facts["counter_data_available"]:
        rule_verdict = _trace_only_fallback_verdict(facts, db_error=db_error)
    else:
        rule_verdict = bottleneck.classify_from_metrics(facts["metrics_for_classifier"])
        rule_verdict = _promote_sparse_compute_verdict(facts, rule_verdict)
        rule_verdict = _override_runtime_dominant_verdict(facts, rule_verdict)

    # Step 3: LLM refinement (optional).
    agent = build_analysis_agent()
    raw = run_agent(
        agent,
        input_payload={**payload.model_dump(), "facts": facts, "rule_verdict": rule_verdict},
        provider=provider,
        airgap=airgap,
    )

    if raw.get("_mode") == "airgap":
        return schemas.AnalysisOutput(
            primary_bottleneck=_validated_bottleneck_type(rule_verdict["type"]),
            confidence=rule_verdict["confidence"],
            time_breakdown=facts["time_breakdown"],
            hot_kernels=facts["hot_kernels"],
            counter_data_available=facts["counter_data_available"],
        )

    so = raw.get("structured_output") or {}
    return schemas.AnalysisOutput(
        primary_bottleneck=_validated_bottleneck_type(
            so.get("primary_bottleneck", rule_verdict["type"])
        ),
        confidence=so.get("confidence", rule_verdict["confidence"]),
        time_breakdown=so.get("time_breakdown", facts["time_breakdown"]),
        hot_kernels=so.get("hot_kernels", facts["hot_kernels"]),
        counter_data_available=so.get("counter_data_available", facts["counter_data_available"]),
    )


__all__ = ["build_analysis_agent", "run_analysis"]
