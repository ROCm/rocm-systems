#!/usr/bin/env python3
"""Runs one potentially destructive ConSan GPU row in a contained process."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from consan_coverage_gate import (
    CoverageParseError,
    parse_coverage_evidence,
    parse_coverage_site_records,
)
from consan_run_provenance import load_contract, validate_current_contract
from consan_validation_support import (
    RESULT_SCHEMA_VERSION,
    atomic_write_json,
    git_identity,
    sha256_file,
)

MAX_GPU_JOBS = 4
MAX_RESOURCE_PLAN_ALTERNATIVES = 4096
MAX_RESOURCE_PLAN_ALTERNATIVE_ERRORS = 16
QUARANTINE_FILE = ".gpu-quarantine.json"
GLOBAL_DESTRUCTIVE_LOCK_ENV = "CONSAN_DESTRUCTIVE_GPU_LOCK"
DEFAULT_GLOBAL_DESTRUCTIVE_LOCK = "/tmp/rocjitsu-consan-destructive-gpu.lock"
ROW_RESULT_ENV = "CONSAN_ROW_RESULT_PATH"
UNSPECIFIED = "unspecified"
INVENTORY = "inventory"
INCOMPLETE_BARRIER_DROP_ENV = (
    "RJ_CONSAN_FAULT_ALLOW_DESTRUCTIVE_INCOMPLETE_BARRIER_DROP"
)
DIVERGENT_BARRIER_MOVE_ENV = "RJ_CONSAN_FAULT_ALLOW_DESTRUCTIVE_DIVERGENT_BARRIER_MOVE"

_CONSAN_PREFIX = "[rocjitsu-dbi-hooks] ConSan "
_MUTATION_PATCH_KINDS = {
    "inline-barrier-nop-rewrite",
    "inline-barrier-move-source-rewrite",
    "inline-barrier-move-target-rewrite",
    "inline-atomic-address-rewrite",
    "inline-atomic-order-rewrite",
    "inline-atomic-scope-rewrite",
}

_RESOURCE_PLAN_ALTERNATIVE_KINDS = {
    "guest_operand_overlap_spill",
    "spill_backed_operand_recovery",
}
_RESOURCE_PLAN_ALTERNATIVE_SOURCES = {
    "unsupported",
    "explicit",
    "dead",
    "descriptor-growth",
    "spill",
}
_RESOURCE_PLAN_ALTERNATIVE_OUTCOMES = (
    "selected",
    "rejected",
    "superseded",
    "contributed",
    "vetoed",
)

_MODE_SELECTIONS = {
    "record-replay": ("moi", "record_replay"),
    "record_replay": ("moi", "record_replay"),
    "inline-shadow": ("moi", "inline_shadow"),
    "inline_shadow": ("moi", "inline_shadow"),
    "sampled": ("moi", "sampled"),
    "supercollider": ("supercollider", "supercollider"),
}


def _selection_from_environment(environment: dict[str, str]) -> tuple[str, str]:
    mode = environment.get("RJ_CONSAN_MODE", "").strip().lower()
    if mode in _MODE_SELECTIONS:
        return _MODE_SELECTIONS[mode]
    return (
        environment.get("RJ_CONSAN_FLAVOR", UNSPECIFIED),
        environment.get("RJ_CONSAN_MOI_ENGINE", UNSPECIFIED),
    )


def _key_values(text: str) -> dict[str, str]:
    """Parses whitespace-delimited key=value fields from a ConSan log record."""
    return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(\"[^\"]*\"|\S+)", text))


def _integer(fields: dict[str, str], name: str, default: int = 0) -> int:
    try:
        return int(fields.get(name, str(default)), 0)
    except ValueError:
        return default


def _required_integer(fields: dict[str, str], name: str) -> int | None:
    value = fields.get(name)
    if value is None:
        return None
    try:
        return int(value, 0)
    except ValueError:
        return None


def _append_resource_plan_error(errors: list[str], message: str) -> None:
    if len(errors) < MAX_RESOURCE_PLAN_ALTERNATIVE_ERRORS:
        errors.append(message)


def _parse_inline_release_evidence(fields: dict[str, str]) -> dict[str, object] | None:
    """Parses one complete, stable Inline release plus causal snapshot."""
    names = (
        "index",
        "version",
        "owner",
        "epoch_plus_one",
        "workgroup",
        "address",
        "dispatch",
        "snapshot_count",
        "snapshot_flags",
    )
    values = {name: _required_integer(fields, name) for name in names}
    if any(value is None for value in values.values()):
        return None
    index = values["index"]
    version = values["version"]
    owner = values["owner"]
    epoch = values["epoch_plus_one"]
    workgroup = values["workgroup"]
    address = values["address"]
    dispatch = values["dispatch"]
    snapshot_count = values["snapshot_count"]
    snapshot_flags = values["snapshot_flags"]
    if (
        index < 0
        or version == 0
        or version & 1
        or owner == 0
        or epoch <= 0
        or epoch > 1023
        or workgroup == 0
        or address == 0
        or dispatch == 0
        or snapshot_count < 0
        or snapshot_count > 4
        or snapshot_flags != 0
    ):
        return None
    snapshot = []
    prior_owner = 0
    for entry_index in range(snapshot_count):
        ancestor_owner = _required_integer(fields, f"snapshot{entry_index}_owner")
        ancestor_epoch = _required_integer(
            fields, f"snapshot{entry_index}_epoch_plus_one"
        )
        if (
            ancestor_owner is None
            or ancestor_epoch is None
            or ancestor_owner <= prior_owner
            or ancestor_owner == owner
            or ancestor_epoch <= 0
            or ancestor_epoch > 1023
        ):
            return None
        snapshot.append(
            {
                "owner": ancestor_owner,
                "epoch_plus_one": ancestor_epoch,
            }
        )
        prior_owner = ancestor_owner
    for entry_index in range(snapshot_count, 4):
        extra_owner = _required_integer(fields, f"snapshot{entry_index}_owner")
        extra_epoch = _required_integer(fields, f"snapshot{entry_index}_epoch_plus_one")
        if (extra_owner is None) != (extra_epoch is None):
            return None
        if extra_owner not in {None, 0} or extra_epoch not in {None, 0}:
            return None
    return {
        "index": index,
        "version": version,
        "owner": owner,
        "epoch_plus_one": epoch,
        "workgroup": workgroup,
        "address": address,
        "dispatch": dispatch,
        "snapshot_flags": snapshot_flags,
        "snapshot": snapshot,
    }


def _parse_inline_token_evidence(fields: dict[str, str]) -> dict[str, object] | None:
    """Parses one complete, stable direct or inherited Inline token."""
    names = (
        "index",
        "version",
        "consumer",
        "producer",
        "epoch_plus_one",
        "workgroup",
        "dispatch",
        "source_address",
        "source_version",
    )
    values = {name: _required_integer(fields, name) for name in names}
    kind = fields.get("kind")
    if any(value is None for value in values.values()) or kind not in {
        "direct",
        "inherited",
    }:
        return None
    if (
        values["index"] < 0
        or values["version"] == 0
        or values["version"] & 1
        or values["consumer"] == 0
        or values["producer"] == 0
        or values["consumer"] == values["producer"]
        or values["epoch_plus_one"] <= 0
        or values["epoch_plus_one"] > 1023
        or values["workgroup"] == 0
        or values["dispatch"] == 0
        or values["source_address"] == 0
        or values["source_version"] == 0
        or values["source_version"] & 1
    ):
        return None
    return {
        "index": values["index"],
        "version": values["version"],
        "kind": kind,
        "consumer": values["consumer"],
        "producer": values["producer"],
        "epoch_plus_one": values["epoch_plus_one"],
        "workgroup": values["workgroup"],
        "dispatch": values["dispatch"],
        "source_address": values["source_address"],
        "source_version": values["source_version"],
    }


def _parse_consan_log(log_text: str) -> dict[str, dict[str, object]]:
    """Extracts qualification evidence only from known ConSan record prefixes."""
    try:
        has_coverage_summary = any(
            _CONSAN_PREFIX + "coverage " in line for line in log_text.splitlines()
        )
        parsed_coverage = (
            parse_coverage_evidence(log_text) if has_coverage_summary else None
        )
        sites = (
            parsed_coverage.sites
            if parsed_coverage
            else parse_coverage_site_records(log_text)
        )
        coverage_sites = [asdict(site) for site in sites]
        analysis_verdict = asdict(parsed_coverage.verdict) if parsed_coverage else None
        static_coverage_records = (
            [asdict(record) for record in parsed_coverage.coverage]
            if parsed_coverage
            else []
        )
        coverage_site_parse_error = None
    except CoverageParseError as error:
        coverage_sites = []
        analysis_verdict = None
        static_coverage_records = []
        coverage_site_parse_error = str(error)
    fault_summaries: list[dict[str, str]] = []
    fault_install_summaries: list[dict[str, str]] = []
    fault_load_selections: list[dict[str, str]] = []
    fault_load_summaries: list[dict[str, str]] = []
    fault_sites = 0
    proof_patch_kinds: list[str] = []
    fault_patch_kinds: list[str] = []
    instrumentation_patch_kinds: list[str] = []
    patch_count = 0
    patch_outcomes: list[dict[str, object]] = []
    supported_sites = 0
    skipped_sites = 0
    rejected_sites = 0
    spill_patch_count = 0
    spill_slot_bytes = 0
    resource_plan_alternative_counts = {
        "attempts": 0,
        "selected": 0,
        "rejected": 0,
        "superseded": 0,
        "contributed": 0,
        "vetoed": 0,
    }
    resource_plan_alternatives: list[dict[str, object]] = []
    resource_plan_alternative_tallies = {
        outcome: 0 for outcome in _RESOURCE_PLAN_ALTERNATIVE_OUTCOMES
    }
    resource_plan_alternative_record_count = 0
    resource_plan_alternatives_truncated = 0
    resource_plan_alternative_errors: list[str] = []
    resource_plan_next_attempt: dict[tuple[str, str, int, int], int] = {}
    resource_plan_selected_per_plan: dict[tuple[str, str, int, int], int] = {}
    report_buffer_bytes = 0
    report_plan_count = 0
    report_memory_summary_count = 0
    report_required_bytes = 0
    report_allocated_bytes = 0
    report_live_before_cleanup_bytes = 0
    report_live_after_cleanup_bytes = 0
    report_peak_live_bytes = 0
    report_allocation_failures = 0
    report_capacity_failures = 0
    report_cleanup_failures = 0
    report_region_capacity_entries = {
        "access": 0,
        "barrier": 0,
        "atomic": 0,
        "fence": 0,
        "diagnostic": 0,
        "exact_shadow": 0,
        "inline_atomic_release": 0,
        "inline_acquired_token": 0,
        "inline_causal_snapshot": 0,
        "sampled_watchpoint": 0,
        "sampled_causal_window": 0,
        "sampled_sync_metadata": 0,
        "sampled_pending_acquire": 0,
    }
    report_plans: list[dict[str, object]] = []
    report_memory_summaries: list[dict[str, int]] = []
    shadow_capacity_entries = 0
    diagnostic_capacity_entries = 0
    inline_atomic_release_capacity_entries = 0
    inline_acquired_token_capacity_entries = 0
    inline_causal_snapshot_capacity_entries = 0
    report_buffers = 0
    event_counts = {
        "access": 0,
        "barrier": 0,
        "atomic": 0,
        "diagnostic": 0,
        "exact_shadow": 0,
        "inline_atomic_release": 0,
        "inline_acquired_token": 0,
        "sampled": 0,
    }
    overflow_counts = {
        "access": 0,
        "barrier": 0,
        "atomic": 0,
        "diagnostic": 0,
        "sampled": 0,
    }
    sampled_snapshot_counts = {
        "stale": 0,
        "incomplete": 0,
        "changed": 0,
        "malformed": 0,
    }
    exact_snapshot_counts = {"incomplete": 0, "changed": 0, "malformed": 0}
    inline_release_snapshot_counts = {
        "incomplete": 0,
        "changed": 0,
        "overflow": 0,
        "source_incomplete": 0,
        "malformed": 0,
    }
    inline_token_snapshot_counts = {"incomplete": 0, "changed": 0, "malformed": 0}
    inline_evidence_counts = {
        "release_records": 0,
        "token_records": 0,
        "malformed_records": 0,
        "duplicate_records": 0,
        "count_mismatches": 0,
        "capacity_violations": 0,
        "state_mismatches": 0,
    }
    inline_release_evidence: list[dict[str, object]] = []
    inline_token_evidence: list[dict[str, object]] = []
    inline_release_keys: set[tuple[str, int]] = set()
    inline_token_keys: set[tuple[str, int]] = set()
    inline_coverage_counts = {
        "undercoverage": 0,
        "overflow": 0,
        "unsupported": 0,
        "malformed": 0,
    }
    inline_diagnostics = 0
    replay_diagnostics = 0
    sampled_conflicts = 0
    sampled_immediate_conflicts = 0
    sampled_claimed_windows = 0
    sampled_reader_access_events = 0
    sampled_writer_access_events = 0
    supercollider_mismatches = 0
    sc_report_buffer_count = 0
    sc_report_allocation_failures = 0
    sc_report_read_failures = 0
    sc_report_cleanup_failures = 0
    saw_instrumentation_evidence = False
    readers: dict[str, dict[str, object]] = {}

    def reader_record(fields: dict[str, str]) -> dict[str, object] | None:
        reader = fields.get("reader")
        if not reader:
            return None
        return readers.setdefault(
            reader,
            {
                "reader": reader,
                "patches": 0,
                "fault_patches": 0,
                "instrumentation_patches": 0,
                "proof_patch_kinds": [],
                "report_buffer_count": 0,
                "event_counts": {key: 0 for key in event_counts},
                "overflow_counts": {key: 0 for key in overflow_counts},
                "exact_snapshot_counts": {key: 0 for key in exact_snapshot_counts},
                "inline_release_snapshot_counts": {
                    key: 0 for key in inline_release_snapshot_counts
                },
                "inline_token_snapshot_counts": {
                    key: 0 for key in inline_token_snapshot_counts
                },
                "inline_coverage_counts": {key: 0 for key in inline_coverage_counts},
                "inline_evidence_counts": {key: 0 for key in inline_evidence_counts},
                "inline_release_evidence": [],
                "inline_token_evidence": [],
                "inline_release_capacity": 0,
                "inline_token_capacity": 0,
                "inline_snapshot_capacity": 0,
                "inline_summary_records": 0,
                "inline_summary_complete": True,
                "report_plans": [],
                "report_buffers": [],
                "spilled_vgpr_count": 0,
                "private_segment_bytes": 0,
                "workgroup_shadow_bytes": 0,
                "group_segment_bytes": 0,
            },
        )

    for line in log_text.splitlines():
        marker = line.find(_CONSAN_PREFIX)
        if marker < 0:
            continue
        record = line[marker + len(_CONSAN_PREFIX) :]
        fields = _key_values(record)
        if record.startswith("fault summary "):
            fault_summaries.append(fields)
        elif record.startswith("fault install "):
            fault_install_summaries.append(fields)
        elif record.startswith("fault load selection "):
            fault_load_selections.append(fields)
        elif record.startswith("fault load summary "):
            fault_load_summaries.append(fields)
        elif record.startswith("fault site "):
            fault_sites += 1
        elif record.startswith("proof patch "):
            kind = fields.get("kind")
            if kind:
                proof_patch_kinds.append(kind)
                if kind in _MUTATION_PATCH_KINDS:
                    fault_patch_kinds.append(kind)
                else:
                    instrumentation_patch_kinds.append(kind)
                    saw_instrumentation_evidence = True
                per_reader = reader_record(fields)
                if per_reader is not None:
                    per_reader["proof_patch_kinds"].append(kind)
                    per_reader["fault_patches"] += int(kind in _MUTATION_PATCH_KINDS)
                    per_reader["instrumentation_patches"] += int(
                        kind not in _MUTATION_PATCH_KINDS
                    )
                    per_reader["spilled_vgpr_count"] += _integer(
                        fields, "spilled_vgprs"
                    )
                    per_reader["private_segment_bytes"] = max(
                        per_reader["private_segment_bytes"],
                        _integer(fields, "private_bytes"),
                    )
                    per_reader["workgroup_shadow_bytes"] = max(
                        per_reader["workgroup_shadow_bytes"],
                        _integer(fields, "workgroup_shadow_bytes"),
                    )
                    per_reader["group_segment_bytes"] = max(
                        per_reader["group_segment_bytes"],
                        _integer(fields, "group_bytes"),
                    )
        elif record.startswith("patch end "):
            patch_count += _integer(fields, "patches")
            patch_outcomes.append(
                {
                    "reader": fields.get("reader", UNSPECIFIED),
                    "outcome": fields.get("outcome", "unknown"),
                    "errors": _integer(fields, "errors"),
                    "warnings": _integer(fields, "warnings"),
                    "patches": _integer(fields, "patches"),
                }
            )
            per_reader = reader_record(fields)
            if per_reader is not None:
                per_reader["patches"] += _integer(fields, "patches")
        elif record.startswith("summary "):
            supported_sites += _integer(fields, "supported_lds_sites")
            supported_sites += _integer(fields, "function_supported_lds_sites")
            skipped_sites += _integer(fields, "skips")
            rejected_sites += _integer(fields, "rejects")
        elif record.startswith("MOI resources "):
            spill_patch_count += _integer(fields, "emitted_spill_patches")
            spill_slot_bytes += _integer(fields, "emitted_spill_slot_bytes")
            for key, field in (
                ("attempts", "alternative_attempts"),
                ("selected", "alternative_selected"),
                ("rejected", "alternative_rejected"),
                ("superseded", "alternative_superseded"),
                ("contributed", "alternative_contributed"),
                ("vetoed", "alternative_vetoed"),
            ):
                value = _required_integer(fields, field)
                if value is None or value < 0:
                    _append_resource_plan_error(
                        resource_plan_alternative_errors,
                        f"invalid aggregate field {field}",
                    )
                else:
                    resource_plan_alternative_counts[key] += value
        elif record.startswith("MOI resource-alternative "):
            resource_plan_alternative_record_count += 1
            reader = fields.get("reader", "")
            site = fields.get("site", "")
            kind = fields.get("kind", "")
            source = fields.get("source", "")
            outcome = fields.get("outcome", "")
            numbers = {
                name: _required_integer(fields, name)
                for name in ("candidate", "text_offset", "attempt", "scratch_count")
            }
            for name, value in numbers.items():
                if value is None or value < 0:
                    _append_resource_plan_error(
                        resource_plan_alternative_errors,
                        f"invalid detail field {name} at record "
                        f"{resource_plan_alternative_record_count}",
                    )
            if not reader:
                _append_resource_plan_error(
                    resource_plan_alternative_errors,
                    f"missing reader at record {resource_plan_alternative_record_count}",
                )
            if site not in {"access", "barrier", "atomic", "fence"}:
                _append_resource_plan_error(
                    resource_plan_alternative_errors,
                    f"unknown site {site or '<missing>'}",
                )
            if kind not in _RESOURCE_PLAN_ALTERNATIVE_KINDS:
                _append_resource_plan_error(
                    resource_plan_alternative_errors,
                    f"unknown kind {kind or '<missing>'}",
                )
            if source not in _RESOURCE_PLAN_ALTERNATIVE_SOURCES:
                _append_resource_plan_error(
                    resource_plan_alternative_errors,
                    f"unknown source {source or '<missing>'}",
                )
            if outcome not in resource_plan_alternative_tallies:
                _append_resource_plan_error(
                    resource_plan_alternative_errors,
                    f"unknown outcome {outcome or '<missing>'}",
                )
            else:
                resource_plan_alternative_tallies[outcome] += 1

            candidate = numbers["candidate"] if numbers["candidate"] is not None else 0
            text_offset = (
                numbers["text_offset"] if numbers["text_offset"] is not None else 0
            )
            attempt = numbers["attempt"] if numbers["attempt"] is not None else 0
            plan_identity = (reader, site, candidate, text_offset)
            if outcome == "selected":
                resource_plan_selected_per_plan[plan_identity] = (
                    resource_plan_selected_per_plan.get(plan_identity, 0) + 1
                )
            expected_attempt = resource_plan_next_attempt.get(plan_identity, 0)
            if attempt != expected_attempt:
                _append_resource_plan_error(
                    resource_plan_alternative_errors,
                    f"non-chronological attempt for {reader}:{site}:{candidate}:"
                    f"0x{text_offset:x}: expected {expected_attempt}, got {attempt}",
                )
            resource_plan_next_attempt[plan_identity] = attempt + 1

            parsed_alternative = {
                "reader": reader,
                "site": site or "unknown",
                "candidate": candidate,
                "text_offset": text_offset,
                "attempt": attempt,
                "kind": kind or "unknown",
                "scratch_count": (
                    numbers["scratch_count"]
                    if numbers["scratch_count"] is not None
                    else 0
                ),
                "source": source or "unknown",
                "reason": fields.get("reason", "unknown"),
                "outcome": outcome or "unknown",
            }
            if len(resource_plan_alternatives) < MAX_RESOURCE_PLAN_ALTERNATIVES:
                resource_plan_alternatives.append(parsed_alternative)
            else:
                resource_plan_alternatives_truncated += 1
        elif record.startswith("MOI auto report plan "):
            plan = {
                "reader": fields.get("reader", ""),
                "outcome": fields.get("outcome", "unknown"),
                "reason": fields.get("reason", "unknown"),
                "required_bytes": _integer(fields, "required_bytes"),
                "cap_bytes": _integer(fields, "cap_bytes"),
                "per_buffer_ceiling": _integer(fields, "per_buffer_ceiling"),
                "process_ceiling": _integer(fields, "process_ceiling"),
                "access_ranges": _integer(fields, "access_ranges"),
                "barriers": _integer(fields, "barriers"),
                "atomics": _integer(fields, "atomics"),
                "fences": _integer(fields, "fences"),
                "diagnostics": _integer(fields, "diagnostics"),
                "sampled_banks": _integer(fields, "sampled_banks"),
                "sampled_watchpoints": _integer(fields, "sampled_watchpoints"),
                "inline_lds_bytes": _integer(fields, "inline_lds_bytes"),
                "inline_releases": _integer(fields, "inline_releases"),
                "inline_snapshots": _integer(fields, "inline_snapshots"),
                "inline_tokens": _integer(fields, "inline_tokens"),
            }
            report_plans.append(plan)
            report_plan_count += 1
            per_reader = reader_record(fields)
            if per_reader is not None:
                per_reader["report_plans"].append(plan)
        elif record.startswith("MOI auto report buffer "):
            if fields.get("allocation_outcome") != "allocated":
                continue
            report_buffer_bytes += _integer(fields, "bytes")
            shadow_capacity_entries += _integer(fields, "exact_shadow_entry_capacity")
            diagnostic_capacity_entries += _integer(fields, "diagnostic_capacity")
            inline_atomic_release_capacity_entries += _integer(
                fields, "inline_atomic_release_capacity"
            )
            inline_acquired_token_capacity_entries += _integer(
                fields, "inline_acquired_epoch_token_capacity"
            )
            inline_causal_snapshot_capacity_entries += _integer(
                fields, "inline_causal_snapshot_capacity"
            )
            capacities = {
                "access": _integer(fields, "access_record_capacity"),
                "barrier": _integer(fields, "barrier_record_capacity"),
                "atomic": _integer(fields, "atomic_record_capacity"),
                "fence": _integer(fields, "fence_record_capacity"),
                "diagnostic": _integer(fields, "diagnostic_capacity"),
                "exact_shadow": _integer(fields, "exact_shadow_entry_capacity"),
                "inline_atomic_release": _integer(
                    fields, "inline_atomic_release_capacity"
                ),
                "inline_acquired_token": _integer(
                    fields, "inline_acquired_epoch_token_capacity"
                ),
                "inline_causal_snapshot": _integer(
                    fields, "inline_causal_snapshot_capacity"
                ),
                "sampled_watchpoint": _integer(fields, "sampled_watchpoint_capacity"),
                "sampled_causal_window": _integer(
                    fields, "sampled_causal_window_capacity"
                ),
                "sampled_sync_metadata": _integer(
                    fields, "sampled_sync_metadata_capacity"
                ),
                "sampled_pending_acquire": _integer(
                    fields, "sampled_pending_acquire_capacity"
                ),
            }
            for kind, capacity in capacities.items():
                report_region_capacity_entries[kind] += capacity
            report_buffers += 1
            saw_instrumentation_evidence = True
            per_reader = reader_record(fields)
            if per_reader is not None:
                per_reader["report_buffer_count"] += 1
                per_reader["report_buffers"].append(
                    {
                        "bytes": _integer(fields, "bytes"),
                        "required_bytes": _integer(fields, "required_bytes"),
                        "cap_bytes": _integer(fields, "cap_bytes"),
                        "process_current_bytes": _integer(
                            fields, "process_current_bytes"
                        ),
                        "process_peak_bytes": _integer(fields, "process_peak_bytes"),
                        "process_ceiling_bytes": _integer(
                            fields, "process_ceiling_bytes"
                        ),
                        "allocation_outcome": fields.get(
                            "allocation_outcome", "unknown"
                        ),
                        "capacities": capacities,
                    }
                )
                per_reader["inline_release_capacity"] += _integer(
                    fields, "inline_atomic_release_capacity"
                )
                per_reader["inline_token_capacity"] += _integer(
                    fields, "inline_acquired_epoch_token_capacity"
                )
                per_reader["inline_snapshot_capacity"] += _integer(
                    fields, "inline_causal_snapshot_capacity"
                )
        elif record.startswith("MOI report memory "):
            summary: dict[str, int] = {
                "required_bytes": _integer(fields, "required_bytes"),
                "allocated_bytes": _integer(fields, "allocated_bytes"),
                "live_before_cleanup": _integer(fields, "live_before_cleanup"),
                "live_after_cleanup": _integer(fields, "live_after_cleanup"),
                "peak_live_bytes": _integer(fields, "peak_live_bytes"),
                "per_buffer_ceiling": _integer(fields, "per_buffer_ceiling"),
                "process_ceiling": _integer(fields, "process_ceiling"),
                "allocation_failures": _integer(fields, "allocation_failures"),
                "capacity_failures": _integer(fields, "capacity_failures"),
                "cleanup_failures": _integer(fields, "cleanup_failures"),
            }
            report_memory_summaries.append(summary)
            report_memory_summary_count += 1
            report_required_bytes += summary["required_bytes"]
            report_allocated_bytes += summary["allocated_bytes"]
            report_live_before_cleanup_bytes += summary["live_before_cleanup"]
            report_live_after_cleanup_bytes += summary["live_after_cleanup"]
            report_peak_live_bytes = max(
                report_peak_live_bytes, summary["peak_live_bytes"]
            )
            report_allocation_failures += summary["allocation_failures"]
            report_capacity_failures += summary["capacity_failures"]
            report_cleanup_failures += summary["cleanup_failures"]
        elif record.startswith("MOI auto report reader="):
            required_inline_summary_fields = (
                "visible_inline_atomic_releases",
                "visible_inline_acquired_tokens",
                "release_incomplete_snapshots",
                "release_changed_snapshots",
                "release_overflow_snapshots",
                "release_source_incomplete_snapshots",
                "release_malformed_snapshots",
                "token_incomplete_snapshots",
                "token_changed_snapshots",
                "token_malformed_snapshots",
            )
            inline_diagnostics += _integer(fields, "visible_diagnostics")
            sampled_conflicts += _integer(fields, "sampled_conflicts")
            sampled_immediate_conflicts += _integer(
                fields, "sampled_immediate_conflicts"
            )
            sampled_claimed_windows += _integer(fields, "sampled_claimed_windows")
            sampled_snapshot_counts["stale"] += _integer(
                fields, "sampled_stale_snapshots"
            )
            sampled_snapshot_counts["incomplete"] += _integer(
                fields, "sampled_incomplete_snapshots"
            )
            sampled_snapshot_counts["changed"] += _integer(
                fields, "sampled_changed_snapshots"
            )
            sampled_snapshot_counts["malformed"] += _integer(
                fields, "sampled_malformed_snapshots"
            )
            exact_snapshot_counts["incomplete"] += _integer(
                fields, "exact_incomplete_snapshots"
            )
            exact_snapshot_counts["changed"] += _integer(
                fields, "exact_changed_snapshots"
            )
            exact_snapshot_counts["malformed"] += _integer(
                fields, "exact_malformed_snapshots"
            )
            for key, field in (
                ("incomplete", "release_incomplete_snapshots"),
                ("changed", "release_changed_snapshots"),
                ("overflow", "release_overflow_snapshots"),
                ("source_incomplete", "release_source_incomplete_snapshots"),
                ("malformed", "release_malformed_snapshots"),
            ):
                inline_release_snapshot_counts[key] += _integer(fields, field)
            for key, field in (
                ("incomplete", "token_incomplete_snapshots"),
                ("changed", "token_changed_snapshots"),
                ("malformed", "token_malformed_snapshots"),
            ):
                inline_token_snapshot_counts[key] += _integer(fields, field)
            inline_undercoverage_field = (
                "inline_undercoverage"
                if "inline_undercoverage" in fields
                else "inline_unsupported_workgroups"
            )
            inline_coverage_counts["undercoverage"] += _integer(
                fields, inline_undercoverage_field
            )
            inline_coverage_counts["overflow"] += _integer(fields, "inline_overflow")
            inline_coverage_counts["unsupported"] += _integer(
                fields, "inline_unsupported"
            )
            inline_coverage_counts["malformed"] += _integer(fields, "inline_malformed")
            event_counts["access"] += _integer(fields, "visible_records")
            event_counts["barrier"] += _integer(fields, "visible_barriers")
            event_counts["atomic"] += _integer(fields, "visible_atomics")
            event_counts["diagnostic"] += _integer(fields, "visible_diagnostics")
            event_counts["exact_shadow"] += _integer(fields, "visible_exact_shadow")
            event_counts["inline_atomic_release"] += _integer(
                fields, "visible_inline_atomic_releases"
            )
            event_counts["inline_acquired_token"] += _integer(
                fields, "visible_inline_acquired_tokens"
            )
            event_counts["sampled"] += _integer(fields, "visible_sampled")
            overflow_counts["access"] += _integer(fields, "dropped_records")
            overflow_counts["barrier"] += _integer(fields, "dropped_barriers")
            overflow_counts["atomic"] += _integer(fields, "dropped_atomics")
            overflow_counts["diagnostic"] += _integer(fields, "dropped_diagnostics")
            overflow_counts["sampled"] += _integer(fields, "sampled_dropped_windows")
            saw_instrumentation_evidence = True
            per_reader = reader_record(fields)
            if per_reader is not None:
                per_reader["inline_summary_records"] += 1
                per_reader["inline_summary_complete"] &= all(
                    _required_integer(fields, field) is not None
                    for field in required_inline_summary_fields
                )
                per_events = per_reader["event_counts"]
                per_overflow = per_reader["overflow_counts"]
                per_exact_snapshots = per_reader["exact_snapshot_counts"]
                per_inline_coverage = per_reader["inline_coverage_counts"]
                for key, field in (
                    ("access", "visible_records"),
                    ("barrier", "visible_barriers"),
                    ("atomic", "visible_atomics"),
                    ("diagnostic", "visible_diagnostics"),
                    ("exact_shadow", "visible_exact_shadow"),
                    ("inline_atomic_release", "visible_inline_atomic_releases"),
                    ("inline_acquired_token", "visible_inline_acquired_tokens"),
                    ("sampled", "visible_sampled"),
                ):
                    per_events[key] += _integer(fields, field)
                for key, field in (
                    ("access", "dropped_records"),
                    ("barrier", "dropped_barriers"),
                    ("atomic", "dropped_atomics"),
                    ("diagnostic", "dropped_diagnostics"),
                    ("sampled", "sampled_dropped_windows"),
                ):
                    per_overflow[key] += _integer(fields, field)
                for key, field in (
                    ("incomplete", "exact_incomplete_snapshots"),
                    ("changed", "exact_changed_snapshots"),
                    ("malformed", "exact_malformed_snapshots"),
                ):
                    per_exact_snapshots[key] += _integer(fields, field)
                for key, field in (
                    ("incomplete", "release_incomplete_snapshots"),
                    ("changed", "release_changed_snapshots"),
                    ("overflow", "release_overflow_snapshots"),
                    ("source_incomplete", "release_source_incomplete_snapshots"),
                    ("malformed", "release_malformed_snapshots"),
                ):
                    per_reader["inline_release_snapshot_counts"][key] += _integer(
                        fields, field
                    )
                for key, field in (
                    ("incomplete", "token_incomplete_snapshots"),
                    ("changed", "token_changed_snapshots"),
                    ("malformed", "token_malformed_snapshots"),
                ):
                    per_reader["inline_token_snapshot_counts"][key] += _integer(
                        fields, field
                    )
                for key, field in (
                    ("undercoverage", inline_undercoverage_field),
                    ("overflow", "inline_overflow"),
                    ("unsupported", "inline_unsupported"),
                    ("malformed", "inline_malformed"),
                ):
                    per_inline_coverage[key] += _integer(fields, field)
        elif record.startswith("MOI auto replay "):
            replay_diagnostics += _integer(fields, "diagnostics")
            saw_instrumentation_evidence = True
        elif record.startswith("MOI auto sampled "):
            kind = _integer(fields, "kind")
            sampled_reader_access_events += int(kind in {1, 3})
            sampled_writer_access_events += int(kind in {2, 3})
            saw_instrumentation_evidence = True
        elif record.startswith("SC auto report "):
            per_reader = reader_record(fields)
            if per_reader is not None:
                per_reader["supercollider_mismatch"] = fields.get("mismatch") == "true"
            saw_instrumentation_evidence = True
        elif record.startswith("SC report summary "):
            sc_report_buffer_count += _integer(fields, "buffers")
            supercollider_mismatches += _integer(fields, "mismatches")
            sc_report_allocation_failures += _integer(fields, "allocation_failures")
            sc_report_read_failures += _integer(fields, "read_failures")
            sc_report_cleanup_failures += _integer(fields, "cleanup_failures")
            saw_instrumentation_evidence |= _integer(fields, "buffers") != 0
        elif record.startswith("MOI auto inline-atomic-release "):
            per_reader = reader_record(fields)
            parsed = _parse_inline_release_evidence(fields)
            if per_reader is None or parsed is None:
                inline_evidence_counts["malformed_records"] += 1
                if per_reader is not None:
                    per_reader["inline_evidence_counts"]["malformed_records"] += 1
                continue
            key = (str(fields["reader"]), int(parsed["index"]))
            if key in inline_release_keys:
                inline_evidence_counts["duplicate_records"] += 1
                per_reader["inline_evidence_counts"]["duplicate_records"] += 1
                continue
            inline_release_keys.add(key)
            evidence = {"reader": fields["reader"], **parsed}
            inline_release_evidence.append(evidence)
            per_reader["inline_release_evidence"].append(parsed)
            inline_evidence_counts["release_records"] += 1
            per_reader["inline_evidence_counts"]["release_records"] += 1
            saw_instrumentation_evidence = True
        elif record.startswith("MOI auto inline-acquired-token "):
            per_reader = reader_record(fields)
            parsed = _parse_inline_token_evidence(fields)
            if per_reader is None or parsed is None:
                inline_evidence_counts["malformed_records"] += 1
                if per_reader is not None:
                    per_reader["inline_evidence_counts"]["malformed_records"] += 1
                continue
            key = (str(fields["reader"]), int(parsed["index"]))
            if key in inline_token_keys:
                inline_evidence_counts["duplicate_records"] += 1
                per_reader["inline_evidence_counts"]["duplicate_records"] += 1
                continue
            inline_token_keys.add(key)
            evidence = {"reader": fields["reader"], **parsed}
            inline_token_evidence.append(evidence)
            per_reader["inline_token_evidence"].append(parsed)
            inline_evidence_counts["token_records"] += 1
            per_reader["inline_evidence_counts"]["token_records"] += 1
            saw_instrumentation_evidence = True

    def reject_reader_evidence(record: dict[str, object], reason: str) -> None:
        inline_evidence_counts[reason] += 1
        record["inline_evidence_counts"][reason] += 1

    for record in readers.values():
        releases = record["inline_release_evidence"]
        tokens = record["inline_token_evidence"]
        if record["event_counts"]["inline_atomic_release"] != len(releases):
            reject_reader_evidence(record, "count_mismatches")
        if record["event_counts"]["inline_acquired_token"] != len(tokens):
            reject_reader_evidence(record, "count_mismatches")
        release_capacity = min(
            record["inline_release_capacity"], record["inline_snapshot_capacity"]
        )
        for release in releases:
            if release_capacity == 0 or release["index"] >= release_capacity:
                reject_reader_evidence(record, "capacity_violations")
        for token in tokens:
            if (
                record["inline_token_capacity"] == 0
                or token["index"] >= record["inline_token_capacity"]
            ):
                reject_reader_evidence(record, "capacity_violations")
            sources = [
                release
                for release in releases
                if (
                    release["dispatch"] == token["dispatch"]
                    and release["workgroup"] == token["workgroup"]
                    and release["address"] == token["source_address"]
                    and release["version"] == token["source_version"]
                )
            ]
            if len(sources) != 1:
                reject_reader_evidence(record, "malformed_records")
                continue
            source = sources[0]
            if token["kind"] == "direct":
                valid_source = (
                    token["producer"] == source["owner"]
                    and token["epoch_plus_one"] == source["epoch_plus_one"]
                )
            else:
                valid_source = any(
                    ancestor["owner"] == token["producer"]
                    and ancestor["epoch_plus_one"] == token["epoch_plus_one"]
                    for ancestor in source["snapshot"]
                )
            if not valid_source:
                reject_reader_evidence(record, "malformed_records")

        release_states = sum(record["inline_release_snapshot_counts"].values())
        token_states = sum(record["inline_token_snapshot_counts"].values())
        has_inline_capacity = (
            record["inline_release_capacity"]
            or record["inline_snapshot_capacity"]
            or record["inline_token_capacity"]
        )
        if has_inline_capacity and (
            record["inline_summary_records"] == 0
            or not record["inline_summary_complete"]
        ):
            reject_reader_evidence(record, "state_mismatches")
        if (
            record["event_counts"]["inline_atomic_release"] + release_states
            > release_capacity
            or record["event_counts"]["inline_acquired_token"] + token_states
            > record["inline_token_capacity"]
        ):
            reject_reader_evidence(record, "state_mismatches")

    mutation_reader_map: dict[tuple[str, str], dict[str, object]] = {}
    for fields in fault_summaries:
        process = fields.get("process", UNSPECIFIED)
        reader = fields.get("reader", UNSPECIFIED)
        record = mutation_reader_map.setdefault(
            (process, reader),
            {
                "process": process,
                "reader": reader,
                "requested": 0,
                "planned": 0,
                "raw_applied": 0,
                "applied": 0,
                "discarded_applied": 0,
            },
        )
        record["requested"] = max(record["requested"], _integer(fields, "requested"))
        record["planned"] += _integer(fields, "planned")
        record["raw_applied"] += _integer(fields, "applied")

    install_by_identity: dict[tuple[str, str], list[dict[str, str]]] = {}
    valid_install_evidence = True
    for fields in fault_install_summaries:
        process = fields.get("process")
        reader = fields.get("reader")
        installed = fields.get("installed")
        applied_count = _integer(fields, "applied")
        if (
            not process
            or not reader
            or installed not in {"true", "false"}
            or applied_count <= 0
        ):
            valid_install_evidence = False
            continue
        install_by_identity.setdefault((process, reader), []).append(fields)

    # A strict load rejection can terminate after the producer flushes
    # installed=false evidence but before the ordinary fault summary is
    # emitted. An applied-but-discarded mutation proves its own requested and
    # planned counts; retain that distinction instead of misclassifying the
    # producer log as incomplete.
    for identity, installs in install_by_identity.items():
        if identity in mutation_reader_map or any(
            fields["installed"] != "false" for fields in installs
        ):
            continue
        process, reader = identity
        discarded = sum(_integer(fields, "applied") for fields in installs)
        mutation_reader_map[identity] = {
            "process": process,
            "reader": reader,
            "requested": 1,
            "planned": discarded,
            "raw_applied": discarded,
            "applied": 0,
            "discarded_applied": 0,
        }

    raw_by_identity = {
        identity: int(record["raw_applied"])
        for identity, record in mutation_reader_map.items()
        if record["raw_applied"]
    }
    install_total_by_identity = {
        identity: sum(_integer(fields, "applied") for fields in records)
        for identity, records in install_by_identity.items()
    }
    installation_evidence_complete = (
        not raw_by_identity and not fault_install_summaries
    ) or (
        bool(fault_install_summaries)
        and valid_install_evidence
        and install_total_by_identity == raw_by_identity
    )
    for identity, record in mutation_reader_map.items():
        if installation_evidence_complete and identity in install_by_identity:
            installs = install_by_identity[identity]
            record["applied"] = sum(
                _integer(fields, "applied")
                for fields in installs
                if fields["installed"] == "true"
            )
            record["discarded_applied"] = sum(
                _integer(fields, "applied")
                for fields in installs
                if fields["installed"] == "false"
            )
        else:
            record["applied"] = record["raw_applied"]

    mutation_readers = sorted(
        mutation_reader_map.values(),
        key=lambda record: (record["process"], record["reader"]),
    )
    classified_alternative_count = sum(
        resource_plan_alternative_counts[outcome]
        for outcome in _RESOURCE_PLAN_ALTERNATIVE_OUTCOMES
    )
    if classified_alternative_count != resource_plan_alternative_counts["attempts"]:
        _append_resource_plan_error(
            resource_plan_alternative_errors,
            "aggregate alternative attempts do not equal classified outcomes",
        )
    if (
        resource_plan_alternative_record_count
        != resource_plan_alternative_counts["attempts"]
    ):
        _append_resource_plan_error(
            resource_plan_alternative_errors,
            "detail record count does not equal aggregate alternative attempts",
        )
    for outcome in _RESOURCE_PLAN_ALTERNATIVE_OUTCOMES:
        if (
            resource_plan_alternative_tallies[outcome]
            != resource_plan_alternative_counts[outcome]
        ):
            _append_resource_plan_error(
                resource_plan_alternative_errors,
                f"detail {outcome} count does not match aggregate",
            )
    if any(count > 1 for count in resource_plan_selected_per_plan.values()):
        _append_resource_plan_error(
            resource_plan_alternative_errors,
            "more than one selected alternative was recorded for a plan",
        )
    resource_plan_alternative_parse_error = (
        "; ".join(resource_plan_alternative_errors)
        if resource_plan_alternative_errors
        else None
    )
    resource_plan_alternatives_complete = (
        resource_plan_alternative_parse_error is None
        and resource_plan_alternatives_truncated == 0
    )

    requested = max(
        (int(record["requested"]) for record in mutation_readers), default=0
    )
    planned = sum(int(record["planned"]) for record in mutation_readers)
    raw_applied = sum(int(record["raw_applied"]) for record in mutation_readers)
    applied = sum(int(record["applied"]) for record in mutation_readers)
    discarded_applied = sum(
        int(record["discarded_applied"]) for record in mutation_readers
    )
    applied_readers = sum(record["applied"] > 0 for record in mutation_readers)
    process_evidence_complete = bool(fault_summaries) and all(
        fields.get("process") for fields in fault_summaries
    )
    fault_processes: dict[str, dict[str, object]] = {}
    for reader_record in mutation_readers:
        process = str(reader_record["process"])
        process_record = fault_processes.setdefault(
            process,
            {
                "process": process,
                "requested": 0,
                "planned": 0,
                "raw_applied": 0,
                "applied": 0,
                "discarded_applied": 0,
                "readers": {},
            },
        )
        process_record["requested"] = max(
            process_record["requested"], reader_record["requested"]
        )
        process_record["planned"] += reader_record["planned"]
        process_record["raw_applied"] += reader_record["raw_applied"]
        process_record["applied"] += reader_record["applied"]
        process_record["discarded_applied"] += reader_record["discarded_applied"]
        process_record["readers"][reader_record["reader"]] = {
            key: value for key, value in reader_record.items() if key != "process"
        }
    fault_process_evidence = []
    for process_record in fault_processes.values():
        process_record["readers"] = sorted(
            process_record["readers"].values(), key=lambda record: record["reader"]
        )
        fault_process_evidence.append(process_record)
    fault_process_evidence.sort(key=lambda record: record["process"])
    selected_loads = sum(
        fields.get("selected") == "true" for fields in fault_load_selections
    )
    load_selection = {
        "schema_version": 1,
        "requested_occurrences": sorted(
            {
                _integer(fields, "requested_occurrence")
                for fields in (*fault_load_selections, *fault_load_summaries)
            }
            - {0}
        ),
        "matching_readers": sum(
            fields.get("matched") == "true" for fields in fault_load_selections
        ),
        "observed_occurrences": sorted(
            _integer(fields, "observed_occurrence")
            for fields in fault_load_selections
            if fields.get("matched") == "true"
        ),
        "readers": [
            {
                "reader": fields.get("reader", UNSPECIFIED),
                "matched": fields.get("matched") == "true",
                "observed_occurrence": _integer(fields, "observed_occurrence"),
                "selected": fields.get("selected") == "true",
                "overflowed": fields.get("overflow") == "true",
            }
            for fields in fault_load_selections
        ],
        "selected_loads": selected_loads,
        "overflowed": any(
            fields.get("overflow") == "true" for fields in fault_load_selections
        )
        or any(fields.get("overflow") == "true" for fields in fault_load_summaries),
        "accepted_summaries": sum(
            fields.get("accepted") == "true" for fields in fault_load_summaries
        ),
        "summary_count": len(fault_load_summaries),
    }
    if applied:
        applicability = "applied"
    elif planned:
        applicability = "planned_not_applied"
    elif requested:
        applicability = "not_applicable"
    else:
        applicability = "not_requested"

    static_diagnostics = proof_patch_kinds.count("inline-malformed-barrier-abort")
    # SuperCollider's report marker is a detector-owned value-instability
    # diagnostic. It does not establish a happens-before edge or identify the
    # exact racing pair, but excluding it from the sanitizer outcome made a
    # genuine redundant-access mismatch impossible to qualify as detection.
    supercollider_diagnostics = supercollider_mismatches
    diagnostic_count = (
        inline_diagnostics
        + replay_diagnostics
        + sampled_conflicts
        + sampled_immediate_conflicts
        + static_diagnostics
        + supercollider_diagnostics
    )
    if diagnostic_count:
        sanitizer_outcome = "detected"
    elif saw_instrumentation_evidence:
        sanitizer_outcome = "not_detected"
    else:
        sanitizer_outcome = "not_exercised"

    instrumentation_patch_count = sum(
        kind not in _MUTATION_PATCH_KINDS for kind in proof_patch_kinds
    )
    redundant_access_patch_count = sum(
        kind == "local-cave-lds-store-check-trap" for kind in proof_patch_kinds
    )
    reader_coverage = []
    for record in readers.values():
        record["inline_release_evidence"].sort(key=lambda value: value["index"])
        record["inline_token_evidence"].sort(key=lambda value: value["index"])
        record["inline_evidence_capacities"] = {
            "release": record.pop("inline_release_capacity"),
            "snapshot": record.pop("inline_snapshot_capacity"),
            "token": record.pop("inline_token_capacity"),
        }
        record["overflowed"] = (
            any(record["overflow_counts"].values())
            or any(record["exact_snapshot_counts"].values())
            or any(record["inline_release_snapshot_counts"].values())
            or any(record["inline_token_snapshot_counts"].values())
            or any(record["inline_coverage_counts"].values())
            or any(
                record["inline_evidence_counts"][key]
                for key in (
                    "malformed_records",
                    "duplicate_records",
                    "count_mismatches",
                    "capacity_violations",
                    "state_mismatches",
                )
            )
        )
        reader_coverage.append(record)
    reader_coverage.sort(key=lambda record: record["reader"])
    return {
        "mutation": {
            "accounting_schema_version": 2,
            "requested": requested,
            "planned": planned,
            "raw_applied": raw_applied,
            "applied": applied,
            "discarded_applied": discarded_applied,
            "installation_evidence_complete": installation_evidence_complete,
            "applicability": applicability,
            "inventoried_sites": fault_sites,
            "proof_patch_kinds": proof_patch_kinds,
            "fault_patch_kinds": fault_patch_kinds,
            "fault_patch_applications": len(fault_patch_kinds),
            "applied_readers": applied_readers,
            "readers": mutation_readers,
            "process_evidence_complete": process_evidence_complete,
            "processes": fault_process_evidence,
            "load_selection": load_selection,
        },
        "sanitizer": {
            "outcome": sanitizer_outcome,
            "inline_diagnostics": inline_diagnostics,
            "replay_diagnostics": replay_diagnostics,
            "sampled_conflicts": sampled_conflicts,
            "sampled_immediate_conflicts": sampled_immediate_conflicts,
            "supercollider_mismatches": supercollider_mismatches,
            "supercollider_diagnostics": supercollider_diagnostics,
            "measured_instability_count": supercollider_mismatches,
            "static_diagnostics": static_diagnostics,
            "diagnostic_count": diagnostic_count,
            "trap_attribution": "none",
        },
        "coverage": {
            "analysis_verdict": analysis_verdict,
            "analysis_complete": (
                analysis_verdict.get("analysis_complete")
                if isinstance(analysis_verdict, dict)
                else None
            ),
            "static_coverage_records": static_coverage_records,
            "patch_outcomes": patch_outcomes,
            "site_dispositions": coverage_sites,
            "site_disposition_parse_error": coverage_site_parse_error,
            "site_dispositions_complete": coverage_site_parse_error is None,
            "evidence_complete": (
                coverage_site_parse_error is None
                and resource_plan_alternative_parse_error is None
            ),
            "inventoried_fault_sites": fault_sites,
            "supported_sites": supported_sites,
            "skipped_sites": skipped_sites,
            "rejected_sites": rejected_sites,
            "patches": patch_count,
            "instrumentation_patches": instrumentation_patch_count,
            "redundant_access_patches": redundant_access_patch_count,
            "instrumentation_patch_kinds": instrumentation_patch_kinds,
            "selected_watchpoints": event_counts["sampled"],
            "sampled_claimed_windows": sampled_claimed_windows,
            "reader_access_events": sampled_reader_access_events,
            "writer_access_events": sampled_writer_access_events,
            "event_counts": event_counts,
            "overflow_counts": overflow_counts,
            "sampled_snapshot_counts": sampled_snapshot_counts,
            "exact_snapshot_counts": exact_snapshot_counts,
            "inline_release_snapshot_counts": inline_release_snapshot_counts,
            "inline_token_snapshot_counts": inline_token_snapshot_counts,
            "inline_coverage_counts": inline_coverage_counts,
            "inline_evidence_counts": inline_evidence_counts,
            "inline_release_evidence": sorted(
                inline_release_evidence,
                key=lambda value: (value["reader"], value["index"]),
            ),
            "inline_token_evidence": sorted(
                inline_token_evidence,
                key=lambda value: (value["reader"], value["index"]),
            ),
            "overflowed": any(overflow_counts.values())
            or any(sampled_snapshot_counts.values())
            or any(exact_snapshot_counts.values())
            or any(inline_release_snapshot_counts.values())
            or any(inline_token_snapshot_counts.values())
            or any(inline_coverage_counts.values())
            or any(
                inline_evidence_counts[key]
                for key in (
                    "malformed_records",
                    "duplicate_records",
                    "count_mismatches",
                    "capacity_violations",
                    "state_mismatches",
                )
            ),
            "readers": reader_coverage,
        },
        "metrics": {
            "report_buffer_count": report_buffers,
            "report_buffer_bytes": report_buffer_bytes,
            "report_plan_count": report_plan_count,
            "report_plans": report_plans,
            "report_region_capacity_entries": report_region_capacity_entries,
            "report_memory_summary_count": report_memory_summary_count,
            "report_memory_summaries": report_memory_summaries,
            "report_required_bytes": report_required_bytes,
            "report_allocated_bytes": report_allocated_bytes,
            "report_live_before_cleanup_bytes": report_live_before_cleanup_bytes,
            "report_live_after_cleanup_bytes": report_live_after_cleanup_bytes,
            "report_peak_live_bytes": report_peak_live_bytes,
            "report_allocation_failures": report_allocation_failures,
            "report_capacity_failures": report_capacity_failures,
            "report_cleanup_failures": report_cleanup_failures,
            "sc_report_buffer_count": sc_report_buffer_count,
            "sc_report_buffer_bytes": sc_report_buffer_count * 4,
            "sc_report_allocation_failures": sc_report_allocation_failures,
            "sc_report_read_failures": sc_report_read_failures,
            "sc_report_cleanup_failures": sc_report_cleanup_failures,
            "shadow_capacity_entries": shadow_capacity_entries,
            "diagnostic_capacity_entries": diagnostic_capacity_entries,
            "inline_atomic_release_capacity_entries": inline_atomic_release_capacity_entries,
            "inline_acquired_token_capacity_entries": inline_acquired_token_capacity_entries,
            "inline_causal_snapshot_capacity_entries": inline_causal_snapshot_capacity_entries,
            "spill_patch_count": spill_patch_count,
            "spill_slot_bytes": spill_slot_bytes,
            "resource_plan_alternative_counts": resource_plan_alternative_counts,
            "resource_plan_alternatives": resource_plan_alternatives,
            "resource_plan_alternative_record_count": (
                resource_plan_alternative_record_count
            ),
            "resource_plan_alternatives_truncated": (
                resource_plan_alternatives_truncated
            ),
            "resource_plan_alternatives_complete": (
                resource_plan_alternatives_complete
            ),
            "resource_plan_alternative_parse_error": (
                resource_plan_alternative_parse_error
            ),
            "private_segment_bytes": sum(
                record["private_segment_bytes"] for record in reader_coverage
            ),
            "workgroup_shadow_bytes": sum(
                record["workgroup_shadow_bytes"] for record in reader_coverage
            ),
            "group_segment_bytes": sum(
                record["group_segment_bytes"] for record in reader_coverage
            ),
        },
    }


def _code_object_metrics(hashes: list[dict[str, object]]) -> dict[str, int]:
    objects: dict[str, dict[str, int]] = {}
    for entry in hashes:
        label = str(entry.get("label", ""))
        match = re.fullmatch(r"captured:(.*)-(original|patched)\.hsaco", label)
        if match is None or "size" not in entry:
            continue
        objects.setdefault(match.group(1), {})[match.group(2)] = int(entry["size"])
    pairs = [
        value for value in objects.values() if {"original", "patched"} <= value.keys()
    ]
    original = sum(value["original"] for value in pairs)
    patched = sum(value["patched"] for value in pairs)
    return {
        "modified_code_object_count": len(pairs),
        "original_code_bytes": original,
        "patched_code_bytes": patched,
        "code_growth_bytes": patched - original,
    }


def _read_oracle_result(
    path: Path, command_log: str = ""
) -> tuple[dict[str, object], dict[str, object]]:
    default: dict[str, object] = {
        "outcome": "not_run",
        "source": "missing",
        "detail": None,
        "result_file": path.name,
    }
    if not path.is_file():
        iree_success = (
            "[SUCCESS] all function outputs matched their expected values."
            in command_log
        )
        iree_failure = "[FAILED] result[" in command_log
        if iree_success != iree_failure:
            default = {
                **default,
                "outcome": "pass" if iree_success else "fail",
                "source": "iree_expected_output_log",
                "detail": "iree-run-module checked every requested expected output",
            }
        else:
            passed = [
                int(value)
                for value in re.findall(
                    r"\[\s+PASSED\s+\]\s+(\d+)\s+tests?", command_log
                )
            ]
            failed = [
                int(value)
                for value in re.findall(
                    r"\[\s+FAILED\s+\]\s+(\d+)\s+tests?", command_log
                )
            ]
            gtest_outcome = (
                "fail"
                if failed and failed[-1] > 0
                else "pass" if passed and passed[-1] > 0 else None
            )
            if gtest_outcome is not None:
                default = {
                    **default,
                    "outcome": gtest_outcome,
                    "source": "gtest_assertions",
                    "detail": "GoogleTest workload assertions supplied the oracle",
                }
        return default, {
            "outcome": "unknown",
            "count": None,
            "expectation": None,
            "source": "missing",
            "detail": "no explicit source-diagnostic evidence",
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return (
            {
                **default,
                "outcome": "unknown",
                "source": "malformed",
                "detail": str(exc),
            },
            {
                "outcome": "unknown",
                "count": None,
                "expectation": None,
                "source": "malformed",
                "detail": str(exc),
            },
        )
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return (
            {
                **default,
                "outcome": "unknown",
                "source": "malformed",
                "detail": "oracle result must be a schema-version-1 JSON object",
            },
            {
                "outcome": "unknown",
                "count": None,
                "expectation": None,
                "source": "malformed",
                "detail": "result must be a schema-version-1 JSON object",
            },
        )
    outcome = value.get("oracle")
    if outcome not in {"pass", "fail", "unknown", "not_run"}:
        return (
            {
                **default,
                "outcome": "unknown",
                "source": "malformed",
                "detail": "oracle must be pass, fail, unknown, or not_run",
            },
            {
                "outcome": "unknown",
                "count": None,
                "expectation": None,
                "source": "malformed",
                "detail": "semantic oracle outcome is malformed",
            },
        )
    oracle = {
        "outcome": outcome,
        "source": "result_file",
        "detail": value.get("detail"),
        "result_file": path.name,
    }
    source = value.get("source_diagnostics")
    if not isinstance(source, dict):
        source_result = {
            "outcome": "unknown",
            "count": None,
            "expectation": None,
            "source": "missing" if source is None else "malformed",
            "detail": "source_diagnostics is missing or malformed",
        }
    else:
        count = source.get("count")
        expectation = source.get("expectation")
        source_outcome = source.get("outcome")
        valid_count = (
            isinstance(count, int) and not isinstance(count, bool) and count >= 0
        )
        if (
            expectation == "not_applicable"
            and count is None
            and source_outcome
            in {
                None,
                "not_applicable",
            }
        ):
            source_result = {
                **source,
                "outcome": "not_applicable",
                "source": "result_file",
                "result_file": path.name,
            }
        elif expectation in {"zero", "nonzero"} and valid_count:
            matched = (count == 0) if expectation == "zero" else (count > 0)
            expected_outcome = "matched" if matched else "mismatch"
            if source_outcome not in {None, expected_outcome}:
                source_result = {
                    "outcome": "unknown",
                    "count": count,
                    "expectation": expectation,
                    "source": "malformed",
                    "detail": "source outcome does not match count/expectation",
                }
            else:
                source_result = {
                    **source,
                    "outcome": expected_outcome,
                    "source": "result_file",
                    "result_file": path.name,
                }
        else:
            source_result = {
                "outcome": "unknown",
                "count": None,
                "expectation": expectation,
                "source": "malformed",
                "detail": "source diagnostic count/expectation is malformed",
            }
    return oracle, source_result


def _parallel_level(explicit_value: str | None = None) -> int:
    value = explicit_value or os.environ.get("CTEST_PARALLEL_LEVEL", str(MAX_GPU_JOBS))
    try:
        level = int(value)
    except ValueError as exc:
        raise ValueError(
            "CTEST_PARALLEL_LEVEL must be an integer from 1 through 4"
        ) from exc
    if level < 1 or level > MAX_GPU_JOBS:
        raise ValueError("CTEST_PARALLEL_LEVEL must be an integer from 1 through 4")
    return level


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _run_command(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout: float,
) -> tuple[int, bool]:
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            return (
                process.returncode
                if process.returncode is not None
                else -signal.SIGKILL
            ), True


def _parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    key, assigned = value.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("environment key must not be empty")
    return key, assigned


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("hash-file label and path must not be empty")
    return label, Path(path)


def _parse_command_json(value: str) -> list[str]:
    try:
        command = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("expected a JSON argv array") from exc
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) for item in command)
    ):
        raise argparse.ArgumentTypeError(
            "expected a non-empty JSON argv array of strings"
        )
    return command


def _hash_files(
    files: list[tuple[str, Path]], object_dir: Path
) -> list[dict[str, object]]:
    paths = list(files)
    if object_dir.exists():
        paths.extend(
            (f"captured:{path.name}", path)
            for path in sorted(object_dir.rglob("*"))
            if path.is_file()
        )
    values = []
    for label, path in paths:
        resolved = path.resolve()
        entry: dict[str, object] = {
            "label": label,
            "path": str(resolved),
            "exists": resolved.is_file(),
        }
        if resolved.is_file():
            entry.update(
                {"size": resolved.stat().st_size, "sha256": sha256_file(resolved)}
            )
        values.append(entry)
    return values


def _revisions(cwd: Path, roots: list[Path]) -> list[dict[str, object]]:
    candidates = [cwd, Path(__file__).resolve().parent, *roots]
    revisions: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        revision = git_identity(candidate, discover_root=True)
        if revision is not None:
            revisions[str(revision["root"])] = revision
    return [revisions[key] for key in sorted(revisions)]


def _relevant_environment(
    environment: dict[str, str], explicit: set[str]
) -> dict[str, str]:
    prefixes = ("RJ_", "HSA_", "ROCM_", "HIP_", "CTEST_")
    names = {
        key
        for key in environment
        if key.startswith(prefixes)
        or key in {"LD_LIBRARY_PATH", "PATH", "PYTHONPATH"}
        or key in explicit
    }
    return {key: environment[key] for key in sorted(names)}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pair-id", default=UNSPECIFIED)
    parser.add_argument(
        "--row-role",
        choices=("inventory", "baseline", "clean", "fault", UNSPECIFIED),
        default=UNSPECIFIED,
    )
    parser.add_argument("--corpus", default=UNSPECIFIED)
    parser.add_argument("--workload", default=UNSPECIFIED)
    parser.add_argument("--flavor")
    parser.add_argument("--engine")
    parser.add_argument("--fault-family", default=UNSPECIFIED)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--destructive", action="store_true")
    parser.add_argument("--allow-destructive", action="store_true")
    parser.add_argument(
        "--serialize-gpu",
        action="store_true",
        help="hold the global GPU lock across health probes and the workload",
    )
    parser.add_argument("--env", action="append", default=[], type=_parse_assignment)
    parser.add_argument("--revision-root", action="append", default=[], type=Path)
    parser.add_argument("--run-contract", type=Path)
    parser.add_argument(
        "--hash-file", action="append", default=[], type=_parse_labeled_path
    )
    parser.add_argument("--site-id", action="append", default=[])
    parser.add_argument("--health-command-json", type=_parse_command_json)
    parser.add_argument("--smoke-command-json", type=_parse_command_json)
    parser.add_argument("--health-timeout", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.health_timeout <= 0:
        parser.error("--health-timeout must be positive")
    if args.destructive and not args.allow_destructive:
        parser.error("--destructive requires --allow-destructive")
    if (args.health_command_json is None) != (args.smoke_command_json is None):
        parser.error(
            "--health-command-json and --smoke-command-json must be provided together"
        )
    if not args.name or args.name in {".", ".."} or Path(args.name).name != args.name:
        parser.error("--name must be one path-component-safe row name")
    return args


def _parse_replay_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a retained ConSan fault-row manifest"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--allow-destructive", action="store_true")
    parser.add_argument("--allow-drift", action="store_true")
    return parser.parse_args(argv)


def _parse_clear_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clear a ConSan GPU quarantine after checks"
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--health-command-json", type=_parse_command_json)
    parser.add_argument("--smoke-command-json", type=_parse_command_json, required=True)
    parser.add_argument("--health-timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def _parse_summarize_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize retained ConSan fault rows")
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    return parser.parse_args(argv)


def _normalized_spec(manifest: dict[str, object]) -> dict[str, str]:
    raw = manifest.get("spec", {})
    spec = raw if isinstance(raw, dict) else {}
    return {
        key: value if isinstance(value := spec.get(key), str) and value else UNSPECIFIED
        for key in (
            "pair_id",
            "row_role",
            "corpus",
            "workload",
            "flavor",
            "engine",
            "fault_family",
        )
    }


def _execution_is_adequate(manifest: dict[str, object]) -> bool:
    execution = manifest.get("execution")
    if not isinstance(execution, dict) or not execution.get("completed"):
        return False
    return execution.get("outcome") not in {
        "preflight_device_quarantined",
        "preflight_device_unhealthy",
        "device_lost",
        "timeout",
        "signal",
        "queue_timeout",
        "trap",
    }


def _pair_classification(
    clean_rows: list[dict[str, object]], fault_rows: list[dict[str, object]]
) -> tuple[str, dict[str, object]]:
    evidence: dict[str, object] = {}
    if len(clean_rows) != 1 or len(fault_rows) != 1:
        return "indeterminate_execution", evidence
    clean = clean_rows[0]
    fault = fault_rows[0]
    clean_oracle = clean.get("oracle", {})
    fault_oracle = fault.get("oracle", {})
    evidence["clean_oracle_outcome"] = (
        clean_oracle.get("outcome") if isinstance(clean_oracle, dict) else "unknown"
    )
    evidence["fault_oracle_outcome"] = (
        fault_oracle.get("outcome") if isinstance(fault_oracle, dict) else "unknown"
    )
    clean_source = clean.get("source_diagnostics", {})
    fault_source = fault.get("source_diagnostics", {})
    evidence["clean_source_diagnostic_outcome"] = (
        clean_source.get("outcome") if isinstance(clean_source, dict) else "unknown"
    )
    evidence["fault_source_diagnostic_outcome"] = (
        fault_source.get("outcome") if isinstance(fault_source, dict) else "unknown"
    )

    if (
        evidence["clean_source_diagnostic_outcome"] == "mismatch"
        or evidence["fault_source_diagnostic_outcome"] == "mismatch"
    ):
        return "source_calibration_failed", evidence

    if not _execution_is_adequate(clean):
        return "clean_control_failed", evidence
    clean_execution = clean.get("execution", {})
    if (
        not isinstance(clean_execution, dict)
        or clean_execution.get("outcome") != "passed"
    ):
        return "clean_control_failed", evidence
    if evidence["clean_oracle_outcome"] == "fail":
        return "clean_control_failed", evidence
    if evidence["clean_oracle_outcome"] != "pass":
        return "indeterminate_execution", evidence
    clean_sanitizer = clean.get("sanitizer", {})
    if (
        isinstance(clean_sanitizer, dict)
        and clean_sanitizer.get("outcome") == "detected"
    ):
        return "clean_control_failed", evidence
    if not _execution_is_adequate(fault):
        return "indeterminate_execution", evidence

    clean_coverage = clean.get("coverage", {})
    fault_coverage = fault.get("coverage", {})
    if (isinstance(clean_coverage, dict) and clean_coverage.get("overflowed")) or (
        isinstance(fault_coverage, dict) and fault_coverage.get("overflowed")
    ):
        return "overflowed", evidence

    mutation = fault.get("mutation")
    if not isinstance(mutation, dict):
        return "indeterminate_execution", evidence
    evidence["mutation_requested"] = mutation.get("requested")
    evidence["mutation_planned"] = mutation.get("planned")
    evidence["mutation_applied"] = mutation.get("applied")
    evidence["mutation_applicability"] = mutation.get("applicability")
    evidence["mutation_accounting_schema_version"] = mutation.get(
        "accounting_schema_version"
    )
    evidence["mutation_installation_evidence_complete"] = mutation.get(
        "installation_evidence_complete"
    )
    evidence["mutation_discarded_applied"] = mutation.get("discarded_applied", 0)
    if mutation.get("applicability") == "not_applicable":
        return "unsupported", evidence
    if (
        mutation.get("accounting_schema_version") != 2
        or mutation.get("installation_evidence_complete") is not True
        or mutation.get("discarded_applied", 0) != 0
        or mutation.get("requested") != 1
        or mutation.get("planned") != 1
        or mutation.get("applied") != 1
        or mutation.get("applicability") != "applied"
    ):
        return "fault_not_applied", evidence

    sanitizer = fault.get("sanitizer")
    if not isinstance(sanitizer, dict):
        return "indeterminate_execution", evidence
    sanitizer_outcome = sanitizer.get("outcome")
    evidence["sanitizer_outcome"] = sanitizer_outcome
    if sanitizer_outcome == "detected":
        return "qualified_detected", evidence
    if sanitizer_outcome == "not_detected":
        return "qualified_undetected", evidence
    return "indeterminate_execution", evidence


def _summarize(args: argparse.Namespace) -> int:
    root = args.artifact_root.resolve()
    manifests: list[tuple[Path, dict[str, object], dict[str, str]]] = []
    skipped_manifests = 0
    for path in sorted(root.rglob("result.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            skipped_manifests += 1
            continue
        if not isinstance(value, dict) or value.get("state") != "complete":
            skipped_manifests += 1
            continue
        manifests.append((path, value, _normalized_spec(value)))

    grouped: dict[
        tuple[str, ...], list[tuple[Path, dict[str, object], dict[str, str]]]
    ] = {}
    dimension_keys = (
        "pair_id",
        "corpus",
        "workload",
        "flavor",
        "engine",
        "fault_family",
    )
    for path, manifest, spec in manifests:
        if spec["pair_id"] == UNSPECIFIED:
            key = (f"unpaired:{path}", *(spec[name] for name in dimension_keys[1:]))
        else:
            key = tuple(spec[name] for name in dimension_keys)
        grouped.setdefault(key, []).append((path, manifest, spec))

    groups = []
    for key in sorted(grouped):
        rows = grouped[key]
        clean_rows = [
            manifest for _, manifest, spec in rows if spec["row_role"] == "clean"
        ]
        fault_rows = [
            manifest for _, manifest, spec in rows if spec["row_role"] == "fault"
        ]
        classification, evidence = _pair_classification(clean_rows, fault_rows)
        groups.append(
            {
                "pair_id": (
                    key[0] if not key[0].startswith("unpaired:") else UNSPECIFIED
                ),
                **dict(zip(dimension_keys[1:], key[1:])),
                "classification": classification,
                "clean_rows": [
                    str(path.relative_to(root))
                    for path, _, spec in rows
                    if spec["row_role"] == "clean"
                ],
                "fault_rows": [
                    str(path.relative_to(root))
                    for path, _, spec in rows
                    if spec["row_role"] == "fault"
                ],
                "other_rows": [
                    str(path.relative_to(root))
                    for path, _, spec in rows
                    if spec["row_role"] not in {"clean", "fault"}
                ],
                **evidence,
            }
        )

    summary = {
        "schema_version": 1,
        "artifact_root": str(root),
        "manifest_count": len(manifests),
        "skipped_manifest_count": skipped_manifests,
        "group_count": len(groups),
        "groups": groups,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "pair_id",
            "corpus",
            "workload",
            "flavor",
            "engine",
            "fault_family",
            "classification",
            "clean_oracle_outcome",
            "fault_oracle_outcome",
            "clean_source_diagnostic_outcome",
            "fault_source_diagnostic_outcome",
            "mutation_requested",
            "mutation_applied",
            "mutation_applicability",
            "sanitizer_outcome",
        ]
        with args.csv_out.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(groups)
    sys.stdout.write(rendered)
    return 0


def _default_health_command(environment: dict[str, str]) -> list[str] | None:
    rocm_path = environment.get("ROCM_PATH")
    if rocm_path:
        candidate = Path(rocm_path) / "bin" / "rocminfo"
        if candidate.is_file():
            return [str(candidate)]
    executable = shutil.which("rocminfo", path=environment.get("PATH"))
    return [executable] if executable else None


def _health_check(
    label: str,
    health_command: list[str],
    smoke_command: list[str],
    cwd: Path,
    environment: dict[str, str],
    row_dir: Path,
    timeout: float,
) -> dict[str, object]:
    # Health probes are control-plane commands, not part of the measured row.
    # Never let a smoke workload overwrite the row's retained oracle.
    environment = environment.copy()
    environment.pop(ROW_RESULT_ENV, None)
    environment.pop("CONSAN_WORKLOAD_RESULT_PATH", None)
    # The row environment intentionally enables DBI and may require a patch or
    # report record. Applying that contract to rocminfo or the smoke command
    # turns a healthy control into a false device-loss signal. Health gates
    # must observe the device without the sanitizer under test.
    environment.pop("HSA_TOOLS_LIB", None)
    for key in tuple(environment):
        if key.startswith("RJ_CONSAN_"):
            environment.pop(key)
    health_return, health_timeout = _run_command(
        health_command, cwd, environment, row_dir / f"{label}-rocminfo.log", timeout
    )
    smoke_return = None
    smoke_timeout = False
    if health_return == 0 and not health_timeout:
        smoke_return, smoke_timeout = _run_command(
            smoke_command, cwd, environment, row_dir / f"{label}-smoke.log", timeout
        )
    return {
        "healthy": health_return == 0
        and not health_timeout
        and smoke_return == 0
        and not smoke_timeout,
        "rocminfo_return_code": health_return,
        "rocminfo_timed_out": health_timeout,
        "smoke_return_code": smoke_return,
        "smoke_timed_out": smoke_timeout,
    }


def _write_quarantine(root: Path, row: str, reason: str) -> None:
    atomic_write_json(
        root / QUARANTINE_FILE,
        {"schema_version": 1, "row": row, "reason": reason, "created_at": _utc_now()},
    )


def _unique_replay_name(root: Path, requested: str) -> str:
    if not (root / requested).exists():
        return requested
    index = 2
    while (root / f"{requested}-{index}").exists():
        index += 1
    return f"{requested}-{index}"


def _check_replay_drift(manifest: dict[str, object]) -> list[str]:
    drift = []
    for entry in manifest.get("hashes_before", []):
        if not isinstance(entry, dict) or str(entry.get("label", "")).startswith(
            "captured:"
        ):
            continue
        if not entry.get("exists"):
            continue
        path = Path(str(entry["path"]))
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            drift.append(f"hash changed: {path}")
    for expected in manifest.get("git_revisions", []):
        if not isinstance(expected, dict):
            continue
        current = git_identity(Path(str(expected["root"])), discover_root=True)
        if current is None or current.get("head") != expected.get("head"):
            drift.append(f"git revision changed: {expected['root']}")
    return drift


def _replay_to_run_args(args: argparse.Namespace) -> list[str]:
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "complete":
        raise ValueError("only a complete manifest can be replayed")
    if manifest.get("destructive") and not args.allow_destructive:
        raise ValueError("replaying a destructive row requires --allow-destructive")
    drift = _check_replay_drift(manifest)
    if drift and not args.allow_drift:
        raise ValueError("replay input drift detected:\n  " + "\n  ".join(drift))

    root = args.artifact_root or manifest_path.parent.parent / "replays"
    requested_name = args.name or f"{manifest['name']}-replay"
    name = _unique_replay_name(root, requested_name)
    run_args = [
        "--artifact-root",
        str(root),
        "--name",
        name,
        "--timeout",
        str(manifest["timeout_seconds"]),
        "--cwd",
        str(manifest["cwd"]),
    ]
    spec = manifest.get("spec", {})
    if isinstance(spec, dict):
        spec_keys = (
            "pair_id",
            "row_role",
            "corpus",
            "workload",
            "flavor",
            "engine",
            "fault_family",
        )
        for key in spec_keys:
            value = spec.get(key)
            if isinstance(value, str) and value:
                run_args.extend(["--" + key.replace("_", "-"), value])
    if manifest.get("destructive"):
        run_args.extend(["--destructive", "--allow-destructive"])
    if manifest.get("gpu_serialized"):
        run_args.append("--serialize-gpu")
    if manifest.get("health_command") and manifest.get("smoke_command"):
        run_args.extend(
            ["--health-command-json", json.dumps(manifest["health_command"])]
        )
        run_args.extend(["--smoke-command-json", json.dumps(manifest["smoke_command"])])
    if manifest.get("health_timeout_seconds"):
        run_args.extend(["--health-timeout", str(manifest["health_timeout_seconds"])])
    for key, value in dict(manifest.get("environment", {})).items():
        if key not in {"RJ_CONSAN_DUMP_DIR", ROW_RESULT_ENV}:
            run_args.extend(["--env", f"{key}={value}"])
    for revision in manifest.get("git_revisions", []):
        run_args.extend(["--revision-root", str(revision["root"])])
    for hashed in manifest.get("hash_files", []):
        run_args.extend(["--hash-file", f"{hashed['label']}={hashed['path']}"])
    for site_id in manifest.get("site_identities", []):
        run_args.extend(["--site-id", str(site_id)])
    run_args.append("--")
    run_args.extend(str(item) for item in manifest["command"])
    return run_args


def _clear_quarantine(args: argparse.Namespace) -> int:
    environment = os.environ.copy()
    try:
        _parallel_level()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    health_command = args.health_command_json or _default_health_command(environment)
    if health_command is None:
        print("rocminfo not found; provide --health-command-json", file=sys.stderr)
        return 2
    root = args.artifact_root.resolve()
    quarantine = root / QUARANTINE_FILE
    if not quarantine.is_file():
        print(f"no GPU quarantine exists under {root}")
        return 0
    lock_path = Path(
        environment.get(GLOBAL_DESTRUCTIVE_LOCK_ENV, DEFAULT_GLOBAL_DESTRUCTIVE_LOCK)
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        # A prior clearer may have recovered the device while this process was
        # waiting. Conversely, holding the same lock as destructive rows keeps
        # a healthy probe from racing a newly submitted ill-formed workload.
        if not quarantine.is_file():
            print(f"no GPU quarantine exists under {root}")
            return 0
        check_dir = root / f"quarantine-clear-{time.time_ns()}"
        check_dir.mkdir(parents=True)
        check = _health_check(
            "clear",
            health_command,
            args.smoke_command_json,
            args.cwd.resolve(),
            environment,
            check_dir,
            args.health_timeout,
        )
        atomic_write_json(check_dir / "health.json", check)
        if not check["healthy"]:
            print(
                f"GPU quarantine retained; health checks failed: {check_dir}",
                file=sys.stderr,
            )
            return 70
        quarantine.unlink()
    print(f"GPU quarantine cleared after healthy checks: {check_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    effective_argv = sys.argv[1:] if argv is None else argv
    if effective_argv and effective_argv[0] == "summarize":
        return _summarize(_parse_summarize_args(effective_argv[1:]))
    if effective_argv and effective_argv[0] == "clear-quarantine":
        return _clear_quarantine(_parse_clear_args(effective_argv[1:]))
    if effective_argv and effective_argv[0] == "replay":
        replay_args = _parse_replay_args(effective_argv[1:])
        try:
            effective_argv = _replay_to_run_args(replay_args)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    args = _parse_args(effective_argv)
    try:
        parallel = _parallel_level(dict(args.env).get("CTEST_PARALLEL_LEVEL"))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    run_provenance = None
    if args.run_contract is not None:
        try:
            contract_path = args.run_contract.resolve()
            contract = load_contract(contract_path)
            drift = validate_current_contract(contract)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"invalid run contract: {error}", file=sys.stderr)
            return 2
        if contract_path.parent != args.artifact_root.resolve():
            print("run contract must reside in the artifact root", file=sys.stderr)
            return 2
        if args.name not in contract["declared_rows"]:
            print(f"row is not declared by run contract: {args.name}", file=sys.stderr)
            return 2
        if drift:
            print("run contract input drift:\n  " + "\n  ".join(drift), file=sys.stderr)
            return 2
        records = {record["label"]: record for record in contract["files"]}
        run_provenance = {
            "run_id": contract["run_id"],
            "contract_sha256": contract["contract_sha256"],
            "plan_canonical_sha256": records["plan"]["canonical_sha256"],
            "manifest_canonical_sha256": records["manifest"]["canonical_sha256"],
            "files": contract["files"],
            "sources": contract["sources"],
        }

    environment = os.environ.copy()
    explicit_environment = {key for key, _ in args.env}
    environment.update(args.env)
    for destructive_env in (INCOMPLETE_BARRIER_DROP_ENV, DIVERGENT_BARRIER_MOVE_ENV):
        enabled = environment.get(destructive_env, "").strip().lower()
        if enabled in {"1", "true", "yes", "on"} and not args.destructive:
            print(
                f"{destructive_env} requires --destructive containment",
                file=sys.stderr,
            )
            return 2
    health_command = args.health_command_json or _default_health_command(environment)
    smoke_command = args.smoke_command_json
    if args.destructive and health_command is None:
        print(
            "destructive rows require rocminfo or --health-command-json",
            file=sys.stderr,
        )
        return 2
    if args.destructive and smoke_command is None:
        print("destructive rows require --smoke-command-json", file=sys.stderr)
        return 2

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    quarantine_path = args.artifact_root / QUARANTINE_FILE
    if quarantine_path.exists():
        print(
            f"GPU is quarantined by {quarantine_path}; run clear-quarantine after recovery",
            file=sys.stderr,
        )
        return 75
    row_dir = args.artifact_root / args.name
    if row_dir.exists():
        print(f"artifact row already exists: {row_dir}", file=sys.stderr)
        return 2
    row_dir.mkdir()

    object_dir = row_dir / "code-objects"
    object_dir.mkdir()
    environment["RJ_CONSAN_DUMP_DIR"] = str(object_dir.resolve())
    explicit_environment.add("RJ_CONSAN_DUMP_DIR")
    oracle_path = row_dir / "oracle.json"
    environment[ROW_RESULT_ENV] = str(oracle_path.resolve())
    explicit_environment.add(ROW_RESULT_ENV)
    cwd = args.cwd.resolve()

    environment_flavor, environment_engine = _selection_from_environment(environment)
    spec = {
        "pair_id": args.pair_id,
        "row_role": args.row_role,
        "corpus": args.corpus,
        "workload": args.workload,
        "flavor": args.flavor or environment_flavor,
        "engine": args.engine or environment_engine,
        "fault_family": args.fault_family,
    }

    started_wall = _utc_now()
    started_mono = time.monotonic()
    gpu_serialized = args.serialize_gpu or args.destructive
    lock_path = Path(
        environment.get(GLOBAL_DESTRUCTIVE_LOCK_ENV, DEFAULT_GLOBAL_DESTRUCTIVE_LOCK)
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "name": args.name,
        "spec": spec,
        "command": args.command,
        "cwd": str(cwd),
        "environment": _relevant_environment(environment, explicit_environment),
        "git_revisions": _revisions(cwd, args.revision_root),
        "hash_files": [
            {"label": label, "path": str(path.resolve())}
            for label, path in args.hash_file
        ],
        "hashes_before": _hash_files(args.hash_file, object_dir),
        "site_identities": args.site_id,
        "timeout_seconds": args.timeout,
        "destructive": args.destructive,
        "destructive_lock": str(lock_path) if args.destructive else None,
        "gpu_serialized": gpu_serialized,
        "gpu_lock": str(lock_path) if gpu_serialized else None,
        "health_command": health_command,
        "smoke_command": smoke_command,
        "health_timeout_seconds": args.health_timeout,
        "ctest_parallel_level": parallel,
        "started_at": started_wall,
        "state": "running",
        "run_provenance": run_provenance,
    }
    result_path = row_dir / "result.json"
    atomic_write_json(result_path, result)

    lock_file = lock_path.open("a+b")
    pre_health = None
    post_health = None
    quarantined_after_lock = False
    return_code = 0
    timed_out = False
    command_ran = False
    command_elapsed = None
    try:
        if gpu_serialized:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        # The unlocked early check gives immediate feedback. This check is the
        # safety boundary: a preceding serialized row may have quarantined the
        # device while this row was waiting for the global GPU lock.
        if quarantine_path.exists():
            quarantined_after_lock = True
            (row_dir / "command.log").write_text(
                "row not launched: GPU became quarantined while waiting for the GPU lock\n",
                encoding="utf-8",
            )
        elif health_command is not None and smoke_command is not None:
            pre_health = _health_check(
                "before",
                health_command,
                smoke_command,
                cwd,
                environment,
                row_dir,
                args.health_timeout,
            )
        if quarantined_after_lock:
            pass
        elif pre_health is not None and not pre_health["healthy"]:
            (row_dir / "command.log").write_text(
                "row not launched: pre-run GPU health check failed\n", encoding="utf-8"
            )
            _write_quarantine(
                args.artifact_root, args.name, "pre-run health check failed"
            )
        else:
            command_ran = True
            command_started = time.monotonic()
            return_code, timed_out = _run_command(
                args.command, cwd, environment, row_dir / "command.log", args.timeout
            )
            command_elapsed = time.monotonic() - command_started
            if timed_out:
                _write_quarantine(args.artifact_root, args.name, "workload timeout")
            if health_command is not None and smoke_command is not None:
                post_health = _health_check(
                    "after",
                    health_command,
                    smoke_command,
                    cwd,
                    environment,
                    row_dir,
                    args.health_timeout,
                )
                if not post_health["healthy"]:
                    _write_quarantine(
                        args.artifact_root, args.name, "post-run health check failed"
                    )
    finally:
        if gpu_serialized:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

    log_text = (row_dir / "command.log").read_text(encoding="utf-8", errors="replace")
    normalized_log = log_text.lower()
    if quarantined_after_lock:
        outcome = "preflight_device_quarantined"
    elif pre_health is not None and not pre_health["healthy"]:
        outcome = "preflight_device_unhealthy"
    elif post_health is not None and not post_health["healthy"]:
        outcome = "device_lost"
    elif timed_out:
        outcome = "timeout"
    elif return_code < 0:
        outcome = "signal"
    elif "output mismatch" in normalized_log:
        outcome = "output_mismatch"
    elif "queue timeout" in normalized_log:
        outcome = "queue_timeout"
    elif (
        "gpu trap" in normalized_log
        or "device-side assert" in normalized_log
        or "hsa_status_error_illegal_instruction" in normalized_log
        or "illegal shader instruction" in normalized_log
        or "unspecified launch failure" in normalized_log
    ):
        outcome = "trap"
    elif return_code == 0:
        outcome = "passed"
    else:
        outcome = "failed"
    patch_inventory = [
        line
        for line in log_text.splitlines()
        if "ConSan proof patch" in line
        or "ConSan patch end" in line
        or "ConSan replacement reader" in line
    ]
    parsed = _parse_consan_log(log_text)
    if not parsed["coverage"]["evidence_complete"] and outcome == "passed":
        outcome = "evidence_incomplete"
    oracle, source_diagnostics = _read_oracle_result(oracle_path, log_text)
    hashes_after = _hash_files(args.hash_file, object_dir)
    parsed["metrics"].update(_code_object_metrics(hashes_after))
    parsed["metrics"]["elapsed_seconds"] = time.monotonic() - started_mono
    parsed["metrics"]["command_elapsed_seconds"] = command_elapsed
    execution = {
        "outcome": outcome,
        "completed": command_ran and not timed_out and return_code >= 0,
        "command_ran": command_ran,
        "return_code": return_code,
        "signal": -return_code if return_code < 0 else None,
        "timed_out": timed_out,
        "health_before": pre_health,
        "health_after": post_health,
        "quarantined_after_lock": quarantined_after_lock,
    }
    if outcome == "trap":
        parsed["sanitizer"]["trap_attribution"] = "unattributed"
    result.update(
        {
            "state": "complete",
            "outcome": outcome,
            "return_code": return_code,
            "command_ran": command_ran,
            "timed_out": timed_out,
            "health_before": pre_health,
            "health_after": post_health,
            "ended_at": _utc_now(),
            "elapsed_seconds": parsed["metrics"]["elapsed_seconds"],
            "hashes_after": hashes_after,
            "patch_inventory": patch_inventory,
            "execution": execution,
            "oracle": oracle,
            "source_diagnostics": source_diagnostics,
            **parsed,
            "stdout_stderr": "command.log",
            "replay": [
                sys.executable,
                str(Path(__file__).resolve()),
                "replay",
                str(result_path.resolve()),
            ],
        }
    )
    atomic_write_json(result_path, result)
    print(result_path)
    if outcome == "preflight_device_quarantined":
        return 75
    if outcome in {"preflight_device_unhealthy", "device_lost"}:
        return 70
    if timed_out:
        return 124
    return 1 if outcome != "passed" and return_code == 0 else return_code


if __name__ == "__main__":
    raise SystemExit(main())
