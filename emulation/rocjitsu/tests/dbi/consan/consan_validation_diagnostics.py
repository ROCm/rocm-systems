"""ConSan diagnostic, coverage, timing, and retained-code-object parsing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess

from consan_coverage_gate import CoverageParseError, parse_coverage_evidence
from consan_validation_catalog import (
    LLVM_READELF_ENV,
    MOI_DIAGNOSTIC_KINDS,
    MOI_SHADOW_ACCESS_WRITE,
    ValidationError,
)
from consan_validation_support import SITE_KINDS, sha256_file


_COVERAGE_DUMP_NAME = re.compile(
    r"rj-dbi-(?P<dump_id>[0-9]{6,})-reader-(?P<reader>[0-9]+)-"
    r"(?P<kind>original|patched)\.hsaco"
)


def _retained_relative_path(row_dir: Path, path: Path) -> str:
    row = row_dir.resolve()
    workload_root = row.parents[1]
    resolved = path.resolve()
    if not resolved.is_relative_to(workload_root):
        raise ValidationError(
            f"retained artifact path escapes workload artifacts: {resolved}"
        )
    return os.path.relpath(resolved, row)



ReplayIdentity = tuple[int, int]


@dataclass(frozen=True)
class _ReplayDiagnosticRecord:
    signature: str
    reader: int | None
    report_generation: int | None
    generation: int | None
    code_object_fingerprint: str | None
    index: int | None
    kind: int | None
    first_owner: int | None
    second_owner: int | None
    first_instruction: int | None
    second_instruction: int | None
    first_instruction_raw: str | None
    second_instruction_raw: str | None
    first_lds: str | None
    second_lds: str | None
    first_access_kind: int | None
    second_access_kind: int | None

    @property
    def report_identity(self) -> ReplayIdentity | None:
        if self.reader is None or self.report_generation is None:
            return None
        return self.reader, self.report_generation


@dataclass(frozen=True)
class DiagnosticRecord:
    signature: str
    source_identity: str | None
    code_object_fingerprint: str | None
    first_instruction: int | None
    second_instruction: int | None
    first_instruction_raw: str | None
    second_instruction_raw: str | None
    artifact_fields: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class DiagnosticSourceSummary:
    identity: str
    diagnostic_count: int
    record_count: int
    code_object_fingerprint: str
    artifact_fields: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class ParsedDiagnosticOutput:
    profile: str
    sources: tuple[DiagnosticSourceSummary, ...]
    records: tuple[DiagnosticRecord, ...]
    diagnostic_count: int
    pre_output_count: int
    structural_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosticPolicy:
    diagnostics: tuple[str, ...]
    max_diagnostics: int
    kind: str = "clean"
    instruction_groups: tuple[tuple[int, ...], ...] = ()
    code_object_fingerprint: str | None = None

    @classmethod
    def clean(cls) -> DiagnosticPolicy:
        return cls(diagnostics=(), max_diagnostics=0)


@dataclass(frozen=True)
class _ReplayReport:
    identity: ReplayIdentity
    code_object_fingerprint: str
    reported: int
    sampled_conflicts: int
    replay_required: bool
    visible_records: int
    diagnostic_capacity: int


@dataclass(frozen=True)
class _ReplaySummary:
    identity: ReplayIdentity
    code_object_fingerprint: str
    reported: int
    replay_input_access: int | None
    diagnostic_capacity_exhausted: bool | None
    diagnostic_capacity: int | None
    replay_scratch_diagnostic_capacity: int | None
    conflict: bool | None
    metadata_full: bool | None
    provenance_repaired: int
    provenance_unresolved: int | None


@dataclass(frozen=True)
class _ReplaySkipped:
    identity: ReplayIdentity
    code_object_fingerprint: str
    required_shadow_entries: int
    limit: int


class _DiagnosticFieldsError(ValueError):
    pass


def _log_fields(payload: str, context: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in payload.split():
        if "=" not in token:
            raise _DiagnosticFieldsError(
                f"malformed {context}: malformed field {token!r}"
            )
        key, value = token.split("=", 1)
        if re.fullmatch(r"[a-z_]+", key) is None or not value:
            raise _DiagnosticFieldsError(
                f"malformed {context}: malformed field {token!r}"
            )
        if key in fields:
            raise _DiagnosticFieldsError(
                f"malformed {context}: duplicate field {key!r}"
            )
        fields[key] = value
    return fields


def _parse_log_fields(
    line: str, marker: str, context: str, reasons: list[str]
) -> dict[str, str] | None:
    try:
        return _log_fields(line.split(marker, 1)[1], context)
    except (IndexError, _DiagnosticFieldsError) as error:
        reasons.append(str(error) or f"malformed {context}")
        return None


def _unsigned(fields: dict[str, str], name: str) -> int | None:
    try:
        value = int(fields[name], 0)
    except (KeyError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _boolean(fields: dict[str, str], name: str) -> bool | None:
    value = fields.get(name)
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _code_object_fingerprint(fields: dict[str, str]) -> str | None:
    value = fields.get("code_object")
    if value is None or re.fullmatch(r"fnv1a64:[0-9a-f]{16}", value) is None:
        return None
    return value


def _lds_range(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"\[(\d+),(\d+)\)", value)
    if match is None:
        return None
    begin, end = int(match.group(1)), int(match.group(2))
    return (begin, end) if begin < end else None


def _replay_identity(
    fields: dict[str, str], generation_field: str = "generation"
) -> ReplayIdentity | None:
    reader = _unsigned(fields, "reader")
    generation = _unsigned(fields, generation_field)
    if reader is None or generation is None:
        return None
    return reader, generation


def _identity_label(identity: ReplayIdentity) -> str:
    return f"reader={identity[0]},generation={identity[1]}"


def _replay_diagnostic_record(fields: dict[str, str]) -> _ReplayDiagnosticRecord:
    kind = _unsigned(fields, "kind")
    first_lds = _lds_range(fields.get("first_lds"))
    second_lds = _lds_range(fields.get("second_lds"))
    first_owner = _unsigned(fields, "first_owner")
    second_owner = _unsigned(fields, "second_owner")
    generation = _unsigned(fields, "generation")
    first_access_kind = _unsigned(fields, "first_kind")
    second_access_kind = _unsigned(fields, "second_kind")
    signature = (
        "exact-lds-write-write"
        if (
            kind == 1
            and generation is not None
            and fields.get("first_lds_known") == "true"
            and first_lds is not None
            and first_lds == second_lds
            and first_owner is not None
            and second_owner is not None
            and first_owner != second_owner
            and first_access_kind == MOI_SHADOW_ACCESS_WRITE
            and second_access_kind == MOI_SHADOW_ACCESS_WRITE
        )
        else (
            "malformed"
            if kind is None
            else MOI_DIAGNOSTIC_KINDS.get(kind, f"unknown-{kind}")
        )
    )
    return _ReplayDiagnosticRecord(
        signature=signature,
        reader=_unsigned(fields, "reader"),
        report_generation=_unsigned(fields, "report_generation"),
        generation=generation,
        code_object_fingerprint=_code_object_fingerprint(fields),
        index=_unsigned(fields, "index"),
        kind=kind,
        first_owner=first_owner,
        second_owner=second_owner,
        first_instruction=_unsigned(fields, "first_inst"),
        second_instruction=_unsigned(fields, "second_inst"),
        first_instruction_raw=fields.get("first_inst"),
        second_instruction_raw=fields.get("second_inst"),
        first_lds=fields.get("first_lds"),
        second_lds=fields.get("second_lds"),
        first_access_kind=first_access_kind,
        second_access_kind=second_access_kind,
    )


def _bool_label(value: bool | None) -> str:
    if value is None:
        return "missing"
    return "true" if value else "false"


def _instruction_label(value: int | None) -> str:
    return f"0x{value:x}" if value is not None else "missing"


def _parse_record_replay_diagnostic_output(log_text: str) -> ParsedDiagnosticOutput:
    reports: dict[ReplayIdentity, _ReplayReport] = {}
    replays: dict[ReplayIdentity, _ReplaySummary] = {}
    skipped_replays: dict[ReplayIdentity, _ReplaySkipped] = {}
    replay_records: list[_ReplayDiagnosticRecord] = []
    reasons: list[str] = []

    for line in log_text.splitlines():
        if "ConSan MOI auto report reader=" in line:
            reader_match = re.search(r"\breader=(\d+)", line)
            reader_label = (
                f"reader={reader_match.group(1)}"
                if reader_match is not None
                else "reader=missing"
            )
            if " needs hsa_memory_copy for coarse-grained summary" in line:
                reasons.append(
                    "pre-replay report requires hsa_memory_copy: " + reader_label
                )
                continue
            if " hsa_memory_copy failed status=" in line:
                status_match = re.search(r"\bstatus=([^ ]+)", line)
                status = (
                    status_match.group(1) if status_match is not None else "missing"
                )
                reasons.append(
                    "pre-replay hsa_memory_copy failed: "
                    f"{reader_label}, status={status}"
                )
                continue
            if " has invalid header magic=" in line:
                reasons.append("pre-replay report header invalid: " + reader_label)
                continue
            if " has inconsistent ABI-v" in line:
                reasons.append("pre-replay report layout inconsistent: " + reader_label)
                continue
            if " addr=" not in line or " bytes=" not in line:
                reasons.append("malformed pre-replay diagnostic summary")
                continue
            fields = _parse_log_fields(
                line,
                "ConSan MOI auto report ",
                "pre-replay diagnostic summary",
                reasons,
            )
            if fields is None:
                continue
            identity = _replay_identity(fields)
            fingerprint = _code_object_fingerprint(fields)
            diagnostics = _unsigned(fields, "diagnostics")
            diagnostic_capacity = _unsigned(fields, "diagnostic_capacity")
            address = _unsigned(fields, "addr")
            byte_count = _unsigned(fields, "bytes")
            # Older report ABIs emitted every engine's counters in one common
            # summary.  Current mode-local renderers correctly omit Sampled
            # counters from Record/Replay reports.  Preserve validation of a
            # legacy counter when present without requiring another mode's
            # schema from a Record/Replay producer.
            sampled_counts = tuple(
                _unsigned(fields, name) if name in fields else 0
                for name in ("sampled_conflicts", "sampled_immediate_conflicts")
            )
            visible_counts = tuple(
                _unsigned(fields, name)
                for name in (
                    "visible_records",
                    "visible_barriers",
                    "visible_atomics",
                    "visible_fences",
                )
            )
            if (
                identity is None
                or fingerprint is None
                or diagnostics is None
                or diagnostic_capacity is None
                or address is None
                or byte_count is None
                or any(count is None for count in sampled_counts)
                or any(count is None for count in visible_counts)
            ):
                reasons.append("malformed pre-replay diagnostic summary")
            elif identity in reports:
                reasons.append(
                    f"duplicate pre-replay summary {_identity_label(identity)}"
                )
            else:
                reports[identity] = _ReplayReport(
                    identity=identity,
                    code_object_fingerprint=fingerprint,
                    reported=diagnostics,
                    sampled_conflicts=sum(
                        count for count in sampled_counts if count is not None
                    ),
                    replay_required=any(
                        count for count in visible_counts if count is not None
                    ),
                    visible_records=visible_counts[0],
                    diagnostic_capacity=diagnostic_capacity,
                )
        elif "ConSan MOI auto replay diagnostic reader=" in line:
            fields = _parse_log_fields(
                line,
                "ConSan MOI auto replay diagnostic ",
                "replay diagnostic detail",
                reasons,
            )
            if fields is not None:
                replay_records.append(_replay_diagnostic_record(fields))
        elif "ConSan MOI auto replay reader=" in line:
            payload_line = line
            if " skipped " in f" {line} ":
                payload_line = line.replace(" skipped ", " ", 1)
            fields = _parse_log_fields(
                payload_line,
                "ConSan MOI auto replay ",
                (
                    "replay-skipped summary"
                    if payload_line != line
                    else "replay diagnostic summary"
                ),
                reasons,
            )
            if fields is None:
                continue
            identity = _replay_identity(fields)
            fingerprint = _code_object_fingerprint(fields)
            if payload_line != line:
                required = _unsigned(fields, "required_shadow_entries")
                limit = _unsigned(fields, "limit")
                if (
                    identity is None
                    or fingerprint is None
                    or required is None
                    or limit is None
                ):
                    reasons.append("malformed replay-skipped summary")
                elif identity in skipped_replays:
                    reasons.append(
                        f"duplicate replay-skipped summary {_identity_label(identity)}"
                    )
                else:
                    skipped_replays[identity] = _ReplaySkipped(
                        identity=identity,
                        code_object_fingerprint=fingerprint,
                        required_shadow_entries=required,
                        limit=limit,
                    )
                continue
            diagnostics = _unsigned(fields, "diagnostics")
            provenance_repaired = _unsigned(fields, "provenance_repaired")
            if (
                identity is None
                or fingerprint is None
                or diagnostics is None
                or provenance_repaired is None
            ):
                reasons.append("malformed replay diagnostic summary")
            elif identity in replays:
                reasons.append(f"duplicate replay summary {_identity_label(identity)}")
            else:
                replays[identity] = _ReplaySummary(
                    identity=identity,
                    code_object_fingerprint=fingerprint,
                    reported=diagnostics,
                    replay_input_access=_unsigned(fields, "replay_input_access"),
                    diagnostic_capacity_exhausted=_boolean(
                        fields, "diagnostic_capacity_exhausted"
                    ),
                    diagnostic_capacity=_unsigned(fields, "diagnostic_capacity"),
                    replay_scratch_diagnostic_capacity=_unsigned(
                        fields, "replay_scratch_diagnostic_capacity"
                    ),
                    conflict=_boolean(fields, "conflict"),
                    metadata_full=_boolean(fields, "metadata_full"),
                    provenance_repaired=provenance_repaired,
                    provenance_unresolved=_unsigned(fields, "provenance_unresolved"),
                )

    details_by_identity: dict[ReplayIdentity | None, list[_ReplayDiagnosticRecord]] = {}
    for record in replay_records:
        if record.generation is None:
            reasons.append("replay diagnostic detail has malformed generation")
        details_by_identity.setdefault(record.report_identity, []).append(record)

    if not reports:
        reasons.append("missing pre-replay report summary")
    # This is the exact producer guard around automatic replay: a report needs
    # replay only when it contains a visible access or synchronization record.
    required_replay = {
        identity for identity, summary in reports.items() if summary.replay_required
    }
    skipped_identities = set(skipped_replays)
    missing_replay = sorted(required_replay - set(replays) - skipped_identities)
    unexpected_replay = sorted(set(replays) - required_replay)
    unexpected_skipped = sorted(skipped_identities - required_replay)
    for identity in sorted(skipped_identities & required_replay):
        skipped = skipped_replays[identity]
        reasons.append(
            f"replay skipped: {_identity_label(identity)}, "
            f"required_shadow_entries={skipped.required_shadow_entries}, "
            f"limit={skipped.limit}"
        )
    if missing_replay:
        reasons.append(
            "missing replay summaries: "
            + ", ".join(_identity_label(identity) for identity in missing_replay)
        )
    if unexpected_replay:
        reasons.append(
            "unexpected replay summaries: "
            + ", ".join(_identity_label(identity) for identity in unexpected_replay)
        )
    if unexpected_skipped:
        reasons.append(
            "unexpected replay-skipped summaries: "
            + ", ".join(_identity_label(identity) for identity in unexpected_skipped)
        )
    conflicting_replay = sorted(set(replays) & skipped_identities)
    if conflicting_replay:
        reasons.append(
            "replay both skipped and summarized: "
            + ", ".join(_identity_label(identity) for identity in conflicting_replay)
        )

    source_summaries = []
    for identity, summary in sorted(replays.items()):
        details = details_by_identity.get(identity, ())
        detailed = len(details)
        report = reports.get(identity)
        report_fingerprint = (
            report.code_object_fingerprint if report is not None else None
        )
        if report_fingerprint != summary.code_object_fingerprint:
            reasons.append(
                "replay code-object fingerprint mismatch: "
                f"{_identity_label(identity)}, "
                f"pre_report={report_fingerprint or 'missing'}, "
                f"replay={summary.code_object_fingerprint}"
            )
        detail_fingerprints = {record.code_object_fingerprint for record in details}
        if details and detail_fingerprints != {summary.code_object_fingerprint}:
            reasons.append(
                "replay diagnostic code-object fingerprint mismatch: "
                f"{_identity_label(identity)}"
            )
        if summary.reported != detailed:
            reasons.append(
                "replay diagnostic inventory incomplete: "
                f"{_identity_label(identity)}, "
                f"reported={summary.reported}, detailed={detailed}"
            )
        indices = [record.index for record in details]
        concrete_indices = [index for index in indices if index is not None]
        indices_are_contiguous = (
            not indices
            if summary.reported == 0
            else (
                len(indices) == summary.reported
                and len(set(concrete_indices)) == summary.reported
                and min(concrete_indices, default=-1) == 0
                and max(concrete_indices, default=-1) == summary.reported - 1
            )
        )
        if any(index is None for index in indices) or not indices_are_contiguous:
            actual_indices = ",".join(
                "malformed" if index is None else str(index) for index in indices
            )
            expected_indices = (
                f"0..{summary.reported - 1}" if summary.reported else "none"
            )
            reasons.append(
                "replay diagnostic indices invalid: "
                f"{_identity_label(identity)}, "
                f"expected={expected_indices}, "
                f"actual={actual_indices or 'none'}"
            )
        is_legacy_capacity_summary = (
            summary.replay_input_access is None
            and summary.replay_scratch_diagnostic_capacity is None
        )
        expected_diagnostic_capacity = None
        if report is not None:
            expected_diagnostic_capacity = (
                min(report.diagnostic_capacity, report.visible_records)
                if is_legacy_capacity_summary
                else report.diagnostic_capacity
            )
        if summary.diagnostic_capacity != expected_diagnostic_capacity:
            reasons.append(
                "replay diagnostic capacity mismatch: "
                f"{_identity_label(identity)}, "
                f"value={summary.diagnostic_capacity}, "
                f"expected={expected_diagnostic_capacity}"
            )
        expected_scratch_diagnostic_capacity = (
            min(report.diagnostic_capacity, summary.replay_input_access)
            if report is not None and summary.replay_input_access is not None
            else None
        )
        if (
            summary.replay_scratch_diagnostic_capacity is not None
            and expected_scratch_diagnostic_capacity is not None
            and summary.replay_scratch_diagnostic_capacity
            != expected_scratch_diagnostic_capacity
        ):
            reasons.append(
                "replay scratch diagnostic capacity mismatch: "
                f"{_identity_label(identity)}, "
                f"value={summary.replay_scratch_diagnostic_capacity}, "
                f"expected={expected_scratch_diagnostic_capacity}"
            )
        expected_conflict = summary.reported != 0
        if summary.conflict is not expected_conflict:
            reasons.append(
                f"replay conflict status invalid: {_identity_label(identity)}, "
                f"value={_bool_label(summary.conflict)}, "
                f"expected={_bool_label(expected_conflict)}"
            )
        if summary.diagnostic_capacity_exhausted is not False:
            reasons.append(
                "replay diagnostic capacity status invalid: "
                f"{_identity_label(identity)}, "
                f"value={_bool_label(summary.diagnostic_capacity_exhausted)}"
            )
        if summary.metadata_full is not False:
            reasons.append(
                f"replay metadata status invalid: {_identity_label(identity)}, "
                f"value={_bool_label(summary.metadata_full)}"
            )
        if summary.provenance_unresolved != 0:
            reasons.append(
                f"replay provenance unresolved: {_identity_label(identity)}, "
                f"count={summary.provenance_unresolved}"
            )
        if summary.provenance_repaired > summary.reported:
            reasons.append(
                "replay provenance repaired exceeds diagnostics: "
                f"{_identity_label(identity)}, "
                f"repaired={summary.provenance_repaired}, "
                f"diagnostics={summary.reported}"
            )
        source_summaries.append(
            DiagnosticSourceSummary(
                identity=_identity_label(identity),
                diagnostic_count=summary.reported,
                record_count=detailed,
                code_object_fingerprint=summary.code_object_fingerprint,
                artifact_fields=(
                    ("reader", identity[0]),
                    ("replay_input_access", summary.replay_input_access),
                    (
                        "diagnostic_capacity_exhausted",
                        summary.diagnostic_capacity_exhausted,
                    ),
                    ("diagnostic_capacity", summary.diagnostic_capacity),
                    (
                        "replay_scratch_diagnostic_capacity",
                        summary.replay_scratch_diagnostic_capacity,
                    ),
                    ("conflict", summary.conflict),
                    ("metadata_full", summary.metadata_full),
                    ("provenance_repaired", summary.provenance_repaired),
                    ("provenance_unresolved", summary.provenance_unresolved),
                ),
            )
        )

    for identity in details_by_identity:
        if identity is None:
            reasons.append("replay diagnostic detail has malformed identity")
        elif identity not in replays:
            reasons.append(
                "replay diagnostic detail has no summary: "
                f"{_identity_label(identity)}"
            )

    pre_output_count = sum(summary.reported for summary in reports.values())
    sampled_conflict_count = sum(
        summary.sampled_conflicts for summary in reports.values()
    )
    if pre_output_count:
        reasons.append(f"pre-replay diagnostics={pre_output_count}")
    if sampled_conflict_count:
        reasons.append(f"pre-replay sampled conflicts={sampled_conflict_count}")
    records = tuple(
        DiagnosticRecord(
            signature=record.signature,
            source_identity=(
                _identity_label(record.report_identity)
                if record.report_identity is not None
                else None
            ),
            code_object_fingerprint=record.code_object_fingerprint,
            first_instruction=record.first_instruction,
            second_instruction=record.second_instruction,
            first_instruction_raw=record.first_instruction_raw,
            second_instruction_raw=record.second_instruction_raw,
            artifact_fields=(
                ("reader", record.reader),
                ("report_generation", record.report_generation),
                ("generation", record.generation),
                ("index", record.index),
                ("kind", record.kind),
                ("first_owner", record.first_owner),
                ("second_owner", record.second_owner),
                ("first_lds", record.first_lds),
                ("second_lds", record.second_lds),
                ("first_access_kind", record.first_access_kind),
                ("second_access_kind", record.second_access_kind),
            ),
        )
        for record in replay_records
    )
    return ParsedDiagnosticOutput(
        profile="record-replay",
        sources=tuple(source_summaries),
        records=records,
        diagnostic_count=sum(source.diagnostic_count for source in source_summaries),
        pre_output_count=pre_output_count,
        structural_reasons=tuple(reasons),
    )


DIAGNOSTIC_OUTPUT_PARSERS = {
    "record-replay": _parse_record_replay_diagnostic_output,
}


def _diagnostic_record_result(record: DiagnosticRecord) -> dict:
    result = {
        "signature": record.signature,
        "source_identity": record.source_identity,
        "code_object_fingerprint": record.code_object_fingerprint,
        "first_instruction": (
            _instruction_label(record.first_instruction)
            if record.first_instruction is not None
            else None
        ),
        "second_instruction": (
            _instruction_label(record.second_instruction)
            if record.second_instruction is not None
            else None
        ),
        "first_instruction_raw": record.first_instruction_raw,
        "second_instruction_raw": record.second_instruction_raw,
    }
    result.update(record.artifact_fields)
    return result


def _diagnostic_source_result(summary: DiagnosticSourceSummary) -> dict:
    result = {
        "reported": summary.diagnostic_count,
        "detailed": summary.record_count,
        "code_object_fingerprint": summary.code_object_fingerprint,
    }
    result.update(summary.artifact_fields)
    return result


def _evaluate_diagnostic_output(
    output: ParsedDiagnosticOutput,
    policy: DiagnosticPolicy,
) -> dict:
    reasons = list(output.structural_reasons)
    if policy.diagnostics and not policy.instruction_groups:
        reasons.append(
            "policy declares diagnostics without qualified instruction groups"
        )
    observed_signatures = {record.signature for record in output.records}
    unexpected = sorted(observed_signatures - set(policy.diagnostics))
    if output.diagnostic_count > policy.max_diagnostics:
        reasons.append(
            "replay diagnostics exceed declared maximum: "
            f"observed={output.diagnostic_count}, maximum={policy.max_diagnostics}"
        )
    if unexpected:
        reasons.append(f"unexpected diagnostics={','.join(unexpected)}")

    if policy.code_object_fingerprint is not None:
        for source in output.sources:
            if (
                source.diagnostic_count != 0 or source.record_count != 0
            ) and source.code_object_fingerprint != policy.code_object_fingerprint:
                reasons.append(
                    "diagnostic code-object fingerprint does not match contract: "
                    f"{source.identity}, "
                    f"observed={source.code_object_fingerprint}, "
                    f"contract={policy.code_object_fingerprint}"
                )
        if not any(
            source.code_object_fingerprint == policy.code_object_fingerprint
            for source in output.sources
        ):
            reasons.append(
                "contract code-object fingerprint missing from diagnostic summaries: "
                f"expected={policy.code_object_fingerprint}"
            )

    unexpected_sites = []
    if policy.instruction_groups:
        for record in output.records:
            same_qualified_group = (
                record.first_instruction is not None
                and record.second_instruction is not None
                and any(
                    record.first_instruction in group
                    and record.second_instruction in group
                    for group in policy.instruction_groups
                )
            )
            if not same_qualified_group:
                unexpected_sites.append(
                    f"{record.source_identity or 'missing'}:"
                    f"{_instruction_label(record.first_instruction)}->"
                    f"{_instruction_label(record.second_instruction)}"
                )
    if unexpected_sites:
        reasons.append(f"unexpected diagnostic sites={','.join(unexpected_sites)}")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "profile": output.profile,
        "policy": {
            "kind": policy.kind,
            "diagnostics": list(policy.diagnostics),
            "max_diagnostics": policy.max_diagnostics,
            "code_object_fingerprint": policy.code_object_fingerprint,
            "instruction_groups": [list(group) for group in policy.instruction_groups],
        },
        "observed_signatures": sorted(observed_signatures),
        "observed_code_object_fingerprints": sorted(
            {source.code_object_fingerprint for source in output.sources}
        ),
        "diagnostic_count": output.diagnostic_count,
        "replay_count": output.diagnostic_count,
        "pre_replay_count": output.pre_output_count,
        "readers": {
            source.identity: _diagnostic_source_result(source)
            for source in output.sources
        },
        "records": [_diagnostic_record_result(record) for record in output.records],
    }


def _diagnostic_output_summary(
    log_text: str,
    profile: str,
) -> dict:
    parser = DIAGNOSTIC_OUTPUT_PARSERS.get(profile)
    if parser is None:
        raise ValidationError(
            f"no complete diagnostic-output parser for profile: {profile}"
        )
    output = parser(log_text)
    return _evaluate_diagnostic_output(output, DiagnosticPolicy.clean())


def _coverage_summary(
    log_text: str,
    profile: str | None = None,
) -> dict:
    try:
        evidence = parse_coverage_evidence(log_text)
    except CoverageParseError as error:
        rejection_prefix = "[rocjitsu-dbi-hooks] ConSan load rejection "
        rejection_lines = [
            line[len(rejection_prefix) :]
            for line in log_text.splitlines()
            if line.startswith(rejection_prefix)
        ]
        if rejection_lines:
            fields = dict(re.findall(r"([a-z_]+)=([^ ]+)", rejection_lines[-1]))
            return {
                "accepted": False,
                "error": "ConSan rejected a code object before execution",
                "load_rejection": fields,
            }
        return {"accepted": False, "error": str(error)}
    verdict = evidence.verdict
    reasons = []
    if not verdict.applicable:
        reasons.append("no applicable code object")
    if not verdict.analysis_complete:
        reasons.append("analysis incomplete")
    elif not verdict.static_complete:
        reasons.append("static coverage incomplete")
    if not verdict.dynamic_complete:
        reasons.append("dynamic coverage incomplete")
    if verdict.counts["dynamic_incomplete"] != 0:
        reasons.append(f"dynamic_incomplete={verdict.counts['dynamic_incomplete']}")
    for kind in SITE_KINDS:
        patched, supported = verdict.patched_supported[kind]
        if patched != supported:
            reasons.append(f"{kind}={patched}/{supported}")
    if any(record.expert_limit for record in evidence.coverage):
        reasons.append("expert patch limit enabled")
    summary = {
        "accepted": not reasons,
        "reasons": reasons,
        "analysis_complete": verdict.analysis_complete,
        "static_complete": verdict.static_complete,
        "dynamic_complete": verdict.dynamic_complete,
        "patched_supported": {
            kind: list(verdict.patched_supported[kind]) for kind in SITE_KINDS
        },
        "dynamic_incomplete": verdict.counts["dynamic_incomplete"],
    }
    if profile in DIAGNOSTIC_OUTPUT_PARSERS:
        diagnostics = _diagnostic_output_summary(log_text, profile)
        summary["diagnostics"] = diagnostics
        summary["reasons"].extend(diagnostics["reasons"])
        summary["accepted"] = not summary["reasons"]
    return summary


def _benchmark_samples(path: Path) -> list[float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("benchmarks", [])
    iterations = [
        row
        for row in rows
        if row.get("run_type") == "iteration"
        and str(row.get("name", "")).startswith("BM_main/")
    ]
    selected = iterations
    if not selected:
        selected = [
            row
            for row in rows
            if row.get("aggregate_name") == "median"
            and str(row.get("name", "")).startswith("BM_main/")
        ]
    if not selected:
        raise ValidationError(f"expected Qwen benchmark timing rows in {path}")
    benchmark_names = {str(row.get("name", "")) for row in selected}
    if len(benchmark_names) != 1:
        raise ValidationError(
            f"expected one Qwen benchmark identity in {path}, found "
            f"{sorted(benchmark_names)}"
        )
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
    try:
        return [float(row["real_time"]) * scale[row["time_unit"]] for row in selected]
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError(f"malformed Qwen benchmark timing in {path}") from error


def _benchmark_median(path: Path) -> float:
    return statistics.median(_benchmark_samples(path))


def _json_timing_samples(log_text: str, workload_kind: str) -> dict[str, list[float]]:
    documents = []
    for line in log_text.splitlines():
        if line.startswith("{"):
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(documents) != 1:
        raise ValidationError(f"expected one {workload_kind} JSON result")
    document = documents[0]
    timings = {}
    for key, value in document.items():
        if not isinstance(value, dict):
            continue
        if "median_ms" in value:
            samples = value.get("samples_ms")
            if samples is None:
                samples = [value["median_ms"]]
            if (
                not isinstance(samples, list)
                or not samples
                or any(
                    not isinstance(sample, (int, float))
                    or not math.isfinite(float(sample))
                    or sample <= 0
                    for sample in samples
                )
            ):
                raise ValidationError(
                    f"invalid {workload_kind} timing samples for {key}"
                )
            timings[key] = [float(sample) for sample in samples]
        device_samples = value.get("device_samples_ms")
        if device_samples is None and "device_median_ms" in value:
            device_samples = [value["device_median_ms"]]
        if device_samples is not None:
            if (
                not isinstance(device_samples, list)
                or not device_samples
                or any(
                    not isinstance(sample, (int, float))
                    or not math.isfinite(float(sample))
                    or sample <= 0
                    for sample in device_samples
                )
            ):
                raise ValidationError(
                    f"invalid {workload_kind} device timing samples for {key}"
                )
            timings[f"{key}:device"] = [float(sample) for sample in device_samples]
    return timings


def _json_measurements(log_text: str, workload_kind: str) -> dict[str, dict]:
    documents = []
    for line in log_text.splitlines():
        if line.startswith("{"):
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(documents) != 1:
        raise ValidationError(f"expected one {workload_kind} JSON result")
    measurements = {
        key: value
        for key, value in documents[0].items()
        if isinstance(key, str) and isinstance(value, dict)
    }
    if not measurements:
        raise ValidationError(f"expected {workload_kind} measurement rows")
    return measurements


def _json_medians(log_text: str, workload_kind: str) -> dict[str, float]:
    return {
        key: statistics.median(values)
        for key, values in _json_timing_samples(log_text, workload_kind).items()
    }


def _sharktank_medians(log_text: str) -> dict[str, float]:
    return _json_medians(log_text, "Sharktank")


def _gtest_timing_samples(log_texts: list[str]) -> dict[str, list[float]]:
    pattern = re.compile(r"\[==========\].*\(([0-9]+) ms total\)")
    values = []
    for log_text in log_texts:
        matches = pattern.findall(log_text)
        if not matches:
            raise ValidationError("missing GTest total latency")
        values.append(float(matches[-1]))
    return {"process": values}


def _gtest_median(log_texts: list[str]) -> dict[str, float]:
    return {
        mode: statistics.median(values)
        for mode, values in _gtest_timing_samples(log_texts).items()
    }


def _gtest_device_measurement(
    log_text: str, benchmark: str, expected_iterations: int
) -> tuple[float, dict[str, object]]:
    number = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    matches = re.findall(
        rf"^hip_moi_gpu_timing benchmark={re.escape(benchmark)} timer=hip-event "
        rf"aggregate_ms=({number}) iterations=([0-9]+) "
        rf"per_iteration_ms=({number})$",
        log_text,
        re.MULTILINE,
    )
    if len(matches) != 1:
        raise ValidationError(
            f"expected one {benchmark} GPU timing row, found {len(matches)}"
        )
    aggregate_ms = float(matches[0][0])
    iterations = int(matches[0][1])
    per_iteration_ms = float(matches[0][2])
    if iterations != expected_iterations:
        raise ValidationError(
            f"{benchmark} GPU timing iteration mismatch: "
            f"{iterations} != {expected_iterations}"
        )
    if (
        not math.isfinite(aggregate_ms)
        or not math.isfinite(per_iteration_ms)
        or aggregate_ms <= 0.0
        or per_iteration_ms <= 0.0
        or not math.isclose(aggregate_ms / iterations, per_iteration_ms, rel_tol=2.0e-6)
    ):
        raise ValidationError(f"{benchmark} GPU timing row is inconsistent")
    return per_iteration_ms, {
        "benchmark_iterations": iterations,
        "timed_aggregate_ms": aggregate_ms,
        "timing_source": "hip-event",
    }


def _discard_first_sample_per_process(
    per_run: list[dict[str, list[float]]],
) -> list[dict[str, list[float]]]:
    discarded = []
    for item in per_run:
        if any(len(values) < 2 for values in item.values()):
            raise ValidationError(
                "cannot discard one warmup sample from a process with fewer than "
                "two timing samples"
            )
        discarded.append({key: values[1:] for key, values in item.items()})
    return discarded


def _nonnegative_float(fields: dict[str, str], name: str) -> float | None:
    try:
        value = float(fields[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def _empirical_structural_metrics(log_text: str) -> dict[str, object]:
    readers: dict[int, dict[str, object]] = {}
    process_memory: dict[str, int] = {}
    reasons = []

    def parse(line: str, marker: str, context: str) -> dict[str, str] | None:
        return _parse_log_fields(line, marker, context, reasons)

    for line in log_text.splitlines():
        fields = None
        if "ConSan waitcheck timing reader=" in line:
            fields = parse(line, "ConSan waitcheck timing ", "waitcheck timing")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                elapsed = _nonnegative_float(fields, "elapsed_ms")
                if reader is None or elapsed is None:
                    reasons.append("malformed waitcheck timing")
                else:
                    readers.setdefault(reader, {})["waitcheck_ms"] = elapsed
        elif "ConSan MOI inventory end reader=" in line:
            fields = parse(line, "ConSan MOI inventory end ", "MOI inventory timing")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                elapsed = _nonnegative_float(fields, "elapsed_ms")
                if reader is None or elapsed is None:
                    reasons.append("malformed MOI inventory timing")
                else:
                    readers.setdefault(reader, {})["inventory_ms"] = elapsed
        elif "ConSan patch begin reader=" in line:
            fields = parse(line, "ConSan patch begin ", "patch begin")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                byte_count = _unsigned(fields, "bytes")
                if reader is None or byte_count is None:
                    reasons.append("malformed patch begin")
                else:
                    readers.setdefault(reader, {})["original_bytes"] = byte_count
        elif "ConSan patch end reader=" in line:
            fields = parse(line, "ConSan patch end ", "patch end")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                elapsed = _nonnegative_float(fields, "patch_ms")
                patches = _unsigned(fields, "patches")
                if reader is None or elapsed is None or patches is None:
                    reasons.append("patch end lacks empirical timing fields")
                else:
                    record = readers.setdefault(reader, {})
                    record["patch_ms"] = elapsed
                    record["patches"] = patches
                    record["modified"] = _boolean(fields, "modified")
                    record["outcome"] = fields.get("outcome")
        elif "ConSan replacement reader=" in line:
            fields = parse(line, "ConSan replacement ", "replacement image")
            if fields is not None:
                reader = _unsigned(fields, "original_reader")
                byte_count = _unsigned(fields, "bytes")
                if reader is None or byte_count is None:
                    reasons.append("malformed replacement image")
                else:
                    readers.setdefault(reader, {})["patched_bytes"] = byte_count
        elif "ConSan MOI resources reader=" in line:
            fields = parse(line, "ConSan MOI resources ", "MOI resources")
            if fields is not None:
                reader = _unsigned(fields, "reader")
                if reader is None:
                    reasons.append("malformed MOI resources")
                    continue
                resource_names = (
                    "explicit",
                    "dead",
                    "descriptor_growth",
                    "spill",
                    "unsupported",
                    "planned_spill_slot_bytes",
                    "emitted_spill_patches",
                    "emitted_spill_slot_bytes",
                    "alternative_attempts",
                    "alternative_selected",
                    "alternative_rejected",
                    "alternative_superseded",
                    "alternative_contributed",
                    "alternative_vetoed",
                )
                resources = {name: _unsigned(fields, name) for name in resource_names}
                if any(value is None for value in resources.values()):
                    reasons.append("malformed MOI resources")
                else:
                    readers.setdefault(reader, {})["resources"] = resources
        elif "ConSan MOI report memory required_bytes=" in line:
            fields = parse(line, "ConSan MOI report memory ", "report memory")
            if fields is not None:
                for source, destination in (
                    ("required_bytes", "report_required_bytes"),
                    ("allocated_bytes", "report_allocated_bytes"),
                    ("peak_live_bytes", "report_peak_live_bytes"),
                    ("allocation_failures", "report_allocation_failures"),
                    ("capacity_failures", "report_capacity_failures"),
                    ("cleanup_failures", "report_cleanup_failures"),
                ):
                    value = _unsigned(fields, source)
                    if value is None:
                        reasons.append(f"malformed report memory field {source}")
                    else:
                        process_memory[destination] = value
        elif "ConSan transform admission memory live_bytes=" in line:
            fields = parse(
                line,
                "ConSan transform admission memory ",
                "transform admission memory",
            )
            if fields is not None:
                value = _unsigned(fields, "peak_reserved_bytes")
                if value is None:
                    reasons.append("malformed transform admission memory")
                else:
                    process_memory["transform_peak_reserved_bytes"] = value
        elif "ConSan patched-image memory live_bytes=" in line:
            fields = parse(line, "ConSan patched-image memory ", "patched image memory")
            if fields is not None:
                value = _unsigned(fields, "peak_image_bytes")
                if value is None:
                    reasons.append("malformed patched image memory")
                else:
                    process_memory["patched_image_peak_bytes"] = value
        elif "ConSan patched-image growth memory live_bytes=" in line:
            fields = parse(
                line,
                "ConSan patched-image growth memory ",
                "patched image growth memory",
            )
            if fields is not None:
                value = _unsigned(fields, "peak_growth_bytes")
                if value is None:
                    reasons.append("malformed patched image growth memory")
                else:
                    process_memory["patched_image_peak_growth_bytes"] = value

    code_objects = []
    for reader, record in sorted(readers.items()):
        original = record.get("original_bytes")
        patched = record.get("patched_bytes", original)
        record["reader"] = reader
        record["patched_bytes"] = patched
        if isinstance(original, int) and isinstance(patched, int):
            record["growth_bytes"] = patched - original
            record["growth_ratio"] = patched / original if original else None
        code_objects.append(record)
    patched = [
        record["patch_ms"]
        for record in code_objects
        if isinstance(record.get("patch_ms"), float)
    ]
    return {
        "accepted": not reasons and bool(patched),
        "reasons": reasons,
        "code_objects": code_objects,
        "total_patch_ms": sum(patched),
        "process_memory": process_memory,
    }


def _empirical_structural_totals(result: dict) -> dict[str, float]:
    if result.get("accepted") is not True:
        raise ValidationError("empirical row was rejected")
    runs = result.get("structural_metrics_runs")
    if not isinstance(runs, list) or not runs:
        raise ValidationError("empirical row has no structural metrics")
    if any(run.get("accepted") is not True for run in runs):
        raise ValidationError("empirical row has rejected structural metrics")
    totals = {"patch_ms": 0.0, "waitcheck_ms": 0.0, "inventory_ms": 0.0}
    for run in runs:
        totals["patch_ms"] += float(run["total_patch_ms"])
        for record in run["code_objects"]:
            for source, destination in (
                ("waitcheck_ms", "waitcheck_ms"),
                ("inventory_ms", "inventory_ms"),
            ):
                value = record.get(source)
                if isinstance(value, (int, float)):
                    totals[destination] += float(value)
    return totals


def _parse_amdgpu_kernel_metadata(text: str) -> dict[str, object]:
    integer_fields = {
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "sgpr_count",
        "sgpr_spill_count",
        "vgpr_count",
        "vgpr_spill_count",
        "agpr_count",
        "accum_offset",
        "wavefront_size",
        "max_flat_workgroup_size",
    }
    kernels = []
    current: dict[str, object] | None = None
    for line in text.splitlines():
        start = re.match(r"^  - \.([a-z_]+):\s*(.*?)\s*$", line)
        if start is not None:
            if current:
                kernels.append(current)
            current = {}
            name, value = start.groups()
            if name in integer_fields and value:
                try:
                    current[name] = int(value, 0)
                except ValueError:
                    current[name] = None
        if current is None:
            continue
        match = re.match(r"^    \.([a-z_]+):\s*(.*?)\s*$", line)
        if match is None:
            continue
        name, value = match.groups()
        if name == "name":
            current[name] = value
        elif name in integer_fields:
            try:
                current[name] = int(value, 0)
            except ValueError:
                current[name] = None
    if current:
        kernels.append(current)
    return {
        "kernels": kernels,
        "kernel_count": len(kernels),
        "fields": sorted(
            {name for kernel in kernels for name in kernel if name != "name"}
        ),
    }


def _llvm_readelf() -> Path | None:
    configured = os.environ.get(LLVM_READELF_ENV)
    if configured:
        return Path(os.path.abspath(Path(configured).expanduser()))
    discovered = shutil.which("llvm-readelf")
    return Path(discovered) if discovered else None


def _amdgpu_kernel_metadata(path: Path) -> dict[str, object]:
    tool = _llvm_readelf()
    if tool is None:
        return {
            "accepted": False,
            "reason": "llvm-readelf is unavailable",
            "tool": None,
        }
    try:
        completed = subprocess.run(
            [str(tool), "--notes", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"accepted": False, "reason": str(error), "tool": str(tool)}
    if completed.returncode != 0:
        return {
            "accepted": False,
            "reason": completed.stderr.strip() or "llvm-readelf failed",
            "tool": str(tool),
        }
    metadata = _parse_amdgpu_kernel_metadata(completed.stdout)
    return {
        "accepted": metadata["kernel_count"] > 0,
        "reason": None if metadata["kernel_count"] > 0 else "no AMDGPU kernels found",
        "tool": str(tool),
        **metadata,
    }


def _retained_code_object_inventory(row_dir: Path) -> dict[str, object]:
    records: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(row_dir.glob("code-objects-*/*.hsaco")):
        match = _COVERAGE_DUMP_NAME.fullmatch(path.name)
        if match is None:
            continue
        key = (match.group("dump_id"), match.group("reader"))
        record = records.setdefault(
            key,
            {"dump_id": key[0], "reader": int(key[1])},
        )
        record[match.group("kind")] = {
            "path": _retained_relative_path(row_dir, path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "metadata": _amdgpu_kernel_metadata(path),
        }
    pairs = []
    for record in records.values():
        original = record.get("original")
        patched = record.get("patched")
        if isinstance(original, dict) and isinstance(patched, dict):
            record["growth_bytes"] = patched["size"] - original["size"]
            record["growth_ratio"] = (
                patched["size"] / original["size"] if original["size"] else None
            )
            original_kernels = {
                kernel.get("name"): kernel
                for kernel in original["metadata"].get("kernels", [])
                if isinstance(kernel.get("name"), str)
            }
            patched_kernels = {
                kernel.get("name"): kernel
                for kernel in patched["metadata"].get("kernels", [])
                if isinstance(kernel.get("name"), str)
            }
            original_names = sorted(original_kernels)
            patched_names = sorted(patched_kernels)
            resource_fields = (
                "group_segment_fixed_size",
                "private_segment_fixed_size",
                "sgpr_count",
                "sgpr_spill_count",
                "vgpr_count",
                "vgpr_spill_count",
                "agpr_count",
            )
            record["kernel_metadata_delta"] = {
                "original_names": original_names,
                "patched_names": patched_names,
                "name_sets_match": original_names == patched_names,
                "kernels": {
                    name: {
                        field: patched_kernels[name][field]
                        - original_kernels[name][field]
                        for field in resource_fields
                        if isinstance(original_kernels[name].get(field), int)
                        and isinstance(patched_kernels[name].get(field), int)
                    }
                    for name in sorted(set(original_kernels) & set(patched_kernels))
                },
            }
        pairs.append(record)
    return {
        "pairs": pairs,
        "complete_pairs": sum("growth_bytes" in row for row in pairs),
        "metadata_complete_pairs": sum(
            "growth_bytes" in row
            and row["original"]["metadata"]["accepted"]
            and row["patched"]["metadata"]["accepted"]
            for row in pairs
        ),
    }


def _gtest_test_count(log_text: str) -> int | None:
    matches = re.findall(
        r"\[==========\]\s+Running\s+([0-9]+)\s+tests?\s+from",
        log_text,
    )
    return int(matches[-1]) if matches else None
