#!/usr/bin/env python3
"""Parses ConSan coverage records and fails closed on incomplete analysis."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys

from consan_validation_support import SITE_KINDS


PREFIX = "[rocjitsu-dbi-hooks] ConSan "
COVERAGE_KIND = "coverage"
VERDICT_KIND = "analysis verdict"
SITE_KIND = "coverage_site"
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_COUNT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_PAIR = re.compile(r"(0|[1-9][0-9]*)/(0|[1-9][0-9]*)\Z")
_HEX_COUNT = re.compile(r"0x(?:0|[1-9a-f][0-9a-f]*)\Z")
_UINT64_MAX = (1 << 64) - 1


class CoverageParseError(ValueError):
    """The log does not contain unambiguous, well-formed coverage evidence."""


@dataclass(frozen=True)
class CoverageRecord:
    reader: int
    load: int | None
    flavor: str
    engine: str
    analysis_complete: bool
    expert_limit: bool
    counts: dict[str, int]

    @property
    def applicable(self) -> bool:
        return any(self.counts[f"{kind}_discovered"] != 0 for kind in SITE_KINDS)


@dataclass(frozen=True)
class CoverageSiteRecord:
    reader: int
    load: int | None
    kind: str
    disposition: str
    reason: str
    outcome: str
    lowering_reason: str
    resource_reason: str
    container: str
    scope: str
    text_offset: int
    mnemonic: str


@dataclass(frozen=True)
class AnalysisVerdict:
    applicable: bool
    analysis_complete: bool
    static_complete: bool
    dynamic_complete: bool
    applicable_code_objects: int
    incomplete_code_objects: int
    patched_supported: dict[str, tuple[int, int]]
    counts: dict[str, int]


@dataclass(frozen=True)
class CoverageEvidence:
    coverage: tuple[CoverageRecord, ...]
    sites: tuple[CoverageSiteRecord, ...]
    verdict: AnalysisVerdict


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    reasons: tuple[str, ...]
    evidence: CoverageEvidence


_COVERAGE_COUNT_FIELDS = (
    "access_discovered",
    "access_supported",
    "access_selected",
    "access_patched",
    "access_unsupported",
    "access_resource_failed",
    "access_placement_or_lowering_failed",
    "access_expert_limit_omitted",
    "barrier_discovered",
    "barrier_supported",
    "barrier_selected",
    "barrier_patched",
    "barrier_unsupported",
    "barrier_resource_failed",
    "barrier_placement_or_lowering_failed",
    "barrier_expert_limit_omitted",
    "atomic_discovered",
    "atomic_supported",
    "atomic_selected",
    "atomic_patched",
    "atomic_unsupported",
    "atomic_resource_failed",
    "atomic_placement_or_lowering_failed",
    "atomic_expert_limit_omitted",
    "fence_discovered",
    "fence_supported",
    "fence_selected",
    "fence_patched",
    "fence_unsupported",
    "fence_resource_failed",
    "fence_placement_or_lowering_failed",
    "fence_expert_limit_omitted",
)

_VERDICT_COUNT_FIELDS = (
    "applicable_code_objects",
    "incomplete_code_objects",
    "dynamic_incomplete",
    "replay_unsupported_access",
    "replay_unsupported_atomics",
    "replay_unsupported_fences",
    "replay_metadata_full",
)

_SITE_DISPOSITIONS = {"not_applicable", "supported", "unsupported"}
_SITE_REASONS = {
    "none",
    "non_access_instruction",
    "unsupported_mnemonic",
    "invalid_instruction_size",
    "invalid_access_width",
    "missing_address_operand",
    "instruction_out_of_bounds",
    "unsupported_flat_encoding",
    "nonzero_flat_offset",
    "reserved_flat_address_register",
    "non_group_address_space",
    "flat_provenance_policy_excluded",
    "runtime_kernel_excluded",
    "non_flat_atomic_address",
    "missing_atomic_operands",
    "missing_ordering_metadata",
    "unqualified_sync_sequence",
    "no_preceding_sampled_window",
    "unsupported_memory_role",
    "missing_scope",
    "unsupported_scope",
    "compare_exchange_outcome_unavailable",
    "unsupported_atomic_outcome",
    "unsupported_atomic_encoding",
    "unsupported_metadata_encoding",
    "ineligible_fence",
    "missing_communication_event",
    "invalid_barrier_encoding",
}
_SITE_OUTCOMES = {
    "not_applicable",
    "unsupported",
    "pending",
    "patched",
    "resource_failed",
    "placement_or_lowering_failed",
}
_SITE_LOWERING_REASONS = {
    "none",
    "semantic_not_applicable",
    "semantic_unsupported",
    "unsupported_resource_plan",
    "instrumentation_patch_missing",
}
_RESOURCE_REASONS = {
    "none",
    "invalid_request",
    "explicit_misaligned",
    "explicit_out_of_range",
    "explicit_live",
    "forbidden_overlap",
    "missing_instruction",
    "missing_owner",
    "ambiguous_owners",
    "invalid_descriptor",
    "no_legal_window",
    "dynamic_stack",
}


def _fields(payload: str, context: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in payload.split():
        if "=" not in token:
            raise CoverageParseError(f"{context}: malformed field {token!r}")
        key, value = token.split("=", 1)
        if not _KEY.fullmatch(key) or not value:
            raise CoverageParseError(f"{context}: malformed field {token!r}")
        if key in result:
            raise CoverageParseError(f"{context}: duplicate field {key!r}")
        result[key] = value
    return result


def _require(fields: dict[str, str], names: tuple[str, ...], context: str) -> None:
    missing = [name for name in names if name not in fields]
    if missing:
        raise CoverageParseError(f"{context}: missing fields: {', '.join(missing)}")


def _boolean(fields: dict[str, str], name: str, context: str) -> bool:
    value = fields[name]
    if value not in ("true", "false"):
        raise CoverageParseError(f"{context}: {name} must be true or false, got {value!r}")
    return value == "true"


def _count(fields: dict[str, str], name: str, context: str) -> int:
    value = fields[name]
    if not _COUNT.fullmatch(value):
        raise CoverageParseError(f"{context}: {name} is not an unsigned decimal count: {value!r}")
    parsed = int(value)
    if parsed > _UINT64_MAX:
        raise CoverageParseError(f"{context}: {name} exceeds uint64: {value!r}")
    return parsed


def _load(fields: dict[str, str], context: str) -> int | None:
    if "load" not in fields:
        return None
    value = _count(fields, "load", context)
    if value == 0:
        raise CoverageParseError(f"{context}: load occurrence must be nonzero")
    return value


def _pair(fields: dict[str, str], name: str, context: str) -> tuple[int, int]:
    value = fields[name]
    match = _PAIR.fullmatch(value)
    if match is None:
        raise CoverageParseError(
            f"{context}: {name} is not a patched/supported count pair: {value!r}"
        )
    parsed = int(match.group(1)), int(match.group(2))
    if any(value > _UINT64_MAX for value in parsed):
        raise CoverageParseError(f"{context}: {name} exceeds uint64: {fields[name]!r}")
    return parsed


def _choice(fields: dict[str, str], name: str, choices: set[str], context: str) -> str:
    value = fields[name]
    if value not in choices:
        raise CoverageParseError(f"{context}: unknown {name}: {value!r}")
    return value


def _parse_site(payload: str, line_number: int) -> CoverageSiteRecord:
    context = f"coverage_site line {line_number}"
    fields = _fields(payload, context)
    required = (
        "reader",
        "kind",
        "disposition",
        "reason",
        "outcome",
        "lowering_reason",
        "resource_reason",
        "container",
        "scope",
        "text",
        "mnemonic",
    )
    _require(fields, required, context)
    if not _HEX_COUNT.fullmatch(fields["text"]):
        raise CoverageParseError(
            f"{context}: text is not a canonical hexadecimal offset: {fields['text']!r}"
        )
    text_offset = int(fields["text"], 16)
    if text_offset > _UINT64_MAX:
        raise CoverageParseError(f"{context}: text exceeds uint64: {fields['text']!r}")
    record = CoverageSiteRecord(
        reader=_count(fields, "reader", context),
        load=_load(fields, context),
        kind=_choice(fields, "kind", set(SITE_KINDS), context),
        disposition=_choice(fields, "disposition", _SITE_DISPOSITIONS, context),
        reason=_choice(fields, "reason", _SITE_REASONS, context),
        outcome=_choice(fields, "outcome", _SITE_OUTCOMES, context),
        lowering_reason=_choice(
            fields, "lowering_reason", _SITE_LOWERING_REASONS, context
        ),
        resource_reason=_choice(fields, "resource_reason", _RESOURCE_REASONS, context),
        container=fields["container"],
        scope=_choice(fields, "scope", {"kernel", "function"}, context),
        text_offset=text_offset,
        mnemonic=fields["mnemonic"],
    )
    expected = {
        "patched": ("supported", "none", "none"),
        "unsupported": ("unsupported", "semantic_unsupported", "none"),
        "resource_failed": (
            "supported",
            "unsupported_resource_plan",
            None,
        ),
        "placement_or_lowering_failed": (
            "supported",
            "instrumentation_patch_missing",
            "none",
        ),
    }.get(record.outcome)
    if expected is not None:
        disposition, lowering_reason, resource_reason = expected
        if record.disposition != disposition or record.lowering_reason != lowering_reason:
            raise CoverageParseError(
                f"{context}: inconsistent disposition/lowering reason for {record.outcome}"
            )
        if resource_reason is None:
            if record.resource_reason == "none":
                raise CoverageParseError(
                    f"{context}: resource_failed requires a detailed resource reason"
                )
        elif record.resource_reason != resource_reason:
            raise CoverageParseError(
                f"{context}: {record.outcome} requires resource_reason={resource_reason}"
            )
    if record.outcome == "unsupported" and record.reason == "none":
        raise CoverageParseError(f"{context}: unsupported requires a semantic reason")
    if record.outcome in {
        "patched",
        "resource_failed",
        "placement_or_lowering_failed",
    } and record.reason != "none":
        raise CoverageParseError(f"{context}: {record.outcome} requires reason=none")
    return record


def _parse_coverage(payload: str, line_number: int) -> CoverageRecord:
    context = f"coverage line {line_number}"
    fields = _fields(payload, context)
    required = (
        "reader",
        "flavor",
        "engine",
        "analysis_complete",
        "expert_limit",
        *_COVERAGE_COUNT_FIELDS,
    )
    _require(fields, required, context)
    flavor = _choice(fields, "flavor", {"moi", "supercollider"}, context)
    engines = {
        "moi": {"record_replay", "inline_shadow", "sampled"},
        "supercollider": {"supercollider"},
    }
    record = CoverageRecord(
        reader=_count(fields, "reader", context),
        load=_load(fields, context),
        flavor=flavor,
        engine=_choice(fields, "engine", engines[flavor], context),
        analysis_complete=_boolean(fields, "analysis_complete", context),
        expert_limit=_boolean(fields, "expert_limit", context),
        counts={name: _count(fields, name, context) for name in _COVERAGE_COUNT_FIELDS},
    )
    complete = True
    for kind in SITE_KINDS:
        discovered = record.counts[f"{kind}_discovered"]
        supported = record.counts[f"{kind}_supported"]
        selected = record.counts[f"{kind}_selected"]
        patched = record.counts[f"{kind}_patched"]
        unsupported = record.counts[f"{kind}_unsupported"]
        resource_failed = record.counts[f"{kind}_resource_failed"]
        placement_failed = record.counts[f"{kind}_placement_or_lowering_failed"]
        expert_omitted = record.counts[f"{kind}_expert_limit_omitted"]
        if discovered != supported + unsupported:
            raise CoverageParseError(
                f"{context}: {kind} discovered != supported + unsupported: "
                f"{discovered} != {supported} + {unsupported}"
            )
        if supported != selected + expert_omitted:
            raise CoverageParseError(
                f"{context}: {kind} supported != selected + expert omissions: "
                f"{supported} != {selected} + {expert_omitted}"
            )
        if selected != patched + resource_failed + placement_failed:
            raise CoverageParseError(
                f"{context}: {kind} selected != patched + failed: "
                f"{selected} != {patched} + {resource_failed} + {placement_failed}"
            )
        complete &= unsupported == resource_failed == placement_failed == expert_omitted == 0
    if record.analysis_complete != complete:
        raise CoverageParseError(
            f"{context}: analysis_complete disagrees with independently derived counters"
        )
    return record


def _parse_verdict(payload: str, line_number: int) -> AnalysisVerdict:
    context = f"analysis verdict line {line_number}"
    booleans = ("applicable", "analysis_complete", "static_complete", "dynamic_complete")
    fields = _fields(payload, context)
    _require(fields, (*booleans, *SITE_KINDS, *_VERDICT_COUNT_FIELDS), context)
    pairs = {kind: _pair(fields, kind, context) for kind in SITE_KINDS}
    counts = {name: _count(fields, name, context) for name in _VERDICT_COUNT_FIELDS}
    return AnalysisVerdict(
        applicable=_boolean(fields, "applicable", context),
        analysis_complete=_boolean(fields, "analysis_complete", context),
        static_complete=_boolean(fields, "static_complete", context),
        dynamic_complete=_boolean(fields, "dynamic_complete", context),
        applicable_code_objects=counts["applicable_code_objects"],
        incomplete_code_objects=counts["incomplete_code_objects"],
        patched_supported=pairs,
        counts=counts,
    )


def _aggregate_verdicts(verdicts: list[AnalysisVerdict]) -> AnalysisVerdict:
    """Combines process-final verdicts from one multi-client workload log."""
    applicable = [verdict for verdict in verdicts if verdict.applicable]
    return AnalysisVerdict(
        applicable=bool(applicable),
        analysis_complete=bool(applicable)
        and all(verdict.analysis_complete for verdict in applicable),
        static_complete=bool(applicable)
        and all(verdict.static_complete for verdict in applicable),
        dynamic_complete=all(verdict.dynamic_complete for verdict in verdicts),
        applicable_code_objects=sum(verdict.applicable_code_objects for verdict in verdicts),
        incomplete_code_objects=sum(verdict.incomplete_code_objects for verdict in verdicts),
        patched_supported={
            kind: (
                sum(verdict.patched_supported[kind][0] for verdict in verdicts),
                sum(verdict.patched_supported[kind][1] for verdict in verdicts),
            )
            for kind in SITE_KINDS
        },
        counts={
            name: sum(verdict.counts[name] for verdict in verdicts)
            for name in _VERDICT_COUNT_FIELDS
        },
    )


def parse_coverage_site_records(log_text: str) -> tuple[CoverageSiteRecord, ...]:
    """Strictly retains every machine-readable per-site hook record."""
    sites = []
    for line_number, line in enumerate(log_text.splitlines(), 1):
        marker = line.find(PREFIX)
        if marker < 0:
            continue
        record = line[marker + len(PREFIX) :]
        if record.startswith(SITE_KIND + " "):
            sites.append(_parse_site(record[len(SITE_KIND) + 1 :], line_number))
    return tuple(sites)


def parse_coverage_evidence(log_text: str) -> CoverageEvidence:
    """Returns strict coverage evidence from a complete hook log."""
    coverage: list[CoverageRecord] = []
    sites = parse_coverage_site_records(log_text)
    verdicts: list[AnalysisVerdict] = []
    for line_number, line in enumerate(log_text.splitlines(), 1):
        marker = line.find(PREFIX)
        if marker < 0:
            continue
        record = line[marker + len(PREFIX) :]
        if record.startswith(COVERAGE_KIND + " "):
            coverage.append(_parse_coverage(record[len(COVERAGE_KIND) + 1 :], line_number))
        elif record.startswith(VERDICT_KIND + " "):
            verdicts.append(_parse_verdict(record[len(VERDICT_KIND) + 1 :], line_number))

    if not coverage:
        raise CoverageParseError("missing ConSan coverage record")
    if not verdicts:
        raise CoverageParseError("missing ConSan analysis verdict")
    identities = [(record.reader, record.load) for record in coverage]
    duplicate_identities = sorted(
        identity for identity in set(identities) if identities.count(identity) > 1
    )
    ambiguous_duplicates = [
        identity for identity in duplicate_identities
        if identity[1] is not None
        or any(
            record.flavor != "supercollider"
            for record in coverage
            if (record.reader, record.load) == identity
        )
    ]
    if ambiguous_duplicates:
        raise CoverageParseError(
            "ambiguous duplicate coverage load identities: "
            + ", ".join(f"reader={reader},load={load}" for reader, load in ambiguous_duplicates)
        )

    identity_set = set(identities)
    unknown_site_identities = sorted(
        {(site.reader, site.load) for site in sites} - identity_set
    )
    if unknown_site_identities:
        raise CoverageParseError(
            "coverage_site records reference unknown load identities: "
            + ", ".join(
                f"reader={reader},load={load}" for reader, load in unknown_site_identities
            )
        )
    for coverage_record in coverage:
        reader_sites = tuple(
            site for site in sites
            if (site.reader, site.load) == (coverage_record.reader, coverage_record.load)
        )
        if coverage_record.flavor == "supercollider":
            if reader_sites:
                raise CoverageParseError(
                    f"SuperCollider reader {coverage_record.reader} has MOI coverage_site rows"
                )
            continue
        # Compact production logs intentionally retain only the signed aggregate
        # coverage and verdict records. When verbose per-site evidence is present,
        # keep enforcing its exact agreement with those aggregates; an entirely
        # absent inventory is the compact-log representation, not missing evidence.
        if not reader_sites:
            continue
        for kind in SITE_KINDS:
            retained = tuple(
                site for site in reader_sites if site.kind == kind
            )
            expected_discovered = coverage_record.counts[f"{kind}_discovered"]
            if len(retained) != expected_discovered:
                raise CoverageParseError(
                    f"reader {coverage_record.reader} {kind} coverage_site count "
                    f"does not match discovered: {len(retained)} != {expected_discovered}"
                )
            supported = sum(site.disposition == "supported" for site in retained)
            unsupported = sum(site.disposition == "unsupported" for site in retained)
            if supported != coverage_record.counts[f"{kind}_supported"]:
                raise CoverageParseError(
                    f"reader {coverage_record.reader} {kind} retained supported count mismatch"
                )
            if unsupported != coverage_record.counts[f"{kind}_unsupported"]:
                raise CoverageParseError(
                    f"reader {coverage_record.reader} {kind} retained unsupported count mismatch"
                )
            if not coverage_record.expert_limit:
                for outcome, counter in (
                    ("patched", "patched"),
                    ("resource_failed", "resource_failed"),
                    (
                        "placement_or_lowering_failed",
                        "placement_or_lowering_failed",
                    ),
                ):
                    retained_count = sum(site.outcome == outcome for site in retained)
                    if retained_count != coverage_record.counts[f"{kind}_{counter}"]:
                        raise CoverageParseError(
                            f"reader {coverage_record.reader} {kind} retained {outcome} "
                            "count mismatch"
                        )

    evidence = CoverageEvidence(tuple(coverage), sites, _aggregate_verdicts(verdicts))
    applicable = tuple(record for record in coverage if record.applicable)
    verdict = evidence.verdict
    if len(applicable) != verdict.applicable_code_objects:
        raise CoverageParseError(
            "applicable code-object count disagrees with coverage records: "
            f"{len(applicable)} != {verdict.applicable_code_objects}"
        )
    incomplete = sum(not record.analysis_complete for record in applicable)
    if incomplete != verdict.incomplete_code_objects:
        raise CoverageParseError(
            "incomplete code-object count disagrees with coverage records: "
            f"{incomplete} != {verdict.incomplete_code_objects}"
        )
    for kind in SITE_KINDS:
        aggregate = (
            sum(record.counts[f"{kind}_patched"] for record in applicable),
            sum(record.counts[f"{kind}_supported"] for record in applicable),
        )
        if aggregate != verdict.patched_supported[kind]:
            raise CoverageParseError(
                f"{kind} aggregate disagrees with coverage records: "
                f"{aggregate[0]}/{aggregate[1]} != "
                f"{verdict.patched_supported[kind][0]}/{verdict.patched_supported[kind][1]}"
            )
    return evidence


def acceptance_decision(log_text: str) -> AcceptanceDecision:
    """Accepts only applicable, statically and dynamically complete analysis."""
    evidence = parse_coverage_evidence(log_text)
    verdict = evidence.verdict
    reasons = []
    for name in ("applicable", "static_complete", "dynamic_complete", "analysis_complete"):
        if not getattr(verdict, name):
            reasons.append(f"verdict {name}=false")
    if any(record.expert_limit for record in evidence.coverage):
        reasons.append("expert patch limit is not an ordinary acceptance profile")
    if verdict.applicable_code_objects == 0:
        reasons.append("no applicable code objects")
    if verdict.incomplete_code_objects != 0:
        reasons.append(
            f"incomplete code objects: {verdict.incomplete_code_objects}"
        )
    if verdict.counts["dynamic_incomplete"] != 0:
        reasons.append(
            f"dynamic analysis incomplete: {verdict.counts['dynamic_incomplete']}"
        )
    for kind in SITE_KINDS:
        patched, supported = verdict.patched_supported[kind]
        if patched != supported:
            reasons.append(f"{kind} patched/supported mismatch: {patched}/{supported}")
    for record in evidence.coverage:
        if not record.applicable:
            continue
        if not record.analysis_complete:
            reasons.append(f"reader {record.reader} static analysis incomplete")
        for kind in SITE_KINDS:
            patched = record.counts[f"{kind}_patched"]
            supported = record.counts[f"{kind}_supported"]
            if patched != supported:
                reasons.append(
                    f"reader {record.reader} {kind} patched/supported mismatch: "
                    f"{patched}/{supported}"
                )
            for category in (
                "unsupported",
                "resource_failed",
                "placement_or_lowering_failed",
                "expert_limit_omitted",
            ):
                count = record.counts[f"{kind}_{category}"]
                if count:
                    reasons.append(
                        f"reader {record.reader} {kind} {category}: {count}"
                    )
    return AcceptanceDecision(not reasons, tuple(reasons), evidence)


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path, help="hook log (default: stdin)")
    args = parser.parse_args(argv)
    try:
        text = args.log.read_text(encoding="utf-8") if args.log else sys.stdin.read()
        decision = acceptance_decision(text)
        output = {
            "accepted": decision.accepted,
            "coverage_sites": [asdict(site) for site in decision.evidence.sites],
            "reasons": list(decision.reasons),
        }
        status = 0 if decision.accepted else 1
    except (OSError, UnicodeError, CoverageParseError) as error:
        output = {"accepted": False, "reasons": [str(error)]}
        status = 1
    print(json.dumps(output, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
