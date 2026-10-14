# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Resolve api_classes.json (derived) against api_matrix.yaml (authored).

api_classes.json says what each HIP API's replay behaviour *is*, read out of
HRR's generator. api_matrix.yaml says what we expect it to be, which priority
tier owns it, and which hidden Catch2 workload exercises it. This module joins
the two into one row per API and is the only place that knows the resolution
order.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "api_classes.json"
DEFAULT_OVERLAY = HERE / "api_matrix.yaml"

# Observable replay classes. These are what a test can actually distinguish:
# NOOP and ERROR_STUB each emit a one-time stderr warning naming themselves,
# and HANDLER_ERROR names itself in the line reporting the failed event.
#
# HANDLER_ERROR and CRASH are not classes the generator assigns. They are what
# happens when a real handler runs and does not come back cleanly: the API is
# counted as faithfully replayed, and replaying it fails. HANDLER_ERROR returns
# a HIP error — survivable, which is why the matrix runs hrr-playback with
# --continue-on-error. CRASH takes the process down with a signal, so it also
# costs every event recorded after it, which is why an API declared CRASH is
# given a workload of its own.
#
# UNREPLAYABLE is the generator's fourth assigned class: the API is recorded,
# and replay refuses it by name and reason rather than handing a capture-time
# host pointer to the runtime. An API expected to be UNREPLAYABLE and observed
# UNREPLAYABLE is a PASS — the declared behaviour is "fails loudly here", and
# the reason is the documented scope exclusion.
CLASS_REAL = "REAL"
CLASS_NOOP = "NOOP"
CLASS_ERROR_STUB = "ERROR_STUB"
CLASS_UNREPLAYABLE = "UNREPLAYABLE"
CLASS_HANDLER_ERROR = "HANDLER_ERROR"
CLASS_CRASH = "CRASH"
VALID_CLASSES = (CLASS_REAL, CLASS_NOOP, CLASS_ERROR_STUB, CLASS_UNREPLAYABLE,
                 CLASS_HANDLER_ERROR, CLASS_CRASH)

TIER_ORDER = ["T0", "T1", "T2", "T3", "T4", "T5"]


@dataclass
class ApiRow:
    api: str
    expect_class: str           # what the test asserts
    tier: str
    workloads: List[str]
    gpus: int
    payload_loss: bool
    # False for APIs the workload does call but that never reach the archive
    # under their own name — HIP lowers them to another entry point first, or
    # HRR has no capture wrapper for them. Declaring it turns a permanent
    # coverage hole into an assertion that the folding still happens.
    expect_captured: bool = True
    # Tiers where this API's real handler is known to return an error for the
    # arguments that tier records, while working everywhere else. A whole-API
    # class cannot express that, and without it the difference reads as two
    # tiers disagreeing about what the API is.
    handler_error_in: List[str] = field(default_factory=list)
    skip: bool = False
    skip_reason: str = ""
    # Measured reason this API cannot be made to write an event on this stack,
    # from the unreachable: groups. Empty for every API that is expected to be
    # exercised, which is what makes an undeclared gap a reportable problem
    # rather than a silent one.
    unreachable_reason: str = ""
    unreachable_group: str = ""
    note: str = ""


class MatrixError(RuntimeError):
    pass


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Dict[str, Any]:
    if not path.is_file():
        raise MatrixError(
            f"{path} not found — run ./derive_manifest.py first")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MatrixError(f"{path} is not valid JSON: {exc}")
    except (OSError, UnicodeDecodeError) as exc:
        raise MatrixError(f"cannot read {path}: {exc}")
    if not isinstance(manifest, dict) or not isinstance(
            manifest.get("apis"), list):
        raise MatrixError(f"{path} has no 'apis' list — re-run "
                          "./derive_manifest.py")
    return manifest


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys.

    The overrides block is one flat mapping keyed by API name, and
    related entries for the same API naturally get written in different
    sections — the IPC truncation note next to the other payload-loss notes,
    the not-captured note next to the other capture findings. Plain YAML
    silently keeps the last of those and drops the rest, which loses an
    expectation without any error. Refusing is the only safe behaviour.
    """


def _no_duplicate_keys(loader, node, deep=False):
    seen, dupes = set(), []
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            dupes.append(key)
        seen.add(key)
    if dupes:
        raise MatrixError(
            f"duplicate keys in the overlay: {', '.join(map(str, dupes))}. "
            "Merge them into one entry; YAML would otherwise keep only the "
            "last and silently drop the others.")
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)


# ---------------------------------------------------------------------------
# Overlay schema
#
# Every key the overlay may carry is listed here, because a misspelt key is
# otherwise silently ignored: `payload_los: true` would drop an expectation and
# the matrix would go on passing. Each checker returns None when the value is
# fine, or a description of what was expected.
# ---------------------------------------------------------------------------

_Check = Callable[[Any], Optional[str]]
_EXPECT_VALUES = ("derived",) + VALID_CLASSES


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _string(value: Any) -> Optional[str]:
    return None if isinstance(value, str) else "a string"


def _name(value: Any) -> Optional[str]:
    return None if isinstance(value, str) and value else "a non-empty string"


def _boolean(value: Any) -> Optional[str]:
    return None if isinstance(value, bool) else "a boolean"


def _positive_int(value: Any) -> Optional[str]:
    return None if _is_int(value) and value > 0 else "an integer > 0"


def _version(value: Any) -> Optional[str]:
    return None if _is_int(value) and value == 1 else "the supported version 1"


def _tier(value: Any) -> Optional[str]:
    return None if value in TIER_ORDER else f"one of {', '.join(TIER_ORDER)}"


def _expect(value: Any) -> Optional[str]:
    return (None if value in _EXPECT_VALUES
            else f"one of {', '.join(_EXPECT_VALUES)}")


def _names(value: Any) -> Optional[str]:
    ok = isinstance(value, list) and all(isinstance(v, str) and v for v in value)
    return None if ok else "a list of non-empty strings"


def _nonempty_names(value: Any) -> Optional[str]:
    return None if _names(value) is None and value else (
        "a non-empty list of non-empty strings")


def _tier_names(value: Any) -> Optional[str]:
    ok = isinstance(value, list) and all(
        isinstance(t, str) and t in TIER_ORDER for t in value)
    return None if ok else f"a list of tiers from {', '.join(TIER_ORDER)}"


def _checked_below(value: Any) -> Optional[str]:
    """Section whose shape _validate_sections checks entry by entry."""
    return None


def _regex(value: Any) -> Optional[str]:
    if _name(value) is not None:
        return "a non-empty regular expression"
    try:
        re.compile(value)
    except re.error as exc:
        return f"a valid regular expression ({exc})"
    return None


_DEFAULTS_KEYS: Dict[str, _Check] = {
    "tier": _tier, "expect": _expect, "workloads": _nonempty_names,
}
_TIER_KEYS: Dict[str, _Check] = {
    "title": _string, "rationale": _string, "min_covered": _positive_int,
    "gpus": _positive_int,
    "workloads": _nonempty_names, "apis": _names,
}
_TIER_REQUIRED = ("min_covered", "workloads")
_PATTERN_KEYS: Dict[str, _Check] = {
    "match": _regex, "tier": _tier, "reason": _string,
}
_UNREACHABLE_KEYS: Dict[str, _Check] = {
    "group": _name, "reason": _name, "apis": _names,
}
_OVERRIDE_KEYS: Dict[str, _Check] = {
    "tier": _tier, "expect": _expect, "workloads": _nonempty_names,
    "gpus": _positive_int, "payload_loss": _boolean, "captured": _boolean,
    "handler_error_in": _tier_names, "skip": _boolean, "reason": _string,
    "note": _string,
}


def _check_mapping(where: str, value: Any, keys: Dict[str, _Check],
                   required: tuple, problems: List[str]) -> bool:
    if not isinstance(value, dict):
        problems.append(f"{where}: expected a mapping, "
                        f"got {type(value).__name__}")
        return False
    for key, item in value.items():
        if key not in keys:
            problems.append(f"{where}: unknown key '{key}'")
            continue
        wanted = keys[key](item)
        if wanted is not None:
            problems.append(f"{where}.{key}: expected {wanted}, got {item!r}")
    for key in required:
        if key not in value:
            problems.append(f"{where}: missing required key '{key}'")
    return True


def validate_overlay(overlay: Any) -> None:
    """Reject an overlay that is not exactly the documented shape.

    Raises MatrixError listing every problem found, so one run shows them all.
    """
    problems: List[str] = []
    top_keys: Dict[str, _Check] = {
        "version": _version, "defaults": _checked_below,
        "tiers": _checked_below, "patterns": _checked_below,
        "unreachable": _checked_below, "overrides": _checked_below,
    }
    if _check_mapping("overlay", overlay, top_keys, ("tiers",), problems):
        _validate_sections(overlay, problems)
    if problems:
        raise MatrixError("api_matrix.yaml is malformed:\n  " +
                          "\n  ".join(problems))


def _validate_sections(overlay: Dict[str, Any], problems: List[str]) -> None:
    tiers = overlay.get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        problems.append("tiers: expected a non-empty mapping of tier name "
                        "to its definition")
        tiers = {}
    explicit: Dict[str, str] = {}
    for tier, meta in tiers.items():
        if _tier(tier) is not None:
            problems.append(f"tiers: '{tier}' is not one of "
                            f"{', '.join(TIER_ORDER)}")
        if (_check_mapping(f"tiers.{tier}", meta, _TIER_KEYS, _TIER_REQUIRED,
                           problems) and _names(meta.get("apis", [])) is None):
            for api in meta.get("apis", []):
                explicit.setdefault(api, tier)

    if "defaults" in overlay:
        defaults = overlay["defaults"]
        if _check_mapping("defaults", defaults, _DEFAULTS_KEYS, (), problems):
            _check_tier_configured("defaults.tier", defaults.get("tier"),
                                   tiers, problems)

    if "patterns" in overlay:
        patterns = overlay["patterns"]
        if not isinstance(patterns, list):
            problems.append("patterns: expected a list, "
                            f"got {type(patterns).__name__}")
        else:
            for i, pat in enumerate(patterns):
                where = f"patterns[{i}]"
                if _check_mapping(where, pat, _PATTERN_KEYS,
                                  ("match", "tier"), problems):
                    _check_tier_configured(f"{where}.tier", pat.get("tier"),
                                           tiers, problems)

    if "unreachable" in overlay:
        groups = overlay["unreachable"]
        if not isinstance(groups, list):
            problems.append("unreachable: expected a list, "
                            f"got {type(groups).__name__}")
        else:
            for i, group in enumerate(groups):
                _check_mapping(f"unreachable[{i}]", group, _UNREACHABLE_KEYS,
                               tuple(_UNREACHABLE_KEYS), problems)

    if "overrides" in overlay:
        overrides = overlay["overrides"]
        if not isinstance(overrides, dict):
            problems.append("overrides: expected a mapping of API name to "
                            f"its override, got {type(overrides).__name__}")
        else:
            for api, ov in overrides.items():
                where = f"overrides.{api}"
                if _name(api) is not None:
                    problems.append(f"overrides: key {api!r} is not an API "
                                    "name")
                if not _check_mapping(where, ov, _OVERRIDE_KEYS, (),
                                      problems):
                    continue
                if "tier" in ov:
                    _check_tier_configured(f"{where}.tier", ov["tier"], tiers,
                                           problems)
                    if api in explicit and explicit[api] != ov["tier"]:
                        problems.append(
                            f"{where}.tier: {ov['tier']} conflicts with "
                            f"{api} being listed under tiers.{explicit[api]}"
                            ".apis; an API belongs to exactly one tier")


def _check_tier_configured(where: str, tier: Any, tiers: Dict[str, Any],
                           problems: List[str]) -> None:
    # A tier outside TIER_ORDER is already reported by the key's own checker.
    if tier is not None and _tier(tier) is None and tier not in tiers:
        problems.append(f"{where}: {tier} has no entry under tiers:")


def load_overlay(path: Path = DEFAULT_OVERLAY) -> Dict[str, Any]:
    if not path.is_file():
        raise MatrixError(f"{path} not found")
    try:
        overlay = yaml.load(path.read_text(encoding="utf-8"), _StrictLoader)
    except yaml.YAMLError as exc:
        raise MatrixError(f"{path} is not valid YAML: {exc}")
    except (OSError, UnicodeDecodeError) as exc:
        raise MatrixError(f"cannot read {path}: {exc}")
    validate_overlay(overlay)
    return overlay


def _tier_meta(overlay: Dict[str, Any], tier: str) -> Dict[str, Any]:
    return (overlay.get("tiers") or {}).get(tier, {}) or {}


def unreachable_index(overlay: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """api -> {group, reason}, from the overlay's unreachable: groups.

    An API listed here has been called and measured not to reach the archive on
    this stack. Grouping is not cosmetic: the reason is the same measurement for
    every API in a group (imageSupport=0, absent from the SDK headers, behind a
    crash), so writing it once keeps the declarations honest and makes it
    obvious when a whole group becomes reachable again.
    """
    index: Dict[str, Dict[str, str]] = {}
    for group in overlay.get("unreachable") or []:
        name = group.get("group") or ""
        reason = " ".join((group.get("reason") or "").split())
        if not name or not reason:
            raise MatrixError(
                "every unreachable: entry needs a group and a reason")
        for api in group.get("apis") or []:
            if api in index:
                raise MatrixError(
                    f"{api} is listed in unreachable groups "
                    f"{index[api]['group']} and {name}; one reason per API")
            index[api] = {"group": name, "reason": reason}
    return index


def resolve(manifest: Dict[str, Any], overlay: Dict[str, Any]) -> List[ApiRow]:
    """Join derived classification with the authored overlay.

    Resolution order, first match wins:
      1. overrides:<api>            (may set tier, expect, workload, skip, ...)
      2. tiers:<T>:apis membership
      3. patterns, in file order
      4. defaults
    """
    defaults = overlay.get("defaults") or {}
    overrides = overlay.get("overrides") or {}
    patterns = overlay.get("patterns") or []
    tiers = overlay.get("tiers") or {}

    # api -> tier, from the explicit tier lists.
    explicit: Dict[str, str] = {}
    for tier, meta in tiers.items():
        for api in (meta or {}).get("apis") or []:
            if api in explicit:
                raise MatrixError(
                    f"{api} is listed in both {explicit[api]} and {tier}; "
                    "an API belongs to exactly one tier")
            explicit[api] = tier

    compiled = [(re.compile(p["match"]), p) for p in patterns]
    unreachable = unreachable_index(overlay)

    rows: List[ApiRow] = []
    for entry in manifest["apis"]:
        api = entry["api"]
        ov = overrides.get(api) or {}

        if "tier" in ov:
            tier = ov["tier"]
        elif api in explicit:
            tier = explicit[api]
        else:
            tier = defaults.get("tier", "T4")
            for rx, pat in compiled:
                if rx.search(api):
                    tier = pat["tier"]
                    break

        meta = _tier_meta(overlay, tier)
        workloads = list(ov.get("workloads") or meta.get("workloads")
                         or defaults.get("workloads") or [])
        gpus = int(ov.get("gpus") or meta.get("gpus") or 1)

        expect = ov.get("expect", defaults.get("expect", "derived"))
        derived = entry["observable_class"]
        expect_class = derived if expect == "derived" else expect
        if expect_class not in VALID_CLASSES:
            raise MatrixError(
                f"{api}: expect '{expect}' is not one of "
                f"derived/{'/'.join(VALID_CLASSES)}")

        # The overlay is authoritative on payload loss: the mechanical detector
        # is a superset (it flags every const-struct pointer regardless of
        # whether the loss is consequential) and it misses non-const shapes such
        # as hipStreamBatchMemOp's op-list. An explicit false suppresses it.
        if "payload_loss" in ov:
            payload_loss = bool(ov["payload_loss"])
        else:
            payload_loss = bool(entry["payload_loss"])

        rows.append(ApiRow(
            api=api,
            expect_class=expect_class,
            tier=tier,
            workloads=workloads,
            gpus=gpus,
            payload_loss=payload_loss,
            # A passthrough-only capture shim writes no event at all, so the
            # API can never appear in an archive. That is derivable, so derive
            # it; the overlay only has to declare the cases where a shim does
            # write an event but under a different API's name.
            expect_captured=bool(
                ov.get("captured", entry["capture_class"] != "PASSTHROUGH_ONLY")),
            handler_error_in=list(ov.get("handler_error_in") or []),
            # Only a per-API skip means "never assert this". A tier that is
            # not run by default is already expressed by whether it produced
            # observations — folding it in here would mark every API of a
            # deprioritised tier as skipped even when that tier was run.
            skip=bool(ov.get("skip", False)),
            skip_reason=ov.get("reason", ""),
            unreachable_reason=unreachable.get(api, {}).get("reason", ""),
            unreachable_group=unreachable.get(api, {}).get("group", ""),
            note=ov.get("note", ""),
        ))
    return rows


def validate(rows: List[ApiRow], overlay: Dict[str, Any]) -> List[str]:
    """Structural checks that must hold before the matrix means anything."""
    problems: List[str] = []
    known_tiers = set(overlay.get("tiers") or {})
    by_name = {r.api: r for r in rows}

    for r in rows:
        if r.tier not in known_tiers:
            problems.append(f"{r.api}: tier {r.tier} has no entry under tiers:")
        if not r.workloads:
            problems.append(f"{r.api}: no owning workload")
        for tier in r.handler_error_in:
            if tier not in known_tiers:
                problems.append(
                    f"{r.api}: handler_error_in names {tier}, which is not a "
                    "tier")
        if r.handler_error_in and r.expect_class != CLASS_REAL:
            problems.append(
                f"{r.api}: handler_error_in only means anything for an API "
                f"whose class is {CLASS_REAL}; this one expects "
                f"{r.expect_class}")

    # Every API named in the overlay must exist, or the overlay has rotted
    # against the generator — the same failure derive_manifest.py's baseline
    # catches from the other direction.
    for tier, meta in (overlay.get("tiers") or {}).items():
        for api in (meta or {}).get("apis") or []:
            if api not in by_name:
                problems.append(
                    f"tiers.{tier}: '{api}' is not in api_classes.json "
                    "(renamed or removed upstream?)")
    for api in (overlay.get("overrides") or {}):
        if api not in by_name:
            problems.append(
                f"overrides: '{api}' is not in api_classes.json "
                "(renamed or removed upstream?)")
    for api, entry in unreachable_index(overlay).items():
        if api not in by_name:
            problems.append(
                f"unreachable.{entry['group']}: '{api}' is not in "
                "api_classes.json (renamed or removed upstream?)")
        elif by_name[api].skip:
            problems.append(
                f"{api}: declared both skip and unreachable. skip means the "
                "matrix asserts nothing; unreachable means it asserts the API "
                "cannot be reached. Pick one.")

    return problems


def tier_workloads(overlay: Dict[str, Any], tier: str) -> List[str]:
    return list(_tier_meta(overlay, tier).get("workloads") or [])


def tier_summary(rows: List[ApiRow]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for r in rows:
        bucket = out.setdefault(r.tier, {
            "apis": 0, CLASS_REAL: 0, CLASS_NOOP: 0, CLASS_ERROR_STUB: 0,
            CLASS_UNREPLAYABLE: 0, CLASS_HANDLER_ERROR: 0, CLASS_CRASH: 0,
            "payload_loss": 0, "skipped": 0, "multi_gpu": 0,
        })
        bucket["apis"] += 1
        bucket[r.expect_class] += 1
        if r.payload_loss:
            bucket["payload_loss"] += 1
        if r.skip:
            bucket["skipped"] += 1
        if r.gpus > 1:
            bucket["multi_gpu"] += 1
    return {t: out[t] for t in TIER_ORDER if t in out}


def load_rows(manifest_path: Path = DEFAULT_MANIFEST,
              overlay_path: Path = DEFAULT_OVERLAY) -> List[ApiRow]:
    manifest = load_manifest(manifest_path)
    overlay = load_overlay(overlay_path)
    rows = resolve(manifest, overlay)
    problems = validate(rows, overlay)
    if problems:
        raise MatrixError("api_matrix.yaml does not resolve cleanly:\n  " +
                          "\n  ".join(problems))
    return rows
