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
from typing import Any, Dict, List, Optional

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

# Device capability an unreachable: group's measurement was taken under. Most
# declarations hold on any part; the texture ones do not, because whether a
# texture or array handle can exist at all is a property of the device.
UNREACHABLE_ALWAYS = "always"
UNREACHABLE_NO_IMAGE_SUPPORT = "no_image_support"
VALID_UNREACHABLE_WHEN = (UNREACHABLE_ALWAYS, UNREACHABLE_NO_IMAGE_SUPPORT)


@dataclass
class ApiRow:
    api: str
    table: str
    capture_class: str
    replay_class: str
    derived_class: str          # observable class per the generator
    expect_class: str           # what the test asserts
    tier: str
    workloads: List[str]
    gpus: int
    payload_loss: bool
    payload_loss_shapes: List[str] = field(default_factory=list)
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
    skip_unless_multi_gpu: bool = False
    # Measured reason this API cannot be made to write an event on this stack,
    # from the unreachable: groups. Empty for every API that is expected to be
    # exercised, which is what makes an undeclared gap a reportable problem
    # rather than a silent one.
    unreachable_reason: str = ""
    unreachable_group: str = ""
    # Device capability the group's measurement was taken under, from the
    # group's when:. "always" for a declaration that holds anywhere; otherwise
    # the reason only describes parts with that capability, and on any other
    # part the API is unmeasured rather than unreachable.
    unreachable_when: str = UNREACHABLE_ALWAYS
    note: str = ""
    source: str = ""            # how the tier was decided, for the report

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api": self.api,
            "table": self.table,
            "capture_class": self.capture_class,
            "replay_class": self.replay_class,
            "derived_class": self.derived_class,
            "expect_class": self.expect_class,
            "tier": self.tier,
            "workloads": self.workloads,
            "gpus": self.gpus,
            "payload_loss": self.payload_loss,
            "payload_loss_shapes": self.payload_loss_shapes,
            "expect_captured": self.expect_captured,
            "handler_error_in": self.handler_error_in,
            "skip": self.skip,
            "skip_reason": self.skip_reason,
            "skip_unless_multi_gpu": self.skip_unless_multi_gpu,
            "unreachable_reason": self.unreachable_reason,
            "unreachable_group": self.unreachable_group,
            "unreachable_when": self.unreachable_when,
            "note": self.note,
            "tier_source": self.source,
        }


class MatrixError(RuntimeError):
    pass


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Dict[str, Any]:
    if not path.is_file():
        raise MatrixError(
            f"{path} not found — run ./derive_manifest.py first")
    return json.loads(path.read_text())


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys.

    The overrides block is one flat mapping of 551 possible API names, and
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


def load_overlay(path: Path = DEFAULT_OVERLAY) -> Dict[str, Any]:
    if not path.is_file():
        raise MatrixError(f"{path} not found")
    return yaml.load(path.read_text(), _StrictLoader)


def _tier_meta(overlay: Dict[str, Any], tier: str) -> Dict[str, Any]:
    return (overlay.get("tiers") or {}).get(tier, {}) or {}


def unreachable_index(overlay: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """api -> {group, reason}, from the overlay's unreachable: groups.

    An API listed here has been called and measured not to reach the archive on
    this stack. Grouping is not cosmetic: the reason is the same measurement for
    every API in a group (imageSupport=0, absent from the SDK headers, behind a
    crash), so writing it once keeps the 149 declarations honest and makes it
    obvious when a whole group becomes reachable again.
    """
    index: Dict[str, Dict[str, str]] = {}
    for group in overlay.get("unreachable") or []:
        name = group.get("group") or ""
        reason = " ".join((group.get("reason") or "").split())
        if not name or not reason:
            raise MatrixError(
                "every unreachable: entry needs a group and a reason")
        when = group.get("when") or UNREACHABLE_ALWAYS
        if when not in VALID_UNREACHABLE_WHEN:
            raise MatrixError(
                f"unreachable.{name}: when: '{when}' is not one of "
                f"{', '.join(VALID_UNREACHABLE_WHEN)}")
        for api in group.get("apis") or []:
            if api in index:
                raise MatrixError(
                    f"{api} is listed in unreachable groups "
                    f"{index[api]['group']} and {name}; one reason per API")
            index[api] = {"group": name, "reason": reason, "when": when}
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
            tier, source = ov["tier"], "override"
        elif api in explicit:
            tier, source = explicit[api], "tier-list"
        else:
            tier, source = defaults.get("tier", "T4"), "default"
            for rx, pat in compiled:
                if rx.search(api):
                    tier, source = pat["tier"], f"pattern:{pat['match']}"
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

        shapes = sorted({p["shape"] for p in entry["payload_loss"]})
        # The overlay is authoritative on payload loss: the mechanical detector
        # is a superset (it flags every const-struct pointer regardless of
        # whether the loss is consequential) and it misses non-const shapes such
        # as hipStreamBatchMemOp's op-list. An explicit false suppresses it.
        if "payload_loss" in ov:
            payload_loss = bool(ov["payload_loss"])
        else:
            payload_loss = bool(shapes)

        rows.append(ApiRow(
            api=api,
            table=entry["table"],
            capture_class=entry["capture_class"],
            replay_class=entry["replay_class"],
            derived_class=derived,
            expect_class=expect_class,
            tier=tier,
            workloads=workloads,
            gpus=gpus,
            payload_loss=payload_loss,
            payload_loss_shapes=shapes,
            # A passthrough-only capture shim writes no event at all, so the
            # API can never appear in an archive. That is derivable, so derive
            # it; the overlay only has to declare the cases where a shim does
            # write an event but under a different API's name.
            expect_captured=bool(
                ov.get("captured", entry["capture_class"] != "PASSTHROUGH_ONLY")),
            handler_error_in=list(ov.get("handler_error_in") or []),
            # Only a per-API skip means "never assert this". A tier's
            # skip_by_default means "do not run unless asked", which is already
            # expressed by whether that tier produced observations — folding it
            # in here would make --include-deprioritised run T5 and then report
            # all 92 of its APIs as skipped.
            skip=bool(ov.get("skip", False)),
            skip_reason=ov.get("reason", ""),
            skip_unless_multi_gpu=bool(ov.get("skip_unless_multi_gpu", False)) or gpus > 1,
            unreachable_reason=unreachable.get(api, {}).get("reason", ""),
            unreachable_group=unreachable.get(api, {}).get("group", ""),
            unreachable_when=unreachable.get(api, {}).get(
                "when", UNREACHABLE_ALWAYS),
            note=ov.get("note", ""),
            source=source,
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
