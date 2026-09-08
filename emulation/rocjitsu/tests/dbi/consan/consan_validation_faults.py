"""ConSan fault inventory, qualification, execution, and acceptance."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time

from consan_validation_catalog import (
    FAULT_FAMILY_ENVIRONMENTS,
    FAULT_FAMILY_SITE_KINDS,
    ORDINARY_FORBIDDEN_ENVIRONMENT,
    ORDINARY_MOI_RUNTIME_DEFAULTS,
    PROFILE_IDS,
    PROFILES,
    SCHEMA_VERSION,
    SETTING_CATEGORIES,
    ValidationError,
    Workload,
    _fault_families,
    _fault_family_environment,
    _resolved_workload,
    _workload_for_target,
)
from consan_validation_commands import (
    _audited_settings,
    _audited_unsets,
    _clean_environment,
    _controlled_environment,
    _corpus_root,
    _doctor,
    _effective_workload,
    _fault_workload_command,
    _health_smoke_command,
    _hook_path,
    _outer_repetitions,
    _profile_runtime_defaults,
    _resolve_workload_selection,
    _run_environment,
    _with_launcher,
    _workload_command,
    _workspace_from_environment,
    _write_provenance,
)
from consan_validation_execution import _stop_process_group
from consan_validation_support import (
    FAULT_RESERVATION_QUALIFIED,
    RESULT_SCHEMA_VERSION,
    atomic_write_json,
    fault_reservation_qualification,
    sha256_file,
)


def _inventory_records(
    log_text: str, family: str | None = None
) -> dict[str, list[str]]:
    event_kind = FAULT_FAMILY_SITE_KINDS.get(family) if family else None
    prefixes = {
        "sites": "ConSan fault site ",
        "sequences": "ConSan sync sequence ",
        "destinations": "ConSan barrier destination ",
    }
    records = {key: [] for key in prefixes}
    for line in log_text.splitlines():
        for key, prefix in prefixes.items():
            if prefix not in line:
                continue
            match = re.search(r"\bidentity=(\S+)", line)
            if match:
                identity = match.group(1)
                if family and key == "sites" and f"|kind={event_kind}|" not in identity:
                    continue
                if (
                    family
                    and key == "sequences"
                    and f"|event={event_kind}|" not in identity
                ):
                    continue
                if family and key == "destinations" and family != "barrier-move":
                    continue
                records[key].append(identity)
        if "ConSan fault site " in line:
            match = re.search(r"\bsync_sequence=(\S+)", line)
            if match:
                identity = match.group(1)
                if identity != "-" and (
                    not family or f"|event={event_kind}|" in identity
                ):
                    records["sequences"].append(identity)
    return {key: sorted(set(values)) for key, values in records.items()}


def _inventory_line_completes(
    line: str, family: str, relevant_readers: set[str]
) -> bool:
    """Tracks a relevant site through the matching code-object coverage record."""
    site_kind = FAULT_FAMILY_SITE_KINDS[family]
    if "ConSan fault site " in line and f" kind={site_kind} " in line:
        match = re.search(r"\breader=(\S+)", line)
        if match:
            relevant_readers.add(match.group(1))
    if "ConSan coverage " not in line:
        return False
    match = re.search(r"\breader=(\S+)", line)
    return bool(match and match.group(1) in relevant_readers)


def _inventory_collection_complete(log_text: str, family: str) -> bool:
    relevant_readers: set[str] = set()
    return any(
        _inventory_line_completes(line, family, relevant_readers)
        for line in log_text.splitlines()
    )


def _run_inventory_process(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    timeout: int,
    family: str,
) -> tuple[int, float, str, bool, str]:
    """Collects static identities without waiting for workload execution."""
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    lines: list[str] = []
    pending = ""
    relevant_readers: set[str] = set()
    collection_complete = False
    timed_out = False
    deadline = start + timeout
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        events = selector.select(min(remaining, 0.25))
        if not events:
            continue
        chunk = os.read(process.stdout.fileno(), 65536)
        if not chunk:
            continue
        pending += chunk.decode(errors="replace")
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            line += "\n"
            lines.append(line)
            if _inventory_line_completes(line, family, relevant_readers):
                collection_complete = True
                break
        if collection_complete:
            break

    outcome = "natural-exit"
    if collection_complete:
        outcome = "static-inventory-complete"
        _stop_process_group(process, signal.SIGTERM)
    elif timed_out:
        outcome = "timeout"
        _stop_process_group(process, signal.SIGTERM)
    try:
        remainder, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _stop_process_group(process, signal.SIGKILL)
        remainder, _ = process.communicate()
    selector.close()
    if remainder:
        pending += remainder.decode(errors="replace")
    if pending:
        lines.append(pending)
    if timed_out:
        lines.append(f"\nvalidation timeout after {timeout}s\n")
    output = "".join(lines)
    collection_complete = not timed_out and _inventory_collection_complete(
        output, family
    )
    elapsed = time.monotonic() - start
    log_path.write_text(output, encoding="utf-8")
    returncode = 124 if timed_out else process.returncode
    return returncode, elapsed, output, collection_complete, outcome


def _fault_template(target: str, workload: Workload) -> dict:
    profile_policies = {
        profile: {"detector": "REVIEW_REQUIRED", "oracle": "any"}
        for profile in PROFILE_IDS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "workload": workload.id,
        "review_required": True,
        "faults": [
            {
                "id": family,
                "family": family,
                "environment": _fault_family_environment(target, family),
                "profiles": profile_policies,
            }
            for family in _fault_families(target, workload)
        ],
    }


def _fault_inventory_environment(family: str) -> dict[str, str]:
    """Enables family-specific analysis without selecting or applying a site."""
    return {
        name: value
        for name, value in FAULT_FAMILY_ENVIRONMENTS[family].items()
        if not name.endswith("_IDENTITY")
    }


def _inventory(args: argparse.Namespace) -> int:
    selection = _resolve_workload_selection(args, allow_all=False)
    target = selection.target
    workload = selection.require_workload()
    workspace = _workspace_from_environment()
    if not _doctor(workspace, target, (workload.id,), args.launcher)["ok"]:
        raise ValidationError("workspace doctor failed; run the doctor subcommand")
    root = args.artifact_root.resolve() / workload.id / "inventory"
    root.mkdir(parents=True, exist_ok=False)
    provenance = _write_provenance(
        workspace, target, workload, root, args.launcher
    )
    hook = _hook_path(workspace)
    command = _fault_workload_command(
        workspace, target, workload, root / "unused.json"
    )
    command = _with_launcher(args.launcher, command)
    family_runs = []
    aggregate_records = {"sites": set(), "sequences": set(), "destinations": set()}
    for family in _fault_families(target, workload):
        environment = _clean_environment(
            "supercollider", workload, hook, target, workspace
        )
        # Clean qualification uses compact level-1 summaries. Fault inventory
        # explicitly requests level 2 because it consumes per-site identities.
        environment["RJ_CONSAN_LOG"] = "2"
        environment["RJ_CONSAN_FAULT_DRY_RUN"] = "1"
        environment.update(_fault_inventory_environment(family))
        log_path = root / f"command-{family}.log"
        returncode, elapsed, output, collection_complete, outcome = (
            _run_inventory_process(command, environment, log_path, args.timeout, family)
        )
        records = _inventory_records(output, family)
        for kind, values in records.items():
            aggregate_records[kind].update(values)
        family_runs.append(
            {
                "family": family,
                "environment": _controlled_environment(environment),
                "returncode": returncode,
                "outcome": outcome,
                "collection_complete": collection_complete,
                "elapsed_seconds": elapsed,
                "records": records,
                "log": str(log_path),
            }
        )
    records = {kind: sorted(values) for kind, values in aggregate_records.items()}
    document = {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "workload": workload.id,
        "command": command,
        "family_runs": family_runs,
        "returncode": 0,
        "elapsed_seconds": sum(run["elapsed_seconds"] for run in family_runs),
        "records": records,
        "hook": {"path": str(hook), "sha256": sha256_file(hook)},
        "provenance": str(provenance),
    }
    document["accepted"] = all(
        run["collection_complete"] and bool(run["records"]["sites"])
        for run in family_runs
    )
    if not document["accepted"]:
        document["returncode"] = next(
            (
                run["returncode"]
                for run in family_runs
                if not run["collection_complete"] and run["returncode"] != 0
            ),
            1,
        )
    atomic_write_json(root / "inventory.json", document)
    atomic_write_json(
        root / "fault-spec.template.json", _fault_template(target, workload)
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if document["accepted"] else 1


def _load_fault(
    path: Path,
    target: str,
    workload: Workload,
    fault_id: str,
) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError("fault spec has unsupported schema_version")
    if document.get("target") != target:
        raise ValidationError("fault spec target does not match --target")
    if "workloads" in document:
        workloads = document.get("workloads")
        if not isinstance(workloads, dict) or workload.id not in workloads:
            raise ValidationError("fault spec does not define --workload")
        workload_document = workloads[workload.id]
    else:
        if document.get("workload") != workload.id:
            raise ValidationError("fault spec workload does not match --workload")
        workload_document = document
    if not isinstance(workload_document, dict):
        raise ValidationError("fault workload policy must be an object")
    if document.get("review_required") is not False:
        raise ValidationError("fault spec must set review_required=false after review")
    faults = workload_document.get("faults", [])
    if not isinstance(faults, list):
        raise ValidationError("faults must be a list")
    matches = [
        fault
        for fault in faults
        if isinstance(fault, dict) and fault.get("id") == fault_id
    ]
    if len(matches) != 1:
        raise ValidationError(f"fault spec must define exactly one {fault_id!r}")
    fault = matches[0]
    if fault.get("family") not in _fault_families(target, workload):
        raise ValidationError("fault family is not admitted by the workload manifest")
    environment = fault.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise ValidationError("fault environment must be a non-empty object")
    if any(
        not isinstance(key, str)
        or not key.startswith("RJ_CONSAN_FAULT_")
        or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValidationError(
            "fault environment may contain only string RJ_CONSAN_FAULT_* values"
        )
    mutations = [
        key
        for key, value in environment.items()
        if value == "1"
        and key
        in {
            "RJ_CONSAN_FAULT_DROP_BARRIER",
            "RJ_CONSAN_FAULT_MOVE_BARRIER",
            "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER",
            "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE",
            "RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS",
            "RJ_CONSAN_FAULT_LDS_WRONG_ADDRESS",
        }
    ]
    if len(mutations) != 1:
        raise ValidationError("fault spec must enable exactly one mutation family")
    if "RJ_CONSAN_FAULT_SITE_IDENTITY" not in environment:
        raise ValidationError("fault spec must select an exact site identity")
    if any("REPLACE_FROM_INVENTORY" in value for value in environment.values()):
        raise ValidationError("fault spec still contains an inventory placeholder")
    site_provenance = fault.get("site_provenance")
    if site_provenance is not None:
        required_keys = {"corpus_commit", "executable", "inventory_run"}
        if (
            not isinstance(site_provenance, dict)
            or set(site_provenance) != required_keys
            or re.fullmatch(
                r"[0-9a-f]{40}", str(site_provenance.get("corpus_commit", ""))
            )
            is None
            or not isinstance(site_provenance.get("executable"), str)
            or not site_provenance["executable"]
            or Path(site_provenance["executable"]).name != site_provenance["executable"]
            or not isinstance(site_provenance.get("inventory_run"), str)
            or not site_provenance["inventory_run"]
        ):
            raise ValidationError("fault site_provenance is invalid")
    reach_witness = fault.get("reach_witness")
    if reach_witness is not None:
        if (
            not isinstance(reach_witness, dict)
            or set(reach_witness) != {"kind", "evidence"}
            or reach_witness.get("kind") != "reviewed-unconditional-final-isa"
            or not isinstance(reach_witness.get("evidence"), str)
            or not reach_witness["evidence"].strip()
        ):
            raise ValidationError("fault reach_witness is invalid")
    return fault


def _fault_trials(fault: dict, profile: str) -> tuple[dict, list[dict[str, str]]]:
    profiles = fault.get("profiles", {})
    policy = profiles.get(profile, {}) if isinstance(profiles, dict) else {}
    if not isinstance(policy, dict):
        raise ValidationError(f"invalid profile policy for {profile}")
    policy_environment = policy.get("environment", {})
    if not isinstance(policy_environment, dict) or any(
        not isinstance(key, str)
        or not key.startswith("RJ_CONSAN_")
        or not isinstance(value, str)
        for key, value in policy_environment.items()
    ):
        raise ValidationError(f"invalid profile environment for {profile}")
    unset = policy.get("unset", [])
    if not isinstance(unset, list) or any(
        not isinstance(name, str) or not name.startswith("RJ_CONSAN_") for name in unset
    ):
        raise ValidationError(f"invalid profile unset list for {profile}")
    if "trials" in policy and "trial_axis" in policy:
        raise ValidationError(f"{profile} may define trials or trial_axis, not both")
    if "trial_axis" in policy:
        axis = policy["trial_axis"]
        if not isinstance(axis, dict) or len(axis) != 1:
            raise ValidationError(f"fault trial_axis for {profile} needs one setting")
        name, bounds = next(iter(axis.items()))
        if (
            not isinstance(name, str)
            or not name.startswith("RJ_CONSAN_")
            or not isinstance(bounds, dict)
            or not isinstance(bounds.get("start"), int)
            or not isinstance(bounds.get("stop"), int)
            or bounds["start"] >= bounds["stop"]
            or bounds["stop"] - bounds["start"] > 256
        ):
            raise ValidationError(f"invalid trial_axis for {profile}")
        trials = [
            {name: str(value)} for value in range(bounds["start"], bounds["stop"])
        ]
    else:
        trials = policy.get("trials", [{}])
    if not isinstance(trials, list) or not trials:
        raise ValidationError(f"fault trials for {profile} must be a non-empty list")
    for trial in trials:
        if not isinstance(trial, dict) or any(
            not isinstance(key, str)
            or not key.startswith("RJ_CONSAN_")
            or not isinstance(value, str)
            for key, value in trial.items()
        ):
            raise ValidationError(f"invalid trial environment for {profile}")
    return policy, trials


def _wilson_detection_interval(detections: int, trials: int) -> dict[str, float]:
    if (
        type(detections) is not int
        or type(trials) is not int
        or trials <= 0
        or detections < 0
        or detections > trials
    ):
        raise ValidationError("invalid detection count for Wilson interval")
    z = 1.959963984540054
    proportion = detections / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return {
        "confidence": 0.95,
        "lower": 0.0 if detections == 0 else max(0.0, center - radius),
        "upper": 1.0 if detections == trials else min(1.0, center + radius),
    }


def _fault_trial_environment(
    profile: str,
    workload: Workload,
    hook: Path,
    target: str,
    fault: dict,
    policy: dict,
    trial: dict[str, str],
    workspace: Path,
) -> dict[str, str]:
    environment = _clean_environment(profile, workload, hook, target, workspace)
    environment["CTEST_PARALLEL_LEVEL"] = "1"
    environment.update(fault["environment"])
    if policy.get("detector") in {"detected", "statistical"}:
        environment.pop("RJ_CONSAN_MOI_FORBID_DIAGNOSTICS", None)
    environment.update(policy.get("environment", {}))
    for name in policy.get("unset", []):
        environment.pop(name, None)
    environment.update(trial)
    environment["RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE"] = "1"
    return environment


def _faults_from_spec(
    path: Path,
    target: str,
    workload: Workload,
) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("target") != target:
        raise ValidationError("fault spec target does not match --target")
    if "workloads" in document:
        workload_document = document.get("workloads", {}).get(workload.id)
    else:
        workload_document = (
            document if document.get("workload") == workload.id else None
        )
    if workload_document is None:
        return []
    faults = workload_document.get("faults", [])
    if not isinstance(faults, list):
        raise ValidationError("faults must be a list")
    loaded = []
    for fault in faults:
        if not isinstance(fault, dict) or not isinstance(fault.get("id"), str):
            raise ValidationError("every fault in the spec must have a string id")
        loaded.append(
            _load_fault(path, target, workload, fault["id"])
        )
    return loaded


def _required_diagnostic(policy: dict) -> str:
    disposition = policy.get("disposition")
    if disposition == "not-applicable":
        return "not-applicable"
    detector = policy.get("detector")
    if detector == "detected":
        return "one attributable ConSan detection in every trial"
    if detector == "statistical":
        minimum = policy.get("minimum_detections", "REVIEW_REQUIRED")
        return f"at least {minimum} attributable ConSan detections across all trials"
    if detector == "not_detected":
        return "no ConSan detection; this is a precommitted qualified miss"
    return "REVIEW_REQUIRED"


def _fault_audit(
    workspace: Path,
    target: str,
    workload: Workload,
    fault: dict,
    profiles: tuple[str, ...],
    source: str,
) -> dict:
    hook = _hook_path(workspace)
    command = _fault_workload_command(
        workspace,
        target,
        workload,
        Path("$ARTIFACT_ROOT") / workload.id / "fault" / "unused.json",
    )
    if workload.kind == "sharktank":
        command.append("--allow-oracle-failure")
    expectations = []
    for profile in profiles:
        policy, trials = _fault_trials(fault, profile)
        trial_audits = []
        if policy.get("disposition") != "not-applicable":
            for index, trial in enumerate(trials):
                environment = _fault_trial_environment(
                    profile,
                    workload,
                    hook,
                    target,
                    fault,
                    policy,
                    trial,
                    workspace,
                )
                trial_audits.append(
                    {
                        "index": index,
                        "overrides": _audited_settings(trial),
                        "effective_settings": _audited_settings(environment),
                        "implicit_runtime_defaults": _profile_runtime_defaults(
                            profile, environment
                        ),
                    }
                )
        expectations.append(
            {
                "profile": profile,
                "disposition": policy.get("disposition", "applicable"),
                "detector": policy.get("detector", "REVIEW_REQUIRED"),
                "oracle": policy.get("oracle", "any"),
                "required_diagnostic": _required_diagnostic(policy),
                "minimum_detections": policy.get("minimum_detections"),
                "reason": policy.get("reason"),
                "tracking_issue": policy.get("tracking_issue"),
                "policy_settings": _audited_settings(policy.get("environment", {})),
                "policy_unsets": _audited_unsets(policy.get("unset", [])),
                "trial_count": len(trial_audits),
                "trials": trial_audits,
            }
        )
    return {
        "id": fault["id"],
        "family": fault["family"],
        "source": source,
        "payload_argv": command,
        "validator_argv_template": [
            sys.executable,
            str(Path(__file__).resolve()),
            "--target",
            target,
            "fault",
            "--workload",
            workload.id,
            "--profile",
            "all" if len(profiles) > 1 else profiles[0],
            "--spec",
            "$FAULT_SPEC",
            "--fault",
            fault["id"],
            "--artifact-root",
            "$ARTIFACT_ROOT",
            "--allow-destructive",
        ],
        "mutation_settings": _audited_settings(fault["environment"]),
        "profile_expectations": expectations,
    }


def _explain_contract(
    workspace: Path,
    target: str,
    workload_ids: tuple[str, ...],
    profiles: tuple[str, ...],
    spec_path: Path | None,
) -> dict:
    spec_document = None
    spec_metadata = None
    if spec_path is not None:
        spec_path = spec_path.resolve()
        spec_document = json.loads(spec_path.read_text(encoding="utf-8"))
        spec_metadata = {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
            "review_required": spec_document.get("review_required"),
        }
    workloads = []
    script = str(Path(__file__).resolve())
    for workload_id in workload_ids:
        workload = _workload_for_target(target, workload_id)
        effective_workload = _effective_workload(target, workload)
        output_root = Path("$ARTIFACT_ROOT") / workload.id
        commands = {}
        for phase in ("clean", "overhead"):
            profile_artifact_roots = {
                profile: output_root / phase / profile
                for profile in profiles
            }
            commands[phase] = {
                "payload_argv": _workload_command(
                    workspace,
                    target,
                    workload,
                    phase,
                    output_root / phase / "$PROFILE" / "benchmark-0.json",
                ),
                "processes": _outer_repetitions(target, phase, workload),
                "profile_artifact_roots": {
                    profile: str(root)
                    for profile, root in profile_artifact_roots.items()
                },
                "payload_argv_by_profile": {
                    profile: _workload_command(
                        workspace,
                        target,
                        workload,
                        phase,
                        root / "benchmark-0.json",
                    )
                    for profile, root in profile_artifact_roots.items()
                },
                "validator_argv_template": [
                    sys.executable,
                    script,
                    "--target",
                    target,
                    "run",
                    "--workload",
                    workload.id,
                    "--profile",
                    "all" if len(profiles) > 1 else profiles[0],
                    "--phase",
                    phase,
                    "--include-baseline",
                    "--artifact-root",
                    "$ARTIFACT_ROOT",
                ],
            }
        profile_audits = []
        for profile in profiles:
            environment = _run_environment(
                profile,
                workload,
                _hook_path(workspace),
                target,
                "clean",
                workspace,
            )
            inherited = _clean_environment(
                None, workload, _hook_path(workspace), target, workspace
            )
            harness_environment = {
                name: value
                for name, value in environment.items()
                if name == "HIP_TARGET"
                or name not in inherited
                or inherited[name] != value
            }
            settings = _audited_settings(harness_environment)
            runtime_defaults = _profile_runtime_defaults(profile, environment)
            profile_audits.append(
                {
                    "id": profile,
                    "flavor": PROFILES[profile].flavor,
                    "engine": PROFILES[profile].engine,
                    "clean_result_phase": "clean",
                    "clean_artifact_root": str(output_root / "clean" / profile),
                    "settings": settings,
                    "implicit_runtime_defaults": runtime_defaults,
                    "usability_exceptions": [
                        setting
                        for setting in settings
                        if setting["usability_exception"]
                    ],
                }
            )
        if spec_path is None:
            fault_source = "unreviewed-template"
            faults = _fault_template(target, workload)["faults"]
        else:
            fault_source = "reviewed-spec"
            faults = _faults_from_spec(spec_path, target, workload)
        workloads.append(
            {
                **asdict(effective_workload),
                "commands": commands,
                "profiles": profile_audits,
                "faults": [
                    _fault_audit(
                        workspace, target, workload, fault, profiles, fault_source
                    )
                    for fault in faults
                ],
                "fault_spec_status": (
                    fault_source if faults else "workload-not-present-in-spec"
                ),
            }
        )
    workload_tuning = []
    explicit_event_family_overrides = []
    forbidden_present = []
    fault_policy_exceptions = []
    for workload in workloads:
        for profile in workload["profiles"]:
            names = {setting["name"] for setting in profile["settings"]}
            forbidden_present.extend(
                {
                    "workload": workload["id"],
                    "profile": profile["id"],
                    "setting": name,
                }
                for name in sorted(names & set(ORDINARY_FORBIDDEN_ENVIRONMENT))
            )
            tuned = [
                setting["name"]
                for setting in profile["settings"]
                if setting["usability_exception"]
            ]
            if tuned:
                workload_tuning.append(
                    {
                        "workload": workload["id"],
                        "profile": profile["id"],
                        "settings": tuned,
                    }
                )
            selected = [
                setting["name"]
                for setting in profile["settings"]
                if "usability_note" in setting
                and setting["category"] == "instrumentation-selection"
            ]
            if selected:
                explicit_event_family_overrides.append(
                    {
                        "workload": workload["id"],
                        "profile": profile["id"],
                        "settings": selected,
                    }
                )
        for fault in workload["faults"]:
            for expectation in fault["profile_expectations"]:
                tuned_settings = [
                    setting["name"]
                    for setting in expectation["policy_settings"]
                    if setting["usability_exception"]
                ]
                policy_unsets = [
                    setting["name"] for setting in expectation["policy_unsets"]
                ]
                if tuned_settings or policy_unsets:
                    fault_policy_exceptions.append(
                        {
                            "workload": workload["id"],
                            "fault": fault["id"],
                            "profile": expectation["profile"],
                            "settings": tuned_settings,
                            "unsets": policy_unsets,
                        }
                    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "consan-real-workload-validation-audit-v1",
        "target": target,
        "workspace": str(workspace),
        "setting_categories": SETTING_CATEGORIES,
        "ordinary_forbidden_environment": list(ORDINARY_FORBIDDEN_ENVIRONMENT),
        "usability_audit": {
            "coverage_limiting_controls_present": forbidden_present,
            "workload_specific_tuning": workload_tuning,
            "automatic_event_family_defaults": (
                [
                    {
                        "profiles": [
                            profile
                            for profile in profiles
                            if PROFILES[profile].flavor == "moi"
                        ],
                        "settings": sorted(ORDINARY_MOI_RUNTIME_DEFAULTS),
                    }
                ]
                if any(PROFILES[profile].flavor == "moi" for profile in profiles)
                else []
            ),
            "automatic_profile_defaults": [
                {
                    "profile": profile,
                    "settings": {
                        setting["name"]: setting["value"]
                        for setting in _profile_runtime_defaults(profile)
                    },
                }
                for profile in profiles
                if PROFILES[profile].flavor == "moi"
            ],
            "explicit_event_family_overrides": explicit_event_family_overrides,
            "fault_policy_exceptions": fault_policy_exceptions,
        },
        "fault_spec": spec_metadata,
        "workloads": workloads,
    }


def _print_explain(document: dict) -> None:
    print(f"target: {document['target']}")
    print(f"workspace: {document['workspace']}")
    if document["fault_spec"] is None:
        print("fault expectations: REVIEW_REQUIRED templates (no --spec supplied)")
    else:
        print(f"fault expectations: reviewed {document['fault_spec']['path']}")
    usability = document["usability_audit"]
    print(
        "ordinary coverage-limiting controls: "
        + ("PRESENT" if usability["coverage_limiting_controls_present"] else "none")
    )
    print(
        "workload-specific tuning: "
        + (
            ", ".join(
                f"{item['workload']}/{item['profile']}"
                for item in usability["workload_specific_tuning"]
            )
            if usability["workload_specific_tuning"]
            else "none"
        )
    )
    print("exact selectors and effective per-trial environments: use --json")
    for workload in document["workloads"]:
        print(f"\n{workload['priority']} {workload['id']}")
        for phase in ("clean", "overhead"):
            phase_command = workload["commands"][phase]
            processes = phase_command["processes"]
            if phase_command["payload_argv"] is not None:
                command = shlex.join(phase_command["payload_argv"])
                print(f"  {phase} ({processes} process(es)): {command}")
            else:
                for profile in workload["profiles"]:
                    profile_id = profile["id"]
                    command = shlex.join(
                        phase_command["payload_argv_by_profile"][profile_id]
                    )
                    print(
                        f"  {phase}/{profile_id} "
                        f"({processes} process(es)): {command}"
                    )
        for profile in workload["profiles"]:
            controls = ", ".join(
                f"{setting['name']}={setting['value']} [{setting['category']}]"
                for setting in profile["settings"]
            )
            defaults = ", ".join(
                f"{setting['name']}={setting['value']}"
                for setting in profile["implicit_runtime_defaults"]
            )
            marker = " USABILITY EXCEPTION" if profile["usability_exceptions"] else ""
            print(f"  {profile['id']}{marker}: {controls}")
            if defaults:
                print(f"    automatic runtime defaults: {defaults}")
        for fault in workload["faults"]:
            outcomes = ", ".join(
                f"{item['profile']}={item['detector']}/{item['oracle']}"
                f" ({item['trial_count']} trial(s))"
                for item in fault["profile_expectations"]
            )
            print(f"  fault {fault['id']} [{fault['source']}]: {outcomes}")


def _fault_acceptance(result: dict, policy: dict) -> tuple[bool, list[str]]:
    reasons = []
    mutation = result.get("mutation", {})
    if mutation.get("requested") != 1:
        reasons.append(f"requested={mutation.get('requested')}")
    if mutation.get("planned") != 1:
        reasons.append(f"planned={mutation.get('planned')}")
    if mutation.get("applied") != 1:
        reasons.append(f"applied={mutation.get('applied')}")
    accounting_schema_version = mutation.get("accounting_schema_version")
    if accounting_schema_version != 2:
        reasons.append(
            "accounting_schema_version="
            f"{accounting_schema_version}, expected=2; rerun required"
        )
    elif mutation.get("installation_evidence_complete") is not True:
        reasons.append(
            "installation_evidence_complete="
            f"{mutation.get('installation_evidence_complete')}"
        )
    reservation_status, reservation_reasons = fault_reservation_qualification(
        mutation.get("reservation")
    )
    if reservation_status != FAULT_RESERVATION_QUALIFIED:
        reasons.extend(reservation_reasons)
    if mutation.get("discarded_applied", 0):
        reasons.append(f"discarded_applied={mutation['discarded_applied']}")
    expected_detector = policy.get("detector")
    actual_detector = result.get("sanitizer", {}).get("outcome")
    if expected_detector == "statistical":
        pass
    elif expected_detector not in {"detected", "not_detected"}:
        reasons.append(
            "profile policy lacks detector=detected|not_detected|statistical"
        )
    elif actual_detector != expected_detector:
        reasons.append(f"detector={actual_detector}, expected={expected_detector}")
    expected_oracle = policy.get("oracle", "any")
    actual_oracle = result.get("oracle", {}).get("outcome")
    if expected_oracle not in {"any", "pass", "fail"}:
        reasons.append(f"invalid expected oracle={expected_oracle}")
    elif expected_oracle != "any" and actual_oracle != expected_oracle:
        reasons.append(f"oracle={actual_oracle}, expected={expected_oracle}")
    execution = result.get("execution", {})
    if execution.get("timed_out"):
        reasons.append("timed out")
    execution_outcome = execution.get("outcome")
    if execution_outcome in {
        "signal",
        "queue_timeout",
        "device_lost",
        "preflight_device_unhealthy",
        "preflight_device_quarantined",
    }:
        reasons.append(f"invalid execution outcome={execution_outcome}")
    if execution_outcome == "trap" and actual_detector != "detected":
        reasons.append("unattributed trap is not a detection")
    for name in ("health_before", "health_after"):
        health = execution.get(name)
        if not isinstance(health, dict) or not health.get("healthy"):
            reasons.append(f"{name} failed")
    return not reasons, reasons


def _fault_admission_and_reach(
    result: dict, reach_witness: dict | None
) -> tuple[bool, bool, str | None, list[str]]:
    reasons = []
    mutation = result.get("mutation", {})
    if (
        mutation.get("accounting_schema_version") != 2
        or mutation.get("installation_evidence_complete") is not True
        or mutation.get("requested") != 1
        or mutation.get("planned") != 1
        or mutation.get("applied") != 1
        or mutation.get("discarded_applied", 0) != 0
    ):
        reasons.append("mutation installation was not admitted")
    reservation_status, reservation_reasons = fault_reservation_qualification(
        mutation.get("reservation")
    )
    if reservation_status != FAULT_RESERVATION_QUALIFIED:
        reasons.extend(reservation_reasons)
    execution = result.get("execution", {})
    health_before = execution.get("health_before")
    if not isinstance(health_before, dict) or not health_before.get("healthy"):
        reasons.append("pre-execution health check failed")
    admitted = not reasons
    sanitizer = result.get("sanitizer", {})
    sanitizer_outcome = sanitizer.get("outcome")
    runtime_diagnostic_count = sum(
        int(sanitizer.get(name, 0))
        for name in (
            "inline_diagnostics",
            "replay_diagnostics",
            "sampled_conflicts",
            "sampled_immediate_conflicts",
            "supercollider_diagnostics",
        )
        if isinstance(sanitizer.get(name, 0), int)
    )
    command_ran = execution.get("command_ran") is True
    completed = execution.get("completed") is True
    oracle_outcome = result.get("oracle", {}).get("outcome")
    witness_outcome = None
    if (
        admitted
        and command_ran
        and sanitizer_outcome == "detected"
        and runtime_diagnostic_count > 0
    ):
        witness_outcome = "detector-owned-runtime-diagnostic"
    elif admitted and command_ran and oracle_outcome == "fail":
        witness_outcome = "independent-oracle-manifestation"
    elif admitted and command_ran and completed and reach_witness is not None:
        witness_outcome = str(reach_witness["kind"])
    reached = witness_outcome is not None
    if admitted and not reached:
        reasons.append(
            "trial lacks a detector/oracle runtime witness or reviewed reach proof"
        )
    return admitted, reached, witness_outcome, reasons


def _load_resumable_fault_result(
    result_path: Path,
    *,
    name: str,
    command: list[str],
    environment: dict[str, str],
    identities: list[str],
    workload: Workload,
    profile: str,
    fault: dict,
) -> dict:
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(
            f"cannot resume fault row {result_path}: {error}"
        ) from error
    mismatches = []
    expected_scalars = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "state": "complete",
        "name": name,
    }
    mismatches.extend(
        f"{key}={result.get(key)!r}, expected={expected!r}"
        for key, expected in expected_scalars.items()
        if result.get(key) != expected
    )
    if result.get("command") != command:
        mismatches.append("payload command changed")
    if result.get("site_identities") != identities:
        mismatches.append("fault site identities changed")
    expected_spec = {
        "corpus": workload.corpus,
        "workload": workload.id,
        "flavor": PROFILES[profile].flavor,
        "engine": PROFILES[profile].engine,
        "fault_family": fault["family"],
        "row_role": "fault",
    }
    actual_spec = result.get("spec")
    if not isinstance(actual_spec, dict):
        mismatches.append("fault row has no run specification")
    else:
        mismatches.extend(
            f"spec.{key}={actual_spec.get(key)!r}, expected={expected!r}"
            for key, expected in expected_spec.items()
            if actual_spec.get(key) != expected
        )
    expected_environment = _controlled_environment(environment)
    actual_environment = result.get("environment")
    if not isinstance(actual_environment, dict):
        mismatches.append("fault row has no controlled environment")
    else:
        actual_controlled = _controlled_environment(actual_environment)
        # The runner owns this row-local output path rather than the campaign.
        actual_controlled.pop("RJ_CONSAN_DUMP_DIR", None)
        if actual_controlled != expected_environment:
            mismatches.append("controlled environment changed")
    if mismatches:
        raise ValidationError(
            f"fault row conflicts with resume request {result_path}: "
            + "; ".join(mismatches)
        )
    return result


def _fault(args: argparse.Namespace) -> int:
    selection = _resolve_workload_selection(args, allow_all=False)
    target = selection.target
    workload = _resolved_workload(target, selection.require_workload())
    timeout = args.timeout if args.timeout is not None else workload.run_timeout_seconds
    workspace = _workspace_from_environment()
    if not _doctor(workspace, target, (workload.id,), args.launcher)["ok"]:
        raise ValidationError("workspace doctor failed; run the doctor subcommand")
    if not args.allow_destructive:
        raise ValidationError("fault execution requires --allow-destructive")
    spec_path = args.spec.resolve()
    fault = _load_fault(spec_path, target, workload, args.fault)
    profiles = PROFILE_IDS if args.profile == "all" else (args.profile,)
    launcher = args.launcher
    hook = _hook_path(workspace)
    fault_root = args.artifact_root.resolve() / workload.id / "faults" / fault["id"]
    fault_root.mkdir(parents=True, exist_ok=args.resume)
    provenance = _write_provenance(
        workspace, target, workload, fault_root, launcher
    )
    root = fault_root / "rows"
    root.mkdir(exist_ok=args.resume)
    smoke = _health_smoke_command(
        workspace, target, workload, root / "health-smoke.json"
    )
    if args.smoke_command_json is not None:
        smoke = args.smoke_command_json
    health_command = (
        args.health_command_json
        if args.health_command_json is not None
        else [shutil.which("rocminfo") or "rocminfo"]
    )
    if args.smoke_command_json is None:
        smoke = _with_launcher(launcher, smoke)
    if args.health_command_json is None:
        health_command = _with_launcher(launcher, health_command)
    runner = Path(__file__).with_name("consan_fault_runner.py")
    summaries = []
    profile_summaries = []
    for profile in profiles:
        policy, trials = _fault_trials(fault, profile)
        if policy.get("disposition") == "not-applicable":
            row = {
                "profile": profile,
                "accepted": True,
                "disposition": "not-applicable",
                "reason": policy.get("reason"),
                "tracking_issue": policy.get("tracking_issue"),
            }
            summaries.append(row)
            profile_summaries.append(dict(row))
            continue
        profile_rows = []
        for index, trial in enumerate(trials):
            name = f"{fault['id']}-{profile}-{index}"
            environment = _fault_trial_environment(
                profile,
                workload,
                hook,
                target,
                fault,
                policy,
                trial,
                workspace,
            )
            enabled_mutations = [
                key
                for key in (
                    "RJ_CONSAN_FAULT_DROP_BARRIER",
                    "RJ_CONSAN_FAULT_MOVE_BARRIER",
                    "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER",
                    "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE",
                    "RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS",
                    "RJ_CONSAN_FAULT_LDS_WRONG_ADDRESS",
                )
                if environment.get(key) == "1"
            ]
            if len(enabled_mutations) != 1:
                raise ValidationError(
                    f"trial {profile}/{index} enables {len(enabled_mutations)} mutations"
                )
            command = _fault_workload_command(
                workspace, target, workload, root / "unused.json"
            )
            if workload.kind == "sharktank":
                command.append("--allow-oracle-failure")
            command = _with_launcher(launcher, command)
            identities = sorted(
                value
                for key, value in environment.items()
                if key.startswith("RJ_CONSAN_FAULT_") and key.endswith("IDENTITY")
            )
            invocation = [
                sys.executable,
                str(runner),
                "--artifact-root",
                str(root),
                "--name",
                name,
                "--row-role",
                "fault",
                "--corpus",
                workload.corpus,
                "--workload",
                workload.id,
                "--flavor",
                PROFILES[profile].flavor,
                "--engine",
                PROFILES[profile].engine,
                "--fault-family",
                fault["family"],
                "--timeout",
                str(timeout),
                "--health-timeout",
                str(args.health_timeout),
                "--destructive",
                "--allow-destructive",
                "--serialize-gpu",
                "--health-command-json",
                json.dumps(health_command),
                "--smoke-command-json",
                json.dumps(smoke),
                "--revision-root",
                str(_corpus_root(workspace, workload.corpus)),
                "--hash-file",
                f"hook={hook}",
            ]
            for key, value in _controlled_environment(environment).items():
                invocation.extend(["--env", f"{key}={value}"])
            for identity in identities:
                invocation.extend(["--site-id", identity])
            invocation.extend(["--", *command])
            child_environment = _clean_environment(
                None, workload, hook, target, workspace
            )
            child_environment["CTEST_PARALLEL_LEVEL"] = "1"
            result_path = root / name / "result.json"
            if args.resume and result_path.is_file():
                result = _load_resumable_fault_result(
                    result_path,
                    name=name,
                    command=command,
                    environment=environment,
                    identities=identities,
                    workload=workload,
                    profile=profile,
                    fault=fault,
                )
            else:
                if args.resume and result_path.parent.exists():
                    raise ValidationError(
                        "cannot resume incomplete fault row; use a new artifact "
                        f"root instead of overwriting {result_path.parent}"
                    )
                subprocess.run(invocation, env=child_environment, check=False)
                if result_path.is_file():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    result = None
            if result is None:
                row = {
                    "profile": profile,
                    "trial": index,
                    "accepted": False,
                    "reasons": ["fault runner produced no result.json"],
                    "detector": None,
                    "admitted": False,
                    "reached": False,
                    "reach_outcome": None,
                }
                summaries.append(row)
                profile_rows.append(row)
                continue
            accepted, reasons = _fault_acceptance(result, policy)
            admitted, reached, reach_outcome, reach_reasons = (
                _fault_admission_and_reach(result, fault.get("reach_witness"))
            )
            row = {
                "profile": profile,
                "trial": index,
                "accepted": accepted,
                "reasons": reasons,
                "detector": result.get("sanitizer", {}).get("outcome"),
                "oracle": result.get("oracle", {}).get("outcome"),
                "admitted": admitted,
                "reached": reached,
                "reach_outcome": reach_outcome,
                "reach_reasons": reach_reasons,
                "result": str(result_path),
            }
            summaries.append(row)
            profile_rows.append(row)
        reached_rows = [row for row in profile_rows if row.get("reached") is True]
        detected = sum(row.get("detector") == "detected" for row in reached_rows)
        oracle_manifestations = sum(row.get("oracle") == "fail" for row in reached_rows)
        reach_outcomes = {}
        for row in reached_rows:
            outcome = str(row.get("reach_outcome"))
            reach_outcomes[outcome] = reach_outcomes.get(outcome, 0) + 1
        expected_detector = policy.get("detector")
        profile_reasons = []
        if expected_detector == "statistical":
            minimum = policy.get("minimum_detections")
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
                profile_reasons.append(
                    "statistical policy needs minimum_detections >= 1"
                )
            elif detected < minimum:
                profile_reasons.append(
                    f"detections={detected}/{len(reached_rows)}, minimum={minimum}"
                )
        if not reached_rows:
            profile_reasons.append("no admitted trial reached workload execution")
        detection_interval = (
            _wilson_detection_interval(detected, len(reached_rows))
            if reached_rows
            else None
        )
        oracle_interval = (
            _wilson_detection_interval(oracle_manifestations, len(reached_rows))
            if reached_rows
            else None
        )
        profile_summaries.append(
            {
                "profile": profile,
                "accepted": all(row["accepted"] for row in profile_rows)
                and not profile_reasons,
                "detector_policy": expected_detector,
                "attempted_trials": len(profile_rows),
                "admitted_trials": sum(
                    row.get("admitted") is True for row in profile_rows
                ),
                "reached_trials": len(reached_rows),
                "reach_outcomes": reach_outcomes,
                "detections": detected,
                "trials": len(profile_rows),
                "detection_rate": (
                    detected / len(reached_rows) if reached_rows else None
                ),
                "detection_wilson_95": detection_interval,
                "oracle_manifestations": oracle_manifestations,
                "oracle_manifestation_rate": (
                    oracle_manifestations / len(reached_rows) if reached_rows else None
                ),
                "oracle_manifestation_wilson_95": oracle_interval,
                "reasons": profile_reasons,
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "workload": workload.id,
        "fault": fault["id"],
        "fault_spec": {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
        },
        "launcher": launcher,
        "provenance": str(provenance),
        "rows": summaries,
        "profiles": profile_summaries,
        "accepted": all(profile["accepted"] for profile in profile_summaries),
    }
    if "site_provenance" in fault:
        summary["site_provenance"] = fault["site_provenance"]
    if "reach_witness" in fault:
        summary["reach_witness"] = fault["reach_witness"]
    summary_path = fault_root / "summary.json"
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["accepted"] else 1
