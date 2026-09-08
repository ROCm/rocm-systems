from contextlib import contextmanager
from pathlib import Path
import tempfile


@contextmanager
def temporary_root():
    with tempfile.TemporaryDirectory() as temporary:
        yield Path(temporary)


def coverage(reader: int = 7, **updates: str) -> str:
    fields = {
        "reader": str(reader),
        "flavor": "moi",
        "engine": "record_replay",
        "analysis_complete": "true",
        "expert_limit": "false",
        "access_discovered": "20",
        "access_supported": "20",
        "access_selected": "20",
        "access_patched": "20",
        "access_unsupported": "0",
        "access_resource_failed": "0",
        "access_placement_or_lowering_failed": "0",
        "access_expert_limit_omitted": "0",
        "barrier_discovered": "4",
        "barrier_supported": "4",
        "barrier_selected": "4",
        "barrier_patched": "4",
        "barrier_unsupported": "0",
        "barrier_resource_failed": "0",
        "barrier_placement_or_lowering_failed": "0",
        "barrier_expert_limit_omitted": "0",
        "atomic_discovered": "2",
        "atomic_supported": "2",
        "atomic_selected": "2",
        "atomic_patched": "2",
        "atomic_unsupported": "0",
        "atomic_resource_failed": "0",
        "atomic_placement_or_lowering_failed": "0",
        "atomic_expert_limit_omitted": "0",
        "fence_discovered": "1",
        "fence_supported": "1",
        "fence_selected": "1",
        "fence_patched": "1",
        "fence_unsupported": "0",
        "fence_resource_failed": "0",
        "fence_placement_or_lowering_failed": "0",
        "fence_expert_limit_omitted": "0",
    }
    fields.update(updates)
    for kind in ("access", "barrier", "atomic", "fence"):
        supported_name = f"{kind}_supported"
        if supported_name not in updates:
            continue
        supported = int(fields[supported_name])
        unsupported = int(fields[f"{kind}_unsupported"])
        expert_omitted = int(fields[f"{kind}_expert_limit_omitted"])
        if f"{kind}_discovered" not in updates:
            fields[f"{kind}_discovered"] = str(supported + unsupported)
        if f"{kind}_selected" not in updates:
            fields[f"{kind}_selected"] = str(supported - expert_omitted)
    return "[rocjitsu-dbi-hooks] ConSan coverage " + " ".join(
        f"{key}={value}" for key, value in fields.items()
    )


def verdict(**updates: str) -> str:
    fields = {
        "applicable": "true",
        "analysis_complete": "true",
        "static_complete": "true",
        "dynamic_complete": "true",
        "applicable_code_objects": "1",
        "incomplete_code_objects": "0",
        "access": "20/20",
        "barrier": "4/4",
        "atomic": "2/2",
        "fence": "1/1",
        "dynamic_incomplete": "0",
        "replay_unsupported_access": "0",
        "replay_unsupported_atomics": "0",
        "replay_unsupported_fences": "0",
        "replay_metadata_full": "0",
    }
    fields.update(updates)
    return "[rocjitsu-dbi-hooks] ConSan analysis verdict " + " ".join(
        f"{key}={value}" for key, value in fields.items()
    )


def coverage_site(**updates: str) -> str:
    fields = {
        "reader": "7",
        "kind": "access",
        "disposition": "unsupported",
        "reason": "unsupported_mnemonic",
        "outcome": "unsupported",
        "lowering_reason": "semantic_unsupported",
        "resource_reason": "none",
        "container": "kernel",
        "scope": "kernel",
        "text": "0x10",
        "mnemonic": "ds_load_b96",
    }
    fields.update(updates)
    return "[rocjitsu-dbi-hooks] ConSan coverage_site " + " ".join(
        f"{key}={value}" for key, value in fields.items()
    )


def log(*lines: str, synthesize_sites: bool = True) -> str:
    retained = list(lines)
    if synthesize_sites:
        existing: dict[tuple[str, str, str, str], int] = {}
        for line in retained:
            marker = "ConSan coverage_site "
            if marker not in line:
                continue
            fields = dict(token.split("=", 1) for token in line.split(marker, 1)[1].split())
            key = (
                fields["reader"],
                fields.get("load", ""),
                fields["kind"],
                fields["outcome"],
            )
            existing[key] = existing.get(key, 0) + 1
        serial = 0x1000
        for line in lines:
            marker = "ConSan coverage "
            if marker not in line:
                continue
            tokens = line.split(marker, 1)[1].split()
            if any("=" not in token for token in tokens):
                continue
            fields = dict(token.split("=", 1) for token in tokens)
            if fields.get("flavor") != "moi":
                continue
            reader = fields["reader"]
            load = fields.get("load")
            for kind in ("access", "barrier", "atomic", "fence"):
                desired = {
                    "unsupported": int(fields[f"{kind}_unsupported"]),
                    "resource_failed": int(fields[f"{kind}_resource_failed"]),
                    "placement_or_lowering_failed": int(
                        fields[f"{kind}_placement_or_lowering_failed"]
                    ),
                    "patched": int(fields[f"{kind}_patched"]),
                }
                omitted = int(fields[f"{kind}_expert_limit_omitted"])
                if omitted:
                    desired["placement_or_lowering_failed"] += omitted
                for outcome, count in desired.items():
                    missing = count - existing.get((reader, load or "", kind, outcome), 0)
                    for _ in range(max(missing, 0)):
                        serial += 4
                        if outcome == "unsupported":
                            values = {
                                "disposition": "unsupported",
                                "reason": "unsupported_mnemonic",
                                "lowering_reason": "semantic_unsupported",
                                "resource_reason": "none",
                            }
                        elif outcome == "resource_failed":
                            values = {
                                "disposition": "supported",
                                "reason": "none",
                                "lowering_reason": "unsupported_resource_plan",
                                "resource_reason": "dynamic_stack",
                            }
                        elif outcome == "placement_or_lowering_failed":
                            values = {
                                "disposition": "supported",
                                "reason": "none",
                                "lowering_reason": "instrumentation_patch_missing",
                                "resource_reason": "none",
                            }
                        else:
                            values = {
                                "disposition": "supported",
                                "reason": "none",
                                "lowering_reason": "none",
                                "resource_reason": "none",
                            }
                        retained.append(
                            coverage_site(
                                reader=reader,
                                **({"load": load} if load is not None else {}),
                                kind=kind,
                                outcome=outcome,
                                container=f"{kind}_kernel",
                                text=hex(serial),
                                mnemonic=f"{kind}_site",
                                **values,
                            )
                        )
    return "noise before\n" + "\n".join(retained) + "\nnoise after\n"
