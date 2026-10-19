#!/usr/bin/env python3
"""Check VFIO effects against route totals and completed four-rank IPC imports."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

ROUTE = re.compile(
    r"vfu: committed fabric effects (\d+) -> (\d+): "
    r"reads=(\d+), writes=(\d+), read_bytes=(\d+), write_bytes=(\d+)"
)
EFFECT = re.compile(
    r"vfu: committed fabric effect request=(\d+) (\d+) -> (\d+): "
    r"op=(read|write), backing_id=(\d+), offset=(\d+), bytes=(\d+), "
    r"map_id=(\d+), gpu_va=(0x[0-9a-f]+), iova=(0x[0-9a-f]+)"
)
TRACE = re.compile(r"vfu: fabric effect trace captured=(\d+), dropped=(\d+)")
RANK_BUS = re.compile(r"guest: rccl rank=(\d+) bus_id=([0-9a-f]+) rocr_visible=\d+")
RANK_LOG = re.compile(r"guest: === RCCL rank (\d+) log ===")
IMPORT = re.compile(
    r"Imported shareable buffer device \d+ size (\d+) ptr (0x[0-9a-f]+)"
)
RANK_PID = re.compile(r"\S+:(\d+):\d+ \[0\] NCCL INFO RCCL version")
KFD_ID = re.compile(r"gpu_id = (\d+); unique_id = \d+; location_id = (\d+)")
KFD_DMABUF = re.compile(
    r"P4V KFD DMABUF_IMPORT pid=(\d+) gpu_id=(\d+) "
    r"handle=(0x[0-9a-f]+) va=(0x[0-9a-f]+) bytes=(0x[0-9a-f]+)"
)
KFD_MAP = re.compile(
    r"P4V KFD MAP pid=(\d+) handle=(0x[0-9a-f]+) "
    r"gpu_id=(\d+) va=(0x[0-9a-f]+) bytes=(0x[0-9a-f]+)"
)


def guest_imports(guest_log: Path, config_path: Path) -> dict:
    guest = guest_log.read_text(encoding="utf-8", errors="replace")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # The fixture prints PCI bus IDs in hex with one trailing function nibble.
    by_bus = {
        gpu["location_id"] << 4: gpu["gpu_id"]
        for node in config["system"]["nodes"]
        for gpu in node["gpus"]
    }
    ranks = [(int(rank), int(bus, 16)) for rank, bus in RANK_BUS.findall(guest)]
    headers = list(RANK_LOG.finditer(guest))
    if (
        len(ranks) != 4
        or len(headers) != 4
        or [rank for rank, _ in ranks] != list(range(4))
    ):
        raise ValueError("expected four ordered rank bus IDs and four rank logs")
    if [int(header.group(1)) for header in headers] != list(range(4)):
        raise ValueError("rank log order differs from rank bus order")
    if len(by_bus) < 4 or any(bus not in by_bus for _, bus in ranks):
        raise ValueError("rank bus ID does not identify a configured GPU")

    imported = {}
    for index, header in enumerate(headers):
        gpu_id = by_bus[ranks[index][1]]
        if gpu_id in imported:
            raise ValueError("two ranks resolve to the same configured GPU")
        end = headers[index + 1].start() if index + 1 < len(headers) else len(guest)
        ranges = []
        for size, pointer in IMPORT.findall(guest[header.end() : end]):
            size, pointer = int(size), int(pointer, 16)
            if size == 0 or pointer == 0 or pointer + size > 1 << 64:
                raise ValueError(f"invalid imported range for rank {index}")
            ranges.append(
                {
                    "rank": index,
                    "gpu_id": gpu_id,
                    "pointer": pointer,
                    "size": size,
                    "effects": 0,
                    "bytes": 0,
                }
            )
        if not ranges:
            raise ValueError(f"rank {index} has no completed RCCL IPC import")
        imported[gpu_id] = ranges
    return imported


def guest_kfd_maps(guest_log: Path, config_path: Path, imported: dict) -> dict:
    guest = guest_log.read_text(encoding="utf-8", errors="replace")
    if "guest: amdgpu.p4v_ipc_map_trace=Y" not in guest:
        raise ValueError("guest KFD import/MAP trace was not enabled")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    by_location = {
        gpu["location_id"]: gpu["gpu_id"]
        for node in config["system"]["nodes"]
        for gpu in node["gpus"]
    }
    kfd_id_to_gpu = {}
    for kfd_id_text, location_text in KFD_ID.findall(guest):
        kfd_id, location = int(kfd_id_text), int(location_text)
        if location not in by_location:
            raise ValueError(f"KFD ID {kfd_id} has no configured location")
        gpu_id = by_location[location]
        if kfd_id in kfd_id_to_gpu and kfd_id_to_gpu[kfd_id] != gpu_id:
            raise ValueError(f"KFD ID {kfd_id} has conflicting locations")
        kfd_id_to_gpu[kfd_id] = gpu_id
    if len(kfd_id_to_gpu) != 4 or set(kfd_id_to_gpu.values()) != set(imported):
        raise ValueError("KFD GPU IDs do not cover the four RCCL ranks")

    rank_to_gpu = {entries[0]["rank"]: gpu_id for gpu_id, entries in imported.items()}
    headers = list(RANK_LOG.finditer(guest))
    pid_by_gpu = {}
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(guest)
        pids = set(map(int, RANK_PID.findall(guest[header.end() : end])))
        if len(pids) != 1:
            raise ValueError(f"rank {index} has no unique RCCL process ID")
        pid_by_gpu[rank_to_gpu[index]] = pids.pop()
    if len(set(pid_by_gpu.values())) != 4:
        raise ValueError("RCCL ranks reused a process ID")

    dma_imports = {}
    for pid, gpu_id, handle, va, size in KFD_DMABUF.findall(guest):
        key = (int(pid), int(handle, 16))
        if key in dma_imports:
            raise ValueError(f"duplicate KFD DMA-buf import: {key}")
        dma_imports[key] = {
            "gpu_id": int(gpu_id),
            "va": int(va, 16),
            "bytes": int(size, 16),
        }
    completed_maps = {}
    for pid, handle, gpu_id, va, size in KFD_MAP.findall(guest):
        key = (int(pid), int(handle, 16))
        if key in completed_maps:
            raise ValueError(f"duplicate KFD completed MAP: {key}")
        completed_maps[key] = {
            "gpu_id": int(gpu_id),
            "va": int(va, 16),
            "bytes": int(size, 16),
        }
        if key not in dma_imports or (
            completed_maps[key]["va"],
            completed_maps[key]["bytes"],
        ) != (dma_imports[key]["va"], dma_imports[key]["bytes"]):
            raise ValueError(f"KFD MAP has no matching DMA-buf import: {key}")
    if not completed_maps:
        raise ValueError("guest has no completed KFD DMA-buf MAP records")
    return {
        "pid_by_gpu": pid_by_gpu,
        "id_to_gpu": kfd_id_to_gpu,
        "imports": dma_imports,
        "maps": completed_maps,
    }


def check(
    server_log: Path,
    case_summary: Path,
    guest_log: Path,
    config_path: Path,
    require_kfd_map: bool = False,
) -> dict:
    case = json.loads(case_summary.read_text(encoding="utf-8"))
    imported = guest_imports(guest_log, config_path)
    kfd = guest_kfd_maps(guest_log, config_path, imported) if require_kfd_map else None
    kfd_groups = {}
    routes = {}
    traced = defaultdict(
        lambda: {"reads": 0, "writes": 0, "read_bytes": 0, "write_bytes": 0}
    )
    map_groups = {}
    request_ids = set()
    trace_status = []
    for line in server_log.read_text(encoding="utf-8").splitlines():
        if match := ROUTE.search(line):
            source, target, *counts = map(int, match.groups())
            key = f"{source}->{target}"
            if key in routes:
                raise ValueError(f"duplicate route counter: {key}")
            routes[key] = dict(
                zip(("reads", "writes", "read_bytes", "write_bytes"), counts)
            )
        elif match := EFFECT.search(line):
            (
                request,
                source,
                target,
                op,
                backing,
                offset,
                length,
                map_id,
                gpu_va,
                iova,
            ) = match.groups()
            request = int(request)
            source, target, backing, offset, length, map_id = map(
                int, (source, target, backing, offset, length, map_id)
            )
            gpu_va, iova = int(gpu_va, 16), int(iova, 16)
            if request in request_ids:
                raise ValueError(f"duplicate committed request: {request}")
            request_ids.add(request)
            if not all(
                (request, source, target, backing, length, map_id, gpu_va, iova)
            ):
                raise ValueError(
                    f"incomplete target-map coordinates for request {request}"
                )
            matches = [
                entry
                for entry in imported.get(source, ())
                if entry["pointer"] <= gpu_va
                and gpu_va + length <= entry["pointer"] + entry["size"]
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"request {request} matches {len(matches)} RCCL IPC imports"
                )
            matches[0]["effects"] += 1
            matches[0]["bytes"] += length
            if kfd is not None:
                pid = kfd["pid_by_gpu"][source]
                map_matches = [
                    (key, entry)
                    for key, entry in kfd["maps"].items()
                    if key[0] == pid
                    and entry["va"] <= gpu_va
                    and gpu_va + length <= entry["va"] + entry["bytes"]
                ]
                if len(map_matches) != 1:
                    raise ValueError(
                        f"request {request} matches {len(map_matches)} KFD MAPs"
                    )
                kfd_key, kfd_map = map_matches[0]
                kfd_import = kfd["imports"][kfd_key]
                if (
                    kfd["id_to_gpu"].get(kfd_map["gpu_id"]) != source
                    or kfd["id_to_gpu"].get(kfd_import["gpu_id"]) != target
                    or (kfd_map["va"], kfd_map["bytes"])
                    != (matches[0]["pointer"], matches[0]["size"])
                ):
                    raise ValueError(
                        f"KFD requester, owner, or extent differs on request {request}"
                    )
                if kfd_key not in kfd_groups:
                    kfd_groups[kfd_key] = {
                        "pid": pid,
                        "handle": kfd_key[1],
                        "requester_gpu_id": source,
                        "owner_gpu_id": target,
                        "requester_kfd_gpu_id": kfd_map["gpu_id"],
                        "owner_kfd_gpu_id": kfd_import["gpu_id"],
                        "va": kfd_map["va"],
                        "mapped_bytes": kfd_map["bytes"],
                        "effects": 0,
                        "effect_bytes": 0,
                    }
                kfd_groups[kfd_key]["effects"] += 1
                kfd_groups[kfd_key]["effect_bytes"] += length
            key = f"{source}->{target}"
            traced[key][f"{op}s"] += 1
            traced[key][f"{op}_bytes"] += length
            group_key = (source, target, map_id, backing)
            delta = offset - iova
            if group_key not in map_groups:
                map_groups[group_key] = {
                    "effects": 0,
                    "bytes": 0,
                    "first_offset": offset,
                    "last_offset": offset + length,
                    "backing_minus_iova": delta,
                }
            group = map_groups[group_key]
            if group["backing_minus_iova"] != delta:
                raise ValueError(f"inconsistent target-map coordinates: {group_key}")
            group["effects"] += 1
            group["bytes"] += length
            group["first_offset"] = min(group["first_offset"], offset)
            group["last_offset"] = max(group["last_offset"], offset + length)
        elif match := TRACE.search(line):
            trace_status.append(tuple(map(int, match.groups())))

    if len(trace_status) != 1:
        raise ValueError(f"expected one trace status, found {len(trace_status)}")
    captured, dropped = trace_status[0]
    if captured != len(request_ids) or dropped != 0:
        raise ValueError(
            f"trace incomplete: captured={captured}, records={len(request_ids)}, "
            f"dropped={dropped}"
        )
    if not routes or set(routes) != set(case["routes"]):
        raise ValueError("directed route set does not match the case summary")
    active_routes = {
        key for key, counts in routes.items() if counts["reads"] + counts["writes"] != 0
    }
    if set(traced) != active_routes:
        raise ValueError("trace route set differs from nonzero directed counters")
    for key, counts in routes.items():
        if counts != traced[key]:
            raise ValueError(f"trace totals differ from endpoint counters on {key}")
        for op in ("reads", "writes"):
            if counts[op] != case["routes"][key][op]:
                raise ValueError(f"case summary differs on {key}: {op}")
    for op in ("reads", "writes"):
        if sum(counts[op] for counts in routes.values()) != case[f"total_{op}"]:
            raise ValueError(f"case total differs: {op}")

    result = {
        "case": case["case"],
        "elements": case["elements"],
        "captured": captured,
        "dropped": dropped,
        "read_bytes": sum(counts["read_bytes"] for counts in routes.values()),
        "write_bytes": sum(counts["write_bytes"] for counts in routes.values()),
        "rank_imports": [
            {
                "rank": entries[0]["rank"],
                "gpu_id": gpu_id,
                "completed_imports": len(entries),
                "imports_with_effects": sum(entry["effects"] != 0 for entry in entries),
                "effects": sum(entry["effects"] for entry in entries),
                "bytes": sum(entry["bytes"] for entry in entries),
            }
            for gpu_id, entries in sorted(imported.items())
        ],
        "import_groups": [
            entry
            for entries in imported.values()
            for entry in entries
            if entry["effects"] != 0
        ],
        "map_groups": [
            {
                "requester": key[0],
                "owner": key[1],
                "map_id": key[2],
                "backing_id": key[3],
                **values,
            }
            for key, values in sorted(map_groups.items())
        ],
    }
    if kfd is not None:
        result["guest_kfd"] = {
            "dma_buf_imports": len(kfd["imports"]),
            "completed_maps": len(kfd["maps"]),
            "handles_with_effects": len(kfd_groups),
            "map_groups": [kfd_groups[key] for key in sorted(kfd_groups)],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_log", type=Path)
    parser.add_argument("case_summary", type=Path)
    parser.add_argument("guest_log", type=Path)
    parser.add_argument("config_json", type=Path)
    parser.add_argument("--require-kfd-map", action="store_true")
    arguments = parser.parse_args()
    try:
        print(
            json.dumps(
                check(
                    arguments.server_log,
                    arguments.case_summary,
                    arguments.guest_log,
                    arguments.config_json,
                    arguments.require_kfd_map,
                ),
                indent=2,
            )
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"fabric trace check failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
