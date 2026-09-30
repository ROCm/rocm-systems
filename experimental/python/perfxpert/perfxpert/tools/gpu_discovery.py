###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
###############################################################################

"""Runtime GPU spec discovery through ROCm command-line tools.

This module gathers the hardware facts that ROCm exposes locally instead of
requiring every installed GPU to already have an entry in ``gpu_specs.yaml``.
Discovery is best-effort and read-only:

* ``rocminfo`` supplies HSA topology such as gfx id, CU count, wave size,
  LDS size, SIMDs per CU, and max clock.
* KFD topology sysfs files supply the same read-only topology fields when
  Python-launched ``rocminfo`` cannot open the device node.
* ``rocm-smi`` supplies SKU-facing fields such as GFX version and VRAM size on
  stacks where its JSON mode is available.
* ``amd-smi`` supplies static and metric JSON such as board, clock, PCIe,
  power-limit, and VRAM fields on newer ROCm stacks.

The ROCm tools do not expose every theoretical instruction throughput value on
all systems, so peak FLOP fields are derived only when runtime topology and a
known architecture family are available. Static knowledge remains the fallback
for offline/non-local architectures.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from perfxpert.tools._class import ToolClass, tool_class


PERFXPERT_DISABLE_RUNTIME_GPU_SPECS = "PERFXPERT_DISABLE_RUNTIME_GPU_SPECS"
PERFXPERT_GPU_DISCOVERY_TIMEOUT = "PERFXPERT_GPU_DISCOVERY_TIMEOUT"
_KFD_TOPOLOGY_ROOT = Path("/sys/class/kfd/kfd/topology/nodes")

_GFX_RE = re.compile(r"\b(gfx[0-9a-z]{3,5})\b")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_NA_VALUES = {"", "N/A", "NA", "none", "None", "unknown", "Unknown"}


def _runtime_discovery_disabled() -> bool:
    value = os.environ.get(PERFXPERT_DISABLE_RUNTIME_GPU_SPECS, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _timeout_seconds() -> float:
    value = os.environ.get(PERFXPERT_GPU_DISCOVERY_TIMEOUT, "").strip()
    if not value:
        return 5.0
    try:
        return max(float(value), 0.1)
    except ValueError:
        return 5.0


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in _NA_VALUES
    return False


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return None
    if value.strip() in _NA_VALUES:
        return None
    match = re.search(r"-?\d+", value.replace(",", ""))
    return int(match.group(0)) if match else None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    if value.strip() in _NA_VALUES:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group(0)) if match else None


def _extract_gfx(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    match = _GFX_RE.search(value)
    return match.group(1) if match else None


def _normalize_name(value: Any) -> Optional[str]:
    if not isinstance(value, str) or value.strip() in _NA_VALUES:
        return None
    return value.strip()


def _max_mhz_from_clock_block(value: Any) -> Optional[float]:
    candidates: List[float] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if "max_clk" in obj:
                parsed = _as_float(obj.get("max_clk"))
                if parsed is not None:
                    candidates.append(parsed)
            for item in obj.values():
                _walk(item)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)
        else:
            parsed = _as_float(obj)
            if parsed is not None:
                candidates.append(parsed)

    _walk(value)
    positive = [candidate for candidate in candidates if candidate > 0]
    return max(positive) if positive else None


def _run_command(argv: List[str]) -> Optional[str]:
    executable = shutil.which(argv[0])
    if executable is None:
        return None
    try:
        proc = subprocess.run(
            [executable, *argv[1:]],
            capture_output=True,
            text=True,
            timeout=_timeout_seconds(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = proc.stdout or proc.stderr or ""
    if proc.returncode != 0 and not output:
        return None
    return _strip_ansi(output)


def _set_if_present(gpu: Dict[str, Any], key: str, value: Any, source: str) -> None:
    if _is_missing(value):
        return
    if isinstance(value, (int, float)) and value <= 0 and key.startswith("max_"):
        return
    if key not in gpu or _is_missing(gpu.get(key)) or key == "name":
        gpu[key] = value
        gpu.setdefault("spec_sources", {})[key] = source


def _parse_rocminfo(text: str) -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    group_pool = False

    def _finish() -> None:
        if current and current.get("device_type") == "GPU" and current.get("gfx_id"):
            current.pop("device_type", None)
            gpus.append(current.copy())

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if re.fullmatch(r"Agent\s+\d+", line):
            _finish()
            current = {"spec_sources": {}}
            group_pool = False
            continue
        if current is None or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key == "Segment" and "GROUP" in value:
            group_pool = True
            continue
        if group_pool and key == "Size":
            size_kb = _as_int(value)
            if size_kb is not None:
                _set_if_present(current, "lds_kb", size_kb, "rocminfo")
                _set_if_present(current, "lds_per_cu_kb", size_kb, "rocminfo")
            group_pool = False
            continue

        if key == "Device Type":
            current["device_type"] = value
        elif key == "Name":
            gfx_id = _extract_gfx(value)
            if gfx_id:
                _set_if_present(current, "gfx_id", gfx_id, "rocminfo")
            elif "name" not in current:
                _set_if_present(current, "name", _normalize_name(value), "rocminfo")
        elif key == "Marketing Name":
            _set_if_present(current, "name", _normalize_name(value), "rocminfo")
        elif key == "Node":
            _set_if_present(current, "node_id", _as_int(value), "rocminfo")
        elif key == "Compute Unit":
            _set_if_present(current, "cu_count", _as_int(value), "rocminfo")
        elif key == "SIMDs per CU":
            _set_if_present(current, "simds_per_cu", _as_int(value), "rocminfo")
        elif key == "Max Clock Freq. (MHz)":
            _set_if_present(current, "max_sclk_mhz", _as_float(value), "rocminfo")
        elif key == "Wavefront Size":
            _set_if_present(current, "wave_size", _as_int(value), "rocminfo")
        elif key == "Max Waves Per CU":
            max_waves = _as_int(value)
            if max_waves is not None:
                _set_if_present(current, "max_waves_per_cu", max_waves, "rocminfo")
        elif key == "BDFID":
            _set_if_present(current, "bdf_id", _as_int(value), "rocminfo")
        elif key == "Uuid":
            _set_if_present(current, "uuid", _normalize_name(value), "rocminfo")

    _finish()
    return gpus


def _load_json(text: Optional[str]) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError, ValueError):
        return None


def _parse_rocm_smi_json(text: Optional[str]) -> List[Dict[str, Any]]:
    raw = _load_json(text)
    if not isinstance(raw, dict):
        return []

    gpus: List[Dict[str, Any]] = []
    for key, rec in raw.items():
        if not isinstance(key, str) or not key.startswith("card") or not isinstance(rec, dict):
            continue
        gpu: Dict[str, Any] = {"spec_sources": {}}
        _set_if_present(gpu, "gpu_id", _as_int(key.replace("card", "")), "rocm-smi")
        _set_if_present(gpu, "node_id", _as_int(rec.get("Node ID")), "rocm-smi")
        _set_if_present(gpu, "gfx_id", _extract_gfx(str(rec.get("GFX Version", ""))), "rocm-smi")
        _set_if_present(gpu, "name", _normalize_name(rec.get("Card Series")), "rocm-smi")
        _set_if_present(gpu, "card_model", _normalize_name(rec.get("Card Model")), "rocm-smi")
        _set_if_present(gpu, "vram_total_bytes", _as_int(rec.get("VRAM Total Memory (B)")), "rocm-smi")
        _set_if_present(gpu, "vram_used_bytes", _as_int(rec.get("VRAM Total Used Memory (B)")), "rocm-smi")

        for rec_key, value in rec.items():
            normalized_key = str(rec_key).lower()
            if "sclk clock speed" in normalized_key:
                _set_if_present(gpu, "current_sclk_mhz", _as_float(value), "rocm-smi")
            elif "mclk clock speed" in normalized_key:
                _set_if_present(gpu, "current_mclk_mhz", _as_float(value), "rocm-smi")
        if gpu.keys() - {"spec_sources"}:
            gpus.append(gpu)
    return gpus


def _parse_amd_smi_list_json(text: Optional[str]) -> List[Dict[str, Any]]:
    raw = _load_json(text)
    if not isinstance(raw, list):
        return []
    gpus: List[Dict[str, Any]] = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        gpu: Dict[str, Any] = {"spec_sources": {}}
        _set_if_present(gpu, "gpu_id", _as_int(rec.get("gpu")), "amd-smi")
        _set_if_present(gpu, "node_id", _as_int(rec.get("node_id")), "amd-smi")
        _set_if_present(gpu, "bdf", _normalize_name(rec.get("bdf")), "amd-smi")
        _set_if_present(gpu, "uuid", _normalize_name(rec.get("uuid")), "amd-smi")
        if gpu.keys() - {"spec_sources"}:
            gpus.append(gpu)
    return gpus


def _parse_amd_smi_gpu_data_json(text: Optional[str], source_command: str) -> List[Dict[str, Any]]:
    raw = _load_json(text)
    if not isinstance(raw, dict):
        return []
    rows = raw.get("gpu_data")
    if not isinstance(rows, list):
        return []

    gpus: List[Dict[str, Any]] = []
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        gpu: Dict[str, Any] = {"spec_sources": {}}
        _set_if_present(gpu, "gpu_id", _as_int(rec.get("gpu")), "amd-smi")

        asic = rec.get("asic") if isinstance(rec.get("asic"), dict) else {}
        _set_if_present(gpu, "name", _normalize_name(asic.get("market_name")), "amd-smi")
        _set_if_present(gpu, "gfx_id", _extract_gfx(str(asic.get("target_graphics_version", ""))), "amd-smi")
        _set_if_present(gpu, "cu_count", _as_int(asic.get("num_compute_units")), "amd-smi")

        board = rec.get("board") if isinstance(rec.get("board"), dict) else {}
        _set_if_present(gpu, "board_product_name", _normalize_name(board.get("product_name")), "amd-smi")

        bus = rec.get("bus") if isinstance(rec.get("bus"), dict) else {}
        _set_if_present(gpu, "bdf", _normalize_name(bus.get("bdf")), "amd-smi")
        _set_if_present(gpu, "pcie_width", _as_int(bus.get("max_pcie_width")), "amd-smi")
        _set_if_present(gpu, "pcie_speed_gts", _as_float(bus.get("max_pcie_speed")), "amd-smi")

        numa = rec.get("numa") if isinstance(rec.get("numa"), dict) else {}
        _set_if_present(gpu, "numa_node", _as_int(numa.get("node")), "amd-smi")

        vram = rec.get("vram") if isinstance(rec.get("vram"), dict) else {}
        _set_if_present(gpu, "vram_type", _normalize_name(vram.get("type")), "amd-smi")
        _set_if_present(gpu, "vram_bit_width", _as_int(vram.get("bit_width")), "amd-smi")
        _set_if_present(gpu, "vram_total_bytes", _bytes_from_unit_value(vram.get("size")), "amd-smi")
        bandwidth_tbs = _bandwidth_tbs_from_unit_value(vram.get("max_bandwidth"))
        _set_if_present(gpu, "memory_bandwidth_tbs", bandwidth_tbs, "amd-smi")

        limit = rec.get("limit") if isinstance(rec.get("limit"), dict) else {}
        ppt0 = limit.get("ppt0") if isinstance(limit.get("ppt0"), dict) else {}
        _set_if_present(gpu, "tdp_w", _as_float(ppt0.get("socket_power_limit")), "amd-smi")

        clock = rec.get("clock") if isinstance(rec.get("clock"), dict) else {}
        sys_clock = clock.get("sys") if isinstance(clock.get("sys"), dict) else None
        mem_clock = clock.get("mem") if isinstance(clock.get("mem"), dict) else None
        _set_if_present(gpu, "max_sclk_mhz", _max_mhz_from_clock_block(sys_clock), "amd-smi")
        _set_if_present(gpu, "max_mclk_mhz", _max_mhz_from_clock_block(mem_clock), "amd-smi")
        _set_if_present(gpu, "current_sclk_mhz", _as_float(_dict_get(sys_clock, "current_frequency")), "amd-smi")
        _set_if_present(gpu, "current_mclk_mhz", _as_float(_dict_get(mem_clock, "current_frequency")), "amd-smi")

        mem_usage = rec.get("mem_usage") if isinstance(rec.get("mem_usage"), dict) else {}
        _set_if_present(gpu, "vram_total_bytes", _bytes_from_unit_value(mem_usage.get("total_vram")), "amd-smi")
        _set_if_present(gpu, "vram_used_bytes", _bytes_from_unit_value(mem_usage.get("used_vram")), "amd-smi")

        if source_command == "amd-smi metric":
            metric_clock = rec.get("clock") if isinstance(rec.get("clock"), dict) else {}
            gfx_clock = metric_clock.get("gfx_0") if isinstance(metric_clock.get("gfx_0"), dict) else None
            mem0_clock = metric_clock.get("mem_0") if isinstance(metric_clock.get("mem_0"), dict) else None
            _set_if_present(gpu, "max_sclk_mhz", _as_float(_dict_get(gfx_clock, "max_clk")), "amd-smi")
            _set_if_present(gpu, "current_sclk_mhz", _as_float(_dict_get(gfx_clock, "clk")), "amd-smi")
            _set_if_present(gpu, "max_mclk_mhz", _as_float(_dict_get(mem0_clock, "max_clk")), "amd-smi")
            _set_if_present(gpu, "current_mclk_mhz", _as_float(_dict_get(mem0_clock, "clk")), "amd-smi")

        if gpu.keys() - {"spec_sources"}:
            gpus.append(gpu)
    return gpus


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_key_value_text(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            result[parts[0]] = parts[1]
    return result


def _gfx_from_target_version(value: Any) -> Optional[str]:
    version = _as_int(value)
    if version is None or version <= 0:
        return None
    digits = f"{version:06d}"
    major = int(digits[:2])
    minor = int(digits[2:4])
    stepping = int(digits[4:])
    if major <= 0:
        return None
    return f"gfx{major}{minor}{stepping}"


def _parse_kfd_topology(root: Path = _KFD_TOPOLOGY_ROOT) -> List[Dict[str, Any]]:
    if not root.exists():
        return []

    gpus: List[Dict[str, Any]] = []
    try:
        node_dirs = sorted(root.iterdir())
    except OSError:
        return []

    for node_dir in node_dirs:
        if not node_dir.is_dir():
            continue
        props = _parse_key_value_text(_read_text(node_dir / "properties"))
        gfx_id = _extract_gfx(_read_text(node_dir / "name")) or _gfx_from_target_version(
            props.get("gfx_target_version")
        )
        if not gfx_id:
            continue

        gpu: Dict[str, Any] = {"spec_sources": {}}
        _set_if_present(gpu, "node_id", _as_int(node_dir.name), "kfd-topology")
        _set_if_present(gpu, "gfx_id", gfx_id, "kfd-topology")
        _set_if_present(gpu, "max_waves_per_simd", _as_int(props.get("max_waves_per_simd")), "kfd-topology")
        _set_if_present(gpu, "lds_kb", _as_int(props.get("lds_size_in_kb")), "kfd-topology")
        _set_if_present(gpu, "lds_per_cu_kb", _as_int(props.get("lds_size_in_kb")), "kfd-topology")
        _set_if_present(gpu, "wave_size", _as_int(props.get("wave_front_size")), "kfd-topology")
        _set_if_present(gpu, "simds_per_cu", _as_int(props.get("simd_per_cu")), "kfd-topology")
        _set_if_present(gpu, "max_sclk_mhz", _as_float(props.get("max_engine_clk_fcompute")), "kfd-topology")
        _set_if_present(gpu, "max_sclk_mhz", _as_float(props.get("max_engine_clk_ccompute")), "kfd-topology")

        simd_count = _as_int(props.get("simd_count"))
        simds_per_cu = _as_int(props.get("simd_per_cu"))
        if simd_count and simds_per_cu:
            _set_if_present(gpu, "cu_count", simd_count // simds_per_cu, "kfd-topology")

        try:
            mem_bank_props = sorted((node_dir / "mem_banks").glob("*/properties"))
        except OSError:
            mem_bank_props = []

        for bank_dir in mem_bank_props:
            bank_props = _parse_key_value_text(_read_text(bank_dir))
            if _as_int(bank_props.get("heap_type")) != 1:
                continue
            _set_if_present(gpu, "vram_total_bytes", _as_int(bank_props.get("size_in_bytes")), "kfd-topology")
            _set_if_present(gpu, "vram_bit_width", _as_int(bank_props.get("width")), "kfd-topology")
            _set_if_present(gpu, "max_mclk_mhz", _as_float(bank_props.get("mem_clk_max")), "kfd-topology")
            _set_if_present(
                gpu,
                "memory_bandwidth_tbs",
                _memory_bandwidth_tbs_from_kfd(
                    bank_props.get("width"),
                    bank_props.get("mem_clk_max"),
                ),
                "derived-from-kfd-topology",
            )
            break

        gpus.append(gpu)
    return gpus


def _dict_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _bytes_from_unit_value(value: Any) -> Optional[int]:
    if isinstance(value, dict):
        amount = _as_float(value.get("value"))
        unit = str(value.get("unit", "")).strip().lower()
    else:
        amount = _as_float(value)
        unit = str(value).strip().lower() if isinstance(value, str) else ""
    if amount is None:
        return None
    if unit in {"kb", "kib"}:
        return int(amount * 1024)
    if unit in {"mb", "mib"}:
        return int(amount * 1024 * 1024)
    if unit in {"gb", "gib"}:
        return int(amount * 1024 * 1024 * 1024)
    if unit in {"tb", "tib"}:
        return int(amount * 1024 * 1024 * 1024 * 1024)
    return int(amount)


def _bandwidth_tbs_from_unit_value(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        amount = _as_float(value.get("value"))
        unit = str(value.get("unit", "")).strip().lower()
    else:
        amount = _as_float(value)
        unit = str(value).strip().lower() if isinstance(value, str) else ""
    if amount is None or amount <= 0:
        return None
    if "gb/s" in unit or "gbps" in unit:
        return round(amount / 1000.0, 3)
    if "mb/s" in unit:
        return round(amount / 1_000_000.0, 3)
    return amount if amount < 100 else round(amount / 1000.0, 3)


def _memory_bandwidth_tbs_from_kfd(width_bits: Any, mem_clk_mhz: Any) -> Optional[float]:
    width = _as_float(width_bits)
    clock = _as_float(mem_clk_mhz)
    if width is None or clock is None or width <= 0 or clock <= 0:
        return None
    bytes_per_cycle = width / 8.0
    gbps = clock * bytes_per_cycle * 2.0 / 1000.0
    return round(gbps / 1000.0, 3)


def _merge_gpu_lists(sources: Iterable[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    identity_keys = ("bdf", "uuid", "node_id", "gpu_id")

    def _has_identity(record: Dict[str, Any]) -> bool:
        return any(record.get(key) is not None for key in identity_keys)

    def _identity_conflicts(
        candidate: Dict[str, Any],
        existing: Dict[str, Any],
        matched_key: str,
    ) -> bool:
        for key in identity_keys:
            if key == matched_key:
                continue
            candidate_value = candidate.get(key)
            existing_value = existing.get(key)
            if candidate_value is not None and existing_value is not None and candidate_value != existing_value:
                return True
        return False

    def _find_match(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for existing in merged:
            for key in identity_keys:
                if candidate.get(key) is None or candidate.get(key) != existing.get(key):
                    continue
                if key == "gpu_id" and _identity_conflicts(candidate, existing, key):
                    continue
                return existing
        if not _has_identity(candidate):
            for existing in merged:
                if (
                    not _has_identity(existing)
                    and candidate.get("gfx_id") is not None
                    and candidate.get("gfx_id") == existing.get("gfx_id")
                ):
                    return existing
        return None

    for source_gpus in sources:
        for candidate in source_gpus:
            match = _find_match(candidate)
            if match is None:
                match = {"spec_sources": {}}
                merged.append(match)
            for key, value in candidate.items():
                if key == "spec_sources":
                    match.setdefault("spec_sources", {}).update(value)
                    continue
                source = candidate.get("spec_sources", {}).get(key, "runtime")
                _set_if_present(match, key, value, source)

    for gpu in merged:
        _fill_arch_defaults(gpu)
        _derive_peak_fields(gpu)
    return merged


def _fill_arch_defaults(gpu: Dict[str, Any]) -> None:
    gfx_id = str(gpu.get("gfx_id") or "")
    if not gfx_id:
        return
    defaults: Dict[str, Any]
    if gfx_id.startswith(("gfx90a", "gfx94", "gfx95")):
        defaults = {
            "wave_size": 64,
            "simds_per_cu": 4,
            "lds_kb": 64,
            "lds_per_cu_kb": 64,
            "max_vgprs_per_thread": 256,
            "vgprs_per_simd": 512,
        }
    elif gfx_id.startswith("gfx908"):
        defaults = {
            "wave_size": 64,
            "simds_per_cu": 4,
            "lds_kb": 64,
            "lds_per_cu_kb": 64,
            "max_vgprs_per_thread": 256,
            "vgprs_per_simd": 256,
        }
    elif gfx_id.startswith("gfx10"):
        defaults = {
            "wave_size": 32,
            "simds_per_cu": 2,
            "lds_kb": 64,
            "lds_per_cu_kb": 64,
            "max_vgprs_per_thread": 256,
            "vgprs_per_simd": 1024,
        }
    elif gfx_id.startswith(("gfx11", "gfx12")):
        defaults = {
            "wave_size": 32,
            "simds_per_cu": 2,
            "lds_kb": 64,
            "lds_per_cu_kb": 64,
            "max_vgprs_per_thread": 256,
            "vgprs_per_simd": 1536,
        }
    else:
        defaults = {}

    if "max_waves_per_simd" not in defaults:
        max_waves_per_cu = _as_int(gpu.get("max_waves_per_cu"))
        simds_per_cu = _as_int(gpu.get("simds_per_cu"))
        if max_waves_per_cu and simds_per_cu:
            defaults["max_waves_per_simd"] = max(max_waves_per_cu // simds_per_cu, 1)

    for key, value in defaults.items():
        if key not in gpu or _is_missing(gpu.get(key)):
            gpu[key] = value
            gpu.setdefault("spec_sources", {})[key] = "architecture-default"


def _derive_peak_fields(gpu: Dict[str, Any]) -> None:
    cu_count = _as_int(gpu.get("cu_count"))
    max_sclk_mhz = _as_float(gpu.get("max_sclk_mhz"))
    gfx_id = str(gpu.get("gfx_id") or "")
    throughput = _throughput_model(gfx_id)
    if not cu_count or not max_sclk_mhz or throughput is None:
        return

    fp32 = round(cu_count * throughput["fp32_ops_per_cu_cycle"] * max_sclk_mhz / 1_000_000.0, 3)
    derived = {
        "peak_fp32_tflops": fp32,
        "peak_fp64_tflops": fp32 * throughput["fp64_ratio"],
        "peak_fp16_tflops": fp32 * throughput["fp16_ratio"],
        "peak_bf16_tflops": fp32 * throughput["bf16_ratio"],
        "peak_fp8_tflops": fp32 * throughput["fp8_ratio"],
        "peak_int8_tops": fp32 * throughput["int8_ratio"],
    }
    for key, value in derived.items():
        _set_if_present(gpu, key, round(value, 3), "derived-from-runtime-topology")


def _throughput_model(gfx_id: str) -> Optional[Dict[str, float]]:
    if gfx_id.startswith("gfx908"):
        return {
            "fp32_ops_per_cu_cycle": 128.0,
            "fp64_ratio": 0.5,
            "fp16_ratio": 8.0,
            "bf16_ratio": 4.0,
            "fp8_ratio": 0.0,
            "int8_ratio": 4.0,
        }
    if gfx_id.startswith("gfx90a"):
        return {
            "fp32_ops_per_cu_cycle": 128.0,
            "fp64_ratio": 1.0,
            "fp16_ratio": 8.0,
            "bf16_ratio": 8.0,
            "fp8_ratio": 0.0,
            "int8_ratio": 8.0,
        }
    if gfx_id.startswith("gfx94"):
        return {
            "fp32_ops_per_cu_cycle": 256.0,
            "fp64_ratio": 0.5,
            "fp16_ratio": 8.0,
            "bf16_ratio": 8.0,
            "fp8_ratio": 16.0,
            "int8_ratio": 16.0,
        }
    if gfx_id.startswith("gfx95"):
        return {
            "fp32_ops_per_cu_cycle": 256.0,
            "fp64_ratio": 0.5,
            "fp16_ratio": 16.0,
            "bf16_ratio": 16.0,
            "fp8_ratio": 32.0,
            "int8_ratio": 32.0,
        }
    if gfx_id.startswith("gfx10"):
        return {
            "fp32_ops_per_cu_cycle": 128.0,
            "fp64_ratio": 1.0 / 32.0,
            "fp16_ratio": 2.0,
            "bf16_ratio": 0.0,
            "fp8_ratio": 0.0,
            "int8_ratio": 0.0,
        }
    if gfx_id.startswith(("gfx11", "gfx12")):
        return {
            "fp32_ops_per_cu_cycle": 256.0,
            "fp64_ratio": 1.0 / 32.0,
            "fp16_ratio": 2.0,
            "bf16_ratio": 2.0,
            "fp8_ratio": 0.0,
            "int8_ratio": 2.0,
        }
    return None


@lru_cache(maxsize=1)
def _discover_runtime_gpu_specs_cached() -> Dict[str, Any]:
    if _runtime_discovery_disabled():
        return {"source": [], "gpus": [], "errors": ["runtime GPU discovery disabled"]}

    rocminfo_text = _run_command(["rocminfo"])
    rocm_smi_info = _run_command(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    rocm_smi_clocks = _run_command(["rocm-smi", "--showclocks", "--showclkfrq", "--json"])
    amd_smi_list = _run_command(["amd-smi", "list", "--json"])
    amd_smi_static = _run_command(["amd-smi", "static", "--json"])
    amd_smi_metric = _run_command(["amd-smi", "metric", "--json"])

    rocminfo_gpus = _parse_rocminfo(rocminfo_text or "")
    kfd_topology_gpus = _parse_kfd_topology()
    rocm_smi_gpus = _parse_rocm_smi_json(rocm_smi_info) + _parse_rocm_smi_json(rocm_smi_clocks)
    amd_smi_gpus = (
        _parse_amd_smi_list_json(amd_smi_list)
        + _parse_amd_smi_gpu_data_json(amd_smi_static, "amd-smi static")
        + _parse_amd_smi_gpu_data_json(amd_smi_metric, "amd-smi metric")
    )

    source_names = []
    source_names.extend(["rocminfo"] if rocminfo_gpus else [])
    source_names.extend(["kfd-topology"] if kfd_topology_gpus else [])
    source_names.extend(["rocm-smi"] if rocm_smi_gpus else [])
    source_names.extend(["amd-smi"] if amd_smi_gpus else [])

    gpus = _merge_gpu_lists(
        [
            rocminfo_gpus,
            kfd_topology_gpus,
            rocm_smi_gpus,
            amd_smi_gpus,
        ]
    )
    return {"source": source_names, "gpus": gpus, "errors": []}


@tool_class(ToolClass.READ_ONLY)
def discover_runtime_gpu_specs(gfx_id: str = "") -> Dict[str, Any]:
    """Return process-host GPU specs discovered from ROCm tools.

    Discovery is local to the host where this Python process runs. For SSH
    optimization workflows, collect GPU facts on the remote execution host or
    run PerfXpert on that host; controller-host runtime specs must not be used
    as remote-target SoL bounds.

    Args:
        gfx_id: Optional architecture id filter, for example ``gfx942``.

    Returns:
        A dict with ``source`` and ``gpus``. The ``gpus`` entries include
        ``spec_sources`` so callers can see which command supplied each field.
        Missing tools or unsupported fields produce an empty/fallback result
        rather than raising.
    """
    result = _discover_runtime_gpu_specs_cached()
    if not gfx_id:
        return result
    filtered = [gpu for gpu in result["gpus"] if gpu.get("gfx_id") == gfx_id]
    return {**result, "gpus": filtered}


def runtime_specs_for_gfx(gfx_id: str) -> Optional[Dict[str, Any]]:
    """Return the first runtime-discovered spec for ``gfx_id`` if present."""
    for gpu in _discover_runtime_gpu_specs_cached().get("gpus", []):
        if gpu.get("gfx_id") == gfx_id:
            return dict(gpu)
    return None


def first_runtime_gpu_specs() -> Optional[Dict[str, Any]]:
    """Return the first locally discovered GPU spec if any."""
    gpus = _discover_runtime_gpu_specs_cached().get("gpus", [])
    return dict(gpus[0]) if gpus else None


__all__ = [
    "PERFXPERT_DISABLE_RUNTIME_GPU_SPECS",
    "PERFXPERT_GPU_DISCOVERY_TIMEOUT",
    "discover_runtime_gpu_specs",
    "first_runtime_gpu_specs",
    "runtime_specs_for_gfx",
]
