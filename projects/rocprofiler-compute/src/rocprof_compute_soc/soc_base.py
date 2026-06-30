# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from __future__ import annotations

import argparse
import functools
import math
import os
import shutil
import sys
from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import config
from roofline.run_benchmark import run_roofline_benchmark
from utils import amdsmi_interface
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import MachineSpecs
from utils.utils_common import (
    METRIC_ID_RE,
    add_counter_extra_config_input_yaml,
    canonical_config_arch,
    convert_metric_id_to_panel_info,
    create_temp_rocprofiler_metrics_path,
    get_arch_alias_to_panel_id,
    is_only_pc_sampling,
    is_tcc_channel_counter,
    parse_sets_yaml,
    resolve_rocm_library_path,
    validate_roofline_csv,
)
from utils.utils_counter_defs import (
    counter_to_block,
    extract_counters_and_variables,
)
from vendored import yaml


def _same_bucket_priority_ids_from_policy_value(
    arch_name: str,
    ids: object,
) -> tuple[str, ...] | None:
    """
    Normalize ``same_bucket_priority_metric_ids`` from YAML: list of ids, or
    mapping id -> { name: ... } (id is the key; order preserved).
    Returns None if invalid (caller should warn).
    """
    if ids is None:
        return ()
    if isinstance(ids, list):
        return tuple(str(x) for x in ids)
    if isinstance(ids, dict):
        ordered: list[str] = []
        for key, meta in ids.items():
            token = str(key).strip()
            if not token:
                continue
            if meta is not None and not isinstance(meta, dict):
                console_warning(
                    "profiling",
                    (
                        "Ignoring same_bucket_priority_metric_ids["
                        f"{arch_name!r}][{token!r}]: "
                        "expected a mapping or null."
                    ),
                )
                continue
            ordered.append(token)
        return tuple(ordered)
    return None


@functools.lru_cache(maxsize=1)
def _load_same_bucket_priority_policy_map() -> dict[str, tuple[str, ...]]:
    """Load counter grouping policy YAML into arch -> metric id tuple."""
    path = (
        config.rocprof_compute_home
        / "rocprof_compute_soc"
        / "analysis_configs"
        / "profiling_counter_grouping_policy.yaml"
    )
    if not path.is_file():
        console_warning(
            "profiling",
            f"Profiling counter grouping policy missing ({path}); "
            "same-bucket priority metrics disabled.",
        )
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    archs = raw.get("architectures")
    if not isinstance(archs, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for arch_name, cfg in archs.items():
        if not isinstance(cfg, dict):
            continue
        key = str(arch_name)
        parsed = _same_bucket_priority_ids_from_policy_value(
            key,
            cfg.get("same_bucket_priority_metric_ids"),
        )
        if parsed is None:
            console_warning(
                "profiling",
                f"Ignoring same_bucket_priority_metric_ids for {key!r}: "
                "expected a list of ids or a mapping id -> metadata.",
            )
            continue
        out[key] = parsed
    return out


class OmniSoC_Base:
    def __init__(self, args: argparse.Namespace, mspec: MachineSpecs) -> None:
        # new info field will contain rocminfo or sysinfo to populate properties
        console_debug("[omnisoc init]")
        self.__args = args
        self.__arch: Optional[str] = None
        self._mspec = mspec
        # Per IP block, max number of simultaneous counters. GFX IP Blocks.
        self.__perfmon_config: dict[str, int] = {}
        self.__compatible_profilers: list[str] = []  # Store SoC compatible profilers
        self.populate_mspec()

    def __hash__(self) -> int:
        return hash(self.__arch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.__arch == other.get_soc()

    def set_perfmon_config(self, config: dict[str, int]) -> None:
        self.__perfmon_config = config

    def set_arch(self, arch: str) -> None:
        self.__arch = arch

    def set_compatible_profilers(self, profiler_names: list[str]) -> None:
        self.__compatible_profilers = profiler_names

    def get_arch(self) -> Optional[str]:
        return self.__arch

    def get_args(self) -> argparse.Namespace:
        return self.__args

    def get_compatible_profilers(self) -> list[str]:
        return self.__compatible_profilers

    def populate_mspec(self) -> None:
        from utils.specs import search, total_sqc

        if (
            not hasattr(self._mspec, "rocminfo_lines")
            or self._mspec.rocminfo_lines is None
        ):
            return

        # load stats from rocminfo
        self._mspec.gpu_l1 = ""
        self._mspec.gpu_l2 = ""

        for linetext in self._mspec.rocminfo_lines:
            key = search(r"^\s*L1:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.gpu_l1 = key
                continue

            key = search(r"^\s*L2:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.gpu_l2 = key
                continue

            key = search(r"^\s*Max Clock Freq\. \(MHz\):\s+([0-9]+)", linetext)
            if key is not None:
                self._mspec.max_sclk = key
                continue

            key = search(r"^\s*Compute Unit:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.cu_per_gpu = key
                continue

            key = search(r"^\s*SIMDs per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.simd_per_cu = key
                continue

            key = search(r"^\s*Shader Engines:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.se_per_gpu = key
                continue

            key = search(r"^\s*Wavefront Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.wave_size = key
                continue

            key = search(r"^\s*Workgroup Max Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.workgroup_max_size = key
                continue

            key = search(r"^\s*Max Waves Per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key is not None:
                self._mspec.max_waves_per_cu = key
                break

        if self._mspec.gpu_arch and self._mspec.cu_per_gpu and self._mspec.se_per_gpu:
            self._mspec.sqc_per_gpu = str(
                total_sqc(
                    self._mspec.gpu_arch, self._mspec.cu_per_gpu, self._mspec.se_per_gpu
                )
            )

        with amdsmi_interface.amdsmi_ctx():
            self._mspec.max_mclk = str(amdsmi_interface.get_mem_max_clock())

        # These are just max values now, because the parsing was broken and this was
        # inconsistent with how we use the clocks elsewhere (all max, all the time)
        self._mspec.cur_sclk = self._mspec.max_sclk
        self._mspec.cur_mclk = self._mspec.max_mclk

        if self._mspec.gpu_arch:
            self._mspec.gpu_series = mi_gpu_specs.get_gpu_series(self._mspec.gpu_arch)
        # specify gpu model name for gfx942 hardware
        self._mspec.gpu_model = mi_gpu_specs.get_gpu_model(
            self._mspec.gpu_arch, self._mspec.gpu_chip_id
        )

        if not self._mspec.gpu_model:
            self._mspec.gpu_model = self.detect_gpu_model(self._mspec.gpu_arch)

        self._mspec.num_xcd = str(
            mi_gpu_specs.get_num_xcds(
                self._mspec.gpu_arch,
                self._mspec.gpu_model,
                self._mspec.compute_partition,
            )
        )

    @demarcate
    def detect_gpu_model(self, gpu_arch: str) -> Optional[str]:
        """
        Detects the GPU model using various identifiers from 'amd-smi static'.
        Falls back through multiple methods if the primary method fails.
        """
        with amdsmi_interface.amdsmi_ctx():
            gpu_model = "N/A"
            for model in mi_gpu_specs.get_all_gpu_models():
                for amdsmi_gpu_model in amdsmi_interface.get_gpu_model():
                    if model.lower() in amdsmi_gpu_model.lower():
                        gpu_model = model
                        break

        gpu_model = self._adjust_mi300_model(gpu_model.lower(), gpu_arch.lower())

        if gpu_model.lower() not in mi_gpu_specs.get_num_xcds_dict().keys():
            console_warning(f'Unknown GPU model detected: "{gpu_model}".')
            return

        return gpu_model.upper()

    def _adjust_mi300_model(self, gpu_model: str, gpu_arch: str) -> str:
        """
        Applies specific adjustments for MI300 series GPU models based on architecture.
        """

        if gpu_model in ["mi300a", "mi300x"]:
            if gpu_arch in ["gfx940", "gfx941"]:
                gpu_model += "_a0"
            elif gpu_arch == "gfx942":
                gpu_model += "_a1"

        return gpu_model

    def _append_analysis_yaml_for_filter_token(
        self,
        raw_token: str,
        config_filename_dict: dict[str, str],
        config_root_dir: str,
        texts: list[str],
    ) -> None:
        block_id = raw_token
        if METRIC_ID_RE.match(block_id):
            pass
        else:
            alias = block_id
            panel_alias_dict = get_arch_alias_to_panel_id(self._mspec.gpu_arch)
            if alias not in panel_alias_dict:
                raise KeyError(f"Unknown panel alias: {alias!r}")
            block_id = str(panel_alias_dict[alias])
            console_log(f"alias: {alias}, block id: {block_id}")

        file_id, panel_id, metric_id = convert_metric_id_to_panel_info(block_id)

        if file_id not in config_filename_dict:
            console_warning(
                f"Skipping {block_id}: file id {file_id} not found in {config_root_dir}"
            )
            return

        with open(config_filename_dict[file_id], encoding="utf-8") as stream:
            file_config = yaml.safe_load(stream)
        if panel_id is None:
            texts.append(yaml.dump(file_config, sort_keys=False))
            return

        panel_dict = {
            section["metric_table"]["id"]: section["metric_table"]
            for section in file_config["Panel Config"]["data source"]
            if "metric_table" in section
        }
        if panel_id not in panel_dict:
            console_warning(
                f"Skipping {block_id}: metric table {panel_id} not found in "
                f"{config_filename_dict[file_id]}"
            )
            return
        if metric_id is None:
            texts.append(yaml.dump(panel_dict[panel_id], sort_keys=False))
            return

        metric_dict = {
            idx: panel_dict[panel_id]["metric"][metric]
            for idx, metric in enumerate(panel_dict[panel_id]["metric"].keys())
        }
        if metric_id not in metric_dict:
            console_warning(
                f"Skipping {block_id}: metric id {metric_id} not found in "
                f"panel id {panel_id}"
            )
            return
        texts.append(yaml.dump(metric_dict[metric_id], sort_keys=False))

    def _same_bucket_priority_metric_ids(self) -> tuple[str, ...]:
        """Metric ids whose PMCs get tier-0 priority in the greedy coalescing pass.

        Loaded from profiling_counter_grouping_policy.yaml for the current arch.
        Returns an empty tuple when the arch has no grouping policy.
        """
        arch = self.__arch
        if not arch:
            return ()
        return _load_same_bucket_priority_policy_map().get(arch, ())

    def _metric_aware_coalesce_pass(
        self,
        work_set: set[str],
        output_files: list[CounterFile],
        file_count: int,
    ) -> tuple[set[str], list[CounterFile], int]:
        """Greedy heuristic: place each metric's counters into the first feasible
        pmc_perf bucket, else open a new one. Overflow stays for first-fit.

        Accepts:
            work_set        — counters still to be placed (not modified)
            output_files    — existing CounterFile buckets (not modified)
            file_count      — current bucket sequence number
        Returns:
            (remaining_counters, updated_files, file_count)
        """
        if not work_set:
            return work_set, list(output_files), file_count

        # Work on copies so the caller’s originals are untouched.
        remaining = set(work_set)
        files = list(output_files)

        # -- Build priority keys from grouping policy YAML --
        priority_keys: set[tuple[str, Any, int]] = set()
        for token in self._same_bucket_priority_metric_ids():
            tid = token.strip()
            if not METRIC_ID_RE.match(tid):
                continue
            file_id, panel_id, metric_idx = convert_metric_id_to_panel_info(tid)
            if metric_idx is None:
                continue
            priority_keys.add((file_id, panel_id, metric_idx))

        # -- Scan arch YAML metrics and collect (sort_key, counters, label) rows --
        rows: list[tuple[tuple, frozenset[str], str]] = []
        for (
            stem_id,
            panel_id,
            metric_idx,
            metric_name,
            metric_yaml,
        ) in self._iter_arch_analysis_yaml_metrics():
            hw, _ = extract_counters_and_variables(metric_yaml, self._mspec.gpu_series)
            hw = self._expand_tcc_template_counters(hw)
            counters = frozenset(hw & remaining)
            if not counters:
                continue
            tier = 0 if (stem_id, panel_id, metric_idx) in priority_keys else 1
            panel_s = str(panel_id) if panel_id is not None else ""
            sort_key = (tier, -len(counters), stem_id, panel_s, metric_idx)
            label = f"{stem_id}.{panel_s}.{metric_idx} ({metric_name})"
            rows.append((sort_key, counters, label))
        rows.sort(key=lambda r: r[0])

        # -- Place each metric group into an existing or new bucket --
        cfg = self.__perfmon_config
        for _sort_key, group, label in rows:
            # Re-filter: remaining shrinks as earlier groups are placed.
            need_sorted = sorted(c for c in group if c in remaining)
            if not need_sorted:
                continue
            placed = False
            for bucket_idx, bucket in enumerate(files):
                if bucket.name.endswith("_ACCUM"):
                    continue
                trial = _trial_counter_file_with_extra(bucket, cfg, need_sorted)
                if trial is not None:
                    files[bucket_idx] = trial
                    remaining -= set(need_sorted)
                    placed = True
                    break
            if placed:
                continue
            new_bucket = CounterFile(str(file_count), cfg)
            trial = _trial_counter_file_with_extra(new_bucket, cfg, need_sorted)
            if trial is not None and flat_counters_in_perfmon_file(trial):
                files.append(trial)
                file_count += 1
                remaining -= set(need_sorted)
            else:
                console_debug(
                    "profiling",
                    f"Metric-aware pack: cannot fit all PMCs for "
                    f"{label!r} in one bucket; deferring to first-fit.",
                )
        return remaining, files, file_count

    @demarcate
    def detect_counters(self) -> tuple[set[str], list[str]]:
        """
        Create a set of counters required for the selected report sections.
        Parse analysis report configuration files based on the selected report
        sections to be filtered.
        """
        args = self.get_args()

        # File id dict
        config_arch = canonical_config_arch(self.__arch) or self.__arch
        config_root_dir = f"{args.config_dir}/{config_arch}"
        config_filename_dict = {
            filename.name.split("_")[0]: str(filename)
            for filename in Path(config_root_dir).glob("*.yaml")
        }

        filter_blocks = args.filter_blocks
        if args.set_selected and self.__arch:
            sets_info = parse_sets_yaml(self.__arch)
            if args.set_selected not in set(sets_info.keys()):
                console_error(
                    f'argument --set: invalid choice: "{args.set_selected}" '
                    f"(choose from {sets_info.keys()})"
                )
            filter_blocks = [
                next(iter(metric.keys()))
                for metric in sets_info[args.set_selected]["metric"]
            ]
        elif args.roof_only:
            filter_blocks = ["4"]

        texts: list[str] = []
        if not filter_blocks:
            # Do not profile block 30 unless explicitly requested
            exclude_file_ids: set[str] = set()
            if not args.membw_analysis:
                exclude_file_ids.add("3000")

            # Select all sections by default
            for file_id, filename in config_filename_dict.items():
                if file_id in exclude_file_ids:
                    continue

                with open(filename, encoding="utf-8") as stream:
                    texts.append(stream.read())

        for block_id in filter_blocks:
            self._append_analysis_yaml_for_filter_token(
                block_id, config_filename_dict, config_root_dir, texts
            )

        counters, _ = extract_counters_and_variables(
            "\n".join(texts), self._mspec.gpu_series
        )
        counters = self._expand_tcc_template_counters(counters)

        return counters, filter_blocks

    def _expand_tcc_template_counters(self, counters: set[str]) -> set[str]:
        """
        Expand TCC channel templates (name ending with '[') into indexed channel
        counters, matching perfmon allocation.
        """
        out = set(counters)
        num_xcd = int(self._mspec.num_xcd)
        l2_banks = int(self._mspec.l2_banks)
        for counter_name in counters.copy():
            if counter_name.startswith("TCC") and counter_name.endswith("["):
                out.discard(counter_name)
                base = counter_name.split("[")[0]
                out.update(f"{base}[{i}]" for i in range(num_xcd * l2_banks))
        return out

    @demarcate
    def perfmon_filter(self) -> list[str]:
        """Filter default performance counter set based on user arguments"""
        counters, filter_blocks = self.detect_counters()

        if is_only_pc_sampling(filter_blocks):
            console_log(
                "profiling",
                "PC sampling only mode -- skipping counter collection setup",
            )
            return filter_blocks

        # Coalesce and writeback workload specific perfmon
        self.perfmon_coalesce(counters)

        return filter_blocks

    def _allocate_perfmon_counter_files(
        self, counters: set[str]
    ) -> tuple[list[CounterFile], int, int]:
        """Bin-pack counters into perfmon buckets.

        Returns (output_files, file_count, accu_file_count).

        Accumulator counters (ending with _ACCUM) get dedicated files first.
        If the arch has priority metrics in profiling_counter_grouping_policy.yaml,
        a metric-aware greedy pass runs before the final per-counter first-fit.
        """
        output_files: list[CounterFile] = []
        accu_file_count = 0
        work = sorted(list(counters))
        for counter in work.copy():
            if counter.endswith("_ACCUM") and not is_tcc_channel_counter(counter):
                work.remove(counter)
                output_files.append(CounterFile(counter, self.__perfmon_config))
                output_files[-1].add(counter)
                # Paired level-event slot: hardware programs the level counter
                # alongside its accumulator, so hold one extra slot in the
                # same block.
                output_files[-1].reserve(counter, 1)
                accu_file_count += 1

        file_count = 0
        tcc_channel_counter_file_map: dict[str, CounterFile] = {}

        work_set = set(work)
        if self._same_bucket_priority_metric_ids():
            work_set, output_files, file_count = self._metric_aware_coalesce_pass(
                work_set, output_files, file_count
            )
        work = sorted(work_set)
        tcc_channel_counter_file_map = _rebuild_tcc_channel_file_map(output_files)

        for ctr in work:
            if is_tcc_channel_counter(ctr):
                output_file = tcc_channel_counter_file_map.get(ctr.split("[")[0])
                if output_file:
                    output_file.add(ctr)
                    continue

            added = False
            for output_file in output_files:
                if output_file.add(ctr):
                    added = True
                    if is_tcc_channel_counter(ctr):
                        tcc_channel_counter_file_map[ctr.split("[")[0]] = output_file
                    break

            if not added:
                output_files.append(CounterFile(str(file_count), self.__perfmon_config))
                file_count += 1
                output_files[-1].add(ctr)

        return output_files, file_count, accu_file_count

    def _iter_arch_analysis_yaml_metrics(
        self,
    ) -> Iterator[tuple[str, Any, int, str, str]]:
        """Iterate analysis_configs/<arch> YAML metric_table rows.

        Yields:
            stem_id     — YAML filename prefix (e.g. "2" from "2_SQ.yaml")
            panel_id    — metric_table "id" field (may be None)
            metric_idx  — zero-based index of the metric within its table
            metric_name — metric key string
            metric_yaml — metric body serialised as YAML text
        """
        args = self.get_args()
        arch = self.__arch
        if not arch:
            return
        config_root = Path(args.config_dir) / (canonical_config_arch(arch) or arch)
        if not config_root.is_dir():
            return
        exclude: set[str] = set()
        if not args.membw_analysis:
            exclude.add("3000")

        for ypath in sorted(config_root.glob("*.yaml")):
            stem_id = ypath.name.split("_")[0]
            if stem_id in exclude:
                continue
            try:
                with open(ypath, encoding="utf-8") as stream:
                    doc = yaml.safe_load(stream)
            except (OSError, UnicodeError, yaml.YAMLError):
                continue
            if not isinstance(doc, dict):
                continue
            panel_cfg = doc.get("Panel Config")
            if not isinstance(panel_cfg, dict):
                continue
            sources = panel_cfg.get("data source")
            if not isinstance(sources, list):
                continue
            for section in sources:
                if not isinstance(section, dict) or "metric_table" not in section:
                    continue
                mt = section["metric_table"]
                if not isinstance(mt, dict):
                    continue
                metrics = mt.get("metric")
                if not isinstance(metrics, dict):
                    continue
                panel_id = mt.get("id")
                for idx, (metric_name, metric_body) in enumerate(metrics.items()):
                    try:
                        metric_text = yaml.dump(
                            metric_body, sort_keys=False, allow_unicode=True
                        )
                    except (TypeError, yaml.YAMLError):
                        continue
                    yield stem_id, panel_id, idx, metric_name, metric_text

    def get_rocprof_supported_counters(self) -> set[str]:
        args = self.get_args()
        rocprof_counters: set[str] = set()

        # Point to counter definition
        old_rocprofiler_metrics_path = os.environ.get("ROCPROFILER_METRICS_PATH")
        with open(
            config.rocprof_compute_home
            / "rocprof_compute_soc"
            / "profile_configs"
            / "sdk_config.yaml",
            encoding="utf-8",
        ) as filename:
            sdk_config = yaml.safe_load(filename)
        os.environ["ROCPROFILER_METRICS_PATH"] = create_temp_rocprofiler_metrics_path(
            sdk_config
        )

        # Backward compatibility support for sdk avail module moved from
        # <rocm_path>/bin/rocprofv3_avail_module/avail.py to
        # <rocm_path>/lib/python3/site-packages/rocprofv3/avail.py
        new_path = str(
            Path(args.rocprofiler_sdk_tool_path).parents[1] / "python3/site-packages"
        )
        old_path = str(Path(args.rocprofiler_sdk_tool_path).parents[2] / "bin")
        try:
            sys.path.append(new_path)
            from rocprofv3 import avail
        except ImportError:
            console_debug(
                f"Could not import rocprofiler-sdk avail module from {new_path}, "
                f"trying {old_path}"
            )
            try:
                sys.path.remove(new_path)
                sys.path.append(old_path)
                from rocprofv3_avail_module import avail
            except ImportError:
                console_error("Failed to import rocprofiler-sdk avail module.")

        # librocprofv3-list-avail.so location varies by ROCm version:
        #   ROCm >= 7.1: <rocm_path>/lib/rocprofiler-sdk/
        #   ROCm 7.0.x:  <rocm_path>/libexec/rocprofiler-sdk/
        avail_lib_name = "librocprofv3-list-avail.so"
        avail_lib_path = resolve_rocm_library_path(
            str(Path(args.rocprofiler_sdk_tool_path).parent / avail_lib_name)
        )
        if not Path(avail_lib_path).exists():
            avail_lib_path = resolve_rocm_library_path(
                str(
                    Path(args.rocprofiler_sdk_tool_path).parents[2]
                    / "libexec"
                    / "rocprofiler-sdk"
                    / avail_lib_name
                )
            )
        avail.loadLibrary.libname = avail_lib_path
        counters = avail.get_counters()
        rocprof_counters = {
            counter.name
            for counter in counters[list(counters.keys())[0]]
            if hasattr(counter, "block") or hasattr(counter, "expression")
        }
        # Delete counter definition temporary directory
        if os.environ.get("ROCPROFILER_METRICS_PATH"):
            shutil.rmtree(os.environ["ROCPROFILER_METRICS_PATH"], ignore_errors=True)
        # Reset env. var.
        if old_rocprofiler_metrics_path is None:
            del os.environ["ROCPROFILER_METRICS_PATH"]
        else:
            os.environ["ROCPROFILER_METRICS_PATH"] = old_rocprofiler_metrics_path

        return rocprof_counters

    @demarcate
    def perfmon_coalesce(self, counters: set[str]) -> None:
        """
        Sort and bucket all related performance counters to minimize required
        application passes
        """
        workload_perfmon_dir = Path(self.get_args().output_directory) / "perfmon"
        workload_perfmon_dir.mkdir(parents=True, exist_ok=True)

        rocprof_counters = self.get_rocprof_supported_counters()
        # rocprof does not support TCC channel counters in the avail output,
        # so remove channel suffix for comparison
        not_supported_counters = {
            counter.split("[")[0] if is_tcc_channel_counter(counter) else counter
            for counter in counters
        } - rocprof_counters

        if not_supported_counters:
            console_warning(
                "Following counters might not be supported by rocprof: "
                f"{', '.join(not_supported_counters)}"
            )

        # We might be providing definitions of unsupported counters, so still try to
        # collect them
        if not counters:
            console_error(
                "profiling",
                "No performance counters to collect, "
                "please check the provided profiling filters",
            )

        console_debug(f"Collecting following counters: {', '.join(counters)} ")

        output_files, file_count, accu_file_count = (
            self._allocate_perfmon_counter_files(counters)
        )

        console_debug("profiling", f"perfmon_coalesce file_count {file_count}")

        # TODO: rewrite the above logic for spatial_multiplexing later
        if self.get_args().spatial_multiplexing:
            # TODO: more error checking
            if len(self.get_args().spatial_multiplexing) != 3:
                console_error(
                    "profiling",
                    "multiplexing need provide node_idx node_count and gpu_count",
                )

            node_idx, node_count, gpu_count = map(
                int, self.get_args().spatial_multiplexing
            )

            old_group_num = file_count + accu_file_count
            new_bucket_count = node_count * gpu_count
            groups_per_bucket = math.ceil(
                old_group_num / new_bucket_count
            )  # It equals to file num per node
            max_groups_per_node = groups_per_bucket * gpu_count

            group_start = node_idx * max_groups_per_node
            group_end = min((node_idx + 1) * max_groups_per_node, old_group_num)

            console_debug(
                "profiling",
                f"spatial_multiplexing node_idx {node_idx}, node_count {node_count}, "
                f"gpu_count: {gpu_count},\n"
                f"old_group_num {old_group_num}, new_bucket_count {new_bucket_count}, "
                f"groups_per_bucket {groups_per_bucket},\n"
                f"max_groups_per_node {max_groups_per_node}, "
                f"group_start {group_start}, group_end {group_end}",
            )

            for f_idx in range(groups_per_bucket):
                file_name = (
                    Path(workload_perfmon_dir)
                    / f"pmc_perf_node_{node_idx}_{f_idx}.yaml"
                )

                pmc = []
                for g_idx in range(
                    group_start + f_idx * gpu_count,
                    min(group_end, group_start + (f_idx + 1) * gpu_count),
                ):
                    gpu_idx = g_idx % gpu_count
                    for block_name in output_files[g_idx].blocks.keys():
                        for ctr in output_files[g_idx].blocks[block_name].elements:
                            pmc.append(f"{ctr}:device={gpu_idx}")

                # Write counters to file
                with open(file_name, "w", encoding="utf-8") as fd:
                    fd.write(yaml.dump({"jobs": [{"pmc": pmc}]}, sort_keys=False))
        else:
            # Output to files
            for f in output_files:
                pmc_filename = workload_perfmon_dir / f"pmc_perf_{f.name}.yaml"
                counter_def_filename = (
                    workload_perfmon_dir / f"counter_def_{f.name}.yaml"
                )

                pmc = []
                counter_def: dict[str, Any] = {}

                for ctr in [
                    ctr
                    for block_name in f.blocks
                    for ctr in f.blocks[block_name].elements
                ]:
                    pmc.append(ctr)
                    # Add TCC channel counters definitions
                    if is_tcc_channel_counter(ctr):
                        counter_name = ctr.split("[")[0]
                        idx = int(ctr.split("[")[1].split("]")[0])
                        xcd_idx = idx // int(self._mspec.l2_banks)
                        channel_idx = idx % int(self._mspec.l2_banks)
                        expression = (
                            f"select({counter_name},"
                            f"[DIMENSION_XCC=[{xcd_idx}], "
                            f"DIMENSION_INSTANCE=[{channel_idx}]])"
                        )
                        description = (
                            f"{counter_name} on {xcd_idx}th XCC and "
                            f"{channel_idx}th channel"
                        )
                        counter_def = add_counter_extra_config_input_yaml(
                            counter_def,
                            ctr,
                            description,
                            expression,
                            [self.__arch],
                        )

                # Write counters to file
                with open(pmc_filename, "w", encoding="utf-8") as fd:
                    fd.write(yaml.dump({"jobs": [{"pmc": pmc}]}, sort_keys=False))

                # Write counter definitions to file
                if counter_def:
                    with open(counter_def_filename, "w", encoding="utf-8") as fp:
                        fp.write(yaml.dump(counter_def, sort_keys=False))

    # ----------------------------------------------------
    # Required methods to be implemented by child classes
    # ----------------------------------------------------
    @abstractmethod
    def profiling_setup(self) -> Optional[list[str]]:
        """Perform any SoC-specific setup prior to profiling."""
        console_debug("profiling", f"perform SoC profiling setup for {self.__arch}")

    @abstractmethod
    def post_profiling(self) -> None:
        """Perform any SoC-specific post profiling activities."""
        console_debug("profiling", f"perform SoC post processing for {self.__arch}")
        # Roofline can be skipped via --no-roof
        # Roofline not supported on MI 100
        # Roofline not supported on Strix Halo
        # If --filter-blocks is provided, roofline block (block 4) should be mentioned
        if (
            self.get_args().no_roof
            or self.__arch == "gfx908"
            or self.__arch == "gfx1151"
            or (
                self.get_args().filter_blocks
                and "4" not in self.get_args().filter_blocks
                and "roof" not in self.get_args().filter_blocks
            )
        ):
            console_log("roofline", "Skipping roofline")
        else:
            roofline_csv = Path(self.get_args().output_directory) / "roofline.csv"
            console_log(
                "roofline",
                f"Checking for roofline.csv in {self.get_args().output_directory}",
            )
            if not roofline_csv.is_file():
                try:
                    run_roofline_benchmark(
                        self.get_args().device, roofline_csv, self._mspec.cache_sizes
                    )
                except Exception as e:
                    console_error(
                        "roofline",
                        f"Benchmark execution failed: {e}. Skipping roofline.",
                        exit=False,
                    )
                    return

            is_valid, error_msg = validate_roofline_csv(
                self.get_args().output_directory
            )
            if not is_valid:
                console_error(
                    "roofline",
                    f"Invalid roofline.csv: {error_msg}",
                    exit=False,
                )
                return

            console_log(
                "roofline",
                "Roofline data saved to "
                f"{self.get_args().output_directory}/roofline.csv\n"
                "  Run 'rocprof-compute analyze -p "
                f"{self.get_args().output_directory}' "
                "for charts",
            )


# Set with limited size
class LimitedSet:
    def __init__(self, maxsize: int) -> None:
        self.avail: int = maxsize
        self.elements: list[str] = []

    def add(self, element: str) -> bool:
        if element in self.elements:
            return True
        # Store all channels for a TCC channel counter in the same file
        if element.split("[")[0] in {elem.split("[")[0] for elem in self.elements}:
            self.elements.append(element)
            return True
        if self.avail > 0:
            self.avail -= 1
            self.elements.append(element)
            return True
        return False

    def reserve(self, n: int) -> bool:
        if self.avail < n:
            return False
        self.avail -= n
        return True


# Represents a file that lists PMC counters. Number of counters for each
# block limited according to perfmon config.
class CounterFile:
    def __init__(self, name: str, perfmon_config: dict[str, int]) -> None:
        self.name: str = name
        self.blocks: dict[str, LimitedSet] = {
            block: LimitedSet(capacity) for block, capacity in perfmon_config.items()
        }

    def add(self, counter: str) -> bool:
        return self.blocks[counter_to_block(counter)].add(counter)

    def reserve(self, counter: str, n: int) -> bool:
        return self.blocks[counter_to_block(counter)].reserve(n)


def _trial_counter_file_with_extra(
    basis: CounterFile,
    perfmon_config: dict[str, int],
    extra_counters_sorted: list[str],
) -> CounterFile | None:
    """Clone basis, try appending extras; None if any won't fit."""
    trial = CounterFile(basis.name, perfmon_config)
    for ctr in flat_counters_in_perfmon_file(basis):
        if not trial.add(ctr):
            msg = f"clone replay failed for {ctr!r} in bucket {basis.name!r}"
            raise RuntimeError(msg)
    for ctr in extra_counters_sorted:
        if not trial.add(ctr):
            return None
    return trial


def _rebuild_tcc_channel_file_map(
    output_files: list[CounterFile],
) -> dict[str, CounterFile]:
    """Map TCC counter base name to the bucket that holds its channel instances."""
    result: dict[str, CounterFile] = {}
    for bucket in output_files:
        for ctr in flat_counters_in_perfmon_file(bucket):
            if is_tcc_channel_counter(ctr):
                result[ctr.split("[")[0]] = bucket
    return result


def flat_counters_in_perfmon_file(counter_file: CounterFile) -> list[str]:
    """Ordered list of PMC counter names assigned to one perfmon bucket file."""
    return [
        ctr
        for block_name in counter_file.blocks
        for ctr in counter_file.blocks[block_name].elements
    ]
