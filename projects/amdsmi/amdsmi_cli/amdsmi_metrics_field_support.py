# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which ``amd-smi metric`` fields the detected gpu_metrics version can populate.

Backs the default field filtering of human-readable ``amd-smi metric`` output,
which ``--show-unsupported`` turns off. ``--json`` and ``--csv`` are consumed by
scripts and never filter, so ``metric.py`` does not call this module for them.

The field set carried by ``gpu_metrics`` is determined by its header version and
is not cumulative: v1.4 drops 18 of the fields v1.3 carried, v1.6 drops
jpeg_activity/vcn_activity/throttle_status. Each version is therefore an explicit
set, mirroring the per-version ``copy_internal_to_external_metrics()`` in
``rocm_smi_gpu_metrics.cc``.

Three rules keep the feature from ever losing data:

* an unmapped version yields ``None``, and callers then suppress nothing, so a
  future metrics version degrades to today's output;
* :func:`filter_unsupported` drops a path only when every leaf under it already
  reads "N/A", so a mis-mapped path can under-report but can never hide a real
  value;
* a section the user named on the command line is passed as protected and is
  left whole rather than emptied, so an explicit ``--throttle`` never answers
  with silence.

The all-N/A gate is what makes the partition blob safe to approximate. It is the
second eligible source and is versioned independently, by
``amdgpu_partition_metric_version_translation_table`` in the same C++ file, which
this module does not read: every partition-backed path (``xcp_stats.*``,
``temperature_mid``, ``current_socclks_mid``) is judged by the GPU metrics
version instead. A wrong approximation can then only change whether an already
empty column is printed, which on a mixed-version fleet shows up as a key-set
difference between GPUs rather than as a lost value.
"""

from typing import Any, Dict, FrozenSet, Optional, Tuple

NA = "N/A"

# (format_revision, content_revision) as reported by the gpu_metrics header.
VersionKey = Tuple[int, int]
# A (section, key) coordinate in the ``metric`` command's output dict.
CliPath = Tuple[str, str]
# Public ``amdsmi_gpu_metrics_t`` member names.
FieldSet = FrozenSet[str]


def _fields(names: str) -> FieldSet:
    """Field sets are written as whitespace-separated text to keep the table readable."""
    return frozenset(names.split())


# Public ``amdsmi_gpu_metrics_t`` members each header version can populate.
_VERSION_FIELDS: Dict[VersionKey, FieldSet] = {
    (1, 0): _fields(
        """
        average_dclk0_frequency average_dclk1_frequency average_gfx_activity
        average_gfxclk_frequency average_mm_activity average_socclk_frequency average_socket_power
        average_uclk_frequency average_umc_activity average_vclk0_frequency average_vclk1_frequency
        current_dclk0 current_dclk1 current_fan_speed current_gfxclk current_socclk current_uclk
        current_vclk0 current_vclk1 energy_accumulator pcie_link_speed pcie_link_width
        system_clock_counter temperature_edge temperature_hotspot temperature_mem temperature_vrgfx
        temperature_vrmem temperature_vrsoc throttle_status
        """
    ),
    (1, 1): _fields(
        """
        average_dclk0_frequency average_dclk1_frequency average_gfx_activity
        average_gfxclk_frequency average_mm_activity average_socclk_frequency average_socket_power
        average_uclk_frequency average_umc_activity average_vclk0_frequency average_vclk1_frequency
        current_dclk0 current_dclk1 current_fan_speed current_gfxclk current_socclk current_uclk
        current_vclk0 current_vclk1 energy_accumulator gfx_activity_acc mem_activity_acc
        pcie_link_speed pcie_link_width system_clock_counter temperature_edge temperature_hbm
        temperature_hotspot temperature_mem temperature_vrgfx temperature_vrmem temperature_vrsoc
        throttle_status
        """
    ),
    (1, 2): _fields(
        """
        average_dclk0_frequency average_dclk1_frequency average_gfx_activity
        average_gfxclk_frequency average_mm_activity average_socclk_frequency average_socket_power
        average_uclk_frequency average_umc_activity average_vclk0_frequency average_vclk1_frequency
        current_dclk0 current_dclk1 current_fan_speed current_gfxclk current_socclk current_uclk
        current_vclk0 current_vclk1 energy_accumulator firmware_timestamp gfx_activity_acc
        mem_activity_acc pcie_link_speed pcie_link_width system_clock_counter temperature_edge
        temperature_hbm temperature_hotspot temperature_mem temperature_vrgfx temperature_vrmem
        temperature_vrsoc throttle_status
        """
    ),
    (1, 3): _fields(
        """
        average_dclk0_frequency average_dclk1_frequency average_gfx_activity
        average_gfxclk_frequency average_mm_activity average_socclk_frequency average_socket_power
        average_uclk_frequency average_umc_activity average_vclk0_frequency average_vclk1_frequency
        current_dclk0 current_dclk0s current_dclk1 current_fan_speed current_gfxclk current_gfxclks
        current_socclk current_socclks current_uclk current_vclk0 current_vclk0s current_vclk1
        energy_accumulator firmware_timestamp gfx_activity_acc indep_throttle_status
        mem_activity_acc pcie_link_speed pcie_link_width system_clock_counter temperature_edge
        temperature_hbm temperature_hotspot temperature_mem temperature_vrgfx temperature_vrmem
        temperature_vrsoc throttle_status vcn_activity voltage_gfx voltage_mem voltage_soc
        """
    ),
    (1, 4): _fields(
        """
        average_gfx_activity average_umc_activity current_dclk0 current_dclk0s current_dclk1
        current_gfxclk current_gfxclks current_socclk current_socclks current_socket_power
        current_uclk current_vclk0 current_vclk0s current_vclk1 energy_accumulator
        firmware_timestamp gfx_activity_acc gfxclk_lock_status mem_activity_acc pcie_bandwidth_acc
        pcie_bandwidth_inst pcie_l0_to_recov_count_acc pcie_link_speed pcie_link_width
        pcie_replay_count_acc pcie_replay_rover_count_acc system_clock_counter temperature_hotspot
        temperature_mem temperature_vrsoc throttle_status vcn_activity xgmi_link_speed
        xgmi_link_width xgmi_read_data_acc xgmi_write_data_acc
        """
    ),
    (1, 5): _fields(
        """
        average_gfx_activity average_umc_activity current_dclk0 current_dclk0s current_dclk1
        current_gfxclk current_gfxclks current_socclk current_socclks current_socket_power
        current_uclk current_vclk0 current_vclk0s current_vclk1 energy_accumulator
        firmware_timestamp gfx_activity_acc gfxclk_lock_status jpeg_activity mem_activity_acc
        pcie_bandwidth_acc pcie_bandwidth_inst pcie_l0_to_recov_count_acc pcie_link_speed
        pcie_link_width pcie_nak_rcvd_count_acc pcie_nak_sent_count_acc pcie_replay_count_acc
        pcie_replay_rover_count_acc system_clock_counter temperature_hotspot temperature_mem
        temperature_vrsoc throttle_status vcn_activity xgmi_link_speed xgmi_link_width
        xgmi_read_data_acc xgmi_write_data_acc
        """
    ),
    (1, 6): _fields(
        """
        accumulation_counter average_gfx_activity average_umc_activity current_dclk0 current_dclk0s
        current_dclk1 current_gfxclk current_gfxclks current_socclk current_socclks
        current_socket_power current_uclk current_vclk0 current_vclk0s current_vclk1
        energy_accumulator firmware_timestamp gfx_activity_acc gfxclk_lock_status
        hbm_thm_residency_acc mem_activity_acc num_partition pcie_bandwidth_acc pcie_bandwidth_inst
        pcie_l0_to_recov_count_acc pcie_lc_perf_other_end_recovery pcie_link_speed pcie_link_width
        pcie_nak_rcvd_count_acc pcie_nak_sent_count_acc pcie_replay_count_acc
        pcie_replay_rover_count_acc ppt_residency_acc prochot_residency_acc socket_thm_residency_acc
        system_clock_counter temperature_hotspot temperature_mem temperature_vrsoc
        vr_thm_residency_acc xcp_stats.gfx_busy_acc xcp_stats.gfx_busy_inst xcp_stats.jpeg_busy
        xcp_stats.vcn_busy xgmi_link_speed xgmi_link_width xgmi_read_data_acc xgmi_write_data_acc
        """
    ),
    (1, 7): _fields(
        """
        accumulation_counter average_gfx_activity average_umc_activity current_dclk0 current_dclk0s
        current_dclk1 current_gfxclk current_gfxclks current_socclk current_socclks
        current_socket_power current_uclk current_vclk0 current_vclk0s current_vclk1
        energy_accumulator firmware_timestamp gfx_activity_acc gfxclk_lock_status
        hbm_thm_residency_acc mem_activity_acc num_partition pcie_bandwidth_acc pcie_bandwidth_inst
        pcie_l0_to_recov_count_acc pcie_lc_perf_other_end_recovery pcie_link_speed pcie_link_width
        pcie_nak_rcvd_count_acc pcie_nak_sent_count_acc pcie_replay_count_acc
        pcie_replay_rover_count_acc ppt_residency_acc prochot_residency_acc socket_thm_residency_acc
        system_clock_counter temperature_hotspot temperature_mem temperature_vrsoc
        vr_thm_residency_acc vram_max_bandwidth xcp_stats.gfx_below_host_limit_acc
        xcp_stats.gfx_busy_acc xcp_stats.gfx_busy_inst xcp_stats.jpeg_busy xcp_stats.vcn_busy
        xgmi_link_speed xgmi_link_status xgmi_link_width xgmi_read_data_acc xgmi_write_data_acc
        """
    ),
    (1, 8): _fields(
        """
        accumulation_counter average_gfx_activity average_umc_activity current_dclk0 current_dclk0s
        current_dclk1 current_gfxclk current_gfxclks current_socclk current_socclks
        current_socket_power current_uclk current_vclk0 current_vclk0s current_vclk1
        energy_accumulator firmware_timestamp gfx_activity_acc gfxclk_lock_status
        hbm_thm_residency_acc mem_activity_acc num_partition pcie_bandwidth_acc pcie_bandwidth_inst
        pcie_l0_to_recov_count_acc pcie_lc_perf_other_end_recovery pcie_link_speed pcie_link_width
        pcie_nak_rcvd_count_acc pcie_nak_sent_count_acc pcie_replay_count_acc
        pcie_replay_rover_count_acc ppt_residency_acc prochot_residency_acc socket_thm_residency_acc
        system_clock_counter temperature_hotspot temperature_mem temperature_vrsoc
        vr_thm_residency_acc vram_max_bandwidth xcp_stats.gfx_below_host_limit_ppt_acc
        xcp_stats.gfx_below_host_limit_thm_acc xcp_stats.gfx_below_host_limit_total_acc
        xcp_stats.gfx_busy_acc xcp_stats.gfx_busy_inst xcp_stats.gfx_low_utilization_acc
        xcp_stats.jpeg_busy xcp_stats.temperature_xcd xcp_stats.vcn_busy xgmi_link_speed
        xgmi_link_status xgmi_link_width xgmi_read_data_acc xgmi_write_data_acc
        """
    ),
    # APU v2.4 and v3.0 share ApuMetricsBase_v30_t but fill different public members.
    (2, 4): _fields(
        """
        average_dclk0_frequency average_gfx_activity average_gfxclk_frequency average_mm_activity
        average_socclk_frequency average_socket_power average_uclk_frequency average_vclk0_frequency
        current_dclk0 current_gfxclk current_socclk current_uclk current_vclk0 indep_throttle_status
        system_clock_counter throttle_status
        """
    ),
    (3, 0): _fields(
        """
        average_gfx_activity average_gfxclk_frequency average_socclk_frequency average_socket_power
        average_uclk_frequency average_vclk0_frequency system_clock_counter
        """
    ),
}

# (1, >=9) is the kernel-schema-driven dynamic path, so this is a static upper
# bound: the kernel may advertise fewer attributes than the library can emit.
_DYNAMIC_FIELDS: FieldSet = _fields(
    """
    accumulation_counter average_gfx_activity average_umc_activity current_dclk0 current_dclk0s
    current_dclk1 current_gfxclk current_gfxclks current_socclk current_socclks
    current_socclks_mid current_socket_power current_uclk current_uclk_aid current_vclk0
    current_vclk0s current_vclk1 energy_accumulator firmware_timestamp gfx_activity_acc
    gfxclk_lock_status hbm_thm_residency_acc mem_activity_acc pcie_bandwidth_acc
    pcie_bandwidth_inst pcie_l0_to_recov_count_acc pcie_lc_perf_other_end_recovery
    pcie_link_speed pcie_link_width pcie_nak_rcvd_count_acc pcie_nak_sent_count_acc
    pcie_replay_count_acc pcie_replay_rover_count_acc ppt_residency_acc prochot_residency_acc
    socket_thm_residency_acc system_clock_counter temperature_aid temperature_hbm_stacks
    temperature_hotspot temperature_mem temperature_mid temperature_vrsoc vr_thm_residency_acc
    vram_max_bandwidth xcp_stats.gfx_below_host_limit_ppt_acc
    xcp_stats.gfx_below_host_limit_thm_acc xcp_stats.gfx_below_host_limit_total_acc
    xcp_stats.gfx_busy_acc xcp_stats.gfx_busy_inst xcp_stats.gfx_low_utilization_acc
    xcp_stats.jpeg_busy xcp_stats.temperature_xcd xcp_stats.vcn_busy xgmi_link_speed
    xgmi_link_status xgmi_link_width xgmi_read_data_acc xgmi_write_data_acc
    """
)

_DYNAMIC_FORMAT_REVISION = 1
_DYNAMIC_MIN_CONTENT_REVISION = 9

# (section, key) -> every gpu_metrics field the path needs. All of them are
# required, so a missing one makes the path suppressible. An allow-list: a CLI
# path absent here is never suppressed.
#
# A path may be mapped only when the metrics blobs are its only source. If
# metric.py can also fill it from hwmon, sysfs or another API, its absence is not
# determined by a metrics version, and suppression would follow live GPU state
# rather than a header. Notably absent for that reason: temperature
# edge/hotspot/mem and the whole fan section (hwmon), the voltages
# (amdsmi_get_power_info), pcie replay_count and the per-engine clock slots
# (sysfs), and usage.vcn_busy (amdsmi_get_vcn_busy_percent on Navi).
_CLI_PATH_REQUIRED_SOURCES: Dict[CliPath, Tuple[str, ...]] = {
    ("usage", "mm_activity"): ("average_mm_activity",),
    ("usage", "vcn_activity"): ("vcn_activity",),
    ("usage", "jpeg_activity"): ("jpeg_activity",),
    ("usage", "gfx_busy_inst"): ("xcp_stats.gfx_busy_inst",),
    ("usage", "jpeg_busy"): ("xcp_stats.jpeg_busy",),
    ("power", "throttle_status"): ("throttle_status",),
    ("clock", "uclk_aid"): ("current_uclk_aid",),
    ("clock", "socclks_mid"): ("current_socclks_mid",),
    ("temperature", "hbm_stacks"): ("temperature_hbm_stacks",),
    ("temperature", "mid"): ("temperature_mid",),
    ("temperature", "aid"): ("temperature_aid",),
    ("temperature", "xcd"): ("xcp_stats.temperature_xcd",),
    ("pcie", "bandwidth"): ("pcie_bandwidth_inst",),
    ("pcie", "l0_to_recovery_count"): ("pcie_l0_to_recov_count_acc",),
    ("pcie", "replay_roll_over_count"): ("pcie_replay_rover_count_acc",),
    ("pcie", "nak_sent_count"): ("pcie_nak_sent_count_acc",),
    ("pcie", "nak_received_count"): ("pcie_nak_rcvd_count_acc",),
    ("pcie", "lc_perf_other_end_recovery_count"): ("pcie_lc_perf_other_end_recovery",),
    ("throttle", "accumulation_counter"): ("accumulation_counter",),
}

# Throttle rows all come from the residency accumulators, exposed through
# amdsmi_get_violation_status(); each row has accumulated/status/activity forms.
_THROTTLE_SOURCES: Dict[str, str] = {
    "prochot": "prochot_residency_acc",
    "ppt": "ppt_residency_acc",
    "socket_thermal": "socket_thm_residency_acc",
    "vr_thermal": "vr_thm_residency_acc",
    "hbm_thermal": "hbm_thm_residency_acc",
    "gfx_clk_below_host_limit": "xcp_stats.gfx_below_host_limit_acc",
    "gfx_clk_below_host_limit_power": "xcp_stats.gfx_below_host_limit_ppt_acc",
    "gfx_clk_below_host_limit_thermal": "xcp_stats.gfx_below_host_limit_thm_acc",
    "total_gfx_clk_below_host_limit": "xcp_stats.gfx_below_host_limit_total_acc",
    "low_utilization": "xcp_stats.gfx_low_utilization_acc",
}

for _row, _source in _THROTTLE_SOURCES.items():
    _CLI_PATH_REQUIRED_SOURCES[("throttle", f"{_row}_accumulated")] = (_source,)
    # amdsmi_get_violation_status() computes the activity percentage as the
    # accumulator over the counter and derives the status from it, leaving both
    # unset without the counter, so both forms require both fields.
    _CLI_PATH_REQUIRED_SOURCES[("throttle", f"{_row}_violation_status")] = (
        _source,
        "accumulation_counter",
    )
    _CLI_PATH_REQUIRED_SOURCES[("throttle", f"{_row}_violation_activity")] = (
        _source,
        "accumulation_counter",
    )


def known_versions() -> FrozenSet[VersionKey]:
    """The ``(format_revision, content_revision)`` pairs the table maps explicitly."""
    return frozenset(_VERSION_FIELDS)


def populatable_fields(format_revision: Any, content_revision: Any) -> Optional[FieldSet]:
    """Public gpu_metrics fields this version can carry, or None if unmapped.

    Explicit entries win over the dynamic bound, matching
    ``translate_header_to_flag_version()``, which consults
    ``amdgpu_metric_version_translation_table`` first and only falls back to the
    (1, >=9) dynamic reader on a miss.
    """
    try:
        version = (int(format_revision), int(content_revision))
    except (TypeError, ValueError):
        return None
    fields = _VERSION_FIELDS.get(version)
    if fields is not None:
        return fields
    if version[0] == _DYNAMIC_FORMAT_REVISION and version[1] >= _DYNAMIC_MIN_CONTENT_REVISION:
        return _DYNAMIC_FIELDS
    return None


def build_suppression_set(header: Any) -> FrozenSet[CliPath]:
    """CLI paths the version in ``header`` can never populate.

    ``header`` is the dict from ``amdsmi_get_gpu_metrics_header_info()``. Anything
    unreadable or unmapped yields an empty set, which suppresses nothing.
    """
    if not isinstance(header, dict):
        return frozenset()
    fields = populatable_fields(header.get("format_revision"), header.get("content_revision"))
    if fields is None:
        return frozenset()
    return frozenset(
        path
        for path, sources in _CLI_PATH_REQUIRED_SOURCES.items()
        if any(source not in fields for source in sources)
    )


def _is_all_na(value: Any) -> bool:
    """True when nothing under ``value`` carries data."""
    if isinstance(value, str):
        if value == NA:
            return True
        # Human-readable output renders a list as "[N/A, 42 %]".
        if value.startswith("[") and value.endswith("]"):
            # A naive split mis-splits an element containing a comma, which then
            # compares unequal to "N/A" and keeps the field. Fails safe.
            items = value[1:-1].split(",")
            return bool(value[1:-1].strip()) and all(item.strip() == NA for item in items)
        return False
    if isinstance(value, dict):
        return bool(value) and all(_is_all_na(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_is_all_na(item) for item in value)
    return False


def filter_unsupported(
    values_dict: Dict[str, Any],
    suppressed: FrozenSet[CliPath],
    protected_sections: FrozenSet[str] = frozenset(),
) -> Dict[str, Any]:
    """``values_dict`` without the suppressed all-N/A paths.

    An empty ``suppressed`` set is handed straight back as ``values_dict``
    itself. Otherwise the top level is a new dict, and the sections it did not
    have to change are the objects it was given.

    Sections emptied by the filter are dropped, except those in
    ``protected_sections``: the user asked for those by name, so they are handed
    back unfiltered rather than as nothing. A section that arrived empty is kept
    so filtering never changes anything but suppressed fields.
    """
    if not suppressed:
        return values_dict

    filtered = {}
    for section, section_value in values_dict.items():
        if not isinstance(section_value, dict):
            filtered[section] = section_value
            continue
        kept = {
            key: value
            for key, value in section_value.items()
            if (section, key) not in suppressed or not _is_all_na(value)
        }
        if not kept and section_value:
            if section in protected_sections:
                filtered[section] = section_value
            continue
        filtered[section] = kept
    return filtered
