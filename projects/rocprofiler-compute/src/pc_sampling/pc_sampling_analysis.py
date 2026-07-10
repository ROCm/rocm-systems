# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""PC sampling analysis utilities.

Helpers for building a normalized PC sampling dataframe from a parsed
``rocprofiler-sdk-tool[0]`` dict.
"""

from typing import Any, NamedTuple, Optional

import pandas as pd

PC_SAMPLING_NOT_ISSUE_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"
PC_SAMPLING_INST_TYPE_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_"

# Canonical not-issued stall reasons (rocprofiler-sdk enum, prefix stripped).
# Reasons outside this set are dropped during aggregation.
STALL_REASON_KEYS = frozenset({
    "NONE",
    "NO_INSTRUCTION_AVAILABLE",
    "ALU_DEPENDENCY",
    "WAITCNT",
    "INTERNAL_INSTRUCTION",
    "BARRIER_WAIT",
    "ARBITER_NOT_WIN",
    "ARBITER_WIN_EX_STALL",
    "OTHER_WAIT",
    "SLEEP_WAIT",
    "LAST",
})

NORMALIZED_RECORD_COLUMNS = [
    "inst_index",
    "code_object_id",
    "code_object_offset",
    "dispatch_id",
    "kernel_id",
    "wave_issued",
    "stall_reason",
    "inst_type",
]

SOURCE_LINE_MISSING = "N/A"


class InstructionLineRecord(NamedTuple):
    """One sampled offset attributed to one kernel, with aggregated counts."""

    code_object_offset: int
    kernel_name: Optional[str]
    instruction: Optional[str]
    comment: str
    total_count: int
    issue_count: Optional[int]
    stall_count: Optional[int]
    stall_reasons: dict[str, int]
    inst_types: dict[str, int]


class CodeObjectRecord(NamedTuple):
    """A code object and the sampled instruction lines it owns."""

    code_object_id: int
    load_base: Optional[int]
    instruction_lines: list[InstructionLineRecord]


def detect_pc_sampling_method(tool_data: dict[str, Any]) -> Optional[str]:
    """Detect the PC sampling method from the populated buffer record array.

    Prioritizes stochastic over host_trap. Returns None when neither array
    holds any samples.
    """
    buffer_records = tool_data["buffer_records"]
    if buffer_records["pc_sample_stochastic"]:
        return "stochastic"
    if buffer_records["pc_sample_host_trap"]:
        return "host_trap"
    return None


def load_pc_sample_records(tool_data: dict[str, Any]) -> pd.DataFrame:
    """Flatten the PC sample arrays into normalized per-sample rows."""
    buffer_records = tool_data["buffer_records"]
    samples = (
        buffer_records["pc_sample_stochastic"] + buffer_records["pc_sample_host_trap"]
    )
    # kernel_id via dispatch correlation so kernels sharing a code object are
    # attributed correctly.
    dispatch_to_kernel_id = {
        dispatch["dispatch_info"]["dispatch_id"]: dispatch["dispatch_info"]["kernel_id"]
        for dispatch in buffer_records["kernel_dispatch"]
    }

    rows = []
    for sample in samples:
        record = sample.get("record", {})
        pc_info = record.get("pc", {})
        code_object_id = pc_info.get("code_object_id")
        code_object_offset = pc_info.get("code_object_offset")
        inst_index = sample.get("inst_index")
        # Skip records without the keys needed to place and label the sample.
        if None in (code_object_id, code_object_offset, inst_index):
            continue
        dispatch_id = record.get("dispatch_id")
        rows.append({
            "inst_index": inst_index,
            "code_object_id": code_object_id,
            "code_object_offset": code_object_offset,
            "dispatch_id": dispatch_id,
            "kernel_id": dispatch_to_kernel_id.get(dispatch_id),
            "wave_issued": record.get("wave_issued"),
            "stall_reason": record.get("snapshot", {}).get("stall_reason"),
            "inst_type": record.get("inst_type"),
        })

    return pd.DataFrame(rows, columns=NORMALIZED_RECORD_COLUMNS)


def aggregate_pc_sample_records(
    records_df: pd.DataFrame,
    group_by: list[str],
) -> pd.DataFrame:
    """Group normalized records into per-group counts, stall reasons, inst types."""
    # inst_index and kernel_id are constant within a group; carry the first when
    # they are not group keys.
    carried = [
        column for column in ("inst_index", "kernel_id") if column not in group_by
    ]
    if records_df.empty:
        return pd.DataFrame(
            columns=[
                *group_by,
                "count",
                "count_issued",
                "count_stalled",
                "stall_reason",
                "inst_type",
                *carried,
            ]
        )

    aggregations = {
        "count": ("inst_index", "size"),
        "count_issued": ("wave_issued", _aggregate_count_issued),
        "count_stalled": ("wave_issued", _aggregate_count_stalled),
        "stall_reason": ("stall_reason", _aggregate_stall_reason),
        "inst_type": ("inst_type", _aggregate_inst_type),
    }
    for column in carried:
        aggregations[column] = (column, "first")

    # dropna=False keeps samples whose kernel_id is unmapped (None) instead of
    # silently dropping them from the all-kernel view.
    return records_df.groupby(group_by, as_index=False, dropna=False).agg(
        **aggregations
    )


def enrich_with_metadata(
    aggregated_df: pd.DataFrame,
    tool_data: dict[str, Any],
    attach: set[str],
) -> pd.DataFrame:
    """Attach the columns named in *attach* by index into *tool_data*."""
    df = aggregated_df.copy()
    strings = tool_data["strings"]
    instructions = strings["pc_sample_instructions"]
    comments = strings["pc_sample_comments"]

    if "instruction" in attach:
        df["instruction"] = df["inst_index"].apply(
            lambda index: instructions[index] if index < len(instructions) else None
        )
    if "source_line" in attach:
        # Keep the raw comment; "N/A" (not "") when empty or out of range so the
        # caller trims real strings and display code keeps the column.
        df["source_line"] = df["inst_index"].apply(
            lambda index: (
                comments[index]
                if index < len(comments) and comments[index]
                else SOURCE_LINE_MISSING
            )
        )
    if "kernel_name" in attach:
        kernel_id_to_name = {
            symbol["kernel_id"]: symbol["formatted_kernel_name"]
            for symbol in tool_data["kernel_symbols"]
        }
        # .get (not .map) so an unmapped kernel_id yields None, not NaN.
        df["kernel_name"] = df["kernel_id"].apply(kernel_id_to_name.get)

    return df


def load_aggregated_pc_sampling(tool_data: dict[str, Any]) -> list[CodeObjectRecord]:
    """Build the normalized code-object tree the analysis DB inserts.

    Runs load -> aggregate -> enrich, grouping samples by
    (code_object_id, offset, kernel_id) so each line is attributed to the kernel
    its dispatch ran, then joins the code object catalog (load_base).
    """
    records_df = load_pc_sample_records(tool_data)
    aggregated_df = aggregate_pc_sample_records(
        records_df,
        group_by=["code_object_id", "code_object_offset", "kernel_id"],
    )
    aggregated_df = enrich_with_metadata(
        aggregated_df,
        tool_data,
        attach={"instruction", "source_line", "kernel_name"},
    )

    catalog = {
        code_object["code_object_id"]: code_object
        for code_object in tool_data.get("code_objects", [])
    }

    records = []
    for code_object_id, group in aggregated_df.groupby("code_object_id", dropna=False):
        catalog_entry = catalog.get(code_object_id, {})
        records.append(
            CodeObjectRecord(
                code_object_id=int(code_object_id),
                load_base=catalog_entry.get("load_base"),
                instruction_lines=[
                    _to_instruction_line_record(row) for row in group.itertuples()
                ],
            )
        )
    return records


def _to_instruction_line_record(row: Any) -> InstructionLineRecord:  # noqa: ANN401
    """Convert one aggregated dataframe row into an InstructionLineRecord."""
    return InstructionLineRecord(
        code_object_offset=int(row.code_object_offset),
        kernel_name=row.kernel_name,
        instruction=row.instruction,
        comment=row.source_line,
        total_count=int(row.count),
        issue_count=None if pd.isna(row.count_issued) else int(row.count_issued),
        stall_count=None if pd.isna(row.count_stalled) else int(row.count_stalled),
        stall_reasons={} if pd.isna(row.stall_reason) else row.stall_reason,
        inst_types={} if pd.isna(row.inst_type) else row.inst_type,
    )


def _aggregate_count_issued(wave_issued: pd.Series) -> Optional[int]:
    """Sum issued waves; None when no wave_issued info exists (host_trap)."""
    if wave_issued.isnull().all():
        return None
    return int(wave_issued.sum())


def _aggregate_count_stalled(wave_issued: pd.Series) -> Optional[int]:
    """Count not-issued waves; None when no wave_issued info exists (host_trap)."""
    if wave_issued.isnull().all():
        return None
    return int(wave_issued.count() - wave_issued.sum())


def _aggregate_stall_reason(stall_reason: pd.Series) -> Optional[dict[str, int]]:
    """Count valid not-issued stall reasons as a descending {reason: count} dict.

    None when no stall info exists (host_trap). Reasons outside
    ``STALL_REASON_KEYS`` (e.g. malformed or unknown enum values) are dropped.
    """
    present = stall_reason.dropna()
    if present.empty:
        return None
    stripped = present.str[len(PC_SAMPLING_NOT_ISSUE_PREFIX) :]
    valid = stripped[stripped.isin(STALL_REASON_KEYS)]
    return valid.value_counts().to_dict()


def _aggregate_inst_type(inst_type: pd.Series) -> Optional[dict[str, int]]:
    """Count sampled instruction types as a descending {type: count} dict.

    None when no inst_type info exists. The raw SDK enum prefix is stripped so
    stored values read as VALU, FLAT, NO_INST, etc.
    """
    present = inst_type.dropna()
    if present.empty:
        return None
    stripped = present.str[len(PC_SAMPLING_INST_TYPE_PREFIX) :]
    return stripped.value_counts().to_dict()
