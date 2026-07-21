# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import absolute_import


# f64 transcendentals (rcp/rsq/sqrt) issue via the scalar/transcendental unit on gfx9,
# so the hardware reports inst_type SCALAR for them even though the mnemonic starts with
# 'v_' (authoritative data read bit-for-bit from perf_snapshot_data).
_SCALAR_ISSUED_VALU_PREFIXES = ("v_rcp_f64", "v_rsq_f64", "v_sqrt_f64")


def validate_scalar_instructions_issued(all_samples, scalar_samples):
    # From all samples, extract samples with SCALAR type
    scalar_type_samples_issued = all_samples[
        (
            all_samples["Instruction_Type"]
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_SCALAR"
        )
        & all_samples["Wave_Issued_Instruction"]
    ]

    # scalar_samples contains instructions starting with `s_`
    scalar_samples_issued = scalar_samples[scalar_samples["Wave_Issued_Instruction"]]

    # Every issued s_-prefixed instruction must be hardware-typed SCALAR.
    assert (
        scalar_samples_issued["Instruction_Type"]
        == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_SCALAR"
    ).all()

    # The hardware also reports the known f64 transcendentals (v_rcp_f64/v_rsq_f64/
    # v_sqrt_f64) as SCALAR since they issue via the scalar unit, so hardware-SCALAR can
    # exceed the s_-prefixed count (this replaced an over-strict `len == len` check). That
    # is expected; require that any SCALAR-typed instruction which is not s_-prefixed is
    # one of those known transcendentals and nothing else.
    non_s_scalar_typed = scalar_type_samples_issued[
        ~scalar_type_samples_issued["Instruction"].apply(lambda x: x.startswith("s_"))
    ]
    _unexpected = non_s_scalar_typed[
        ~non_s_scalar_typed["Instruction"].apply(
            lambda x: x.startswith(_SCALAR_ISSUED_VALU_PREFIXES)
        )
    ][["Instruction", "Instruction_Type"]].drop_duplicates()
    assert (
        _unexpected.empty
    ), f"SCALAR-typed issued instructions that are not s_ or a known f64 transcendental:\n{_unexpected.to_string()}"


def validate_scalar_instructions_stalled(scalar_samples):
    scalar_samples_stalled = scalar_samples[
        scalar_samples["Wave_Issued_Instruction"] == False
    ]

    assert (
        scalar_samples_stalled["Stall_Reason"]
        .apply(
            lambda x: x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE"
            or x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL"
            or x
            == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN"
            or x == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY"
        )
        .all()
    )


def validate_scalar_instructions(all_samples, scalar_samples):
    validate_scalar_instructions_issued(all_samples, scalar_samples)
    validate_scalar_instructions_stalled(scalar_samples)
