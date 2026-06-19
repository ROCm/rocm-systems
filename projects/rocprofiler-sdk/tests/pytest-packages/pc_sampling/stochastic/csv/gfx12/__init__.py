# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import numpy as np
import pandas as pd

from .s_instructions import validate_s_instructions
from .valu_instructions import validate_valu_instructions
from .vmem_instructions import validate_vmem_instructions
from .matrix_instructions import validate_matrix_instructions
from .lds_instructions import validate_lds_instructions
from .flat_instructions import validate_flat_instructions
from .dual_valu_instructions import validate_dual_valu_instructions


def validate_wave_count(df):
    # Validating number of actives waves on a SIMD
    assert (
        (df["Wave_Count"] >= 1) & (df["Wave_Count"] <= 16)
    ).all(), "Invalid Wave_Count"


def validate_issued_instruction_type_other(samples):
    # ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_OTHER type of instructions still to be determined
    issued_type_other = samples[samples["Instruction_Type"] == "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_OTHER"]
    assert len(issued_type_other) == 0, "OTHER type of instruction observed first time"


def validate_stochastic_samples_csv(df: pd.DataFrame):
    # TODO: Once rocpd is ready, validate the number of invalid vs valid samples.

    # only valid samples reside in df
    valid_samples = df.copy()

    validate_wave_count(valid_samples)

    # The following checks assumes that we were able to decode
    # the instruction, meaning a code object and dispatch must be known.
    valid_samples = valid_samples[valid_samples["Dispatch_Id"] > 0]

    # scalar, barrier, waitcnt, jump, message, branches (taken and not taken)
    # are handled inside `validate_s_instructions` function
    validate_s_instructions(valid_samples)
    validate_valu_instructions(valid_samples)
    validate_vmem_instructions(valid_samples)
    validate_matrix_instructions(valid_samples)
    validate_lds_instructions(valid_samples)
    validate_flat_instructions(valid_samples)
    validate_dual_valu_instructions(valid_samples)

    # validating issued instructions for uncovered types
    valid_samples_issued = valid_samples[
        valid_samples["Wave_Issued_Instruction"] == True
    ].copy()
    validate_issued_instruction_type_other(valid_samples_issued)

    # NOTE: LDS_DIRECT and DUAL_VALU exist on GFX12, so we do not assert they are absent
