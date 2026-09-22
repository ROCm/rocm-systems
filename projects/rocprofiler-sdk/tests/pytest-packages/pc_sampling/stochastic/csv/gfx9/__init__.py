# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
from .texture_instructions import validate_texture_instructions
from .matrix_instructions import validate_matrix_instructions
from .lds_instructions import validate_lds_instructions
from .flat_instructions import validate_flat_instructions


def validate_wave_count(df):
    # Validating number of actives waves on a cu
    assert (
        (df["Wave_Count"] >= 1) & (df["Wave_Count"] <= 32)
    ).all(), "Invalid Wave_Count"


def validate_issued_instruction_type_no_inst(samples):
    # NO_INST type of instructions means instruction is not issued
    issued_type_no_inst = samples[samples["Instruction_Type"] == "NO_INST"]
    assert len(issued_type_no_inst) == 0, "NO_INST implies no instruction is issued"


def validate_issued_instruction_type_other(samples):
    # OTHER type of instructions still to be determined
    issued_type_other = samples[samples["Instruction_Type"] == "OTHER"]
    assert len(issued_type_other) == 0, "OTHER type of instruction observed first time"


def validate_issued_instruction_type_lds_direct(samples):
    # LDS_DIRECT type of instructions do not exist on gfx9
    issued_type_lds_direct = samples[samples["Instruction_Type"] == "LDS_DIRECT"]
    assert (
        len(issued_type_lds_direct) == 0
    ), "LDS direct type of instruction observed on GFX9"


def validate_issued_instruction_type_dual_valu(samples):
    # LDS_DIRECT type of instructions do not exist on gfx9
    issued_type_dual_valu = samples[samples["Instruction_Type"] == "DUAL_VALU"]
    assert (
        len(issued_type_dual_valu) == 0
    ), "DUAL_VALU type of instruction observed on GFX9"


# TODO: add checks for missing instruction types
# - export


def validate_stochastic_samples_csv(df: pd.DataFrame):
    # We expect mode valid than invalid samples
    # TODO: use stats for comparing valid vs invalid samples
    # invalid_samples = df[df["Valid"] == False]
    # valid_samples = df[df["Valid"]].copy()
    # assert len(valid_samples) > len(invalid_samples)

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
    validate_texture_instructions(valid_samples)
    validate_matrix_instructions(valid_samples)
    validate_lds_instructions(valid_samples)
    validate_flat_instructions(valid_samples)

    # validating issued instructions for uncovered types
    valid_samples_issued = valid_samples[
        valid_samples["Wave_Issued_Instruction"] == True
    ].copy()
    validate_issued_instruction_type_no_inst(valid_samples_issued)
    validate_issued_instruction_type_other(valid_samples_issued)

    # The following two types of instructions should not be observed on gfx9
    validate_issued_instruction_type_lds_direct(valid_samples_issued)
    validate_issued_instruction_type_dual_valu(valid_samples_issued)
