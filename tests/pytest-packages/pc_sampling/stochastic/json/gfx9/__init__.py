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

import numpy as np
import pandas as pd
from collections import defaultdict
from .arbiter_state import validate_arbiter_state
from .other_instructions import (
    validate_valu_instructions,
    validate_flat_instructions,
    validate_lds_instructions,
)
from .s_instructions import (
    validate_internal_instructions,
    validate_barrier_instructions,
    validate_waitcnt,
    validate_branch_instructions,
    validate_scalar_instructions,
)

# Using Prefix Tree to classify the instruction type
# I did this instead of the regex becuase I wanted to try if we could
# generalize this approach for other types of instructions.
# The dream scenario: We have a giant list of all instructions and their
# types. Then we parse the list and dynamically determine the checks
# based on the instruction types.


# TODO: extract this outside of the file
class TrieNode:
    def __init__(self):
        self.children = {}
        self.instruction_type = None  # Store the instruction type at the leaf node


class PrefixTree:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, full_prefix, instruction_type):
        """Insert a prefix and its associated instruction type into the Trie."""
        node = self.root
        for char in full_prefix:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.instruction_type = (
            instruction_type  # Assign the instruction type at the leaf
        )

    def get_instruction_type(self, instruction):
        """Get the list of instruction types based on the longest matching prefix."""
        node = self.root
        matched_types = []  # List to store matched types

        # Traverse the instruction one character at a time
        for char in instruction:
            if char not in node.children:
                break  # Stop if no match is found

            node = node.children[char]

            # If we reach a node that has an instruction type, store it
            if node.instruction_type:
                matched_types.append(node.instruction_type)

        return matched_types


instructions_with_types = [
    ("s_", "SCALAR"),  # Scalar instructions (general category)
    ("s_waitcnt", "WAITCNT"),  # WAITCNT (specific)
    ("s_sendmsg", "MESSAGE"),  # MESSAGE (specific)
    ("s_barrier", "BARRIER"),  # BARRIER (specifix)
    ("s_swappc", "JUMP"),  # JUMP (specific)
    ("s_setpc", "JUMP"),  # JUMP
    ("s_setpc", "JUMP"),  # JUMP
    ("s_sleep", "JUMP"),  # JUMP
    ("s_branch", "BRANCH"),  # BRANCH
    ("s_cbranch", "BRANCH"),  # BRANCH (conditional)
    ("s_wakeup", "OTHER"),  # OHTER
    ("s_nop", "INTERNAL"),  # INTERNAL
    ("s_sleep", "INTERNAL"),  # INTERNAL
    ("v_", "VALU"),  # VALU
    ("v_mfma", "MATRIX"),  # MATRIX
    ("flat_", "FLAT"),  # FLAT
    ("global_", "FLAT"),  # FLAT
    ("ds_", "LDS"),  # LDS
    ("buffer_", "TEX"),  # TEX
]


inst_type_verify_functions = {
    "BRANCH": validate_branch_instructions,
    "WAITCNT": validate_waitcnt,
    # "OTHER": validate_other_instructions,
    "SCALAR": validate_scalar_instructions,
    "INTERNAL": validate_internal_instructions,
    # "JUMP": validate_jump_instructions,
    # "MESSAGE": validate_message_instructions,
    "BARRIER": validate_barrier_instructions,
    "VALU": validate_valu_instructions,
    "FLAT": validate_flat_instructions,
    "LDS": validate_lds_instructions,
}


def validate_stochastic_samples_json(data_json):
    # fill in the Prefix Tree
    prefix_tree = PrefixTree()
    for prefix, instruction_type in instructions_with_types:
        prefix_tree.insert(prefix, instruction_type)

    instructions = data_json["strings"]["pc_sample_instructions"]
    comments = data_json["strings"]["pc_sample_comments"]

    insts_per_prefix_type = defaultdict(list)

    for sample in data_json["buffer_records"]["pc_sample_stochastic"]:
        inst_index = sample["inst_index"]
        if inst_index == -1:
            # Ignoring samples from blit kernels
            continue
        record = sample["record"]
        # extend the record with the instruction
        record["inst"] = instructions[inst_index]

        # get the instruction type from prefix tree
        inst_prefix_types = prefix_tree.get_instruction_type(record["inst"])
        # each type must have a type
        assert len(inst_prefix_types) > 0
        # As more then one type can be matched, we take the last one as the most specific.
        inst_prefix_type = inst_prefix_types[-1]
        insts_per_prefix_type[inst_prefix_type].append(record)

        # For each sample, we need to validate wave_cnt and arbiter state
        wave_cnt = record["wave_cnt"]
        assert wave_cnt >= 0 and wave_cnt <= 32, "Invalid wave count"

        # arbiter state check
        snapshot = record["snapshot"]
        validate_arbiter_state(snapshot)

    # Check now the instruction type and arb state correlation.
    # We do that for all samples of a single instruction type all at once
    # to minimize the number of functions calls (one call for all samples, instead of a function
    # call per sample).
    # Please note that each sample is iterated at most twice.
    # The first time to group samples per instruction type, and the second time to validate samples.
    for inst_prefix_type, sample_records in insts_per_prefix_type.items():
        if inst_prefix_type in inst_type_verify_functions:
            inst_type_verify_functions[inst_prefix_type](sample_records)
        else:
            assert False, f"Unhandle instruction type: {inst_prefix_type}"
