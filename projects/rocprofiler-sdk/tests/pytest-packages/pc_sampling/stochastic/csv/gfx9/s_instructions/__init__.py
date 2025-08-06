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

from functools import partial

from .branch_instructions import validate_branch_instructions
from .waitcnt import validate_waitcnt
from .other_instructions import validate_other_instructions
from .scalar_instructions import validate_scalar_instructions
from .internal_instructions import validate_internal_instructions
from .jump_instructions import validate_jump_instructions
from .message_instructions import validate_message_instructions
from .barrier_instructions import validate_barrier_instructions


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
]


inst_type_verify_functions = {
    "BRANCH": validate_branch_instructions,
    "WAITCNT": validate_waitcnt,
    "OTHER": validate_other_instructions,
    "SCALAR": validate_scalar_instructions,
    "INTERNAL": validate_internal_instructions,
    "JUMP": validate_jump_instructions,
    "MESSAGE": validate_message_instructions,
    "BARRIER": validate_barrier_instructions,
}


# Function to classify instructions based on the Trie
def classify_instruction_by_prefix(prefix_tree, instruction):
    # extracting the base of the instruction (e.g., s_mov_*, v_mov_*, s_setpc_*, ...)
    base_instruction = instruction.split()[0]

    # Classify based on the Trie (general classification)
    instruction_types = prefix_tree.get_instruction_type(base_instruction)

    # aways use the specific type
    return instruction_types[-1]


def enforce_type_inheritance(sub_df, parent_df):
    for col in parent_df.columns:
        sub_df[col] = sub_df[col].astype(parent_df[col].dtype)
    return sub_df


def validate_s_instructions(df):
    s_instructions = df[df["Instruction"].apply(lambda x: x.startswith("s_"))].copy()

    # fill in the Prefi Tree
    prefix_tree = PrefixTree()
    for prefix, instruction_type in instructions_with_types:
        prefix_tree.insert(prefix, instruction_type)

    _classify_instruction_by_prefix = partial(classify_instruction_by_prefix, prefix_tree)
    s_instructions["Instruction_Type_From_Name"] = s_instructions["Instruction"].apply(
        _classify_instruction_by_prefix
    )

    for inst_type, subframe in s_instructions.groupby("Instruction_Type_From_Name"):
        # subframe = enforce_type_inheritance(subframe, s_instructions)
        if inst_type in inst_type_verify_functions:
            # Pass all samples and filtered samples to the verification function.
            inst_type_verify_functions[inst_type](df, subframe)
