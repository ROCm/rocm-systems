#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import sys
import pytest

# Each filtered context must receive exactly the records of its selected operation that
# the unfiltered context of the same mode received, and no records of other operations.
FILTERED = {
    "callback_allocate": ("callback_all", "allocate", "free"),
    "callback_free": ("callback_all", "free", "allocate"),
    "buffered_allocate": ("buffered_all", "allocate", "free"),
    "buffered_free": ("buffered_all", "free", "allocate"),
}


def test_unfiltered_contexts(input_data):
    contexts = input_data["contexts"]
    for name in ("callback_all", "buffered_all"):
        assert contexts[name]["allocate"] > 0, f"{name}: {contexts[name]}"
        assert contexts[name]["free"] > 0, f"{name}: {contexts[name]}"
        assert contexts[name]["other"] == 0, f"{name}: {contexts[name]}"

    # callback and buffered tracing see the same calls
    for op in ("allocate", "free"):
        assert contexts["callback_all"][op] == contexts["buffered_all"][op], f"{contexts}"


@pytest.mark.parametrize("name", sorted(FILTERED.keys()))
def test_filtered_contexts(input_data, name):
    contexts = input_data["contexts"]
    control, selected, excluded = FILTERED[name]

    assert (
        contexts[name][selected] == contexts[control][selected]
    ), f"{name} should receive every {selected} record: {contexts[name]} vs {control}: {contexts[control]}"
    assert contexts[name][excluded] == 0, f"{name}: {contexts[name]}"
    assert contexts[name]["other"] == 0, f"{name}: {contexts[name]}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
