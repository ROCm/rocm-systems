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


def _check_pipe(snapshot, pipe_name):
    """Validate issue/stall for a single arbiter pipe.

    Allowed combinations per pipe:
        issue=0, stall=0  (idle)
        issue=1, stall=0  (issued)
        issue=0, stall=1  (stalled)

    The only disallowed combination is issue=1 AND stall=1.
    """
    issue = snapshot[f"arb_state_issue_{pipe_name}"]
    stall = snapshot[f"arb_state_stall_{pipe_name}"]
    assert not (issue == 1 and stall == 1), (
        f"{pipe_name} arbiter state check failed: "
        f"issue=1 and stall=1 is not allowed (issue={issue}, stall={stall})"
    )


_PIPES = [
    "valu",
    "scalar",
    "vmem_tex",
    "lds",
    "exp",
    "lds_direct",
    "brmsg",
]


def validate_arbiter_state(snapshot):
    for pipe in _PIPES:
        _check_pipe(snapshot, pipe)
