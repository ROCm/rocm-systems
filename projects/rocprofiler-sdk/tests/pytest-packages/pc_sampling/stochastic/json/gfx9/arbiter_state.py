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


def validate_arbiter_state(snapshot):
    # VALU pipe checks
    if snapshot["dual_issue_valu"]:
        # (valu_issue = 1 & valu_stall = 0) is the only allowed
        assert (
            snapshot["arb_state_issue_valu"] == 1
            and snapshot["arb_state_stall_valu"] == 0
        ), "Dual issue VALU arbiter state check failed"
    else:
        # (valu_issue = 0 & value_stall = 1) is not allowed
        assert not (
            snapshot["arb_state_issue_valu"] == 0
            and snapshot["arb_state_stall_valu"] == 1
        ), "VALU arbiter state check failed"

    # Matrix pipe checks
    # matrix_issue = 0 & matrix_stall = 1 is not allowed
    assert not (
        snapshot["arb_state_issue_matrix"] == 0
        and snapshot["arb_state_stall_matrix"] == 1
    ), "Matrix arbiter state check failed"

    # scalar pipe checks
    # scalar_issue = 0 & scalar_stall = 1 is not allowed
    assert not (
        snapshot["arb_state_issue_scalar"] == 0
        and snapshot["arb_state_stall_scalar"] == 1
    ), "Scalar arbiter state check failed"

    # texture pipe checks
    # tex_issue = 0 & tex_stall = 1 is not allowed
    assert not (
        snapshot["arb_state_issue_vmem_tex"] == 0
        and snapshot["arb_state_stall_vmem_tex"] == 1
    ), "Texture arbiter state check failed"

    # LDS pipe checks
    # lds_issue = 0 & lds_stall = 1 is not allowed
    assert not (
        snapshot["arb_state_issue_lds"] == 0 and snapshot["arb_state_stall_lds"] == 1
    ), "LDS arbiter state check failed"

    # flat pipe checks
    # flat_issue = 0 & flat_stall = 1 is not allowed
    assert not (
        snapshot["arb_state_issue_flat"] == 0 and snapshot["arb_state_stall_flat"] == 1
    ), "Flat arbiter state check failed"

    # misc pipe checks
    # TODO: verify this
    # According to Joe's slides, the misc_stall cannot be 0.
    # However, the condition representing this case fails for `transpose` application
    #   assert((samples['Arbiter_State_Stall_Misc'] == 0).all())
    # Instead, I had to replace is with the condition belowe
    #   misc_issue = 0 & misc_stall = 1 is not allowed
    assert not (
        snapshot["arb_state_issue_misc"] == 0 and snapshot["arb_state_stall_misc"] == 1
    ), "Misc arbiter state check failed"

    # export pipe checks
    # We assume same conditions for Export pipe as for Misc (Joe's original),
    # so we should TODO: verify
    # exp_issue can take both 1 and 0, so no need to check it
    # exp_stall must be 0
    assert snapshot["arb_state_stall_exp"] == 0, "Export arbiter state check failed"

    # lds_direct pipe checks
    # This pipe doesn't exist on GFX9 so both issue and stall must be 0
    assert (
        snapshot["arb_state_issue_lds_direct"] == 0
    ), "LDS Direct arbiter state check failed"
    assert (
        snapshot["arb_state_stall_lds_direct"] == 0
    ), "LDS Direct arbiter state check failed"

    # brmsg pipe doesn't exist on GFX9 so both issue and stall must be 0
    assert snapshot["arb_state_issue_brmsg"] == 0, "BRMSG arbiter state check failed"
    assert snapshot["arb_state_stall_brmsg"] == 0, "BRMSG arbiter state check failed"
