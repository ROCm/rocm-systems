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

# Validators for the OMPT keep-or-defer role decision as a user meets it: through
# rocprofv3, with a second OMPT tool offered to the OpenMP runtime via
# OMP_TOOL_LIBRARIES. Exactly one tool holds the role, and which one follows from
# the rocprofv3 command line:
#
#   keep  - `rocprofv3 --ompt-trace ...` collects OMPT, so rocprofv3 holds the role
#           and the runtime never goes looking for a second tool.
#   defer - `rocprofv3` without OMPT hands the role to the other tool, and keeps
#           tracing everything else it was asked for.
#
# A tool that does hold the role may also hand it back, by calling the public
# rocprofiler_ompt_start_tool directly. That path bypasses the rocprofv3 entry point
# altogether, so whether OMPT is collected has to follow the command line there too.
#
# OMPT is a rocpd-only trace, so rocprofv3's side is read from the rocpd database
# and the other tool's side from the JSON summary it writes at teardown.

OMPT_CATEGORY = "OMPT"


def _ompt_record_count(conn):
    """Row count for OMPT records in rocpd, spanning both ranged regions and
    instant samples."""
    (count,) = conn.execute(
        "SELECT COUNT(*) FROM regions_and_samples WHERE category = ?",
        (OMPT_CATEGORY,),
    ).fetchone()
    return count


def _kernel_dispatch_count(conn):
    (count,) = conn.execute("SELECT COUNT(*) FROM kernels").fetchone()
    return count


def test_rocprofv3_kept_ompt_role(rocpd_conn):
    """rocprofv3 was asked to collect OMPT, so it must hold the OMPT tool role and
    the rocpd database must contain OMPT records."""
    assert (
        _ompt_record_count(rocpd_conn) > 0
    ), "rocprofv3 was run with --ompt-trace but recorded no OMPT records"


def test_other_tool_did_not_get_role(mock_summary):
    """The second tool was offered via OMP_TOOL_LIBRARIES. Because rocprofv3 kept
    the role, the OpenMP runtime never walked that list, so the tool was never even
    loaded and wrote no summary."""
    assert mock_summary is None, (
        "the second OMPT tool ran even though rocprofv3 kept the OMPT tool role: "
        f"{mock_summary}"
    )


def test_rocprofv3_deferred_ompt_role(rocpd_conn):
    """rocprofv3 was not asked to collect OMPT, so it must have declined the role
    and recorded no OMPT."""
    count = _ompt_record_count(rocpd_conn)
    assert (
        count == 0
    ), f"rocprofv3 recorded {count} OMPT record(s) without being asked to trace OMPT"


def test_other_tool_got_role(mock_summary):
    """With rocprofv3 declining, the OpenMP runtime must fall through to its own
    OMP_TOOL_LIBRARIES search and make the second tool the OMPT tool."""
    assert (
        mock_summary is not None
    ), "the second OMPT tool wrote no summary, so the runtime never loaded it"
    assert mock_summary[
        "initialized"
    ], "the second OMPT tool was loaded but never made the OMPT tool"
    assert (
        mock_summary["events"]["thread_begin"] > 0
    ), "the second OMPT tool held the role but received no callbacks"


def test_deferback_rocprofv3_collected_ompt(rocpd_conn):
    """A tool that holds the role may hand it back by calling
    rocprofiler_ompt_start_tool directly. rocprofv3 was asked to collect OMPT, so
    accepting the hand-back must produce the same records as holding the role from
    the start -- even though the hand-back arrives before main(), so neither of the
    usual rocprofv3 initialization points has run."""
    assert (
        _ompt_record_count(rocpd_conn) > 0
    ), "rocprofv3 was handed the OMPT tool role back but recorded no OMPT records"


def test_deferback_without_request_collects_no_ompt(rocpd_conn):
    """The hand-back reaches rocprofiler-sdk directly, bypassing the rocprofv3 entry
    point that consults the command line. It must still decline: another tool cannot
    switch on collection that was never requested."""
    count = _ompt_record_count(rocpd_conn)
    assert count == 0, (
        f"rocprofv3 recorded {count} OMPT record(s) after a hand-back, without being "
        "asked to trace OMPT"
    )


def test_deferback_other_tool_forwarded(mock_summary):
    """The other tool was loaded first and bound by the runtime, then handed the role
    on rather than keeping it, so it ran but never became the OMPT tool itself."""
    assert (
        mock_summary is not None
    ), "the other OMPT tool wrote no summary, so it was never loaded"
    assert not mock_summary[
        "initialized"
    ], "the other OMPT tool kept the role instead of handing it back"


def test_kernel_dispatch_tracing_intact(rocpd_conn):
    """Declining the OMPT role costs rocprofv3 nothing else: the tracing it was
    asked for on the command line must still land in the rocpd database."""
    assert (
        _kernel_dispatch_count(rocpd_conn) > 0
    ), "rocprofv3 was run with --kernel-trace but recorded no kernel dispatches"
