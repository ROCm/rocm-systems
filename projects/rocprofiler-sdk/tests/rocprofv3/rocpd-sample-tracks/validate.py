#!/usr/bin/env python3

# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Validate that rocpd convert -f pftrace correctly emits Perfetto counter tracks
for PMC sample data stored in a rocpd schema 3.0 database.

Expected counter tracks (produced by perfetto.cpp sample_counter_tracks path):
  "CPU Memory Usage (S)"  — symbol=MemUsg, 20 samples, values in MB
  "GPU Temperature (S)"   — symbol=Temp,   20 samples, values in °C
"""

import sys
import pytest

_CPU_TRACK_NAME = "CPU Memory Usage (S)"
_GPU_TRACK_NAME = "GPU Temperature (S)"

_COUNTER_TRACKS_SQL = """\
SELECT DISTINCT counter_track.name AS track_name
FROM counter
JOIN counter_track ON counter.track_id = counter_track.id
WHERE counter_track.name LIKE '% (S)'
"""


def test_sample_counter_tracks_exist(pftrace_reader):
    """At least two (S)-suffixed counter tracks must be present in the pftrace."""
    df = pftrace_reader.query_tp(_COUNTER_TRACKS_SQL)
    track_names = df["track_name"].tolist()
    assert (
        len(track_names) >= 2
    ), f"Expected at least 2 '(S)' counter tracks, found {len(track_names)}: {track_names}"
    assert (
        _CPU_TRACK_NAME in track_names
    ), f"'{_CPU_TRACK_NAME}' not found in counter tracks: {track_names}"
    assert (
        _GPU_TRACK_NAME in track_names
    ), f"'{_GPU_TRACK_NAME}' not found in counter tracks: {track_names}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
