#!/usr/bin/env python3

import re
import sys
import pytest
import json


class TimeWindow(object):

    def __init__(self, beg, end):
        self.offset = beg
        self.duration = end - beg

    def in_region(self, val):
        return val >= self.offset and val <= (self.offset + self.duration)

    def __repr__(self):
        return f"[{self.offset}:{self.offset+self.duration}]"


def check_traces(data, valid_regions, invalid_regions, corrid_records=None):
    for record in data:
        corr_id = record.correlation_id.internal
        rval = (
            corrid_records[corr_id]
            if corrid_records is not None and corr_id in corrid_records
            else record
        )
        valid = [itr for itr in valid_regions if itr.in_region(rval.start_timestamp)]
        assert (
            len(valid) == 1
        ), f"\nrval:\n\t{rval}\nrecord:\n\t{record}\nnot found in valid regions:\n{valid_regions}"

        invalid = [itr for itr in invalid_regions if itr.in_region(rval.start_timestamp)]
        assert (
            len(invalid) == 0
        ), f"\nrval:\n\t{rval}\nrecord:\n\t{record}\nfound in invalid region(s):\n{invalid}"


def test_collection_period_trace(json_data, collection_period_data):
    # Adding 20 us error margin to handle the time taken for the start/stop context to affect the collection
    time_error_margin = 20 * 1e4
    valid_regions = []
    invalid_regions = []
    for period in collection_period_data:
        _start = None
        _stop = None
        if "start" in period.keys():
            _start = period.start.start - time_error_margin
        if "stop" in period.keys():
            _stop = period.stop.stop + time_error_margin

        if _start and _stop:
            valid_regions.append(TimeWindow(_start, _stop))
        elif "duration" in period.keys():
            valid_regions.append(TimeWindow(period.duration.start, period.duration.stop))
        elif _start and not _stop:
            valid_regions.append(TimeWindow(_start, _start + 10e9))  # add 10 seconds

        if "delay" in period.keys():
            invalid_regions.append(TimeWindow(period.delay.start, period.delay.stop))

    data = json_data["rocprofiler-sdk-tool"]
    corrid_records = {}

    for itr in ["hsa_api", "hip_api", "marker_api", "rccl_api"]:
        grp = data.buffer_records[itr]
        check_traces(grp, valid_regions, invalid_regions)
        for record in grp:
            corrid_records[record.correlation_id.external] = record


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_data(
        pftrace_data, json_data, ("hip", "hsa", "marker", "kernel", "memory_copy")
    )


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_data(
        otf2_data, json_data, ("hip", "hsa", "marker", "kernel", "memory_copy")
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
