# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import otf2
import pandas as pd

from otf2.events import Enter, Leave


class Region(object):
    """ """

    def __init__(self, _enter, _leave, _depth, _location=None):
        if _enter.region != _leave.region:
            _location_info = f"\n{_location}" if _location else ""
            raise ValueError(
                f"enter region != leave region :: '{_enter}' != '{_leave}'{_location_info}"
            )

        if _depth < 0:
            _location_info = f". location: '{_location}'" if _location else ""
            raise ValueError(
                f"negative depth ({_depth})! enter: '{_enter}'. leave: '{_leave}'{_location_info}"
            )

        self.region = _enter.region
        self.depth = _depth
        self.name = _enter.region.name
        self.attributes = [
            itr for itr in [_enter.attributes, _leave.attributes] if itr is not None
        ]
        self.enter_nsec = _enter.time
        self.leave_nsec = _leave.time
        self.delta_nsec = _leave.time - _enter.time
        if self.delta_nsec < 0:
            raise ValueError(
                f"negative timestamp delta :: '{_enter.time}' > '{_leave.time}'"
            )

        for itr in self.attributes:
            for key, val in itr.items():
                _key = f"{key.name}"
                if not hasattr(self, _key):
                    setattr(self, _key, val)

        if not hasattr(self, "category"):
            self.category = "unk"

    def __str__(self):
        return f"{self.name:<35} :: {self.delta_nsec} nsec"


class OTF2Reader:
    """Read in perfetto protobuf output"""

    def __init__(self, filename):
        self.filename = filename if isinstance(filename, (list, tuple)) else [filename]

    def read(self):
        def _read_trace(trace_name):
            trace = otf2.reader.Reader(trace_name)
            # print(f"Read {len(trace.definitions.strings)} string definitions")
            # for string in trace.definitions.strings:
            #     print(f"String definition with value '{string}' in trace.")
            # print("Read {} events".format(len(trace.events)))

            events = [[loc, evt] for loc, evt in trace.events]
            locations = [itr for itr in trace.definitions.locations]
            location_groups = [itr for itr in trace.definitions.location_groups]
            system_tree_nodes = [itr for itr in trace.definitions.system_tree_nodes]

            call_stack = {}
            partial_call_stack = {}

            for itr in system_tree_nodes:
                call_stack[itr] = {}
                partial_call_stack[itr] = {}

            for itr in location_groups:
                call_stack[itr.system_tree_parent][itr] = {}
                partial_call_stack[itr.system_tree_parent][itr] = {}

            for itr in locations:
                call_stack[itr.group.system_tree_parent][itr.group][itr] = []
                partial_call_stack[itr.group.system_tree_parent][itr.group][itr] = []

            for location, event in events:
                _stree = location.group.system_tree_parent
                _group = location.group
                _partial = partial_call_stack[_stree][_group][location]
                if isinstance(event, Enter):
                    # expected length
                    _elen = len(_partial) + 1
                    _partial.append(event)
                elif isinstance(event, Leave):
                    # expected length
                    _elen = len(_partial) - 1
                    _depth = len(_partial)
                    _leave = event

                    # it appears that on MI300, the end of A may exceed the
                    # begin of B kernels very slightly (i.e. overlap in same
                    # stream/queue). This leads to slightly out of order
                    # Enter/Leave regions and thus we need to occasionally
                    # search further back in the callstack to find the correct
                    # Enter region
                    _enter = _partial[-1]
                    if _enter.region == _leave.region:
                        _partial.pop()
                    else:
                        for ridx, ritr in enumerate(reversed(_partial)):
                            if ritr.region == _leave.region:
                                _enter = _partial.pop(len(_partial) - ridx - 1)
                                break

                    # below is what is expected on non-MI300
                    # _enter = _partial.pop()

                    # add the region
                    call_stack[_stree][_group][location].append(
                        Region(_enter, _leave, _depth - 1, location)
                    )

                # modified length
                _mlen = len(partial_call_stack[_stree][_group][location])
                # if modified length != expected length
                if _mlen != _elen:
                    raise RuntimeError(
                        f"Modified length ({_mlen}) != Expected length({_elen}) for {event} at {location}"
                    )

            data = {
                "system_tree_node": [],
                "location_group": [],
                "location": [],
                "region": [],
                "attributes": [],
                "depth": [],
                "name": [],
                "category": [],
                "start_ts": [],
                "end_ts": [],
            }

            for tree, lgitr in call_stack.items():
                for group, gitr in lgitr.items():
                    for loc, ritr in gitr.items():
                        for region in ritr:
                            data["system_tree_node"] += [tree]
                            data["location_group"] += [group]
                            data["location"] += [loc]
                            data["region"] += [region.region]
                            data["attributes"] += [region.attributes]
                            data["depth"] += [region.depth]
                            data["category"] += [region.category]
                            data["name"] += [region.name]
                            data["start_ts"] += [region.enter_nsec]
                            data["end_ts"] += [region.leave_nsec]

            return (trace, pd.DataFrame.from_dict(data))

        readers = []
        df = pd.DataFrame()
        for itr in self.filename:
            _reader, _df = _read_trace(itr)
            readers += [_reader]
            df = pd.concat([df, _df])

        return (df, readers)


def read_trace(filename):
    data = OTF2Reader(filename).read()[0]

    print(f"\nDATA:\n{data}")

    attributes = list(data["attributes"])

    print(f"\nATTRIBUTES:\n{attributes}")
