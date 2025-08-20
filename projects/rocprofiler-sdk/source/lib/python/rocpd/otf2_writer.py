###############################################################################
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
###############################################################################

import os
import collections


class Location:
    def __init__(self, pid, tid, agent_handle=0, loc_type=0, queue_id=0):
        self.pid = pid
        self.tid = tid
        self.agent_handle = agent_handle
        self.type = loc_type
        self.queue_id = queue_id
        self.index = None  # Assigned during session setup

    def key(self):
        return (self.pid, self.tid, self.agent_handle, self.type, self.queue_id)


class Region:
    def __init__(self, name, role, paradigm):
        self.name = name
        self.role = role
        self.paradigm = paradigm


class Event:
    def __init__(self, name, location, phase, timestamp, attributes=None):
        self.name = name
        self.location = location
        self.phase = phase
        self.timestamp = timestamp
        self.attributes = attributes or {}


class OTF2Session:
    def __init__(self, trace_dir, config, min_start, max_fini):
        self.trace_dir = trace_dir
        self.config = config
        self.min_start = min_start
        self.max_fini = max_fini
        self.locations = collections.OrderedDict()
        self.regions = {}
        self.events = []
        self.strings = {}
        self.attributes = {}
        self.opened = False
        self._setup()

    def _setup(self):
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)
        self.opened = True

    def add_location(self, location):
        key = location.key()
        if key not in self.locations:
            location.index = len(self.locations)
            self.locations[key] = location
        return self.locations[key]

    def add_region(self, name, role, paradigm):
        if name not in self.regions:
            self.regions[name] = Region(name, role, paradigm)
        return self.regions[name]

    def add_event(self, event):
        self.events.append(event)

    def add_string(self, string):
        if string not in self.strings:
            self.strings[string] = len(self.strings) + 1
        return self.strings[string]

    def add_attribute(self, name, value):
        if name not in self.attributes:
            self.attributes[name] = value
        return self.attributes[name]

    def close(self):
        self._write_definitions()
        self._write_events()
        self.opened = False

    def _write_definitions(self):
        def_file = os.path.join(self.trace_dir, "definitions.otf2")
        with open(def_file, "w", encoding="utf-8") as f:
            for name, region in self.regions.items():
                f.write(f"REGION {region.name} {region.role} {region.paradigm}\n")
            for key, loc in self.locations.items():
                f.write(
                    f"LOCATION {loc.index} {loc.pid} {loc.tid} {loc.agent_handle} {loc.type} {loc.queue_id}\n"
                )
            for s, idx in self.strings.items():
                f.write(f"STRING {idx} {s}\n")
            for attr, val in self.attributes.items():
                f.write(f"ATTR {attr} {val}\n")

    def _write_events(self):
        event_file = os.path.join(self.trace_dir, "events.otf2")
        with open(event_file, "w", encoding="utf-8") as f:
            # Sort events by timestamp, then phase, then location
            self.events.sort(key=lambda e: (e.timestamp, e.phase, e.location.index))
            for event in self.events:
                attr_str = " ".join(f"{k}:{v}" for k, v in event.attributes.items())
                f.write(
                    f"EVENT {event.name} {event.location.index} {event.phase} {event.timestamp} {attr_str}\n"
                )


# Example: ported write_otf2 logic
def write_otf2(importData, config):
    # Extract config values
    trace_dir = config.get("output_path", "./otf2_trace")
    min_start = getattr(importData, "min_start", 0)
    max_fini = getattr(importData, "max_fini", 0)
    session = OTF2Session(trace_dir, config, min_start, max_fini)

    # Example: iterate over threads, regions, events, etc.
    # Replace with real importData structure
    for thread in getattr(importData, "threads", []):
        loc = Location(thread["pid"], thread["tid"])
        session.add_location(loc)
        session.add_region(f"Thread {thread['tid']}", "FUNCTION", "HIP")
        session.add_event(
            Event(f"Thread {thread['tid']}", loc, "ENTER", thread["start"], {})
        )
        session.add_event(
            Event(f"Thread {thread['tid']}", loc, "EXIT", thread["end"], {})
        )

    # Example: add strings and attributes
    session.add_string("category")
    session.add_attribute("category", "tracing category")

    # Add more events/regions/locations as needed from importData

    session.close()
