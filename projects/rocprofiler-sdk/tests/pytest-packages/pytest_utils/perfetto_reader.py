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

import os
import re
import sys
import time
import pandas as pd

from collections import OrderedDict
from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig


PerfettoTraceProcessorShellPath = os.path.join(
    os.path.dirname(__file__), "trace_processor_shell"
)


class PerfettoReader:
    """Read in perfetto protobuf output"""

    def __init__(self, filename, select=None, **kwargs):
        """Arguments:
        filename (str or list/tuple of str):
            Valid arguments should be a list of files

        verbose (int):
            Information about processes, threads, categories, performance, etc.

        report (list of str):
            alternative for verbose. Accepts:

            - category (report categories in files)
            - process (report process info in files)
            - threads (report thread info in files)
            - track_ids (report track id info in files)
            - profile (report timing info from processing)

        exclude_category (list of str):
            Every slice has an associated category, e.g., slices from sampling may be
            in "sampling" category, slices from instrumentation may be in "instrumentation"
            category. The categories in protobuf is specific to the tool that generated the
            protobuf. Use this option to exclude the slices from specific categories

        include_category (list of str):
            Every slice has an associated category, e.g., slices from sampling may be
            in "sampling" category, slices from instrumentation may be in "instrumentation"
            category. The categories in protobuf is specific to the tool that generated the
            protobuf. Use this option to restrict the slices to the specified categories

        default_categories (list of str):
            Use this option as a safety value when include/exclude are used. These categories
            are used when include/exclude category filtering resulted in no data in the data
            frame. Accepts "all" if you want to fallback to just including all categories.

        patterns (list of regex str):
            Use this option to specify how the function/file/line are extracted from labels.
            For example, the default patterns are:

                r"(?P<func>.*) \\[(?P<file>\\S+):(?P<line>[0-9]+)\\]$"
                r"(?P<func>.*) \\[(?P<file>\\S+)\\]$"
                r"^(?P<file>\\S+):(?P<line>[0-9]+)$"

            To extract the function/file/line from the labeling patterns:

                func [file:line]
                func [file]
                file:line

            respectively.

        thread_index_regex (regex str):
            Use this option to determine thread IDs from the name of the threads. For example,
            Omnitrace labels certain process tracks "Thread X (S)" to indicate that the given
            process track are samples of Thread X. The default pattern is:

                r"(T|t)hread (?P<thread_index>[0-9]+)( |$)"

            to extract the "thread_index" field.

        max_depth (int):
            Set this a positive non-zero number signifying the maximum call-stack depth to
            process. This can significantly reduce the processing time for large traces.
        """

        self.filename = filename if isinstance(filename, (list, tuple)) else [filename]
        self.metadata = {"hatchet_inclusive_suffix": ".inc"}
        self.default_metric = "time{}".format(self.metadata["hatchet_inclusive_suffix"])
        self.verbose = 0
        self.report = []
        self.exclude = []
        self.include = []
        self.categories = []
        self.default_categories = []
        self.df_categories = []
        self.dataframe = pd.DataFrame()
        self.process = pd.DataFrame()
        self.threads = pd.DataFrame()
        self.track_ids = []
        self.trace_processor = []
        self.compiled_patterns = []
        self.thread_index_regex = None
        self.max_depth = None
        self.configure(**kwargs)

    def configure(self, **kwargs):

        # pre-compile the regex patterns for extracting the func, file, and line info
        # users can use their own pattens via patterns=[...]. An empty set of patterns
        # is valid for avoiding parsing func/file/line info
        _default_patterns = [
            # func [file:line]
            r"(?P<func>.*) \[(?P<file>\S+):(?P<line>[0-9]+)\]$",
            # func [file]
            r"(?P<func>.*) \[(?P<file>\S+)\]$",
            # file:line
            r"^(?P<file>\S+):(?P<line>[0-9]+)$",
        ]
        _patterns = kwargs["patterns"] if "patterns" in kwargs else None

        if _patterns is None:
            _patterns = _default_patterns
        elif kwargs.get("use_default_patterns", True):
            _patterns += _default_patterns

        self.compiled_patterns = [re.compile(x) for x in _patterns]
        self.thread_index_regex = re.compile(
            kwargs.get("thread_index_regex", "(T|t)hread (?P<thread_index>[0-9]+)( |$)")
        )

        def report_at_verbosity(key, lvl):
            if self.verbose >= lvl and key not in self.report:
                self.report.append(key)

        self.verbose = self.verbose if "verbose" not in kwargs else kwargs["verbose"]
        self.report = kwargs["report"] if "report" in kwargs else self.report
        report_at_verbosity("category", 1)
        report_at_verbosity("process", 2)
        report_at_verbosity("threads", 2)
        report_at_verbosity("track_ids", 2)

        _filenames = sorted(self.filename)
        self.filename = kwargs.get("filename", sorted(self.filename))

        _new_filenames = [x for x in self.filename if x not in _filenames]

        _timeout = kwargs.get("timeout", 3)

        def construct_trace_processor(trace_v, timeout_v):
            for i in range(4):
                try:
                    verbosity = True if i > 0 else False
                    cfg = TraceProcessorConfig(verbose=verbosity)
                    if hasattr(cfg, "load_timeout"):
                        cfg.load_timeout = timeout_v + i
                    if hasattr(cfg, "bin_path") and os.path.exists(
                        PerfettoTraceProcessorShellPath
                    ):
                        cfg.bin_path = PerfettoTraceProcessorShellPath
                    return TraceProcessor(trace=(trace_v), config=cfg)
                except Exception as e:
                    nwait = i + 1
                    sys.stderr.write(
                        f"{e}\nRetrying trace processor construction after {nwait} seconds...\n"
                    )
                    sys.stderr.flush()
                    time.sleep(nwait)

            raise RuntimeError(f"Failed to construct trace processor for '{trace_v}'")

        if len(self.filename) + len(_new_filenames) != len(self.trace_processor):
            self.trace_processor = [
                construct_trace_processor(f, _timeout) for f in self.filename
            ]
        elif _new_filenames:
            self.trace_processor += [
                construct_trace_processor(f, _timeout) for f in _new_filenames
            ]

        self.max_depth = kwargs.get("max_depth", None)

    def query_tp(self, query, index_name=lambda x: "tp_index"):
        """Simplifies querying the trace processor and always adds a
        "tp_index" column for referencing which trace_processor
        generated the data
        """

        def _append_column(df, name, idx):
            """Used to add a tp_index column to our data which is used to identify
            which trace-processor the query results came from"""
            if name and name not in df:
                df.insert(0, name, idx)
            return df

        def _get_dataframe(tp):
            """workaround for bug in TraceProcessor.QueryResultIterator.as_pandas_dataframe()"""
            query_itr = tp.query(f"{query}")
            # the perfetto trace processor query function looks like this:
            #
            #     def query(self, sql: str):
            #         response = self.http.execute_query(sql)
            #         if response.error:
            #             raise TraceProcessorException(response.error)
            #         return TraceProcessor.QueryResultIterator(response.column_names,
            #                                                   response.batch)
            #
            # unfortunately, data type of response.column_names is RepeatedScalarContainer
            # and in a lot of versions of pandas, this type does not satisfy any of it's
            # checks for whether this is a valid Index-type for the columns:
            #   isinstance(..., Index)
            #   isinstance(..., ABCSeries)
            #   is_iterator(...)
            #   isinstance(..., list)
            #
            # and thus as_pandas_dataframe() raises a TypeError exception. Queries can
            # be VERY expensive for a large database (>> 10s of seconds) so instead of
            # try -> except -> re-query (necessary) -> convert to list -> re-call
            # as_pandas_dataframe() like so:
            #
            #   try:
            #       return query_itr.as_pandas_dataframe()
            #   except TypeError:
            #       query_itr = tp.query(...)
            #       query_itr.__dict__["..."] = list(query_itr.__dict__["..."])
            #       return query_itr.as_pandas_dataframe()
            #
            # which would effectively result in two queries every single time, we just
            # do it upfront if the dict entry that is known to cause the problem exists
            # and is of the type that we know causes problems
            #
            _buggy_dict_entry = "_QueryResultIterator__column_names"
            if (
                _buggy_dict_entry in query_itr.__dict__
                and type(query_itr.__dict__[_buggy_dict_entry]).__name__
                == "RepeatedScalarContainer"
            ):
                query_itr.__dict__[_buggy_dict_entry] = list(
                    query_itr.__dict__[_buggy_dict_entry]
                )

            return query_itr.as_pandas_dataframe()

        return pd.concat(
            [
                _append_column(
                    _get_dataframe(tp),
                    index_name(idx),
                    idx,
                )
                for idx, tp in enumerate(self.trace_processor)
            ]
        )

    def extract_tp_data(self, **kwargs):
        """Extracts all the necessary data from the trace processor"""
        self.configure(**kwargs)

        # generate empty dictionaries for each trace processor
        self.track_ids = [{} for _ in range(len(self.trace_processor))]

        self.dataframe = self.query_tp(
            "SELECT slice_id, track_id, category, depth, stack_id, parent_stack_id, ts, dur, name FROM slice"
        )

        counter_df = self.query_tp(
            """SELECT
                counter.id as slice_id,
                counter.track_id,
                CASE
                    WHEN counter_track.name LIKE '%SCRATCH MEMORY%' THEN 'scratch_memory'
                    WHEN counter_track.name LIKE '%ALLOCATE BYTES%' THEN 'memory_allocation'
                    ELSE 'counter_collection'
                END as category,
                0 as depth,
                0 as stack_id,
                0 as parent_stack_id,
                counter.ts as ts,
                end.ts - counter.ts as dur,
                counter_track.name as name
            FROM counter AS counter
            JOIN counter AS end ON counter.id + 1 == end.id AND end.track_id == counter.track_id
            JOIN counter_track ON counter.track_id = counter_track.id
            WHERE (counter_track.name LIKE 'AGENT%'
                   OR counter_track.name LIKE '%SCRATCH MEMORY%'
                   OR counter_track.name LIKE '%ALLOCATE BYTES%') AND counter.id % 2 == 0"""
        )

        # Transform counter data to match the main dataframe schema
        if not counter_df.empty:
            # Register counter track IDs in self.track_ids before adding to dataframe
            for row in counter_df.itertuples():
                if (
                    row.tp_index < len(self.track_ids)
                    and row.track_id not in self.track_ids[row.tp_index]
                ):
                    # Add the counter track to track_ids with reasonable default values
                    self.track_ids[row.tp_index][row.track_id] = {
                        "tp_index": row.tp_index,
                        "pid": 0,
                        "tid": 0,
                        "rank": 0,
                        "thread": 0,
                        "prio": 2,
                        "process_name": "counter_process",
                        "thread_name": f"counter_track_{row.category}",
                    }

            # Create a new dataframe with the right columns
            counter_collection_df = pd.DataFrame(
                {
                    "tp_index": counter_df["tp_index"],
                    "slice_id": counter_df["slice_id"],
                    "track_id": counter_df["track_id"],
                    "category": counter_df["category"],
                    "depth": 0,
                    "stack_id": 0,
                    "parent_stack_id": 0,
                    "ts": counter_df["ts"],
                    "dur": 0,
                    "name": counter_df["name"].astype(str),
                }
            )

            # Concatenate with main dataframe
            self.dataframe = pd.concat(
                [self.dataframe, counter_collection_df], ignore_index=True
            )

        self.df_categories = sorted(list(self.dataframe["category"].unique()))

        # check for update to include/exclude category
        self.exclude = kwargs.get("exclude_category", self.exclude)
        self.include = kwargs.get("include_category", self.include)

        self.categories = self.df_categories[:]

        # apply include first
        if self.include:
            self.categories = [x for x in self.categories if x in self.include]

        # apply exclude after
        if self.exclude:
            self.categories = [x for x in self.categories if x not in self.exclude]

        self.default_categories = kwargs.get(
            "default_categories", self.default_categories
        )
        _acceptable_default_categories = (
            'default_categories can be set to: "all", ["all"], or [list of categories...]'
        )

        if not self.categories and self.default_categories:
            if not isinstance(self.default_categories, (tuple, list)):
                self.default_categories = [self.default_categories]

            if "all" in self.default_categories:
                self.categories = self.df_categories[:]
            else:
                raise ValueError(
                    f"invalid default_categories value: {self.default_categories}. {_acceptable_default_categories}"
                )

        # filter out any categories that do not exist
        self.categories = sorted([x for x in self.categories if x in self.df_categories])

        if not self.categories:
            raise ValueError(
                f"The application of include_category={self.include} followed by exclude_category={self.exclude} rendered an empty set of categories (available={self.df_categories}). Either clear one of the configs or assign the default_categories. {_acceptable_default_categories}"
            )

        if "category" in self.report:
            _ignore = [x for x in self.df_categories if x not in self.categories]
            print(
                "categories: {}{}".format(
                    ", ".join(self.categories),
                    " (ignored: {})".format(", ".join(_ignore)) if _ignore else "",
                )
            )

        # reduce the dataframe to given specified category data
        # TODO: adjust the parent stack ids. if <user> category entry is child of <host> category entry, we lose <user> category entry
        self.dataframe = self.dataframe[self.dataframe["category"].isin(self.categories)]

        if self.dataframe.empty:
            raise RuntimeError(
                "category filtering resulted in an empty dataframe. categories: include={}, exclude={}, available={}".format(
                    self.include, self.exclude, self.df_categories
                )
            )

        self.process = self.query_tp(
            "SELECT process.upid AS process_upid, process.id AS process_id, process.pid, process.name AS process_name, process_track.upid as track_upid, process_track.id AS track_id, process_track.parent_id as track_parent_id, process_track.name AS track_name from process JOIN process_track ON process_track.upid = process.upid WHERE process.pid > 0"
        )
        self.threads = self.query_tp(
            "SELECT thread.utid AS thread_utid, thread.id AS thread_id, thread.tid, thread.name as thread_name, thread.is_main_thread, thread_track.id AS track_id, thread_track.parent_id AS track_parent_id, thread_track.name AS track_name from thread JOIN thread_track ON thread_track.utid = thread.utid"
        )

        # generate mapping from track IDs to process and thread info.
        # the "pid" and "tid" fields are the system value. we want to
        # assign a "rank" and "thread" value for "pid" and "tid",
        # respectively which start at zero and monotonically increase
        for thread in self.threads.itertuples():
            _thread_name = (
                thread.thread_name if thread.track_name is None else thread.track_name
            )
            if thread.track_parent_id == 0:
                    self.track_ids[thread.tp_index][thread.track_id] = {
                        "tp_index": thread.tp_index,
                        "pid": 0,
                        "tid": thread.tid,
                        "rank": -1,
                        "thread": -1,
                        "prio": 0 if thread.is_main_thread else 1,
                        "process_name": "",
                        "thread_name": _thread_name,
                    }
            else:
                for process in self.process.itertuples():
                    if process.tp_index != thread.tp_index:
                        continue
                    _process_name = (
                        process.process_name
                        if process.track_name is None
                        else process.track_name
                    )
                    if process.track_id == thread.track_parent_id:
                        self.track_ids[thread.tp_index][thread.track_id] = {
                            "tp_index": thread.tp_index,
                            "pid": process.pid,
                            "tid": thread.tid,
                            "rank": -1,
                            "thread": -1,
                            "prio": 0 if thread.is_main_thread else 1,
                            "process_name": _process_name,
                            "thread_name": _thread_name,
                        }
                        break

        # some track ids do not have an associated system thread id so handle them here.
        # for example, omnitrace post-processes sampling data collected on a thread
        # during finalization and is inserted into perfetto on the main thread
        # but not in the main thread track so perfetto does not associate the
        # track_id with a system thread id.
        for process in self.process.itertuples():
            if process.track_id in self.track_ids[process.tp_index].keys():
                continue
            _process_name = (
                process.track_name
                if process.process_name is None
                else process.process_name
            )
            _thread_name = (
                process.process_name if process.track_name is None else process.track_name
            )
            self.track_ids[process.tp_index][process.track_id] = {
                "tp_index": process.tp_index,
                "pid": process.pid,
                "tid": process.pid,
                "rank": -1,
                "thread": -1,
                "prio": 0 if process.track_parent_id is None else 2,
                "process_name": _process_name,
                "thread_name": _thread_name,
            }

        if "track_ids" in self.report and self.verbose >= 3:
            print("\ntrack ids (original):")
            for idx, _track_ids in enumerate(self.track_ids):
                for key, itr in _track_ids.items():
                    print(f"  {idx:2}:  {key:8} :: {itr}")
            print("")

        # since the protobuf just has raw (system) PID and TIDs and there may be multiple PIDs and TIDs
        # in the same file, we need to map the system PIDs to rank IDs starting at zero and, for each
        # PID, map the system TIDs to thread-ids starting at zero
        pid_offset = 0
        for idx, _track_ids in enumerate(self.track_ids):
            _track_ids = dict(
                sorted(
                    _track_ids.items(),
                    key=lambda x: [x[1]["pid"], x[1]["prio"], x[1]["tid"]],
                )
            )
            pids = list(set([x["pid"] for _, x in _track_ids.items()]))

            if self.verbose >= 3:
                tids = list(set([x["tid"] for _, x in _track_ids.items()]))
                print(f"pids: {pids}")
                print(f"tids: {tids}")

            # assign the rank and then increment the rank offset by the number of PIDs in the file
            for pidx, pid in enumerate(pids):
                for _, itr in _track_ids.items():
                    if itr["pid"] == pid:
                        itr["rank"] = pidx + pid_offset
            pid_offset += len(pids)

            for _, pid in enumerate(pids):
                # dictionary containing only the data for this pid
                _pid_track_ids = dict(
                    [[x, y] for x, y in _track_ids.items() if y["pid"] == pid]
                )

                assigned_track_ids = []

                # filter out the main threads (priority == 0) for a given pid and set index to a value of zero
                main_thr_info = set(
                    [x for x, y in _pid_track_ids.items() if y["prio"] == 0]
                )

                # for known "main" threads, assign index to zero
                for track_id, track_id_data in _track_ids.items():
                    if track_id in main_thr_info:
                        track_id_data["thread"] = 0
                        assigned_track_ids.append(track_id)

                # starting value for assignment. set before next step
                offset = 1 if assigned_track_ids else 0

                # search thread name to try to identify which thread it belongs to.
                # needs to come after offset assignment
                for track_id, track_id_data in _track_ids.items():
                    if (
                        track_id in assigned_track_ids
                        or track_id not in _pid_track_ids.keys()
                    ):
                        continue
                    m = re.search(self.thread_index_regex, track_id_data["thread_name"])
                    if m:
                        track_id_data["thread"] = int(m.group("thread_index"))
                        assigned_track_ids.append(track_id)

                # filter out the non-main threads (priority > 0) for a given pid that haven't already been assigned an index
                chld_thr_info = set(
                    [
                        x
                        for x, y in _pid_track_ids.items()
                        if y["prio"] > 0 and x not in assigned_track_ids
                    ]
                )

                # finally, assign remaining tracks thread indexes via incrementing offset value
                for track_id, track_id_data in _track_ids.items():
                    if (
                        track_id in assigned_track_ids
                        or track_id not in _pid_track_ids.keys()
                    ):
                        continue
                    if track_id in chld_thr_info:
                        track_id_data["thread"] = offset
                        assigned_track_ids.append(track_id)
                        offset += 1

                # make sure the thread indexes are monotonically increasing
                # this may not be the case because of the assignment via regex matching
                _pid_track_ids = dict(
                    [[x, y] for x, y in _track_ids.items() if y["pid"] == pid]
                )
                tidx_max = max([y["thread"] for x, y in _pid_track_ids.items()])
                tidx_uniq = len(
                    set(
                        [
                            y["thread"]
                            for x, y in _pid_track_ids.items()
                            if y["thread"] >= 0
                        ]
                    )
                )

                if self.verbose >= 3:
                    print(f"\nTID :: max={tidx_max}, unique={tidx_uniq}\n")

                # add one to comparison since one thread with a value of 0 would be a size of 1
                while tidx_max + 1 > tidx_uniq:
                    for idx in range(tidx_max):
                        # if this is empty, we need to decrement all thread indexes > idx
                        _tidx_loc = [
                            x for x, y in _pid_track_ids.items() if y["thread"] == idx
                        ]
                        if not _tidx_loc:
                            for itr in _pid_track_ids.keys():
                                if _track_ids[itr]["thread"] > idx:
                                    _track_ids[itr]["thread"] -= 1
                            break
                            # exit the loop so that we recalculate tidx_max
                    tidx_max = max([y["thread"] for x, y in _pid_track_ids.items()])

        if "process" in self.report:
            print(f"\nprocess:\n{self.process.to_string()}\n")
        if "threads" in self.report:
            print(f"\nthreads:\n{self.threads.to_string()}\n")
        if "track_ids" in self.report:
            print("\ntrack ids:")
            for idx, _track_ids in enumerate(self.track_ids):
                for key, itr in _track_ids.items():
                    print(f"  {idx:2}:  {key:8} :: {itr}")
            print("")

        if self.verbose >= 3:
            print("\nTID mapping:")
            for idx, _track_ids in enumerate(self.track_ids):
                for track_id, itr in _track_ids.items():
                    pid = itr["pid"]
                    tid = itr["tid"]
                    pidx = itr["rank"]
                    tidx = itr["thread"]
                    print(
                        f"  {idx:2}:  [{track_id:4}] {pid:8} -> {pidx:8} :: {tid:8} -> {tidx:8}"
                    )

                print("")

    def create_graph(self):
        """Create graph and dataframe"""

        def get_frame_attributes(_name):
            """Get the standard set of dictionary entries for a Frame.
            Also, parses the prefix for func-file-line info
            which is typically in the form:
                <FUNC> [<FILE>:<LINE>]
                <FUNC> [<FILE>]
            """

            if not self.compiled_patterns:
                return {"type": "function", "name": _name}

            def _process_regex(_data):
                """Process the regex data for func/file/line info"""
                return _data.groupdict() if _data is not None else None

            def _perform_regex(_name):
                """Performs a search for standard configurations of function + file + line"""
                for _pattern in self.compiled_patterns:
                    _tmp = _process_regex(re.search(_pattern, _name))
                    if _tmp:
                        return _tmp
                return None

            _keys = {"type": "region", "name": _name}
            _extra = {"file": "<unknown>", "line": "0"}
            _pdict = _perform_regex(_name)
            if _pdict is not None:
                _func = _pdict.get("func", None)
                _file = _pdict.get("file", "<unknown>")
                _line = _pdict.get("line", "0")
                _head = _pdict.get("head", None)
                _tail = _pdict.get("tail", None)

                _line_s = f":{_line}" if int(_line) > 0 else ""
                _tail_s = f"/{_tail}" if _tail is not None else ""
                _file_s = f"{_file}{_line_s}" if _file != "<unknown>" else _file

                _extra["file"] = _file_s
                _extra["line"] = _line

                if "head" in _pdict:
                    _keys["name"] = _head.rstrip()
                    if _func is not None:
                        _extra["func"] = _func
                else:
                    if _func is not None:
                        _keys["name"] = _func
                    else:
                        _keys["name"] = _file_s

                _keys["name"] = "{}{}".format(_keys["name"], _tail_s)

            return (_keys, _extra)

        # list_roots = []
        track_id_dict = OrderedDict()
        callpath_to_node = {}

        def_metric = self.default_metric

        df = self.dataframe
        _cols = [
            "tp_index",
            "track_id",
            "category",
            "slice_id",
            "stack_id",
            "parent_stack_id",
            "name",
            "depth",
            "ts",
            "dur",
        ]
        _data = [df[x].to_list() for x in _cols]

        assert min([len(x) for x in _data]) == max([len(x) for x in _data])

        for _tp_index, _track_id in zip(
            _data[_cols.index("tp_index")], _data[_cols.index("track_id")]
        ):
            assert _tp_index < len(self.track_ids)
            assert _track_id in self.track_ids[_tp_index].keys()

            _track_info = self.track_ids[_tp_index][_track_id]
            _rank = _track_info["rank"]
            _thread = _track_info["thread"]

            if _tp_index not in track_id_dict:
                track_id_dict[_tp_index] = OrderedDict()
            if _rank not in track_id_dict[_tp_index]:
                track_id_dict[_tp_index][_rank] = OrderedDict()
            if _thread not in track_id_dict[_tp_index][_rank]:
                track_id_dict[_tp_index][_rank][_thread] = OrderedDict()

            track_id_dict[_tp_index][_rank][_thread] = {0: None}

        for (
            _tp_index,
            _track_id,
            _category,
            _slice_id,
            _stack_id,
            _parent_stack_id,
            _name,
            _depth,
            _ts,
            _dur,
        ) in zip(*_data):
            _track_info = self.track_ids[_tp_index][_track_id]
            _rank = _track_info["rank"]
            _thread = _track_info["thread"]

            _track_id_dict = track_id_dict[_tp_index][_rank][_thread]

            # removed because of filtering
            if _parent_stack_id not in _track_id_dict:
                continue

            # reduce processing time
            if self.max_depth is not None and _depth > self.max_depth:
                continue

            _metrics = {}
            _metrics["rank"] = _track_info["rank"]
            _metrics["thread"] = _track_info["thread"]
            _metrics["pid"] = _track_info["pid"]
            _metrics["tid"] = _track_info["tid"]
            _metrics["track_id"] = _track_id
            _metrics["slice_id"] = _slice_id
            _metrics["stack_id"] = _stack_id
            _metrics["parent_stack_id"] = _parent_stack_id
            _metrics["ts"] = _ts
            _metrics[def_metric] = float(_dur) * 1.0e-9  # nsec -> sec

            _frame_attrs, _extra = get_frame_attributes(_name)
            _extra["tp_index"] = _tp_index
            _extra["category"] = _category
            _extra["depth"] = _depth

            # look up the parent node specific to the TP index, rank, and thread
            # stack ID is assigned by perfetto and parent stack ID is the
            # stack ID of it's parent.
            # _parent_node = _track_id_dict[_parent_stack_id]

            # hnode = Node(Frame(_frame_attrs, **_extra), None)

            # if _parent_node:
            #     _parent_node.add_child(hnode)
            # else:
            #     list_roots.append(hnode)

            # # make sure this stack ID is unique for the
            # # TP index, rank, and thread and is equal to a
            # # previously seen node with the same stack ID
            # if _stack_id not in _track_id_dict:
            #     _track_id_dict[_stack_id] = hnode
            # elif _track_id_dict[_stack_id] != hnode:
            #     _existing = _track_id_dict[_stack_id]
            #     raise RuntimeError(
            #         f"{_stack_id} already exists in track_id_dict[{_tp_index}][{_rank}][{_thread}]. failed to set:\n  {hnode.frame} (current)\n    {_existing.frame} (existing)"
            #     )

            _hash = hash((_tp_index, _slice_id, _track_id))
            if _hash in callpath_to_node:
                raise ValueError(f"{_hash} already exists in callpath_to_node dict")

            _frame_attrs.pop("type")  # should not be a column in dataframe
            callpath_to_node[_hash] = dict(
                **_frame_attrs,
                **_metrics,
                **_extra,
            )

        if not callpath_to_node:
            raise RuntimeError(
                "call-graph is empty. if category filtering was used, you may have filtered out all the root nodes and thus all of it's children"
            )

        return (df, callpath_to_node)
        # graph, dataframe = graphframe_indexing_helper(
        #     list_roots,
        #     data=list(callpath_to_node.values()),
        #     extensions=["rank", "thread"],
        #     fill_value=0,
        # )

        # inc_metrics = [self.default_metric]
        # exc_metrics = []
        # return (graph, dataframe, exc_metrics, inc_metrics, self.default_metric)

    def read(self, **kwargs):
        """Read perfetto json."""

        self.configure(**kwargs)
        self.extract_tp_data(**kwargs)

        (
            dataframe,
            callpath_to_node,
        ) = self.create_graph()

        # def _read(**kwargs):
        #     return self.read(**kwargs)

        # def _configure(**kwargs):
        #     self.configure(**kwargs)

        # def _query(*args, **kwargs):
        #     assert self.trace_processor
        #     return self.query_tp(*args, **kwargs)

        return (
            dataframe,
            callpath_to_node,
        )
        #     default_metric=def_metric,
        #     metadata=self.metadata,
        #     attributes={
        #         "reader": self,
        #         "read": _read,
        #         "query": _query,
        #         "configure": _configure,
        #         "selected_categories": lambda: self.categories,
        #         "available_categories": lambda: self.df_categories,
        #     },
        # )
