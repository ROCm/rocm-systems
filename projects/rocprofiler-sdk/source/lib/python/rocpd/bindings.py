#!/usr/bin/env python3
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


__all__ = [
    "RocpdImportData",
    "connect",
    "supported",
    "read_agents",
    "read_nodes",
    "read_processes",
    "read_threads",
]

import os
import sqlite3

try:
    # support disabling the python bindings via environment variable
    rocpd_disable_bindings = os.environ.get("ROCPD_DISABLE_BINDINGS", None)
    if rocpd_disable_bindings is not None and f"{rocpd_disable_bindings}".lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        raise ImportError(
            "rocpd python bindings explicitly disabled via ROCPD_DISABLE_BINDINGS"
        )

    # import the compiled pybind11 extensions
    from . import libpyrocpd

    def supported():
        """Check if the rocpd python bindings are available."""
        return True

    # class defined by the bindings
    RocpdImportData = libpyrocpd.RocpdImportData
    output_config = libpyrocpd.output_config
    agent_indexing = libpyrocpd.agent_indexing
    schema_jinja_variables = libpyrocpd.schema_jinja_variables
    metadata = libpyrocpd.metadata
    Thread = libpyrocpd.Thread
    Process = libpyrocpd.Process
    Node = libpyrocpd.Node
    Agent = libpyrocpd.Agent

    # enumerations defined by the bindings
    agent_indexing = libpyrocpd.agent_indexing
    sql_option = libpyrocpd.sql_option
    sql_schema = libpyrocpd.sql_schema
    sql_engine = libpyrocpd.sql_engine

    # functions defined by the bindings
    connect = libpyrocpd.connect
    format_path = libpyrocpd.format_path
    read_agents = libpyrocpd.read_agents
    read_nodes = libpyrocpd.read_nodes
    read_processes = libpyrocpd.read_processes
    read_threads = libpyrocpd.read_threads
    write_csv = libpyrocpd.write_csv
    write_perfetto = libpyrocpd.write_perfetto
    write_otf2 = libpyrocpd.write_otf2
    load_schema = libpyrocpd.load_schema

except ImportError:
    from typing import Iterator, Any, Mapping, Sequence, Type

    def supported():
        """Check if the rocpd python bindings are available."""
        return False

    # Function for connecting to the database
    connect = sqlite3.connect

    class RocpdImportData:
        """Fallback class replicating the interface of libpyrocpd.RocpdImportData."""

        def __init__(self, connection, databases=[]):
            if isinstance(connection, RocpdImportData):
                assert (
                    not databases
                ), f"Copy __init__ should not pass list of databases: {databases}"
                self.connection = connection.connection
                self.databases = connection.databases[:]
            else:
                self.connection = connection
                self.databases = (
                    list(databases)
                    if isinstance(databases, (list, tuple))
                    else [databases]
                )

        def empty(self):
            return not self.connection or not self.databases

        def size(self):
            return len(self.databases) if self.connection else 0

        def __len__(self):
            return self.size()

    def _unsupported_function(*args, **kwargs):
        raise RuntimeError("function requires rocpd Python bindings")

    format_path = _unsupported_function
    write_csv = _unsupported_function
    write_perfetto = _unsupported_function
    write_otf2 = _unsupported_function

    class RocpdNamespace:
        def __init__(self, data: dict, allowed_data: dict | None = None):
            """Initialize the RocpdNamespace with data and optional allowed_data filter."""
            if not isinstance(data, dict):
                raise TypeError("Expected a dict")

            for k, v in data.items():
                if allowed_data is None:
                    setattr(self, k, v)
                elif k in allowed_data:
                    setattr(self, allowed_data[k], v)
                elif k == "extdata":
                    import json

                    jdata = json.loads(v) if v else {}
                    if isinstance(jdata, dict):
                        for jk, jv in jdata.items():
                            if jk in allowed_data and jk not in data:
                                setattr(self, allowed_data[jk], jv)

        def __repr__(self):
            return f"{self.__class__.__name__}({self.__dict__})"

    class Agent(RocpdNamespace):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Node(RocpdNamespace):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Process(RocpdNamespace):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Thread(RocpdNamespace):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def _query_as_objects(
        conn,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
        factory: Type[RocpdNamespace] = RocpdNamespace,
        allowed_data: dict | None = None,
    ) -> Iterator[RocpdNamespace]:
        """
        Execute `query` and yield one `factory(dict)` per row without altering `conn.row_factory`.
        """
        cur = conn.execute(query, params or [])
        cols = [d[0] for d in cur.description]
        print(f"Allowed Data: {allowed_data}")
        for row in cur:
            # row is a tuple; pair with column names
            data = {col: row[i] for i, col in enumerate(cols)}
            yield factory(data=data, allowed_data=allowed_data)

    def _fetch_all_objects(
        conn,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
        factory: Type[RocpdNamespace] = RocpdNamespace,
        allowed_data: dict | None = None,
    ) -> list[RocpdNamespace]:
        """Return a list instead of a generator."""
        return list(
            _query_as_objects(
                conn, query, params=params, factory=factory, allowed_data=allowed_data
            )
        )

    def read_agents(conn, conditions):
        """Read agent data from the database."""
        query = (
            "SELECT * FROM rocpd_info_agent WHERE {}".format(conditions)
            if conditions
            else "SELECT * FROM rocpd_info_agent"
        )
        allowed_data = dict(
            (itr, itr)
            for itr in [
                "id",
                "guid",
                "nid",
                "pid",
                "node_id",
                "absolute_index",
                "logical_index",
                "type_index",
                "gpu_index",
                "name",
                "generic_name",
                "model_name",
                "product_name",
                "vendor_name",
            ]
        )
        return _fetch_all_objects(conn, query, factory=Agent, allowed_data=allowed_data)

    def read_nodes(conn, conditions):
        """Read node data from the database."""
        query = (
            "SELECT * FROM rocpd_info_node WHERE {}".format(conditions)
            if conditions
            else "SELECT * FROM rocpd_info_node"
        )
        allowed_data = dict(
            (itr, itr)
            for itr in [
                "id",
                "guid",
                "hash",
                "machine_id",
                "hostname",
                "system_name",
                "system_release",
                "system_version",
            ]
        )
        return _fetch_all_objects(conn, query, factory=Node, allowed_data=allowed_data)

    def read_processes(conn, conditions):
        """Read process data from the database."""
        query = (
            "SELECT * FROM processes WHERE {}".format(conditions)
            if conditions
            else "SELECT * FROM processes"
        )
        allowed_data = dict(
            (itr, itr)
            for itr in [
                "id",
                "guid",
                "nid",
                "machine_id",
                "hostname",
                "system_name",
                "system_release",
                "system_version",
                "ppid",
                "pid",
                "init",
                "start",
                "end",
                "fini",
                "command",
            ]
        )
        return _fetch_all_objects(conn, query, factory=Process, allowed_data=allowed_data)

    def read_threads(conn, conditions):
        """Read thread data from the database."""
        query = (
            "SELECT * FROM threads WHERE {}".format(conditions)
            if conditions
            else "SELECT * FROM threads"
        )
        allowed_data = dict(
            (itr, itr)
            for itr in [
                "id",
                "guid",
                "nid",
                "machine_id",
                "hostname",
                "system_name",
                "system_release",
                "system_version",
                "ppid",
                "tid",
                "start",
                "end",
                "name",
            ]
        )
        return _fetch_all_objects(conn, query, factory=Process, allowed_data=allowed_data)
