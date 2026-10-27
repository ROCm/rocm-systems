###############################################################################
# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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
    "load_schema",
    "query_supported_schema_versions",
    "schema_version",
    "output_config",
    "read_agents",
    "read_nodes",
    "read_processes",
    "read_threads",
]

import os
import sqlite3

# ---------------------------------------------------------------------------
# Pure-python schema loading (mirrors rocpd_sql_load_schema and
# rocpd_query_supported_schema_versions in librocprofiler-sdk-rocpd). These are
# defined at module scope so they can be compared against libpyrocpd in tests.
# ---------------------------------------------------------------------------

# values of sql_schema enum -> key in versions.yml
_SCHEMA_KIND_KEYS = {
    1: "rocpd_tables",
    2: "rocpd_indexes",
    3: "rocpd_views",
    4: "rocpd_data_views",
    5: "rocpd_summary_views",
    6: "rocpd_metadata",
}

_SQLITE3_PRAGMA_FOREIGN_KEYS = 0x02


def _version_tuple(version):
    """Convert str / schema_version-like / (major, minor, patch) into a tuple of ints."""
    if isinstance(version, str):
        parts = [int(x) for x in version.split(".") if x != ""][:3]
        return tuple(parts + [0] * (3 - len(parts)))
    if hasattr(version, "major"):
        return (int(version.major), int(version.minor), int(version.patch))
    return tuple(int(x) for x in version)


def _find_share_path():
    """Locate <prefix>/share/rocprofiler-sdk-rocpd relative to this file."""
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "share" / "rocprofiler-sdk-rocpd"
        if (candidate / "versions.yml").exists():
            return str(candidate)
    return ""


def _schema_search_paths(hints=None):
    paths = [str(x) for x in (hints or [])]
    paths += os.environ.get("ROCPD_SCHEMA_PATH", "").split(":")
    paths += [_find_share_path()]
    return [p for p in paths if p]


def _load_version_file_map(paths):
    """Returns {version_str: {kind_key: relative_file}} from the first versions.yml found."""
    import yaml

    for itr in paths:
        fpath = os.path.join(itr, "versions.yml")
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                data = yaml.safe_load(f)
            result = {}
            for entry in data["rocprofiler-sdk-rocpd"]["rocpd_schemas"]:
                version = str(entry["version"])
                result[version] = {
                    key: entry[key]
                    for key in _SCHEMA_KIND_KEYS.values()
                    if key in entry
                }
            return result
    return {}


def _py_query_supported_schema_versions(engine=1, schema_path_hints=None):
    """Returns sorted list of (major, minor, patch) tuples."""
    if int(engine) != 1:
        raise RuntimeError("unsupported rocpd SQL engine")
    vmap = _load_version_file_map(_schema_search_paths(schema_path_hints))
    return sorted(_version_tuple(v) for v in vmap)


def _py_load_schema(engine, kind, options, version, variables):
    """Pure-python equivalent of libpyrocpd.load_schema."""
    if int(engine) != 1:
        raise RuntimeError("unsupported rocpd SQL engine")

    paths = _schema_search_paths()
    vmap = _load_version_file_map(paths)
    if not vmap:
        raise RuntimeError("rocpd schema versions.yml not found")

    vtuple = _version_tuple(version)
    if vtuple == (0, 0, 0):
        vtuple = max(_version_tuple(v) for v in vmap)
    vkey = ".".join(str(x) for x in vtuple)
    if vkey not in vmap:
        raise RuntimeError(f"rocpd schema version {vkey} is not supported")

    kind_key = _SCHEMA_KIND_KEYS.get(int(kind))
    if kind_key is None or kind_key not in vmap[vkey]:
        raise RuntimeError(f"rocpd schema kind {kind} not available for version {vkey}")

    contents = None
    for itr in paths:
        fpath = os.path.join(itr, vmap[vkey][kind_key])
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                contents = f.read()
            break
    if not contents:
        raise RuntimeError(f"rocpd schema file for {kind_key} {vkey} not found or empty")

    if int(options) & _SQLITE3_PRAGMA_FOREIGN_KEYS:
        contents = "PRAGMA foreign_keys = ON;\n\n" + contents

    uuid = getattr(variables, "uuid", None)
    if uuid is not None:
        uuid = str(uuid)
        if uuid and not uuid.startswith("_"):
            uuid = "_" + uuid
        uuid = uuid.replace("-", "_")
        contents = contents.replace("{{uuid}}", uuid)

    guid = getattr(variables, "guid", None)
    substitutions = {
        "{{guid}}": None if guid is None else str(guid),
        "{{schema_version}}": vkey,
        "{{schema_version_major}}": str(vtuple[0]),
        "{{schema_version_minor}}": str(vtuple[1]),
        "{{schema_version_patch}}": str(vtuple[2]),
    }
    for key, value in substitutions.items():
        if value is not None:
            contents = contents.replace(key, value)

    return contents


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

    # classes defined by the bindings
    RocpdImportData = libpyrocpd.RocpdImportData
    output_config = libpyrocpd.output_config
    schema_jinja_variables = libpyrocpd.schema_jinja_variables
    schema_version = libpyrocpd.schema_version
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
    write_perfetto = libpyrocpd.write_perfetto
    load_schema = libpyrocpd.load_schema
    query_supported_schema_versions = libpyrocpd.query_supported_schema_versions

except ImportError:
    from typing import Iterator, Any, Mapping, Sequence, Type
    import enum

    def supported():
        """Check if the rocpd python bindings are available."""
        return False

    # Function for connecting to the database
    connect = sqlite3.connect

    class schema_version:
        def __init__(self, *args):
            if len(args) == 0:
                self.major, self.minor, self.patch = 0, 0, 0
            elif len(args) == 3:
                self.major = int(args[0])
                self.minor = int(args[1])
                self.patch = int(args[2])
            elif len(args) == 1:
                version = args[0]
                if isinstance(version, str):
                    parts = version.split(".")
                    self.major = int(parts[0]) if len(parts) > 0 else 0
                    self.minor = int(parts[1]) if len(parts) > 1 else 0
                    self.patch = int(parts[2]) if len(parts) > 2 else 0
                elif isinstance(version, schema_version):
                    self.major = version.major
                    self.minor = version.minor
                    self.patch = version.patch
                else:
                    self.major = version.major
                    self.minor = version.minor
                    self.patch = version.patch
            else:
                raise TypeError(
                    f"schema_version() takes 0, 1, or 3 arguments ({len(args)} given)"
                )

        def __str__(self):
            return f"{self.major}.{self.minor}.{self.patch}"

        def __repr__(self):
            return f"schema_version({self.major}, {self.minor}, {self.patch})"

        def _as_tuple(self):
            return (self.major, self.minor, self.patch)

        @staticmethod
        def _to_tuple(obj):
            if hasattr(obj, "_as_tuple"):
                return obj._as_tuple()
            return (obj.major, obj.minor, obj.patch)

        def __eq__(self, other):
            return self._as_tuple() == self._to_tuple(other)

        def __ne__(self, other):
            return self._as_tuple() != self._to_tuple(other)

        def __lt__(self, other):
            return self._as_tuple() < self._to_tuple(other)

        def __gt__(self, other):
            return self._as_tuple() > self._to_tuple(other)

        def __le__(self, other):
            return self._as_tuple() <= self._to_tuple(other)

        def __ge__(self, other):
            return self._as_tuple() >= self._to_tuple(other)

    class RocpdImportData:
        """Fallback class replicating the interface of libpyrocpd.RocpdImportData."""

        def __init__(
            self,
            connection=None,
            databases=[],
            schema_version=None,
            supported_features=[],
            _sv_cls=schema_version,
        ):
            if connection is None:
                self.connection = None
                self.databases = []
                self.schema_version = _sv_cls()
                self.supported_features = []
            elif isinstance(connection, RocpdImportData):
                assert (
                    not databases
                ), f"Copy __init__ should not pass list of databases: {databases}"
                self.connection = connection.connection
                self.databases = connection.databases[:]
                self.schema_version = connection.schema_version
                self.supported_features = list(connection.supported_features)
            else:
                self.connection = connection
                self.databases = (
                    list(databases)
                    if isinstance(databases, (list, tuple))
                    else [databases]
                )
                self.schema_version = (
                    schema_version
                    if schema_version is not None
                    else _sv_cls()
                )
                self.supported_features = list(supported_features)

        def empty(self):
            return not self.connection or not self.databases

        def size(self):
            return len(self.databases) if self.connection else 0

        def __len__(self):
            return self.size()

    def _unsupported_function(*args, **kwargs):
        raise RuntimeError("function requires rocpd Python bindings")

    def format_path(path, tag=None):
        """Fallback for libpyrocpd.format_path: resolve output keys in filepath."""
        import os
        import socket
        import re
        import sys

        if tag is None:
            tag = os.path.basename(sys.executable)

        hostname = socket.gethostname()
        pid = str(os.getpid())
        ppid = str(os.getppid())
        cwd = os.getcwd()

        replacements = {
            "%hostname%": hostname,
            "{hostname}": hostname,
            "%h": hostname,
            "%pid%": pid,
            "{pid}": pid,
            "%p": pid,
            "%ppid%": ppid,
            "{ppid}": ppid,
            "%cwd%": cwd,
            "{cwd}": cwd,
            "%tag%": tag,
            "{tag}": tag,
        }

        for key, value in replacements.items():
            path = path.replace(key, value)

        for pattern in [
            r"%(env|ENV)\{([A-Z0-9_]+)\}%",
            r"\$(env|ENV)\{([A-Z0-9_]+)\}",
            r"%q\{([A-Z0-9_]+)\}",
        ]:
            def _env_repl(m):
                var_name = m.group(m.lastindex)
                return os.environ.get(var_name, "")

            path = re.sub(pattern, _env_repl, path)

        return path

    class agent_indexing(enum.IntEnum):
        node = 0
        logical_node = 1
        logical_node_type = 2

    class sql_engine(enum.IntEnum):
        sqlite3 = 1            # ROCPD_SQL_ENGINE_SQLITE3

    class sql_schema(enum.IntEnum):
        rocpd_tables = 1
        rocpd_indexes = 2
        rocpd_views = 3
        data_views = 4
        summary_views = 5
        rocpd_metadata = 6

    class sql_option(enum.IntEnum):
        none = 0
        sqlite3_pragma_foreign_keys = 0x02

    class output_config:
        def __init__(self):
            self.output_path = "%cwd%"
            self.output_file = "%hostname%/%pid%"
            self.tmp_directory = self.output_path
            self.csv = False
            self.pftrace = False
            self.otf2 = False
            self.kernel_rename = False
            self.group_by_queue = False
            self.annotate_args = False
            self.annotate_kfd = False
            self.annotate_pmc = False
            self.agent_index_value = agent_indexing.logical_node
            self.perfetto_shmem_size_hint = 64
            self.perfetto_buffer_size = (1 * 1024 * 1024)
            self.perfetto_backend = "inprocess"
            self.perfetto_buffer_fill_policy = "discard"

    class schema_jinja_variables:
        def __init__(self):
            self.uuid = None
            self.guid = None

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


    def write_perfetto(*args, **kwargs):
        raise RuntimeError(
            "pftrace output requires the rocpd Python bindings (libpyrocpd); "
            "rebuild with the bindings"
        )

    def query_supported_schema_versions(engine=sql_engine.sqlite3, schema_path_hints=[]):
        """Return supported rocpd SQL schema versions as a list of schema_version."""
        return [
            schema_version(*itr)
            for itr in _py_query_supported_schema_versions(engine, schema_path_hints)
        ]

    def load_schema(engine, kind, options, version, variables):
        """Return the rocpd SQL schema text for the given kind and version."""
        return _py_load_schema(engine, kind, options, version, variables)

    agent = Agent
    node = Node
    process = Process
    thread = Thread

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
        return _fetch_all_objects(conn, query, factory=Thread, allowed_data=allowed_data)
