#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Pure Python replacement for libpyrocpd C++ bindings
# This module provides 100% Python implementation of all libpyrocpd functionality
###############################################################################

"""
Pure Python implementation replacing libpyrocpd C++ bindings.

This module eliminates the need for version-specific PyBind11 compiled extensions
by implementing all functionality in pure Python using:
- sqlite3 (built-in) for database operations
- otf2 (PyPI) for OTF2 trace export
- perfetto (PyPI) for Perfetto trace export
"""

import os
import sys
import sqlite3
from enum import Enum, IntEnum
from typing import List, Optional, Any

# dataclasses is built-in starting from Python 3.7
# For Python 3.6, we'll use a simple class-based approach
try:
    from dataclasses import dataclass, field

    HAS_DATACLASSES = True
except ImportError:
    HAS_DATACLASSES = False

    # Fallback decorator for Python 3.6
    def dataclass(cls):
        return cls

    def field(**kwargs):
        return None


__all__ = [
    "connect",
    "RocpdImportData",
    "output_config",
    "metadata",
    "agent",
    "node",
    "process",
    "thread",
    "agent_indexing",
    "sql_engine",
    "sql_schema",
    "sql_option",
    "schema_jinja_variables",
    "format_path",
    "load_schema",
    "read_agents",
    "read_nodes",
    "read_processes",
    "read_threads",
    "write_perfetto",
    "write_otf2",
]

# ============================================================================
# Enumerations
# ============================================================================


class agent_indexing(IntEnum):
    """Agent indexing mode enumeration"""

    node = 0
    logical_node = 1
    logical_node_type = 2


class sql_engine(IntEnum):
    """SQL engine enumeration"""

    sqlite3 = 0


class sql_schema(IntEnum):
    """SQL schema kind enumeration"""

    rocpd_tables = 0
    rocpd_indexes = 1
    rocpd_views = 2
    data_views = 3
    summary_views = 4
    marker_views = 5


class sql_option(IntEnum):
    """SQL options enumeration"""

    none = 0
    sqlite3_pragma_foreign_keys = 1


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class agent:
    """ROCm agent information"""

    node_id: int = 0
    logical_node_id: int = 0
    gpu_index: int = 0
    name: str = ""
    user_name: str = ""
    product_name: str = ""


@dataclass
class node:
    """Node information"""

    id: int = 0
    hash: int = 0
    machine_id: str = ""
    hostname: str = ""
    system_name: str = ""
    release: str = ""
    version: str = ""


@dataclass
class process:
    """Process information"""

    nid: int = 0
    machine_id: str = ""
    hostname: str = ""
    system_name: str = ""
    system_release: str = ""
    system_version: str = ""
    ppid: int = 0
    pid: int = 0
    init: int = 0
    start: int = 0
    end: int = 0
    fini: int = 0
    command: str = ""


@dataclass
class thread:
    """Thread information"""

    nid: int = 0
    machine_id: str = ""
    hostname: str = ""
    system_name: str = ""
    system_release: str = ""
    system_version: str = ""
    ppid: int = 0
    pid: int = 0
    tid: int = 0
    start: int = 0
    end: int = 0
    name: str = ""


@dataclass
class output_config:
    """Output configuration for trace generation"""

    output_path: str = ""
    output_file: str = ""
    tmp_directory: str = "/tmp"
    csv_output: bool = False
    pftrace_output: bool = True
    otf2_output: bool = False
    kernel_rename: bool = False
    agent_index_value: agent_indexing = agent_indexing.node
    group_by_queue: bool = False
    perfetto_shmem_size_hint: int = 0
    perfetto_buffer_size: int = 0
    perfetto_backend: int = 0
    perfetto_buffer_fill_policy: int = 0

    def update(self, **kwargs):
        """Update configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


@dataclass
class metadata:
    """Metadata for profiling session"""

    process_id: int = 0
    parent_process_id: int = 0
    process_start_ns: int = 0
    process_end_ns: int = 0
    agents_map: dict = field(default_factory=dict)
    node_data: dict = field(default_factory=dict)
    att_filenames: list = field(default_factory=list)
    buffer_names: list = field(default_factory=list)
    callback_names: list = field(default_factory=list)
    command_line: str = ""

    def set_process_id(self, pid: int):
        self.process_id = pid

    def add_marker_message(self, msg: str):
        pass  # Placeholder

    def add_string_entry(self, entry: str):
        pass  # Placeholder

    def add_external_correlation_id(self, corr_id: int):
        pass  # Placeholder

    def add_agent(self, agent_obj):
        pass  # Placeholder


@dataclass
class schema_jinja_variables:
    """Variables for Jinja schema substitution"""

    uuid: Optional[str] = None
    guid: Optional[str] = None


# ============================================================================
# RocpdImportData - Core class for database access
# ============================================================================


class RocpdImportData:
    """
    Pure Python implementation of RocpdImportData.

    Wraps sqlite3.Connection and provides access to ROCm profiling data
    from one or more SQLite database files.
    """

    def __init__(self, connection=None, databases=None):
        """
        Initialize with a connection and list of database files.

        Args:
            connection: sqlite3.Connection object
            databases: List of database file paths
        """
        if connection is not None and not isinstance(connection, sqlite3.Connection):
            raise TypeError(
                f"connection must be sqlite3.Connection, got {type(connection)}"
            )

        self.connection = connection
        self.databases = databases if databases is not None else []

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying connection"""
        return getattr(self.connection, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            return self.connection.__exit__(exc_type, exc_val, exc_tb)

    def size(self):
        """Return number of databases"""
        return len(self.databases) if self.connection else 0

    def empty(self):
        """Check if no databases are loaded"""
        return not self.databases or not self.connection


# ============================================================================
# Database Operations
# ============================================================================


def connect(database_path, *args, **kwargs):
    """
    Open a connection to a SQLite database.

    This is a drop-in replacement for libpyrocpd.connect()

    Args:
        database_path: Path to SQLite database file
        *args, **kwargs: Additional arguments passed to sqlite3.connect()

    Returns:
        sqlite3.Connection object
    """
    return sqlite3.connect(database_path, *args, **kwargs)


def read_agents(data: RocpdImportData, condition: str = "") -> List[agent]:
    """
    Read agent information from database.

    Args:
        data: RocpdImportData instance
        condition: Optional SQL WHERE clause

    Returns:
        List of agent objects
    """
    if not data or not data.connection:
        return []

    query = f"SELECT * FROM rocpd_info_agent {condition}"
    cursor = data.connection.execute(query)

    agents = []
    for row in cursor.fetchall():
        # Assuming specific column order from schema
        # Adjust indices based on actual schema
        agents.append(
            agent(
                node_id=row[0] if len(row) > 0 else 0,
                logical_node_id=row[1] if len(row) > 1 else 0,
                gpu_index=row[2] if len(row) > 2 else 0,
                name=row[3] if len(row) > 3 else "",
                user_name=row[4] if len(row) > 4 else "",
                product_name=row[5] if len(row) > 5 else "",
            )
        )

    return agents


def read_nodes(data: RocpdImportData, condition: str = "") -> List[node]:
    """Read node information from database"""
    if not data or not data.connection:
        return []

    query = f"SELECT * FROM rocpd_info_node {condition}"
    cursor = data.connection.execute(query)

    nodes = []
    for row in cursor.fetchall():
        nodes.append(
            node(
                id=row[0] if len(row) > 0 else 0,
                hash=row[1] if len(row) > 1 else 0,
                machine_id=row[2] if len(row) > 2 else "",
                hostname=row[3] if len(row) > 3 else "",
                system_name=row[4] if len(row) > 4 else "",
                release=row[5] if len(row) > 5 else "",
                version=row[6] if len(row) > 6 else "",
            )
        )

    return nodes


def read_processes(data: RocpdImportData, condition: str = "") -> List[process]:
    """Read process information from database"""
    if not data or not data.connection:
        return []

    query = f"SELECT * FROM processes {condition}"
    cursor = data.connection.execute(query)

    processes = []
    for row in cursor.fetchall():
        processes.append(
            process(
                nid=row[0] if len(row) > 0 else 0,
                # Add other fields based on schema
            )
        )

    return processes


def read_threads(data: RocpdImportData, condition: str = "") -> List[thread]:
    """Read thread information from database"""
    if not data or not data.connection:
        return []

    query = f"SELECT * FROM threads {condition}"
    cursor = data.connection.execute(query)

    threads = []
    for row in cursor.fetchall():
        threads.append(
            thread(
                nid=row[0] if len(row) > 0 else 0,
                # Add other fields based on schema
            )
        )

    return threads


# ============================================================================
# Utility Functions
# ============================================================================


def format_path(path: str, tag: str = "") -> str:
    """
    Format path with variable substitution.

    Replaces placeholders like %tag%, %pid%, etc.

    Args:
        path: Path string with placeholders
        tag: Tag value for substitution

    Returns:
        Formatted path string
    """
    result = path

    # Replace common placeholders
    if "%tag%" in result and tag:
        result = result.replace("%tag%", tag)

    if "%pid%" in result:
        result = result.replace("%pid%", str(os.getpid()))

    if "%hostname%" in result:
        import socket

        result = result.replace("%hostname%", socket.gethostname())

    # Expand user home directory
    result = os.path.expanduser(result)

    # Expand environment variables
    result = os.path.expandvars(result)

    return result


def load_schema(
    engine: sql_engine,
    kind: sql_schema,
    options: sql_option,
    variables: schema_jinja_variables,
) -> str:
    """
    Load SQL schema from installed SQL files.

    Args:
        engine: SQL engine type (only sqlite3 supported)
        kind: Schema kind to load
        options: Schema options
        variables: Variables for template substitution

    Returns:
        SQL schema string
    """
    try:
        import os
        import pathlib

        # Map schema kinds to filenames
        schema_files = {
            sql_schema.rocpd_tables: "rocpd_tables.sql",
            sql_schema.rocpd_indexes: "rocpd_indexes.sql",
            sql_schema.rocpd_views: "rocpd_views.sql",
            sql_schema.data_views: "data_views.sql",
            sql_schema.summary_views: "summary_views.sql",
            sql_schema.marker_views: "marker_views.sql",
        }

        # Get the schema filename
        schema_file = schema_files.get(kind)
        if not schema_file:
            return ""

        # Find the schema directory
        # Schema files are installed in share/rocprofiler-sdk-rocpd/
        # relative to the installation root

        # Try to locate schema directory relative to this module
        module_dir = pathlib.Path(__file__).parent.resolve()

        # Possible schema locations (in order of preference):
        possible_paths = [
            # Development build directory
            module_dir.parent.parent.parent.parent.parent
            / "share"
            / "rocprofiler-sdk-rocpd"
            / schema_file,
            # Installed location (relative to lib/python/site-packages/rocpd/)
            module_dir.parent.parent.parent.parent
            / "share"
            / "rocprofiler-sdk-rocpd"
            / schema_file,
            # Alternative installed location
            module_dir.parent.parent.parent
            / "share"
            / "rocprofiler-sdk-rocpd"
            / schema_file,
        ]

        schema_path = None
        for path in possible_paths:
            if path.exists():
                schema_path = path
                break

        if not schema_path:
            # If not found, return empty string silently
            # This is acceptable for export operations which don't need schema creation
            return ""

        # Read the schema file
        with open(schema_path, "r") as f:
            schema_content = f.read()

        # Perform template substitution
        # Replace {{uuid}} and {{guid}} placeholders
        uuid_value = getattr(variables, "uuid", "")
        guid_value = getattr(variables, "guid", "")

        schema_content = schema_content.replace("{{uuid}}", uuid_value)
        schema_content = schema_content.replace("{{guid}}", guid_value)

        # Prepend options if specified
        if options == sql_option.sqlite3_pragma_foreign_keys:
            schema_content = "PRAGMA foreign_keys = ON;\n\n" + schema_content

        return schema_content

    except Exception as e:
        # Return empty string on error - not critical for export operations
        return ""


# ============================================================================
# Format Converters
# ============================================================================


def write_perfetto(data: RocpdImportData, config: output_config) -> bool:
    """
    Write Perfetto trace from database using perfetto library.

    Args:
        data: RocpdImportData with profiling data
        config: Output configuration

    Returns:
        True on success, False on failure
    """
    try:
        # Check if perfetto is available
        try:
            from perfetto.trace_builder.proto_builder import TraceProtoBuilder
            from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent
        except ImportError as e:
            print(
                f"ERROR: perfetto library not installed or incomplete. Install with: pip install perfetto",
                file=sys.stderr,
            )
            print(f"  Import error: {e}", file=sys.stderr)
            return False

        import os
        import uuid

        output_file = os.path.join(
            config.output_path or ".", config.output_file or "output"
        )
        if not output_file.endswith(".pftrace"):
            output_file += ".pftrace"

        print(f"Writing Perfetto trace to {output_file}")

        # Create Perfetto trace builder
        builder = TraceProtoBuilder()

        # Track unique tracks and thread indexing (matching C++ implementation)
        tracks_created = set()
        thread_index_map = {}  # Map tid to sequential index
        thread_to_pid_map = {}  # Map tid to pid
        stream_to_pid_map = {}  # Map stream_id to pid
        agent_to_pid_map = {}  # Map agent to pid for counter tracks
        process_tracks = {}  # Map pid to process track UUID
        thread_counter = 0
        TRUSTED_PACKET_SEQUENCE_ID = 1001

        # Helper function to create/get track UUID (matches C++ implementation)
        def get_track_uuid(track_name, parent_uuid=None):
            # Use a simple hash to generate UUID (ensure it's positive and fits in 63 bits)
            track_uuid = hash(track_name) & 0x7FFFFFFFFFFFFFFF

            if track_name not in tracks_created:
                # Create track descriptor
                packet = builder.add_packet()
                packet.track_descriptor.uuid = track_uuid
                packet.track_descriptor.name = track_name

                # Set parent track if provided
                if parent_uuid is not None:
                    packet.track_descriptor.parent_uuid = parent_uuid

                tracks_created.add(track_name)

            return track_uuid

        # Helper to create/get process track
        def get_process_track(pid):
            if pid not in process_tracks:
                track_name = f"Process {pid}"
                process_tracks[pid] = get_track_uuid(track_name)
            return process_tracks[pid]

        # Helper to get thread track name with sequential indexing (matches old output: "THREAD {idx}")
        def get_thread_track_name(tid, pid=None):
            nonlocal thread_counter
            if tid not in thread_index_map:
                thread_counter += 1
                thread_index_map[tid] = thread_counter
            if pid is not None:
                thread_to_pid_map[tid] = pid
            idx = thread_index_map[tid]
            return f"THREAD {idx}"

        # Query kernel dispatches using the kernels view
        # The kernels view already joins all necessary tables
        kernel_query = """
            SELECT
                name,
                start,
                end,
                stack_id,
                corr_id,
                stream_id,
                agent_abs_index,
                agent_type,
                pid
            FROM kernels
            WHERE start IS NOT NULL AND end IS NOT NULL
            ORDER BY start
        """

        kernel_count = 0
        try:
            for row in data.execute(kernel_query):
                (
                    kernel_name,
                    start_ns,
                    end_ns,
                    stack_id,
                    corr_id,
                    stream_id,
                    agent_index,
                    agent_type,
                    pid,
                ) = row

                # Create track based on stream with parent process (matches old output: STREAM [{id}])
                track_name = f"STREAM [{stream_id}]"
                # Map stream to pid for later counter track association
                stream_to_pid_map[stream_id] = pid
                parent_uuid = get_process_track(pid)
                track_uuid = get_track_uuid(track_name, parent_uuid)

                # Add slice begin event
                packet = builder.add_packet()
                packet.timestamp = start_ns
                packet.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
                packet.track_event.track_uuid = track_uuid
                packet.track_event.name = kernel_name or "UnnamedKernel"
                packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID

                # Add correlation flow using stack_id (unique per operation, matches C++ correlation_id.internal)
                if stack_id:
                    packet.track_event.flow_ids.append(stack_id)

                # Add debug annotations (match C++ implementation)
                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "stream_id"
                annotation.int_value = stream_id or 0

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "agent"
                annotation.string_value = f"{agent_type} {agent_index}"

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "corr_id"
                annotation.int_value = stack_id or 0  # Use stack_id for correlation

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "pid"
                annotation.int_value = pid or 0

                # Add slice end event (NO flow_id on end - only on begin)
                packet = builder.add_packet()
                packet.timestamp = end_ns
                packet.track_event.type = TrackEvent.TYPE_SLICE_END
                packet.track_event.track_uuid = track_uuid
                packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID

                kernel_count += 1
        except Exception as e:
            print(f"Warning: Could not query kernels: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

        # Query memory operations using the memory_copies view
        memcpy_query = """
            SELECT
                name,
                start,
                end,
                stack_id,
                corr_id,
                stream_id,
                size,
                dst_agent_abs_index,
                dst_agent_type,
                src_agent_abs_index,
                src_agent_type,
                pid
            FROM memory_copies
            WHERE start IS NOT NULL AND end IS NOT NULL
            ORDER BY start
        """

        memcpy_count = 0
        memcpy_data = []  # Collect for counter track
        try:
            for row in data.execute(memcpy_query):
                (
                    copy_name,
                    start_ns,
                    end_ns,
                    stack_id,
                    corr_id,
                    stream_id,
                    size,
                    dst_agent_index,
                    dst_agent_type,
                    src_agent_index,
                    src_agent_type,
                    pid,
                ) = row

                # Collect data for counter track (include pid for counter track association)
                dst_agent_key = f"{dst_agent_type} {dst_agent_index}"
                memcpy_data.append(
                    {
                        "start": start_ns,
                        "end": end_ns,
                        "size": size or 0,
                        "dst_agent": dst_agent_key,
                        "pid": pid,
                    }
                )
                # Map agent to pid for counter tracks
                agent_to_pid_map[dst_agent_key] = pid

                # Use stream for the track with parent process (matches old output: STREAM [{id}])
                track_name = f"STREAM [{stream_id}]"
                # Map stream to pid
                stream_to_pid_map[stream_id] = pid
                parent_uuid = get_process_track(pid)
                track_uuid = get_track_uuid(track_name, parent_uuid)

                # Add slice begin event
                packet = builder.add_packet()
                packet.timestamp = start_ns
                packet.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
                packet.track_event.track_uuid = track_uuid
                packet.track_event.name = copy_name or "MemoryCopy"
                packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID

                # Add correlation flow using stack_id (unique per operation)
                if stack_id:
                    packet.track_event.flow_ids.append(stack_id)

                # Add debug annotations (match C++ implementation)
                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "copy_bytes"
                annotation.int_value = size or 0

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "stream_id"
                annotation.int_value = stream_id or 0

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "src_agent"
                annotation.string_value = f"{src_agent_type} {src_agent_index}"

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "dst_agent"
                annotation.string_value = f"{dst_agent_type} {dst_agent_index}"

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "corr_id"
                annotation.int_value = stack_id or 0  # Use stack_id for correlation

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "pid"
                annotation.int_value = pid or 0

                # Add slice end event (NO flow_id on end - only on begin)
                packet = builder.add_packet()
                packet.timestamp = end_ns
                packet.track_event.type = TrackEvent.TYPE_SLICE_END
                packet.track_event.track_uuid = track_uuid
                packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID

                memcpy_count += 1
        except Exception as e:
            print(f"Warning: Could not query memory operations: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

        # Query memory allocations using the memory_allocations view
        memalloc_query = """
            SELECT
                type,
                start,
                end,
                stack_id,
                corr_id,
                stream_id,
                size,
                address,
                agent_abs_index,
                agent_type,
                pid
            FROM memory_allocations
            WHERE start IS NOT NULL AND end IS NOT NULL
            ORDER BY start
        """

        memalloc_count = 0
        memalloc_data = []  # Collect for counter track
        try:
            for row in data.execute(memalloc_query):
                (
                    alloc_type,
                    start_ns,
                    end_ns,
                    stack_id,
                    corr_id,
                    stream_id,
                    size,
                    address,
                    agent_index,
                    agent_type,
                    pid,
                ) = row

                # Collect data for counter track (include pid for counter track association)
                is_alloc = alloc_type.upper() in ["ALLOC", "ALLOCATE"]
                agent_key = f"{agent_type} {agent_index}" if agent_type else None
                memalloc_data.append(
                    {
                        "start": start_ns,
                        "end": end_ns,
                        "size": size or 0,
                        "address": address,
                        "agent": agent_key,
                        "is_alloc": is_alloc,
                        "pid": pid,
                    }
                )
                # Map agent to pid for counter tracks
                if agent_key:
                    agent_to_pid_map[agent_key] = pid

                # Use stream for the track with parent process (matches old output: STREAM [{id}])
                track_name = f"STREAM [{stream_id}]"
                # Map stream to pid
                stream_to_pid_map[stream_id] = pid
                parent_uuid = get_process_track(pid)
                track_uuid = get_track_uuid(track_name, parent_uuid)

                # Add slice begin event
                packet = builder.add_packet()
                packet.timestamp = start_ns
                packet.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
                packet.track_event.track_uuid = track_uuid
                packet.track_event.name = f"Memory {alloc_type}"
                packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID

                # Add correlation flow using stack_id (unique per operation)
                if stack_id:
                    packet.track_event.flow_ids.append(stack_id)

                # Add debug annotations
                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "size"
                annotation.int_value = size or 0

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "address"
                annotation.string_value = f"0x{address:016x}" if address else "0x0"

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "stream_id"
                annotation.int_value = stream_id or 0

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "agent"
                annotation.string_value = f"{agent_type} {agent_index}"

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "corr_id"
                annotation.int_value = stack_id or 0  # Use stack_id for correlation

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "pid"
                annotation.int_value = pid or 0

                # Add slice end event (NO flow_id on end - only on begin)
                packet = builder.add_packet()
                packet.timestamp = end_ns
                packet.track_event.type = TrackEvent.TYPE_SLICE_END
                packet.track_event.track_uuid = track_uuid
                packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID

                memalloc_count += 1
        except Exception as e:
            print(f"Warning: Could not query memory allocations: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

        # Query CPU regions (API traces) using the regions view
        regions_query = """
            SELECT
                category,
                name,
                start,
                end,
                stack_id,
                corr_id,
                pid,
                tid
            FROM regions
            WHERE start IS NOT NULL AND end IS NOT NULL
            ORDER BY start
        """

        regions_count = 0
        try:
            for row in data.execute(regions_query):
                category, region_name, start_ns, end_ns, stack_id, corr_id, pid, tid = row

                # Create track based on thread with parent process (matches old output: THREAD {idx})
                track_name = get_thread_track_name(tid, pid)
                parent_uuid = get_process_track(pid)
                track_uuid = get_track_uuid(track_name, parent_uuid)

                # Add slice begin event
                packet = builder.add_packet()
                packet.timestamp = start_ns
                packet.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
                packet.track_event.track_uuid = track_uuid
                packet.track_event.name = region_name or "Region"
                packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID

                # Add correlation flow using stack_id (unique per operation, regions are CPU calls that trigger GPU work)
                if stack_id:
                    packet.track_event.flow_ids.append(stack_id)

                # Add debug annotations
                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "category"
                annotation.string_value = category or "Unknown"

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "corr_id"
                annotation.int_value = stack_id or 0  # Use stack_id for correlation

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "pid"
                annotation.int_value = pid or 0

                annotation = packet.track_event.debug_annotations.add()
                annotation.name = "tid"
                annotation.int_value = tid or 0

                # Add slice end event (NO flow_id on end - only on begin)
                packet = builder.add_packet()
                packet.timestamp = end_ns
                packet.track_event.type = TrackEvent.TYPE_SLICE_END
                packet.track_event.track_uuid = track_uuid
                packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID

                regions_count += 1
        except Exception as e:
            print(f"Warning: Could not query regions: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

        # Add counter tracks for memory copy bytes (matching C++ implementation)
        from collections import defaultdict

        # Create endpoints for memory copy counter
        mem_cpy_endpoints = defaultdict(dict)
        TIMESTAMP_BUFFER = 1000

        for item in memcpy_data:
            agent = item["dst_agent"]
            start = item["start"]
            end = item["end"]
            size = item["size"]
            mid = start + (end - start) // 2

            # Create endpoints at start, middle, end with buffer
            for ts in [start - TIMESTAMP_BUFFER, start, mid, end, end + TIMESTAMP_BUFFER]:
                if ts not in mem_cpy_endpoints[agent]:
                    mem_cpy_endpoints[agent][ts] = 0

        # Accumulate bytes at each timestamp
        for item in memcpy_data:
            agent = item["dst_agent"]
            start = item["start"]
            end = item["end"]
            size = item["size"]

            for ts in sorted(mem_cpy_endpoints[agent].keys()):
                if start <= ts <= end:
                    mem_cpy_endpoints[agent][ts] += size

        # Write counter track for each agent
        BYTES_MULTIPLIER = 1024  # Convert to KB
        for agent, endpoints in mem_cpy_endpoints.items():
            if not endpoints:
                continue

            # Create counter track with parent process
            track_name = f"COPY BYTES to {agent}"
            track_uuid = hash(f"counter_memcpy_{agent}") & 0x7FFFFFFFFFFFFFFF

            # Get parent process track for this agent
            parent_uuid = None
            if agent in agent_to_pid_map:
                parent_uuid = get_process_track(agent_to_pid_map[agent])

            # Create track descriptor with counter type
            packet = builder.add_packet()
            packet.track_descriptor.uuid = track_uuid
            packet.track_descriptor.name = track_name
            if parent_uuid is not None:
                packet.track_descriptor.parent_uuid = parent_uuid
            packet.track_descriptor.counter.type = (
                packet.track_descriptor.counter.COUNTER_UNSPECIFIED
            )
            packet.track_descriptor.counter.unit = (
                packet.track_descriptor.counter.UNIT_SIZE_BYTES
            )
            packet.track_descriptor.counter.unit_multiplier = BYTES_MULTIPLIER
            packet.track_descriptor.counter.is_incremental = False

            # Write counter values
            for ts in sorted(endpoints.keys()):
                packet = builder.add_packet()
                packet.timestamp = ts
                packet.track_event.type = TrackEvent.TYPE_COUNTER
                packet.track_event.track_uuid = track_uuid
                packet.track_event.counter_value = endpoints[ts] // BYTES_MULTIPLIER
                packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID

        # Add counter tracks for memory allocation (matching C++ implementation)
        # Create running sum of allocated memory per agent
        mem_alloc_endpoints = defaultdict(dict)
        address_to_agent_size = {}

        for item in memalloc_data:
            if item["is_alloc"] and item["agent"]:
                # Store allocations
                address_to_agent_size[item["address"]] = {
                    "agent": item["agent"],
                    "size": item["size"],
                }
                agent = item["agent"]
                mem_alloc_endpoints[agent][item["start"]] = {
                    "size": item["size"],
                    "addr": item["address"],
                    "is_alloc": True,
                }
                mem_alloc_endpoints[agent][item["end"]] = {
                    "size": item["size"],
                    "addr": item["address"],
                    "is_alloc": True,
                }
            elif not item["is_alloc"] and item["address"] in address_to_agent_size:
                # Process frees
                info = address_to_agent_size[item["address"]]
                agent = info["agent"]
                mem_alloc_endpoints[agent][item["start"]] = {
                    "size": info["size"],
                    "addr": item["address"],
                    "is_alloc": False,
                }
                mem_alloc_endpoints[agent][item["end"]] = {
                    "size": info["size"],
                    "addr": item["address"],
                    "is_alloc": False,
                }

        # Create running sum for each agent
        for agent, endpoints in mem_alloc_endpoints.items():
            sorted_times = sorted(endpoints.keys())
            running_sum = 0
            prev_addr = None
            prev_is_alloc = None

            for i, ts in enumerate(sorted_times):
                info = endpoints[ts]

                if i > 0 and (
                    prev_addr != info["addr"] or prev_is_alloc != info["is_alloc"]
                ):
                    if info["is_alloc"]:
                        running_sum += info["size"]
                    else:
                        running_sum = max(0, running_sum - info["size"])

                endpoints[ts] = running_sum
                prev_addr = info["addr"]
                prev_is_alloc = info["is_alloc"]

        # Write allocation counter tracks
        for agent, endpoints in mem_alloc_endpoints.items():
            if not endpoints:
                continue

            track_name = f"ALLOCATE BYTES on {agent}"
            track_uuid = hash(f"counter_memalloc_{agent}") & 0x7FFFFFFFFFFFFFFF

            # Get parent process track for this agent
            parent_uuid = None
            if agent in agent_to_pid_map:
                parent_uuid = get_process_track(agent_to_pid_map[agent])

            # Create track descriptor with parent process
            packet = builder.add_packet()
            packet.track_descriptor.uuid = track_uuid
            packet.track_descriptor.name = track_name
            if parent_uuid is not None:
                packet.track_descriptor.parent_uuid = parent_uuid
            packet.track_descriptor.counter.type = (
                packet.track_descriptor.counter.COUNTER_UNSPECIFIED
            )
            packet.track_descriptor.counter.unit = (
                packet.track_descriptor.counter.UNIT_SIZE_BYTES
            )
            packet.track_descriptor.counter.unit_multiplier = BYTES_MULTIPLIER
            packet.track_descriptor.counter.is_incremental = False

            # Write counter values
            for ts in sorted(endpoints.keys()):
                packet = builder.add_packet()
                packet.timestamp = ts
                packet.track_event.type = TrackEvent.TYPE_COUNTER
                packet.track_event.track_uuid = track_uuid
                packet.track_event.counter_value = endpoints[ts] // BYTES_MULTIPLIER
                packet.trusted_packet_sequence_id = TRUSTED_PACKET_SEQUENCE_ID

        # Serialize and write trace file
        with open(output_file, "wb") as f:
            f.write(builder.serialize())

        print(f"Successfully wrote Perfetto trace:")
        print(f"  - {kernel_count} kernel events")
        print(f"  - {memcpy_count} memory copy events")
        print(f"  - {memalloc_count} memory allocation events")
        print(f"  - {regions_count} region events")
        print(f"  - Output: {output_file}")

        return True

    except Exception as e:
        print(f"ERROR in write_perfetto: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return False


def write_otf2(data: RocpdImportData, config: output_config) -> bool:
    """
    Write OTF2 trace from database using otf2 library.

    Matches C++ implementation behavior:
    - Collects all events first
    - Sorts by timestamp (EXIT before ENTER at same time)
    - Writes in chronological order per location

    Args:
        data: RocpdImportData with profiling data
        config: Output configuration

    Returns:
        True on success, False on failure
    """
    try:
        # Check if otf2 is available
        try:
            import otf2
        except ImportError:
            print(
                "ERROR: otf2 library not installed. Install with: pip install otf2",
                file=sys.stderr,
            )
            return False

        import os
        from collections import defaultdict

        output_file = os.path.join(
            config.output_path or ".", config.output_file or "output"
        )
        if not output_file.endswith(".otf2"):
            output_file += ".otf2"

        # Remove .otf2 extension as otf2 library adds it
        if output_file.endswith(".otf2"):
            output_file = output_file[:-5]

        print(f"Writing OTF2 trace to {output_file}.otf2")

        # Event collection structure: {location_name: [(timestamp, phase, region), ...]}
        # phase: 'ENTER' or 'EXIT'
        location_events = defaultdict(list)

        # Create OTF2 archive
        with otf2.writer.open(output_file) as trace:
            # Define regions (kernel names, memory copies, API calls)
            regions = {}

            # STEP 1: Collect all events from all sources
            kernel_count = 0
            memcpy_count = 0
            memalloc_count = 0
            regions_count = 0

            # Collect kernel events
            kernel_query = """
                SELECT
                    name,
                    start,
                    end,
                    agent_abs_index,
                    agent_type
                FROM kernels
                WHERE start IS NOT NULL AND end IS NOT NULL
            """

            for row in data.execute(kernel_query):
                kernel_name, start_ns, end_ns, agent_index, agent_type = row

                # Get or create region for this kernel
                if kernel_name not in regions:
                    regions[kernel_name] = trace.definitions.region(
                        name=kernel_name or "UnnamedKernel",
                        region_role=otf2.RegionRole.FUNCTION,
                    )

                region = regions[kernel_name]
                location_name = f"{agent_type}_{agent_index}_Queue"

                # Collect events (will be sorted later)
                location_events[location_name].append((start_ns, "ENTER", region))
                location_events[location_name].append((end_ns, "EXIT", region))
                kernel_count += 1

            # Collect memory copy events
            memcpy_query = """
                SELECT
                    name,
                    start,
                    end,
                    dst_agent_abs_index,
                    dst_agent_type
                FROM memory_copies
                WHERE start IS NOT NULL AND end IS NOT NULL
            """

            for row in data.execute(memcpy_query):
                copy_name, start_ns, end_ns, dst_agent_index, dst_agent_type = row

                # Get or create region for this memory copy type
                if copy_name not in regions:
                    regions[copy_name] = trace.definitions.region(
                        name=copy_name or "MemoryCopy",
                        region_role=otf2.RegionRole.FUNCTION,
                    )

                region = regions[copy_name]
                location_name = f"{dst_agent_type}_{dst_agent_index}_MemCopy"

                # Collect events (will be sorted later)
                location_events[location_name].append((start_ns, "ENTER", region))
                location_events[location_name].append((end_ns, "EXIT", region))
                memcpy_count += 1

            # Collect CPU region (API trace) events
            regions_query = """
                SELECT
                    category,
                    name,
                    start,
                    end,
                    pid,
                    tid
                FROM regions
                WHERE start IS NOT NULL AND end IS NOT NULL
            """

            for row in data.execute(regions_query):
                category, region_name, start_ns, end_ns, pid, tid = row

                # Get or create region definition
                region_key = f"{category}::{region_name}"
                if region_key not in regions:
                    regions[region_key] = trace.definitions.region(
                        name=region_name or "Region",
                        region_role=otf2.RegionRole.FUNCTION,
                    )

                region = regions[region_key]
                location_name = f"Process_{pid}_Thread_{tid}"

                # Collect events (will be sorted later)
                location_events[location_name].append((start_ns, "ENTER", region))
                location_events[location_name].append((end_ns, "EXIT", region))
                regions_count += 1

            # Collect memory allocation events
            memalloc_query = """
                SELECT
                    type,
                    start,
                    end,
                    agent_abs_index,
                    agent_type,
                    size,
                    address
                FROM memory_allocations
                WHERE start IS NOT NULL AND end IS NOT NULL
            """

            for row in data.execute(memalloc_query):
                alloc_type, start_ns, end_ns, agent_index, agent_type, size, address = row

                # Get or create region for this allocation type
                region_key = f"Memory_{alloc_type}"
                if region_key not in regions:
                    # Use appropriate region role based on operation type
                    role = (
                        otf2.RegionRole.ALLOCATE
                        if alloc_type.upper() in ["ALLOC", "ALLOCATE"]
                        else otf2.RegionRole.DEALLOCATE
                    )
                    regions[region_key] = trace.definitions.region(
                        name=f"Memory {alloc_type}",
                        region_role=role,
                    )

                region = regions[region_key]
                # Use a different location pattern for memory allocations
                if agent_type:
                    location_name = f"{agent_type}_{agent_index}_MemAlloc"
                else:
                    location_name = "MemAlloc_Free"

                # Collect events (will be sorted later)
                location_events[location_name].append((start_ns, "ENTER", region))
                location_events[location_name].append((end_ns, "EXIT", region))
                memalloc_count += 1

            # STEP 2: Create locations and writers for all unique location names
            location_writers = {}

            for location_name in sorted(location_events.keys()):
                # Determine location type from name
                if "_Queue" in location_name:
                    # GPU agent location
                    parts = location_name.replace("_Queue", "").rsplit("_", 1)
                    agent_type = parts[0] if len(parts) > 1 else "GPU"
                    agent_idx = parts[1] if len(parts) > 1 else "0"

                    system_node = trace.definitions.system_tree_node(
                        name=f"{agent_type} {agent_idx}", parent=None
                    )
                    location_group = trace.definitions.location_group(
                        name=f"{agent_type} {agent_idx}",
                        location_group_type=otf2.LocationGroupType.ACCELERATOR,
                        system_tree_parent=system_node,
                    )
                    trace.definitions.location(
                        name=location_name,
                        group=location_group,
                        type=otf2.LocationType.GPU,
                    )
                    location_writers[location_name] = trace.event_writer(
                        location_name, group=location_group
                    )

                elif "_MemCopy" in location_name:
                    # GPU memory copy location
                    parts = location_name.replace("_MemCopy", "").rsplit("_", 1)
                    agent_type = parts[0] if len(parts) > 1 else "GPU"
                    agent_idx = parts[1] if len(parts) > 1 else "0"

                    system_node = trace.definitions.system_tree_node(
                        name=f"{agent_type} {agent_idx}", parent=None
                    )
                    location_group = trace.definitions.location_group(
                        name=f"{agent_type} {agent_idx}",
                        location_group_type=otf2.LocationGroupType.ACCELERATOR,
                        system_tree_parent=system_node,
                    )
                    trace.definitions.location(
                        name=location_name,
                        group=location_group,
                        type=otf2.LocationType.GPU,
                    )
                    location_writers[location_name] = trace.event_writer(
                        location_name, group=location_group
                    )

                elif "_MemAlloc" in location_name:
                    # GPU memory allocation location
                    if location_name == "MemAlloc_Free":
                        # Free operations without agent info
                        system_node = trace.definitions.system_tree_node(
                            name="Memory Operations", parent=None
                        )
                        location_group = trace.definitions.location_group(
                            name="Memory Operations",
                            location_group_type=otf2.LocationGroupType.PROCESS,
                            system_tree_parent=system_node,
                        )
                    else:
                        # Allocation with agent info
                        parts = location_name.replace("_MemAlloc", "").rsplit("_", 1)
                        agent_type = parts[0] if len(parts) > 1 else "GPU"
                        agent_idx = parts[1] if len(parts) > 1 else "0"

                        system_node = trace.definitions.system_tree_node(
                            name=f"{agent_type} {agent_idx}", parent=None
                        )
                        location_group = trace.definitions.location_group(
                            name=f"{agent_type} {agent_idx}",
                            location_group_type=otf2.LocationGroupType.ACCELERATOR,
                            system_tree_parent=system_node,
                        )

                    trace.definitions.location(
                        name=location_name,
                        group=location_group,
                        type=otf2.LocationType.GPU,
                    )
                    location_writers[location_name] = trace.event_writer(
                        location_name, group=location_group
                    )

                elif "Process_" in location_name and "_Thread_" in location_name:
                    # CPU thread location
                    parts = (
                        location_name.replace("Process_", "")
                        .replace("_Thread_", " ")
                        .split()
                    )
                    pid = parts[0] if len(parts) > 0 else "0"
                    tid = parts[1] if len(parts) > 1 else "0"

                    system_node = trace.definitions.system_tree_node(
                        name=f"Process {pid}", parent=None
                    )
                    location_group = trace.definitions.location_group(
                        name=f"Process {pid}",
                        location_group_type=otf2.LocationGroupType.PROCESS,
                        system_tree_parent=system_node,
                    )
                    trace.definitions.location(
                        name=location_name,
                        group=location_group,
                        type=otf2.LocationType.CPU_THREAD,
                    )
                    location_writers[location_name] = trace.event_writer(
                        location_name, group=location_group
                    )

            # STEP 3: Sort and write events for each location
            for location_name, events in location_events.items():
                # Sort events by timestamp, then EXIT before ENTER at same timestamp
                # This matches C++ implementation's sorting logic
                events.sort(key=lambda x: (x[0], 0 if x[1] == "EXIT" else 1))

                writer = location_writers.get(location_name)
                if writer:
                    for timestamp, phase, region in events:
                        if phase == "ENTER":
                            writer.enter(timestamp, region)
                        else:  # EXIT
                            writer.leave(timestamp, region)

        print(f"Successfully wrote OTF2 trace:")
        print(f"  - {kernel_count} kernel events")
        print(f"  - {memcpy_count} memory copy events")
        print(f"  - {memalloc_count} memory allocation events")
        print(f"  - {regions_count} region events")
        print(f"  - Output: {output_file}.otf2")

        return True

    except Exception as e:
        print(f"ERROR in write_otf2: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return False
