#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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


import os
import re
import sys
import json
import math
import yaml
import random
import shutil
import hashlib
import argparse
import datetime
import textwrap
import subprocess

# global variable that should never be modified during runtime
CONST_METRIC_LIST = [
    "wall_time",
    "cpu_time",
    "cpu_util",
    "peak_rss",
    "page_rss",
    "virtual_memory",
    "major_page_faults",
    "minor_page_faults",
    "priority_context_switches",
    "voluntary_context_switches",
]

CONST_ROCPROFV3_PROFILE_LIST = [
    "hip_compiler_api",
    "hip_runtime_api",
    "hsa_api",
    "kernel_dispatch",
    "marker_api",
    "memory_allocation",
    "memory_copy",
    "ompt",
    "rccl_api",
    "rocdecode_api",
    "rocjpeg_api",
    "scratch_memory",
]

# global variable that should never be modified during runtime.
# Key is database metric name, mapped value is timem json label
CONST_TIMEM_METRIC_MAP = {
    "wall_time": "wall_clock",
    "cpu_time": "cpu_clock",
    "cpu_util": "cpu_util",
    "peak_rss": "peak_rss",
    "page_rss": "page_rss",
    "virtual_memory": "virtual_memory",
    "major_page_faults": "num_major_page_faults",
    "minor_page_faults": "num_minor_page_faults",
    "priority_context_switches": "priority_context_switch",
    "voluntary_context_switches": "voluntary_context_switch",
}

# global variable that should never be modified during runtime.
# Key is database metric name, mapped value is units
CONST_METRIC_UNIT_MAP = {
    "wall_time": "sec",
    "cpu_time": "sec",
    "cpu_util": "%",
    "peak_rss": "MB",
    "page_rss": "MB",
    "virtual_memory": "MB",
    "major_page_faults": "",
    "minor_page_faults": "",
    "priority_context_switches": "",
    "voluntary_context_switches": "",
}

CONST_VERSION_INFO = {
    "version": "@FULL_VERSION_STRING@",
    "git_revision": "@ROCPROFILER_SDK_GIT_REVISION@",
    "library_arch": "@CMAKE_LIBRARY_ARCHITECTURE@",
    "system_name": "@CMAKE_SYSTEM_NAME@",
    "system_processor": "@CMAKE_SYSTEM_PROCESSOR@",
    "system_version": "@CMAKE_SYSTEM_VERSION@",
    "compiler_id": "@CMAKE_CXX_COMPILER_ID@",
    "compiler_version": "@CMAKE_CXX_COMPILER_VERSION@",
}

# global variable(s) that should never be modified during runtime.
CONST_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONST_LOCAL_TIMEM_PATH = os.path.join(CONST_THIS_DIR, "timem")


def get_linked_libraries(executable_path):
    """
    Run `ldd` on the given executable and return a list of linked library paths.

    :param executable_path: Path to the ELF executable.
    :return: List of Path objects to the linked libraries.
    """
    try:
        result = subprocess.run(
            ["ldd", executable_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"ldd failed: {e}\n")
        sys.stderr.flush()
        return []

    libraries = []
    for line in result.stdout.splitlines():
        # Match lines like: "\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f8a3b2e5000)"
        match = re.search(r"=>\s+(\S+)", line)
        if match:
            lib_path = match.group(1)
            if lib_path != "not":
                libraries.append(os.path.realpath(lib_path))
        else:
            # Match lines like: "/lib64/ld-linux-x86-64.so.2 (0x00007f8a3b5de000)"
            match_direct = re.match(r"^\s*(/[^ ]+)", line)
            if match_direct:
                libraries.append(os.path.realpath(match_direct.group(1)))

    return libraries


def get_needed_libraries(executable_path):
    """
    Extract DT_RPATH, DT_RUNPATH, and DT_NEEDED entries (shared library names) from an ELF file.

    :param executable_path: Path to the ELF executable.
    :return: List of RPATHs, list of RUNPATHs, and list of shared library names (e.g. 'libc.so.6').
    """
    from elftools.elf.elffile import ELFFile
    from elftools.elf.dynamic import DynamicSection

    rpath = []
    runpath = []
    needed = []
    with open(executable_path, "rb") as f:
        elf = ELFFile(f)
        for section in elf.iter_sections():
            if isinstance(section, DynamicSection):
                for tag in section.iter_tags():
                    if tag.entry.d_tag == "DT_NEEDED":
                        needed.append(tag.needed)
                    elif tag.entry.d_tag == "DT_RUNPATH":
                        runpath += tag.runpath.split(":")
                    elif tag.entry.d_tag == "DT_RPATH":
                        rpath += tag.rpath.split(":")

    return (rpath, runpath, needed)


def resolve_library(target, name, rpath, ld_library_path, runpath):
    """
    Attempt to resolve a shared library name to its full path RPATH, LD_LIBRARY_PATH, RUNPATH, and lastly, `ldconfig -p`.

    :param target: Path to target executable (source of DT_NEEDED)
    :param name: Shared library name (e.g., 'libc.so.6')
    :param rpath: List of RPATH directories
    :param ld_library_path: List of LD_LIBRARY_PATH directories
    :param runpath: List of RUNPATH directories
    :return: Path to library or None if not found
    """

    def _verify_resolved_libpath(_libpath):
        _libpath = _libpath.replace("$ORIGIN", os.path.realpath(os.path.dirname(target)))
        _val = os.path.join(_libpath, name)
        if os.path.exists(_val) and os.path.isfile(_val):
            return os.path.realpath(_val)
        return None

    for path_set in [rpath, ld_library_path, runpath]:
        for libpath in path_set:
            lib = _verify_resolved_libpath(libpath)
            if lib is not None:
                return lib

    result = subprocess.run(["ldconfig", "-p"], stdout=subprocess.PIPE, text=True)
    for line in result.stdout.splitlines():
        if name in line:
            parts = [itr.strip() for itr in line.strip().split(" ")]
            if (
                len(parts) >= 2
                and os.path.exists(parts[-1])
                and os.path.basename(parts[-1]) == name
            ):
                return os.path.realpath(parts[-1])

    return None


def get_executable_depends(target, ld_library_path):
    deps = []

    try:
        deps += get_linked_libraries(target)
    except Exception as e:
        sys.stderr.write(f"'{target}' raised an exception: {e}\n")
    finally:
        sys.stderr.flush()

    if len(deps) == 0:
        try:
            from elftools.common.exceptions import ELFError

            # if ldd failed, try using pyelftools
            rpath, runpath, needed = get_needed_libraries(target)
            for nitr in needed:
                resolved_nitr = resolve_library(
                    target, nitr, rpath, ld_library_path, runpath
                )
                if resolved_nitr is not None:
                    deps += [resolved_nitr]
        except ELFError:
            sys.stderr.write(
                f"'{target}' is not an ELF file. md5sum changes will be unreliable\n"
            )
        except Exception as e:
            sys.stderr.write(
                f"'{target}' raised an exception (ignoring): {e}. md5sum changes may be unreliable\n"
            )
        finally:
            sys.stderr.flush()

    return deps


def compute_hash(data):
    _data = f"{data}" if not isinstance(data, str) else data
    return hashlib.md5(_data.encode()).hexdigest()


def log_message(*args):

    sys.stdout.write("\n####\n")
    sys.stdout.write("####  ")
    sys.stdout.write(*args)
    sys.stdout.write("\n####\n")
    sys.stdout.flush()


def patch_message(msg, *args):
    msg = textwrap.dedent(msg)

    if len(args) > 0:
        msg = msg.format(*args)

    return msg.strip("\n").strip()


def fatal_error(msg, *args, exit_code=1):
    msg = patch_message(msg, *args)
    sys.stderr.write(f"Fatal error: {msg}\n")
    sys.stderr.flush()
    sys.exit(exit_code)


# Standard deviation function (sample stddev, i.e., n-1 in denominator)
def stddev_samp(values):
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


# Aggregate class for use with sqlite3
class StdDevSamp:
    def __init__(self):
        self.values = []

    def step(self, value):
        if value is not None:
            self.values.append(value)

    def finalize(self):
        return stddev_samp(self.values)


# Function to connect to the MySQL database
def connect_to_database(args):

    if args.db_backend == "sqlite3":
        import sqlite3

        connection = sqlite3.connect(args.sqlite3_database)
        connection.create_aggregate("STDDEV_SAMP", 1, StdDevSamp)

    elif args.db_backend == "mysql":
        import mysql.connector

        mysql_config = {
            "user": args.mysql_user,
            "password": args.mysql_passwd,
            "host": args.mysql_host,
            "database": args.mysql_database,
        }
        connection = mysql.connector.connect(**mysql_config)
    else:
        raise ValueError(f"unhandled database backend {args.db_backend}")

    # backend agnostic
    cursor = connection.cursor()

    # initial setup
    if args.db_backend == "sqlite3":
        cursor.execute("PRAGMA foreign_keys = ON")
    elif args.db_backend == "mysql":
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {args.mysql_database}")
        cursor.execute(f"USE {args.mysql_database}")
    else:
        raise ValueError(f"unhandled database backend {args.db_backend}")

    def execute_db_script(script_file, backend, repl):
        with open(script_file, "r") as ifs:
            data = ifs.read()
            for rkey, ritr in repl.items():
                data = data.replace(rkey, ritr)

            if backend == "sqlite3":
                cursor.executescript(data)
            elif backend == "mysql":
                statements = data.split(";")
                for statement in statements:
                    statement = statement.strip()
                    if statement:
                        cursor.execute(statement)
            else:
                raise ValueError(f"unhandled database backend {backend}")

    # SQL implementation agnostic schema
    for schema in args.db_schema:
        repl = {}
        if args.db_backend == "sqlite3":
            repl = {
                " INT ": " INTEGER ",
                "AUTO_INCREMENT": "AUTOINCREMENT",
                '("{}")': '"{}"',
                '("[]")': '"[]"',
            }
        execute_db_script(schema, args.db_backend, repl)

    # SQL views substitute {{metric}} syntax with metric list
    for views in args.db_views:
        for metric in CONST_METRIC_LIST:
            repl = {"{{metric}}": f"{metric}"}
            execute_db_script(views, args.db_backend, repl)

    return connection, cursor


def find_benchmarked_app(cursor, cmd, launcher, environment):
    """
    Insert benchmarked application
    Args:
        cursor:
            Database cursor
        cmd:
            Command line executed
        launcher: list
            Command line launcher
        environment: dict
            Environment variables
    Returns:
        id: row id of matching benchmarked application or None
    """

    cmd_json = json.dumps(cmd)
    ld_library_path = environment.get("LD_LIBRARY_PATH", "").split(":")

    # get the executable file
    exe_file = cmd[0]
    if not os.path.exists(exe_file):
        exe_file = shutil.which(exe_file)

    exe_data = []
    with open(exe_file, "rb") as ifs:
        exe_data += [ifs.read()]

    # factor in the md5sum of the linked libraries
    for dep in get_executable_depends(exe_file, ld_library_path):
        with open(dep, "rb") as ifs:
            exe_data += [ifs.read()]

    md5sum = compute_hash("".join([f"{itr}" for itr in exe_data]))

    hash_id = compute_hash(
        f"{launcher}{cmd_json}{environment}".replace("\n", "").replace(" ", "")
    )

    select_stmt = f"SELECT id FROM benchmarked_app WHERE hash_id = '{hash_id}' AND md5sum = '{md5sum}'"

    cursor.execute(select_stmt)
    existing_id = cursor.fetchone()
    if existing_id is not None:
        return existing_id[0]

    return None


def insert_benchmarked_app(cursor, cmd, launcher, environment, app_id, profile, args):
    """
    Insert benchmarked application
    Args:
        cursor:
            Database cursor
        cmd: list
            Command line executed
        launcher: list
            Command line launcher
        environment: dict
            Environment variables
        app_id: integer
            Return value of find_benchmarked_app
        profile: dict
            Return value of execute_profile
        args:
            Command line args (argparse.Namespace)
    Returns:
        id: row id of the insert
    """

    cmd_json = json.dumps(cmd)
    env_json = json.dumps(environment)
    ld_library_path = environment.get("LD_LIBRARY_PATH", "").split(":")

    # get the executable file
    exe_file = cmd[0]
    if not os.path.exists(exe_file):
        exe_file = shutil.which(exe_file)

    exe_data = []
    with open(exe_file, "rb") as ifs:
        exe_data += [ifs.read()]

    # factor in the md5sum of the linked libraries
    for dep in get_executable_depends(exe_file, ld_library_path):
        with open(dep, "rb") as ifs:
            exe_data += [ifs.read()]

    md5sum = compute_hash("".join([f"{itr}" for itr in exe_data]))

    hash_id = compute_hash(
        f"{launcher}{cmd_json}{environment}".replace("\n", "").replace(" ", "")
    )

    select_stmt = f"SELECT id FROM benchmarked_app WHERE hash_id = '{hash_id}' AND md5sum = '{md5sum}'"

    cursor.execute(select_stmt)
    existing_id = cursor.fetchone()
    if existing_id is not None:
        assert (
            app_id is not None
        ), "insert_benchmarked_app should have been passed a non-null app_id from find_benchmarked_app"
        if app_id != existing_id[0]:
            raise RuntimeError(
                f"find_benchmarked_app found app_id={app_id} but insert_benchmarked_app found app_id={existing_id[0]}"
            )
        return existing_id[0]

    if profile is None:
        raise RuntimeError(
            f"insert_benchmarked_app requires an application profile from rocprofv3. cmd: {cmd}"
        )

    values = (
        [
            "hash_id",
            "md5sum",
            "command",
            "threads",
        ]
        + CONST_ROCPROFV3_PROFILE_LIST
        + [
            "environment",
        ]
    )

    # make sure the dict indexing in cursor.execute succeeds
    for col in CONST_ROCPROFV3_PROFILE_LIST:
        if col not in profile:
            profile[col] = {"count": None, "unique": None}

    # combine HSA and Marker APIs into "hsa_api" and "marker_api" entries
    for key, itr in profile.items():
        for api in ["hsa", "marker"]:
            if re.match(f"^{api}_([a-z_]+)_api$", key):
                for col in ["count", "unique"]:
                    if profile[f"{api}_api"][col] is None:
                        profile[f"{api}_api"][col] = 0
                    profile[f"{api}_api"][col] += itr[col]

    insert_stmt = "INSERT INTO benchmarked_app ({}) VALUES ({})".format(
        ", ".join(values), ", ".join([args.db_placeholder for _ in values])
    )

    cursor.execute(
        insert_stmt,
        [
            hash_id,
            md5sum,
            cmd_json,
            profile["threads"],
        ]
        + [profile[itr]["count"] for itr in CONST_ROCPROFV3_PROFILE_LIST]
        + [env_json],
    )

    return cursor.lastrowid


def insert_benchmark_config(cursor, sdk_id, config_record, args):
    """
    Insert rocprofiler-sdk information
    Args:
        cursor:
            Database cursor
        config_record:
            configuration info dict from rocprofv3 config json.
        args:
            Command line args (argparse.Namespace)
    Returns:
        id: row id of the insert
    """

    if config_record is not None:
        if "rocprofiler-sdk-tool" in config_record:
            config_record = config_record["rocprofiler-sdk-tool"][0]

        if "metadata" in config_record:
            config_record = config_record["metadata"]

        config_record = config_record["config"]
    else:
        config_record = {"benchmark_mode": "baseline"}

    info_mapping = {
        "benchmark_mode": [],
        "kernel_rename": [],
        "group_by_queue": [],
        "kernel_trace": [],
        "hsa_trace": [
            "hsa_core_api_trace",
            "hsa_amd_ext_api_trace",
            "hsa_image_ext_api_trace",
            "hsa_finalizer_ext_api_trace",
        ],
        "hip_runtime_trace": ["hip_runtime_api_trace"],
        "hip_compiler_trace": ["hip_compiler_api_trace"],
        "marker_trace": ["marker_api_trace"],
        "memory_copy_trace": [],
        "memory_allocation_trace": [],
        "scratch_memory_trace": [],
        "rccl_trace": ["rccl_api_trace"],
        "rocdecode_trace": ["rocdecode_api_trace"],
        "rocjpeg_trace": ["rocjpeg_api_trace"],
        "dispatch_counter_collection": ["counter_collection"],
        "pc_sampling_host_trap": [],
        "pc_sampling_stocastic": [],
        "advanced_thread_trace": [],
        "pmc_counters": ["counters"],
    }

    name_mapping = {
        "hsa_trace": "HSA",
        "hip_runtime_trace": "HIP (Runtime)",
        "hip_compiler_trace": "HIP (Compiler)",
        "marker_trace": "ROCTx",
        "group_by_queue": "Group-By-Queue",
        "memory_allocation_trace": "Memory Alloc",
        "rccl_trace": "RCCL",
        "rocdecode_trace": "rocDecode",
        "rocjpeg_trace": "rocJPEG",
        "pc_sampling_host_trap": "PC Sampling (Host Trap)",
        "pc_sampling_stocastic": "PC Sampling (Stocastic)",
        "advanced_thread_trace": "ATT",
        "pmc_counters": "PMC",
    }

    def patch_key(val):
        if val in name_mapping:
            return name_mapping[val]
        return " ".join(
            [itr.capitalize() for itr in val.lower().replace("_trace", "").split("_")]
        )

    data = {}
    labels = []
    for col, itr in info_mapping.items():
        if not itr:
            itr = [col]

        for citr in itr:
            if citr in config_record and config_record[citr]:
                val = config_record[citr]
                if isinstance(val, list):
                    data[col] = json.dumps(*val)
                    labels.append(f"{patch_key(col)}={val}")
                elif isinstance(val, dict):
                    data[col] = json.dumps(val)
                    labels.append(f"{patch_key(col)}={val}")
                elif isinstance(val, bool):
                    data[col] = config_record[citr]
                    labels.append(patch_key(col))
                elif isinstance(val, str) and col == "benchmark_mode":
                    data[col] = config_record[citr]
                else:
                    raise ValueError(
                        f"unexpected data type for column '{col}': {type(config_record[citr]).__name__}"
                    )
                break

    data_json = json.dumps(data)

    label = " + ".join(sorted(labels))

    hash_id = compute_hash(f"{data_json}".replace("\n", "").replace(" ", ""))

    select_stmt = f"SELECT id FROM benchmark_config WHERE hash_id = '{hash_id}'"

    cursor.execute(select_stmt)
    existing_id = cursor.fetchone()
    if existing_id is not None:
        return existing_id[0]

    columns = ["hash_id", "sdk_id", "label"] + list(data.keys())
    insert_stmt = "INSERT INTO benchmark_config ({}) VALUES ({})".format(
        ", ".join(columns), ", ".join([args.db_placeholder for _ in columns])
    )

    values = [hash_id, sdk_id, label] + [itr for _, itr in data.items()]
    cursor.execute(insert_stmt, values)

    return cursor.lastrowid


def insert_benchmarked_sdk(cursor, config_record, args):
    """
    Insert rocprofiler-sdk information
    Args:
        cursor:
            Database cursor
        config_record:
            configuration info dict from rocprofv3 config json.
        args:
            Command line args (argparse.Namespace)
    Returns:
        id: row id of the insert
    """

    if "rocprofiler-sdk-tool" in config_record:
        config_record = config_record["rocprofiler-sdk-tool"][0]

    if "metadata" in config_record:
        config_record = config_record["metadata"]

    build_spec = config_record["build_spec"]

    hash_id = compute_hash(f"{build_spec}".replace("\n", "").replace(" ", ""))

    select_stmt = f"SELECT id FROM benchmarked_sdk WHERE hash_id = '{hash_id}'"

    cursor.execute(select_stmt)
    existing_id = cursor.fetchone()
    if existing_id is not None:
        return existing_id[0]

    spec_columns = [
        "version_major",
        "version_minor",
        "version_patch",
        "soversion",
        "compiler_id",
        "compiler_version",
        "git_revision",
        "library_arch",
        "system_name",
        "system_processor",
        "system_version",
    ]

    columns = ["hash_id"] + spec_columns
    insert_stmt = "INSERT INTO benchmarked_sdk ({}) VALUES ({})".format(
        ", ".join(columns), ", ".join([args.db_placeholder for _ in columns])
    )

    values = [build_spec[itr] for itr in spec_columns]
    cursor.execute(insert_stmt, (hash_id, *values))

    return cursor.lastrowid


def insert_performance_metrics(
    cursor, app_id, sdk_id, cfg_id, executed_at, perf_record, args
):
    """
    Insert performance metric record
    Args:
        cursor:
            Database cursor
        app_id:
            application database row index
        sdk_id:
            rocprofiler-sdk database row index
        cfg_id:
            configuration info dict from rocprofv3 config json.
            This value should be None if it is a baseline measurement
        executed_at:
            UTC date and time, e.g. 2025-05-13 18:25:37.554604+00:00
        perf_record:
            performance data dict from timem json
        args:
            Command line args (argparse.Namespace)
    Returns:
        id: row id of the insert
    """

    # Insert query matching the schema
    values = [
        "app_id",
        "cfg_id",
        "sdk_id",
        "executed_at",
    ] + CONST_METRIC_LIST

    insert_stmt = "INSERT INTO benchmark_metrics ({}) VALUES ({})".format(
        ", ".join(values), ", ".join([args.db_placeholder for _ in values])
    )

    _record = perf_record["timemory"]["timem"][0]

    # timem records are _record[<name>]["value"] so use the CONST_TIMEM_METRIC_MAP dict to map
    # the database metric name to the dict entry in the timem record, e.g. wall_time is
    # located in _record["wall_clock"]["value"]
    values = [app_id, cfg_id, sdk_id, executed_at] + [
        _record[CONST_TIMEM_METRIC_MAP[itr]]["value"] for itr in CONST_METRIC_LIST
    ]

    cursor.execute(insert_stmt, values)

    return cursor.lastrowid


def execute_profile(
    cmd,
    args,
    env=dict(os.environ),
    launcher=[],
):

    _argt = os.path.basename(cmd[0])
    _pid = os.getpid()
    _data_dir = args.data_dir
    _cmd = cmd

    _config_file = None
    _rocprof_out = f"profile-{_pid}-{_argt}"
    _config_file = f"{_data_dir}/{_rocprof_out}_config.json"
    _cmd = (
        [
            args.rocprofv3,
            "--output-config",
            "-d",
            _data_dir,
            "-o",
            _rocprof_out,
            "--benchmark-mode",
            "execution-profile",
            "--sys-trace",
        ]
        + ["--"]
        + _cmd
    )

    log_message("Executing profile run '{}'".format(" ".join(_cmd)))

    _cmd = launcher + _cmd

    exit_code = subprocess.check_call(_cmd, env=env)

    if exit_code != 0:
        fatal_error("Application exited with non-zero exit code", exit_code)

    _profile = None
    with open(_config_file, "r") as ifs:
        _cfg = json.load(ifs)
        _profile = _cfg["rocprofiler-sdk-tool"][0]["metadata"]["profile"]

    if not args.keep_data:
        for fname in [_config_file]:
            if fname is not None and os.path.exists(fname):
                try:
                    os.remove(fname)
                except Exception as e:
                    sys.stderr.write(f"Error removing {fname}: {e}\n")
                    sys.stderr.flush()

    return _profile


def execute_run(
    cursor,
    benchmark_mode,
    rocprofv3_args,
    cmd,
    app_id,
    nitr,
    args,
    env=dict(os.environ),
    launcher=[],
    is_warmup=False,
):

    _argt = os.path.basename(cmd[0])
    _pid = os.getpid()
    _data_dir = args.data_dir
    _cmd = cmd

    _timem_file = f"{_data_dir}/perf-{_pid}-{_argt}-{nitr}"
    _config_file = None

    if rocprofv3_args:
        _rocprof_out = f"out-{_pid}-{_argt}-{nitr}"
        _config_file = f"{_data_dir}/{_rocprof_out}_config.json"
        _cmd = (
            [
                args.rocprofv3,
                "--output-config",
                "-d",
                _data_dir,
                "-o",
                _rocprof_out,
                "--benchmark-mode",
                benchmark_mode,
            ]
            + rocprofv3_args
            + ["--"]
            + _cmd
        )

    if is_warmup:
        log_message("Executing warmup iteration '{}'".format(" ".join(launcher + _cmd)))
    else:
        log_message("Executing '{}'".format(" ".join(_cmd)))

    _cmd = launcher + [args.timem, "-o", _timem_file, "--"] + _cmd
    _now = datetime.datetime.now(datetime.timezone.utc)

    exit_code = subprocess.check_call(_cmd, env=env)

    if exit_code != 0:
        fatal_error("Application exited with non-zero exit code", exit_code)

    if not is_warmup:
        with open(f"{_timem_file}.json", "r") as ifs:
            _perf = json.load(ifs)

        if _config_file is not None:
            with open(_config_file, "r") as ifs:
                _cfg = json.load(ifs)
            _sdk_id = insert_benchmarked_sdk(cursor, _cfg, args)
            _cfg_id = insert_benchmark_config(cursor, _sdk_id, _cfg, args)
        else:
            _sdk_id = None
            _cfg_id = insert_benchmark_config(cursor, _sdk_id, None, args)

        _perf_id = insert_performance_metrics(
            cursor, app_id, _sdk_id, _cfg_id, _now, _perf, args
        )

        log_message(
            "Inserted benchmarked sdk (app id {}, cfg id {}, sdk id {}, metrics id {})".format(
                app_id, _cfg_id, _sdk_id, _perf_id
            )
        )
    else:
        _sdk_id = None
        _perf_id = None

    if not args.keep_data:
        for fname in [f"{_timem_file}.json", _config_file]:
            if fname is not None and os.path.exists(fname):
                try:
                    os.remove(fname)
                except Exception as e:
                    sys.stderr.write(f"Error removing {fname}: {e}\n")
                    sys.stderr.flush()

    return (_sdk_id, _perf_id)


def parse_args(argv=None):
    """
    Parse the command line arguments
    Args:
        argv:
            Command line arguments
    Returns:
        argparse.Namespace
    """

    if argv is None:
        argv = sys.argv[1:]

    # default path to timem if user doesn't provide it
    default_timem_path = (
        CONST_LOCAL_TIMEM_PATH if os.path.exists(CONST_LOCAL_TIMEM_PATH) else "timem"
    )

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Execute one or more applications",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the version information and exit",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to YAML input file",
    )

    parser.add_argument(
        "-n",
        "--num-iterations",
        type=int,
        help="Number of iterations to run",
        default=3,
    )

    parser.add_argument(
        "-w",
        "--num-warmup-iterations",
        type=int,
        help="Number of warmup iterations to run",
        default=1,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory of the timem and rocprofv3 output",
        default="/tmp/rocprofv3-benchmark",
        metavar="PATH",
    )

    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Do not delete the timem and rocprofv3 output",
    )

    exe_options = parser.add_argument_group("Executable options")

    exe_options.add_argument(
        "--timem",
        type=str,
        help="Path to timem executable",
        default=default_timem_path,
        metavar="FILEPATH",
    )

    exe_options.add_argument(
        "--rocprofv3",
        type=str,
        help="Path to rocprofv3 executable",
        default="rocprofv3",
        metavar="FILEPATH",
    )

    database_options = parser.add_argument_group("Database options (generic)")

    database_options.add_argument(
        "--db-backend",
        type=str,
        help="Select the database backend",
        choices=("sqlite3", "mysql"),
        default="sqlite3",
    )

    database_options.add_argument(
        "--db-schema",
        type=str,
        help="Database schema file(s)",
        nargs="+",
        default=[
            os.path.join(
                CONST_THIS_DIR, "..", "share", "rocprofiler-sdk", "benchmark_tables.sql"
            )
        ],
        metavar="FILEPATH",
    )

    database_options.add_argument(
        "--db-views",
        type=str,
        help="Database views file(s)",
        nargs="+",
        default=[
            os.path.join(
                CONST_THIS_DIR, "..", "share", "rocprofiler-sdk", "benchmark_views.sql"
            )
        ],
        metavar="FILEPATH",
    )

    job_filter_options = parser.add_argument_group("Job filter options")

    job_filter_options.add_argument(
        "--name",
        "--filter-name",
        type=str,
        nargs="+",
        help="Run a specific job name",
        default=None,
        metavar="REGEX",
        dest="filter_name",
    )

    job_filter_options.add_argument(
        "--group",
        "--filter-group",
        type=str,
        nargs="+",
        help="Run jobs in the designated group(s)",
        default=None,
        metavar="REGEX",
        dest="filter_group",
    )

    job_filter_options.add_argument(
        "--filter-rocprofv3",
        type=str,
        nargs="+",
        help="Run rocprofv3 jobs with matching rocprofv3 config label(s)",
        default=None,
        metavar="REGEX",
    )

    job_filter_options.add_argument(
        "--filter-benchmark",
        type=str,
        nargs="+",
        choices=(
            "baseline",
            "disabled-sdk-contexts",
            "sdk-buffer-overhead",
            "sdk-callback-overhead",
            "tool-runtime-overhead",
        ),
        help="Run rocprofv3 jobs with matching rocprofv3 config label(s)",
        default=None,
        metavar="REGEX",
    )

    shuffle_options = parser.add_argument_group("Randomization options")

    shuffle_options.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly shuffle the ordering of the benchmark modes and rocprofv3 parameters",
    )

    shuffle_options.add_argument(
        "--shuffle-rocprofv3",
        action="store_true",
        help="Randomly shuffle the ordering of the rocprofv3 parameters",
    )

    shuffle_options.add_argument(
        "--shuffle-benchmark",
        action="store_true",
        help="Randomly shuffle the ordering of the benchmark modes",
    )

    sqlite3_options = parser.add_argument_group("SQLite3 options")

    sqlite3_options.add_argument(
        "--sqlite3-database",
        help="SQLite3 database filename",
        type=str,
        default="benchmark.db",
        metavar="FILENAME",
    )

    mysql_options = parser.add_argument_group("MySQL options")

    mysql_options.add_argument(
        "--mysql-user",
        type=str,
        help="Database user (MySQL only)",
        default="root",
        metavar="USERNAME",
    )
    mysql_options.add_argument(
        "--mysql-passwd",
        type=str,
        help="Database password (MySQL only)",
        default=None,
        metavar="PASSWORD",
    )
    mysql_options.add_argument(
        "--mysql-host",
        type=str,
        help="Database remote host (MySQL only)",
        default="db.rocprofiler-benchmarking.svc.cluster.local",
        metavar="URL",
    )
    mysql_options.add_argument(
        "--mysql-database",
        type=str,
        help="Database name (MySQL only)",
        default="benchmark_db",
        metavar="DATABASE_NAME",
    )

    args = parser.parse_args(argv)

    if args.db_backend == "sqlite3":
        setattr(args, "db_placeholder", "?")
    elif args.db_backend == "mysql":
        setattr(args, "db_placeholder", "%s")
    else:
        raise ValueError(f"db_placeholder needs to be specified for {args.db_backend}")

    return args


def execute_input(connection, cursor, args):
    """
    Executes YAML input
    """

    def apply_filter(filters, configs):
        if not isinstance(filters, list):
            filters = [filters]
        if not isinstance(configs, list):
            configs = [configs]

        matching = []
        for fitr in filters:
            matching += [itr for itr in configs if re.match(fitr, itr)]
        return matching

    def shuffle(data):
        if isinstance(data, dict):
            return dict(
                [[itr, data[itr]] for itr in random.sample(list(data.keys()), len(data))]
            )
        return random.sample(data, len(data))

    with open(args.input, "r") as ifs:
        spec = yaml.safe_load(ifs)

        for job in spec["jobs"]:
            # populate these specs from the default if they were not explicitly provided
            for param in [
                "command",
                "benchmark",
                "rocprofv3",
                "name",
                "group",
                "environment",
                "environment-ignore",
                "launcher",
            ]:
                if param not in job and param in spec["defaults"]:
                    job[param] = spec["defaults"][param]

            # job name filtering
            if args.filter_name is not None and job["name"]:
                if not apply_filter(args.filter_name, job["name"]):
                    continue

            # job group filtering
            if args.filter_group and job["group"]:
                if not apply_filter(args.filter_group, job["group"]):
                    continue

            benchmarks = job["benchmark"] if "benchmark" in job else []
            if args.filter_benchmark and benchmarks:
                benchmarks = apply_filter(args.filter_benchmark, benchmarks)

            if not benchmarks:
                continue

            launcher = job["launcher"] if "launcher" in job else []
            environ = dict(os.environ)

            if "environment" in job:
                for key, eitr in job["environment"].items():
                    environ[key] = f"{eitr}"

            if "environment-ignore" in job:
                environ_keys = list(environ.keys())
                matching = []
                for eitr in job["environment-ignore"]:
                    matching += [
                        itr for itr in environ_keys if re.match(f"^({eitr})$", itr)
                    ]

                for mitr in matching:
                    del environ[mitr]
                    assert environ.get(mitr, None) is None

            # env substitution
            for key, eitr in dict(os.environ).items():
                # env spec has this value
                if key in environ:
                    if key in environ[key]:
                        environ[key] = (
                            environ[key]
                            .replace(f"${key}", eitr)
                            .replace("${}{}{}".format("{", key, "}"), eitr)
                        )

            launcher = [f"{itr}" for itr in launcher]
            cmd = [f"{itr}" for itr in job["command"]]

            app_id = find_benchmarked_app(cursor, cmd, launcher, environ)
            profile = None

            if app_id is None:
                profile = execute_profile(
                    cmd,
                    args,
                    env=environ,
                    launcher=launcher,
                )

            app_id = insert_benchmarked_app(
                cursor, cmd, launcher, environ, app_id, profile, args
            )

            for nitr in range(0, args.num_warmup_iterations):
                execute_run(
                    cursor,
                    "baseline",
                    None,
                    cmd,
                    app_id,
                    nitr,
                    args,
                    env=environ,
                    launcher=launcher,
                    is_warmup=True,
                )

            for nitr in range(0, args.num_iterations):

                if args.shuffle or args.shuffle_benchmark:
                    _orig_benchmarks = benchmarks[:]
                    benchmarks = shuffle(_orig_benchmarks)
                    log_message(
                        f"Shuffled benchmarks: {_orig_benchmarks} => {benchmarks}"
                    )

                for benchmark_mode in benchmarks:

                    # baseline requires special handling since it doesn't use rocprofv3
                    if benchmark_mode == "baseline":

                        # generate the baselines
                        execute_run(
                            cursor,
                            benchmark_mode,
                            None,
                            cmd,
                            app_id,
                            nitr,
                            args,
                            env=environ,
                            launcher=launcher,
                        )

                    elif benchmark_mode in (
                        "disabled-sdk-contexts",
                        "sdk-buffer-overhead",
                        "sdk-callback-overhead",
                        "tool-runtime-overhead",
                    ):

                        # loop over the rocprofv3 configurations
                        rocprofv3_args_config = job["rocprofv3"]
                        if isinstance(rocprofv3_args_config, (list, tuple)):
                            rocprofv3_args_config = dict(
                                [["", itr] for itr in rocprofv3_args_config]
                            )

                        if args.shuffle or args.shuffle_rocprofv3:
                            _orig_rocprofv3_cfg = rocprofv3_args_config.keys()
                            rocprofv3_args_config = shuffle(rocprofv3_args_config)
                            log_message(
                                f"Shuffled rocprofv3: {list(_orig_rocprofv3_cfg)} => {list(rocprofv3_args_config.keys())}"
                            )

                        for label, rocprofv3_args in rocprofv3_args_config.items():

                            # job name filtering
                            if args.filter_rocprofv3 is not None and label:
                                if not apply_filter(args.filter_rocprofv3, [label]):
                                    continue

                            execute_run(
                                cursor,
                                benchmark_mode,
                                rocprofv3_args,
                                cmd,
                                app_id,
                                nitr,
                                args,
                                env=environ,
                                launcher=launcher,
                            )

                    else:
                        raise ValueError(f"Unsupported benchmark mode: {benchmark_mode}")

                # Commit the transaction
                connection.commit()

    # Commit the transaction
    connection.commit()


def main():
    """
    Main control flow function
    """
    args = parse_args(sys.argv[1:])

    if args.version:
        for key, itr in CONST_VERSION_INFO.items():
            print(f"    {key:>16}: {itr}")
        return

    # Connect to the database
    connection, cursor = connect_to_database(args)

    if args.input is not None:
        execute_input(connection, cursor, args)

        log_message("Data has been inserted into the database successfully!")

    # Generate the analysis
    generate_statistics(connection, cursor, args)

    log_message("Statistics data has been successfully updated!")

    generate_analysis(connection, cursor, args)

    log_message("Analysis has been completed successfully!")

    # Close the connection
    cursor.close()
    connection.close()

    log_message("Connection has been closed successfully!")


def generate_statistics(connection, cursor, args):

    def construct_where_condition(data, conditional="AND"):
        conditions = []

        for key, value in data.items():
            if value is None:
                conditions.append(f"{key} IS NULL")
            else:
                if isinstance(value, str):
                    value_str = f"'{value}'"
                else:
                    value_str = str(value)
                conditions.append(f"{key} = {value_str}")

        return f" {conditional} ".join(conditions)

    select_stmt = "SELECT DISTINCT app_id, cfg_id, sdk_id FROM benchmark_metrics"
    cursor.execute(select_stmt)

    selection_stmts = []
    for row in cursor.fetchall():
        params = dict(zip(["app_id", "cfg_id", "sdk_id"], row))
        where_cond = construct_where_condition(params)

        for metric in CONST_METRIC_LIST:
            selections = ", ".join(
                [
                    f"{itr}({metric})"
                    for itr in ["COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV_SAMP"]
                ]
            )
            select_stmt = f"SELECT {selections} FROM benchmark_metrics WHERE {where_cond}"
            selection_stmts.append([metric, params, select_stmt])

    for metric, params, select_stmt in selection_stmts:
        cursor.execute(select_stmt)

        rows = cursor.fetchall()
        if len(rows) != 1:
            fatal_error(
                f"select statement should only have returned one row. found {len(rows)}. {metric} :: {params}"
            )

        stats = ["count", "sum", "mean", "min", "max", "std_dev"]
        data = dict(zip(stats, rows[0]))

        where_cond = "metric_name LIKE '{}' AND {}".format(
            metric, construct_where_condition(params)
        )

        existing_stmt = f"SELECT id FROM benchmark_statistics WHERE {where_cond}"

        cursor.execute(existing_stmt)
        existing_id = cursor.fetchone()
        if existing_id is not None:

            columns = stats[:]
            insert_stmt = "UPDATE benchmark_statistics SET {} WHERE {}".format(
                ", ".join([f"{itr} = {args.db_placeholder}" for itr in columns]),
                where_cond,
            )

            values = [data[itr] for itr in stats]
            cursor.execute(insert_stmt, values)

        else:

            columns = ["app_id", "cfg_id", "sdk_id", "metric_name", "metric_unit"] + stats
            insert_stmt = "INSERT INTO benchmark_statistics ({}) VALUES ({})".format(
                ", ".join(columns), ", ".join([args.db_placeholder for _ in columns])
            )

            values = [
                params["app_id"],
                params["cfg_id"],
                params["sdk_id"],
                metric,
                CONST_METRIC_UNIT_MAP[metric],
            ] + [data[itr] for itr in stats]

            cursor.execute(insert_stmt, values)

    # Commit the transaction
    connection.commit()


def generate_analysis(connection, cursor, args):
    pass


# Example usage
if __name__ == "__main__":
    main()
