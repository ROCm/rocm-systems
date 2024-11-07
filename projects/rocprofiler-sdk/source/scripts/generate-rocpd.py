#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

import sys
import json
import time
import sqlite3
import argparse

__author__ = "AMD"
__copyright__ = "Copyright 2023, Advanced Micro Devices, Inc."
__license__ = "MIT"
__maintainer__ = "AMD"
__status__ = "Development"

"""
This script converts one or more JSON output files from rocprofv3 into a
single SQLite database conforming to the rocpd SQL Schema.
"""

# this is the list of APIs whose records are inserted into API table which
# needs to be updated whenever tracing support for a new API is added
rocprofv3_apis = ("hip_api", "hsa_api", "marker_api", "rccl_api")


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, d):
        super(dotdict, self).__init__(d)
        for k, v in self.items():
            if isinstance(v, dict):
                self.__setitem__(k, dotdict(v))
            elif isinstance(v, (list, tuple)):
                self.__setitem__(
                    k,
                    [dotdict(i) if isinstance(i, (list, tuple, dict)) else i for i in v],
                )


def dump_table(table):
    cursor.execute(f"SELECT * FROM {table};")
    results = cursor.fetchall()
    print(f"\n\n##### {table} #####\n")
    for itr in results:
        print("  | {}".format(" | ".join([f"{val}" for val in list(itr)])))
    print("")


def execute_raw_sql_statements(cursor, statements):
    """Helper function for executing a sequence of raw SQL statements"""

    for itr in [
        "{};".format(itr.strip()) for itr in statements.strip().split(";") if itr
    ]:
        try:
            cursor.execute(f"{itr}")
        except sqlite3.Error as err:
            sys.stderr.write(f"SQLite3 error: {err}\nStatement:\n\t{itr}\n")
            sys.stderr.flush()
            raise err


def create_schema(cursor):

    # Create table
    table_schema = """
    CREATE TABLE IF NOT EXISTS "rocpd_metadata" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "tag" varchar(4096) NOT NULL, "value" varchar(4096) NOT NULL);
    CREATE TABLE IF NOT EXISTS "rocpd_string" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "string" varchar(4096) NOT NULL UNIQUE ON CONFLICT IGNORE);
    CREATE TABLE IF NOT EXISTS "rocpd_op" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "gpuId" integer NOT NULL, "queueId" integer NOT NULL, "sequenceId" integer NOT NULL, "completionSignal" varchar(18) NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "description_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "opType_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);
    CREATE TABLE IF NOT EXISTS "rocpd_api" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "pid" integer NOT NULL, "tid" integer NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "apiName_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "args_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);
    CREATE TABLE IF NOT EXISTS "rocpd_api_ops" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_id" integer NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "op_id" integer NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED);
    -- optional
    CREATE TABLE IF NOT EXISTS "rocpd_kernelcodeobject" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "vgpr" integer NOT NULL, "sgpr" integer NOT NULL, "fbar" integer NOT NULL, "kernel_id" integer NOT NULL);
    CREATE TABLE IF NOT EXISTS "rocpd_kernelapi" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_ptr_id" integer NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "stream" varchar(18) NOT NULL, "gridX" integer NOT NULL, "gridY" integer NOT NULL, "gridZ" integer NOT NULL, "workgroupX" integer NOT NULL, "workgroupY" integer NOT NULL, "workgroupZ" integer NOT NULL, "groupSegmentSize" integer NOT NULL, "privateSegmentSize" integer NOT NULL, "kernelArgAddress" varchar(18) NOT NULL, "aquireFence" varchar(8) NOT NULL, "releaseFence" varchar(8) NOT NULL, "codeObject_id" integer NOT NULL REFERENCES "rocpd_kernelcodeobject" ("id") DEFERRABLE INITIALLY DEFERRED, "kernelName_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED);
    CREATE TABLE IF NOT EXISTS "rocpd_copyapi" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_ptr_id" integer NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "stream" varchar(18) NOT NULL, "size" integer NOT NULL, "width" integer NOT NULL, "height" integer NOT NULL, "kind" integer NOT NULL, "dst" varchar(18) NOT NULL, "src" varchar(18) NOT NULL, "dstDevice" integer NOT NULL, "srcDevice" integer NOT NULL, "sync" bool NOT NULL, "pinned" bool NOT NULL);

    INSERT INTO "rocpd_metadata"(tag, value) VALUES ("schema_version", "2");

    --CREATE TABLE IF NOT EXISTS "rocpd_monitor" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "deviceType" varchar(16) NOT NULL, "deviceId" integer NOT NULL, "monitorType" varchar(16) NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "value" varchar(255) NOT NULL);
    --CREATE TABLE IF NOT EXISTS "rocpd_barrierop" ("op_ptr_id" integer NOT NULL PRIMARY KEY REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED, "signalCount" integer NOT NULL, "aquireFence" varchar(8) NOT NULL, "releaseFence" varchar(8) NOT NULL);
    --CREATE TABLE IF NOT EXISTS "rocpd_op_inputSignals" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "from_op_id" integer NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED, "to_op_id" integer NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED);
    """

    execute_raw_sql_statements(cursor, table_schema)


def finalize_schema(cursor):

    table_schema = """
    CREATE VIEW api AS SELECT rocpd_api.id,pid,tid,start,end,A.string AS apiName, B.string AS args FROM rocpd_api
        INNER JOIN rocpd_string A ON A.id = rocpd_api.apiName_id
        INNER JOIN rocpd_string B ON B.id = rocpd_api.args_id;
    CREATE VIEW op AS SELECT rocpd_op.id,gpuId,queueId,sequenceId,start,end,A.string AS description, B.string AS opType FROM rocpd_op
        INNER JOIN rocpd_string A ON A.id = rocpd_op.description_id
        INNER JOIN rocpd_string B ON B.id = rocpd_op.opType_id;
    CREATE VIEW busy AS SELECT A.gpuId, GpuTime, WallTime, GpuTime*1.0/WallTime AS Busy FROM (SELECT gpuId, sum(end-start) AS GpuTime FROM rocpd_op GROUP BY gpuId) A
        INNER JOIN (SELECT max(end) - min(start) AS WallTime FROM rocpd_op);
    CREATE VIEW top AS SELECT C.string AS Name, count(C.string) AS TotalCalls, sum(A.end-A.start) / 1000 AS TotalDuration, (sum(A.end-A.start)/count(C.string))/ 1000 AS Ave, sum(A.end-A.start) * 100.0 / (SELECT sum(A.end-A.start) FROM rocpd_op A) AS Percentage FROM (SELECT opType_id AS name_id, start, end FROM rocpd_op WHERE description_id in (SELECT id FROM rocpd_string WHERE string='')
        UNION SELECT description_id, start, end FROM rocpd_op WHERE description_id not in (SELECT id FROM rocpd_string WHERE string='')) A
        JOIN rocpd_string C on C.id = A.name_id GROUP BY Name ORDER BY TotalDuration desc;
    CREATE VIEW ktop AS SELECT C.string AS Name, count(C.string) AS TotalCalls, sum(A.end-A.start) / 1000 AS TotalDuration, (sum(A.end-A.start)/count(C.string))/ 1000 AS Ave, sum(A.end-A.start) * 100.0 / (SELECT sum(A.end-A.start) FROM rocpd_api A
        JOIN rocpd_kernelapi B on B.api_ptr_id = A.id) AS Percentage FROM rocpd_api A
        JOIN rocpd_kernelapi B on B.api_ptr_id = A.id
        JOIN rocpd_string C on C.id = B.kernelname_id GROUP BY Name ORDER BY TotalDuration desc;
    CREATE VIEW kernel AS SELECT B.id, gpuId, queueId, sequenceId, start, end, (end-start) AS duration, stream, gridX, gridY, gridz, workgroupX, workgroupY, workgroupZ, groupSegmentSize, privateSegmentSize, D.string AS kernelName FROM rocpd_api_ops A
        JOIN rocpd_op B on B.id = A.op_id
        JOIN rocpd_kernelapi C ON C.api_ptr_id = A.api_id
        JOIN rocpd_string D on D.id = kernelName_id;
    CREATE VIEW copy AS SELECT B.id, pid, tid, start, end, C.string AS apiName, stream, size, width, height, kind, dst, src, dstDevice, srcDevice, sync, pinned FROM rocpd_copyApi A
        JOIN rocpd_api B ON B.id = A.api_ptr_id
        JOIN rocpd_string C on C.id = B.apiname_id;
    CREATE VIEW copyop AS SELECT B.id, gpuId, queueId, sequenceId, B.start, B.end, (B.end-B.start) AS duration, stream, size, width, height, kind, dst, src, dstDevice, srcDevice, sync, pinned, E.string AS apiName FROM rocpd_api_ops A
        JOIN rocpd_op B ON B.id = A.op_id
        JOIN rocpd_copyapi C ON C.api_ptr_id = A.api_id
        JOIN rocpd_api D on D.id = A.api_id
        JOIN rocpd_string E ON E.id = D.apiName_id;
    """

    execute_raw_sql_statements(cursor, table_schema)


def normalize_timestamps(itr):
    """Make all timestamps relative to the time of rocprofv3 initialization within the application"""

    def _normalize_timestamp_impl(value):
        return value - itr.metadata.init_time

    min_val = None
    for aitr in [
        "hip_api",
        "hsa_api",
        "marker_api",
        "rccl_api",
        "kernel_dispatch",
        "memory_copy",
    ]:
        for ritr in itr.buffer_records[aitr]:
            ritr.start_timestamp = _normalize_timestamp_impl(ritr.start_timestamp)
            ritr.end_timestamp = _normalize_timestamp_impl(ritr.end_timestamp)
            min_val = (
                min([ritr.start_timestamp, min_val])
                if min_val is not None
                else ritr.start_timestamp
            )

    print(f"  - starting timestamp normalized down to a minimum of {min_val} nsec")
    sys.stdout.flush()

    return itr


def insert_strings(cursor, itr):
    """Populate the strings table with all the strings which will be referenced by various records"""

    strings = []

    def append_strings(*args):
        nonlocal strings

        for aitr in args:
            if isinstance(aitr, list):
                strings += aitr
            else:
                strings.append(aitr)

    append_strings("UserMarker")

    for aitr in itr.agents:
        append_strings(aitr.name, aitr.vendor_name, aitr.product_name, aitr.model_name)
    for ritr in itr.strings.callback_records:
        append_strings(ritr.kind, ritr.operations)
    for ritr in itr.strings.buffer_records:
        append_strings(ritr.kind, ritr.operations)
    for ritr in itr.strings.marker_api:
        append_strings(ritr.value)
    for ritr in itr.strings.counters.dimension_ids:
        append_strings(ritr.name)
    for ritr in itr.strings.correlation_id.external:
        append_strings(ritr.value)
    for ritr in itr.kernel_symbols:
        append_strings(ritr.kernel_name)
        append_strings(ritr.formatted_kernel_name)
        append_strings(ritr.demangled_kernel_name)
        append_strings(ritr.truncated_kernel_name)
    for ritr in itr.code_objects:
        append_strings(ritr.uri)

    for itr in sorted(list(set(strings))):
        cursor.execute(f"""INSERT INTO rocpd_string (string) VALUES ('{itr}')""")


def insert_api_data(cursor, itr, corr_id_offset, **kwargs):
    """Add all the HIP, HSA, marker, and RCCL API records to the database.
    Eventually we might want to abstract a way to iterate over the APIs covered
    here instead of maintaining an explicit list.
    """

    marker_message_strings = dict(
        [[eitr.key, eitr.value] for eitr in itr.strings.marker_api]
    )

    def get_api_name(kind, op):
        return itr.strings.buffer_records[kind].operations[op]

    def get_marker_message(name, corr_id):
        return marker_message_strings.get(corr_id, name)

    max_corr_id = 0
    for aitr in rocprofv3_apis:
        for hitr in itr.buffer_records[aitr]:
            corr_id = hitr.correlation_id
            corr_id.internal += corr_id_offset
            name = None
            args = None

            if aitr == "marker_api":
                apiname = get_api_name(hitr.kind, hitr.operation)
                message = get_marker_message(apiname, corr_id.internal)
                mode = kwargs.get("marker_mode", "message")
                assert mode in ("message", "generic", "api")
                if mode == "message":
                    name = message
                    args = 1
                elif mode == "api":
                    name = apiname
                    args = f"(SELECT id FROM rocpd_string WHERE string = '{message}')"
                elif mode == "generic":
                    name = "UserMarker"
                    args = f"(SELECT id FROM rocpd_string WHERE string = '{message}')"
            else:
                name = get_api_name(hitr.kind, hitr.operation)
                args = 1

            assert name is not None
            assert args is not None
            cursor.execute(
                f"""INSERT INTO rocpd_api(id, pid, tid, start, end, apiName_id, args_id)
                        VALUES ({corr_id.internal},
                                {itr.metadata.pid},
                                {hitr.thread_id},
                                {hitr.start_timestamp},
                                {hitr.end_timestamp},
                                (SELECT id FROM rocpd_string WHERE string = '{name}'),
                                {args});
                """
            )
            max_corr_id = max([max_corr_id, corr_id.internal])

    return max_corr_id


def insert_async_data(cursor, itr, corr_id_offset, op_id_offset):
    """Add all the kernel and memory copy records to the database.
    Eventually we might want to handle page-migration, scratch-memory, etc. but,
    at present, rocpd_schema does not support it.
    """

    external_corr_id_strings = dict(
        [[eitr.key, eitr.value] for eitr in itr.strings.correlation_id.external]
    )

    def get_api_name(kind, op=None):
        return (
            itr.strings.buffer_records[kind].operations[op]
            if op is not None
            else itr.strings.buffer_records[kind].kind
        )

    def get_kernel_symbol(kernid):
        return itr.kernel_symbols[kernid]

    def get_kernel_name(kernid, externid):
        if externid > 0:
            return external_corr_id_strings[externid]
        return get_kernel_symbol(kernid).formatted_kernel_name

    def get_agent_id(agent_id):
        for aitr in itr.agents:
            if aitr.id.handle == agent_id.handle:
                return aitr.node_id
        return None

    for kitr in itr.kernel_symbols:
        sgpr = kitr.sgpr_count if "sgpr_count" in kitr.keys() else 0
        arch_vgpr = kitr.arch_vgpr_count if "arch_vgpr_count" in kitr.keys() else 0
        accum_vgpr = kitr.accum_vgpr_count if "accum_vgpr_count" in kitr.keys() else 0
        vgpr = arch_vgpr + accum_vgpr

        cursor.execute(
            f"""INSERT INTO rocpd_kernelcodeobject(vgpr, sgpr, fbar, kernel_id)
                    VALUES ({vgpr}, {sgpr}, 0, {kitr.kernel_id});
            """
        )

    op_id = op_id_offset
    for kitr in itr.buffer_records.kernel_dispatch:
        kind_name = get_api_name(kitr.kind)
        info = kitr.dispatch_info
        kernel_id = info.kernel_id
        queue_id = info.queue_id.handle
        corr_id = kitr.correlation_id
        grid = info.grid_size
        workgroup = info.workgroup_size
        kern_name = get_kernel_name(kernel_id, corr_id.external)
        gpu_id = get_agent_id(info.agent_id)
        ksym = get_kernel_symbol(kernel_id)
        kernel_arg_addr = "{:#x}".format(ksym.kernel_object)
        corr_id.internal += corr_id_offset

        cursor.execute(
            f"""INSERT INTO rocpd_kernelapi(api_ptr_id,
                                            stream,
                                            gridX, gridY, gridZ,
                                            workgroupX, workgroupY, workgroupZ,
                                            groupSegmentSize, privateSegmentSize,
                                            kernelArgAddress, aquireFence, releaseFence,
                                            codeObject_id, kernelName_id)
                VALUES ({corr_id.internal},
                        {queue_id},
                        {grid.x}, {grid.y}, {grid.z},
                        {workgroup.x}, {workgroup.y}, {workgroup.z},
                        {info.group_segment_size}, {info.private_segment_size},
                        '{kernel_arg_addr}', '', '',
                        (SELECT id FROM rocpd_kernelcodeobject WHERE kernel_id = {ksym.kernel_id}),
                        (SELECT id FROM rocpd_string WHERE string = '{kern_name}'));
            """
        )
        cursor.execute(
            f"""INSERT INTO rocpd_op(id, gpuId, queueId, sequenceId, completionSignal, start, end, description_id, opType_id)
                    VALUES ({op_id},
                            {gpu_id},
                            {queue_id},
                            {corr_id.internal},
                            "",
                            {kitr.start_timestamp},
                            {kitr.end_timestamp},
                            (SELECT id FROM rocpd_string WHERE string = '{kern_name}'),
                            (SELECT id FROM rocpd_string WHERE string = '{kind_name}'));
            """
        )
        cursor.execute(
            f"""INSERT INTO rocpd_api_ops(api_id, op_id)
                    VALUES ({corr_id.internal},
                            {op_id});
            """
        )
        op_id += 1

    for mitr in itr.buffer_records.memory_copy:
        kind_name = get_api_name(mitr.kind)
        op_name = get_api_name(mitr.kind, mitr.operation)
        dst_id = get_agent_id(mitr.dst_agent_id)
        src_id = get_agent_id(mitr.src_agent_id)
        corr_id = mitr.correlation_id
        synced = False
        pinned = False
        corr_id.internal += corr_id_offset

        cursor.execute(
            f"""INSERT INTO rocpd_copyapi(api_ptr_id, stream, size, width, height, kind, src, dst, srcDevice, dstDevice, sync, pinned)
                    VALUES ({corr_id.internal},
                            "",
                            {mitr.bytes},
                            {mitr.bytes},
                            1,
                            (SELECT id FROM rocpd_string WHERE string = '{op_name}'),
                            "",
                            "",
                            {src_id},
                            {dst_id},
                            {synced},
                            {pinned});
            """
        )
        cursor.execute(
            f"""INSERT INTO rocpd_op(id, gpuId, queueId, sequenceId, completionSignal, start, end, description_id, opType_id)
                    VALUES ({op_id},
                            {dst_id},
                            0,
                            {corr_id.internal},
                            "",
                            {mitr.start_timestamp},
                            {mitr.end_timestamp},
                            (SELECT id FROM rocpd_string WHERE string = '{op_name}'),
                            (SELECT id FROM rocpd_string WHERE string = '{kind_name}'));
            """
        )
        cursor.execute(
            f"""INSERT INTO rocpd_api_ops(api_id, op_id)
                    VALUES ({corr_id.internal},
                            {op_id});
            """
        )
        op_id += 1

    return op_id


if __name__ == "__main__":

    rocpd_tables = [
        "metadata",
        "string",
        "api",
        "op",
        "api_ops",
        "copyapi",
        "kernelapi",
        "kernelcodeobject",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Input rocprofv3 JSON files",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "-o", "--output", help="Output database name", type=str, default="example.db"
    )
    parser.add_argument(
        "-n",
        "--normalize-timestamps",
        help="Normalize timestamps relative to the app start time",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--marker-mode",
        help="'generic' is classical rocpd behavior: all marker regions have 'UserMarker' name with message in args; 'message' uses the message as the region name; 'api' uses the name of the marker function with the message in args",
        choices=("generic", "message", "api"),
        type=str,
        default="message",
    )
    parser.add_argument(
        "-d",
        "--dump-tables",
        help="Dump generate rocpd tables to console (for debugging)",
        type=str,
        default=None,
        nargs="*",
        choices=set(rocpd_tables),
    )

    args = parser.parse_args(sys.argv[1:])

    start = time.monotonic_ns()
    print(f"Opening '{args.output}'...")

    # Connect to an SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(args.output)

    # Create a cursor object using the cursor() method
    cursor = conn.cursor()

    create_schema(cursor)

    corr_id_offset = 0
    op_id_offset = 0
    for itr in args.input:
        print(f"Reading '{itr}'...")
        with open(itr, "rb") as f:
            data = dotdict(json.load(f))["rocprofiler-sdk-tool"]
            for ditr in data:
                # normalize the timestamps if requested
                ditr = normalize_timestamps(ditr) if args.normalize_timestamps else ditr

                # create the strings table
                insert_strings(cursor, ditr)

                # insert the api data
                _corr_id_offset = insert_api_data(
                    cursor, ditr, corr_id_offset, marker_mode=args.marker_mode
                )

                # insert the kernel and memory copy data
                _op_id_offset = insert_async_data(
                    cursor, ditr, corr_id_offset, op_id_offset
                )

                # Save (commit) the changes
                conn.commit()

                # update the offsets
                corr_id_offset = _corr_id_offset
                op_id_offset = _op_id_offset

    if args.dump_tables is not None and len(args.dump_tables) == 0:
        args.dump_tables = rocpd_tables

    if args.dump_tables is not None:
        for itr in args.dump_tables:
            dump_table(f"rocpd_{itr}")

    finalize_schema(cursor)
    conn.commit()

    print(f"Closing '{args.output}'...")
    # Close the connection
    conn.close()

    end = time.monotonic_ns()
    elapsed_nsec = end - start
    elapsed_sec = elapsed_nsec / 1.0e9
    print(f"Runtime time (nsec): {elapsed_nsec}")
    print(f"Runtime time (sec) : {elapsed_sec}")
