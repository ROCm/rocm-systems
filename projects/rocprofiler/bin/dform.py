################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

import os
from sqlitedb import SQLiteDB


def gen_message(outfile):
    if outfile != "":
        print("File '" + outfile + "' is generating")


def post_process_data(db, table_name, outfile=""):
    #  db.add_data_column('A', 'DispDurNs', 'INTEGER', 'BeginNs - DispatchNs')
    #  db.add_data_column('A', 'ComplDurNs', 'INTEGER', 'CompleteNs - EndNs')
    #  db.add_data_column('A', 'TotalDurNs', 'INTEGER', 'CompleteNs - DispatchNs')
    #  db.add_data_column(table_name, 'TimeNs', 'INTEGER', 'BeginNs - %d' % start_ns)
    db.add_data_column(table_name, "DurationNs", "INTEGER", "EndNs - BeginNs")
    if outfile != "":
        db.dump_csv(table_name, outfile)
    gen_message(outfile)


def gen_data_bins(db, outfile):
    db.execute(
        "create view C as select Name, Calls, TotalDurationNs, TotalDurationNs/Calls as AverageNs, TotalDurationNs*100.0/(select sum(TotalDurationNs) from %s) as Percentage from %s order by TotalDurationNs desc;"
        % ("B", "B")
    )
    db.dump_csv("C", outfile)
    db.execute("DROP VIEW C")


def gen_table_bins(db, table, outfile, name_var, dur_ns_var):
    db.execute(
        "create view B as select (%s) as Name, count(%s) as Calls, sum(%s) as TotalDurationNs from %s group by %s"
        % (name_var, name_var, dur_ns_var, table, name_var)
    )
    gen_data_bins(db, outfile)
    db.execute("DROP VIEW B")
    gen_message(outfile)


def gen_api_json_trace(db, table, start_ns, outfile):
    db.execute(
        'create view B as select "Index", Name as name, __section as pid, __lane as tid, ((BeginNs - %d)/1000) as ts, (DurationNs/1000) as dur from %s;'
        % (start_ns, table)
    )
    db.dump_json("B", table, outfile)
    db.execute("DROP VIEW B")
    gen_message(outfile)


def gen_ext_json_trace(db, table, start_ns, outfile):
    db.execute(
        "create view B as select Name as name, __section as pid, __lane as tid, ((BeginNs - %d)/1000) as ts, ((EndNs - BeginNs)/1000) as dur from %s;"
        % (start_ns, table)
    )
    db.dump_json("B", table, outfile)
    db.execute("DROP VIEW B")
    gen_message(outfile)


def gen_ops_json_trace(db, table, base_pid, start_ns, outfile):
    db.execute(
        'create view B as select "Index", "%s" as name, ("dev-id" + %d) as pid, __lane as tid, ((BeginNs - %d)/1000) as ts, (DurationNs/1000) as dur from %s;'
        % (
            "roctx-range" if "ROCP_RENAME_KERNEL" in os.environ else "Name",
            base_pid,
            start_ns,
            table,
        )
    )
    db.dump_json("B", table, outfile)
    db.execute("DROP VIEW B")
    gen_message(outfile)


def gen_kernel_json_trace(db, table, base_pid, start_ns, outfile):
    db.execute(
        'create view B as select "Index", KernelName as name, ("gpu-id" + %d) as pid, tid, ((BeginNs - %d)/1000) as ts, (DurationNs/1000) as dur from %s;'
        % (base_pid, start_ns, table)
    )
    db.dump_json("B", table, outfile)
    db.execute("DROP VIEW B")
    gen_message(outfile)


##############################################################################################
