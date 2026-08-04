#!/usr/bin/env python3
"""RCCL dispatch-timing traces: the same measurements rocprofv3 gave us, taken
by RCCL itself.

With ``RCCL_KERNEL_TIMING=1`` the library records each collective dispatch's
hardware start/end timestamps, and with ``RCCL_TESTS_KERNEL_TIMING=<prefix>``
rccl-tests writes them to ``<prefix>_pid<pid>.csv``, one file per rank, each row
tagged with the ``Config`` string the rocTX marker used to carry.  That makes a
row self-describing: no marker/kernel correlation, no profiler, and the
timestamps are the packet processor's own, in CLOCK_BOOTTIME like rocprof's.

The point of this module is that everything downstream stays as it was.  A
dispatch trace is turned into exactly the ``regions`` structure
``roctx_analyze.correlate`` produces -- (size, in_place) -> {kernel name ->
[durations_ns]} -- so reports, comparisons, records and plots consume it without
knowing where it came from.

What a dispatch trace cannot show is kernels RCCL did not launch: the rocclr
copy/fill helpers and rccl-tests' own verification kernels have no dispatch
record, so the category breakdown holds collectives only and the non-collective
overhead is zero.  That accounting still needs a profiler.

What it adds over rocprof is per-dispatch identity -- rank, launch sequence
number, channel and thread counts -- which is what makes rank skew and dropped
dispatches visible.
"""

import csv
import glob
import os
import re
from collections import defaultdict

# Suffix rccl-tests gives every trace file, one per process.
FILE_SUFFIX_RE = re.compile(r"_pid\d+\.csv$")

# Columns that identify a dispatch trace, distinguishing it from a rocprof
# kernel trace (which also has timestamps) or a baseline CSV.
REQUIRED_COLUMNS = ("Comm_Hash", "Start_Timestamp", "Config")

# Subdirectory holding one run's traces, e.g. all_reduce_bfloat16_rep3_dispatch.
DIR_SUFFIX = "dispatch"

FUNC_NAMES = {
    0: "Broadcast", 1: "Reduce", 2: "AllGather", 3: "ReduceScatter",
    4: "AllReduce", 5: "SendRecv", 6: "Send", 7: "Recv", 8: "AlltoAll",
    9: "Scatter", 10: "Gather", 11: "AlltoAllPivot", 12: "AlltoAllGda",
    13: "AlltoAllvGda", 14: "AllGatherV", 15: "PutSignal", 16: "Signal",
    17: "WaitSignal",
}

DTYPE_NAMES = {
    0: "int8", 1: "uint8", 2: "int32", 3: "uint32", 4: "int64", 5: "uint64",
    6: "half", 7: "float", 8: "double", 9: "bfloat16", 10: "fp8e4m3",
    11: "fp8e5m2",
}


def kernel_name(func, datatype):
    """Name a dispatch the way the kernel it launched would be named.

    ``roctx_analyze`` categorizes by kernel name, so a record has to answer to
    the same ``ncclDevKernel`` pattern to be counted as a collective.
    """
    return "ncclDevKernel_{}_{}".format(
        FUNC_NAMES.get(func, "Func%d" % func),
        DTYPE_NAMES.get(datatype, "type%d" % datatype),
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def is_dispatch_csv(path):
    """True if *path* is a dispatch trace, judged by its header rather than its
    name, so hand-run traces with any prefix are still found."""
    try:
        with open(path, newline="") as f:
            header = f.readline()
    except OSError:
        return False
    return all('"%s"' % col in header or col in header.split(",")
               for col in REQUIRED_COLUMNS)


def discover_dispatch_files(run_dir):
    """Every dispatch trace at or below *run_dir*, sorted."""
    out = []
    for path in sorted(glob.glob(os.path.join(run_dir, "**", "*.csv"), recursive=True)):
        if FILE_SUFFIX_RE.search(os.path.basename(path)) and is_dispatch_csv(path):
            out.append(path)
    return out


def has_dispatch_traces(run_dir):
    if not os.path.isdir(run_dir):
        return False
    return bool(discover_dispatch_files(run_dir))


def _group_regex():
    # Imported lazily: roctx_analyze imports nothing from here, but this module
    # is also usable on its own.
    import roctx_analyze as ra
    collectives = "|".join(re.escape(k) for k in sorted(ra.BUS_BW_FACTOR, key=len, reverse=True))
    return re.compile(
        r"^(?P<collective>" + collectives + r")"
        r"_(?P<dtype>.+)_rep(?P<rep>\d+)_" + DIR_SUFFIX + r"$"
    )


def discover_dispatch_groups(run_dir):
    """(collective, dtype) -> [subdir, ...] for a run laid out by the runner.

    Mirrors ``roctx_analyze.discover_multi_run_groups`` so a dispatch run reads
    like a profiled one.  Returns None when the directory holds no such
    subdirectories.
    """
    if not os.path.isdir(run_dir):
        return None
    pattern = _group_regex()
    groups = defaultdict(list)
    for entry in sorted(os.listdir(run_dir)):
        m = pattern.match(entry)
        if m and os.path.isdir(os.path.join(run_dir, entry)):
            groups[(m.group("collective"), m.group("dtype"))].append(os.path.join(run_dir, entry))
    return dict(groups) if groups else None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_dispatch_csv(path, include_setup=False):
    """Parse one dispatch trace into a list of record dicts.

    Records tagged ``phase=setup`` are the warmup and correctness dispatches
    rccl-tests makes before the timed loop; they are dropped unless asked for,
    which is what the rocTX marker window did for the profiled path.
    """
    import roctx_analyze as ra

    records = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            config = row.get("Config") or ""
            m = ra.MARKER_MSG_RE.search(config)
            if not m:
                continue
            setup = "phase=setup" in config
            if setup and not include_setup:
                continue
            try:
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
                func = int(row["Func"])
                datatype = int(row["Datatype"])
            except (KeyError, TypeError, ValueError):
                continue
            records.append({
                # A file is one process, and a rep is one directory: launch
                # sequence numbers only mean anything within a process, and two
                # ranks are only comparable when they ran together.
                "file": path,
                "rep": os.path.dirname(os.path.abspath(path)),
                "rank": int(row.get("Rank") or 0),
                "seq": int(row.get("Seq") or 0),
                "comm_hash": row.get("Comm_Hash") or "",
                "func": func,
                "datatype": datatype,
                "count": int(row.get("Count") or 0),
                "nchannels": int(row.get("nChannels") or 0),
                "nthreads": int(row.get("nThreads") or 0),
                "ncolls": int(row.get("nColls") or 0),
                "start": start,
                "end": end,
                "name": kernel_name(func, datatype),
                "size": int(m.group("size")),
                "in_place": int(m.group("in_place")),
                "proc": int(m.group("proc")),
                "setup": setup,
            })
    return records


def load_records(paths, include_setup=False):
    """Parse several traces into one list, in file order."""
    out = []
    for p in paths:
        out.extend(parse_dispatch_csv(p, include_setup=include_setup))
    return out


# ---------------------------------------------------------------------------
# Shapes the rest of the tools already understand
# ---------------------------------------------------------------------------

def regions_from_records(records):
    """(size, in_place) -> {kernel name -> [durations_ns]}, as ``correlate``."""
    regions = defaultdict(lambda: defaultdict(list))
    for r in records:
        regions[(r["size"], r["in_place"])][r["name"]].append(r["end"] - r["start"])
    return regions


def collect_regions(dirs, include_setup=False):
    """Regions merged over every dispatch trace under *dirs*.

    Returns (regions, n_files, n_records).
    """
    import roctx_analyze as ra

    regions = defaultdict(lambda: defaultdict(list))
    n_files = n_records = 0
    for d in dirs:
        for path in discover_dispatch_files(d):
            n_files += 1
            recs = parse_dispatch_csv(path, include_setup=include_setup)
            n_records += len(recs)
            ra.merge_regions(regions, regions_from_records(recs))
    return regions, n_files, n_records


def timing_samples(dirs, include_setup=False):
    """Per (size, in_place) launch sequences, as ``collective_timing_samples``.

    ``dur`` is the on-GPU duration of a dispatch and ``gap`` the interval
    between one dispatch ending and the next starting on the same rank.  Unlike
    the profiled path, "same rank" is read off the record instead of inferred
    from which file it came out of, and consecutive dispatches are identified by
    their launch sequence number, so a dispatch RCCL could not time leaves a
    hole rather than a spurious gap.

    Adds ``skew``: for each launch sequence number, how far apart the ranks
    started, which only per-dispatch identity makes measurable.
    """
    out = defaultdict(lambda: {"dur": [], "gap": [], "skew": []})
    by_key = defaultdict(list)
    for d in dirs:
        for path in discover_dispatch_files(d):
            for r in parse_dispatch_csv(path, include_setup=include_setup):
                by_key[(r["size"], r["in_place"])].append(r)

    for key, recs in by_key.items():
        # One series per rank: a process may hold several ranks of the same
        # communicator, each launching on its own stream with its own sequence.
        by_stream = defaultdict(list)
        for r in recs:
            by_stream[(r["file"], r["comm_hash"], r["rank"])].append(r)
        for series in by_stream.values():
            series.sort(key=lambda r: r["seq"])
            prev = None
            for r in series:
                out[key]["dur"].append(r["end"] - r["start"])
                if prev is not None and r["seq"] == prev["seq"] + 1:
                    out[key]["gap"].append(r["start"] - prev["end"])
                prev = r

        # Rank skew needs the same dispatch seen from several ranks, so it only
        # exists for a multi-rank run whose traces were collected together.
        starts_by_seq = defaultdict(list)
        for r in recs:
            starts_by_seq[(r["rep"], r["comm_hash"], r["seq"])].append(r["start"])
        for starts in starts_by_seq.values():
            if len(starts) > 1:
                out[key]["skew"].append(max(starts) - min(starts))
    return dict(out)


def dispatch_stats(dirs, include_setup=False):
    """Run-level facts about the trace itself, for the report header."""
    ranks = set()
    comms = set()
    nchannels = set()
    nthreads = set()
    seq_seen = defaultdict(set)
    total = 0
    for d in dirs:
        for path in discover_dispatch_files(d):
            for r in parse_dispatch_csv(path, include_setup=include_setup):
                total += 1
                ranks.add(r["rank"])
                comms.add(r["comm_hash"])
                nchannels.add(r["nchannels"])
                nthreads.add(r["nthreads"])
                seq_seen[(r["file"], r["comm_hash"], r["rank"],
                          r["size"], r["in_place"])].add(r["seq"])

    # A dispatch RCCL counted but could not time leaves a gap in the sequence.
    missing = 0
    for seqs in seq_seen.values():
        if seqs:
            missing += (max(seqs) - min(seqs) + 1) - len(seqs)
    return {
        "records": total,
        "ranks": len(ranks),
        "comms": len(comms),
        "nchannels": sorted(nchannels),
        "nthreads": sorted(nthreads),
        "untimed": missing,
    }
