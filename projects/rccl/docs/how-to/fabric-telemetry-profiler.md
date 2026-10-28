# Finding whether a slow collective is the fabric's fault or the host's

This guide shows how to run RCCL with the accl profiler plugin and network
telemetry together, and how to read the report they produce. The report
answers one question: **when a collective was slow, was the network fabric
holding it back, or was a host?**

The main signal is PFC pause:

- **rx pause**: the switch told our NIC to stop sending. The fabric was
  congested.
- **tx pause**: our NIC told the switch to stop sending. This host could not
  take data in fast enough.

The report also looks at ECN marks and CNPs, packet loss, receiver-not-ready
errors, and whether other traffic was sharing the link.

Two outputs of the same run are joined:

| Output | Comes from | Holds |
|---|---|---|
| `accl_profiler_rank*.jsonl` | accl profiler plugin | When each collective ran, and its RCCL bytes per NIC |
| `rccl_telemetry_*.json` | RCCL network telemetry | NIC counters (PFC, ECN, bytes) sampled every few ms |

Both use `CLOCK_MONOTONIC`, so on one host they line up directly.

## Requirements

- An RCCL build with network telemetry (in-tree, no build flag) and the accl
  profiler plugin (`librccl-profiler-accl.so`, built by default on Linux).
- The `IB-CAST` network transport (`NCCL_NET=IB-CAST`). Telemetry only covers
  this transport.
- Python 3 on the machine where you make the report. No extra packages.
- For clean per-collective numbers: one GPU per process (`-g 1`) and one
  communicator active at a time. See [Limitations](#limitations).

## Build the plugin

The plugin is built with RCCL. To build it alone:

```bash
cd projects/rccl/plugins/profiler/accl
cmake -S . -B build && cmake --build build
```

An installed RCCL puts it at `lib/librccl-profiler-accl.so`.

## Run

Set these on every rank. Point both output directories at the same shared
path, so every node's files end up in one place.

| Variable | Value | Why |
|---|---|---|
| `NCCL_PROFILER_PLUGIN` | path to `librccl-profiler-accl.so` | Loads the accl profiler |
| `ACCL_PROFILER_OUTPUT_DIR` | shared directory | Where the `.jsonl` files go |
| `RCCL_TELEMETRY_ENABLE` | `1` | Turns telemetry on |
| `RCCL_TELEMETRY_OUTPUT_DIR` | same shared directory | Where the telemetry `.json` files go |
| `RCCL_TELEMETRY_SAMPLE_MS` | `10` | Samples the NIC counters every 10 ms. Without this there is no time series and the report cannot work |
| `NCCL_NET` | `IB-CAST` | The transport telemetry covers |
| `ACCL_PROFILER_MIN_SIZE_BYTES` | optional | Skip collectives smaller than this |
| `ACCL_PROFILER_NET` | optional, `0` | Turn off the per-collective NIC snapshots |

Example, 2 nodes with 8 GPUs each:

```bash
OUT=/shared/run1
mpirun -np 16 -H node1:8,node2:8 \
  -x NCCL_PROFILER_PLUGIN=/path/to/librccl-profiler-accl.so \
  -x ACCL_PROFILER_OUTPUT_DIR=$OUT \
  -x RCCL_TELEMETRY_ENABLE=1 -x RCCL_TELEMETRY_OUTPUT_DIR=$OUT \
  -x RCCL_TELEMETRY_SAMPLE_MS=10 \
  -x NCCL_NET=IB-CAST \
  ./all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 50 -w 5
```

When the job ends, `$OUT` holds one `accl_profiler_rank*.jsonl` per rank and
communicator, and one `rccl_telemetry_*.json` per process.

The sampler keeps at most 100,000 samples per process: about 16 minutes at
10 ms for a rank that drives one NIC. For longer runs, raise
`RCCL_TELEMETRY_SAMPLE_MS`.

## Make the report and the trace

```bash
cd projects/rccl/plugins/profiler/accl
python3 accl_fabric_report.py report --accl $OUT --telemetry $OUT > report.txt
python3 accl_fabric_report.py trace  --accl $OUT --telemetry $OUT -o trace.json
```

Open `trace.json` in [Perfetto](https://ui.perfetto.dev). It has one track per
rank with its collectives as slices, and counter tracks per NIC: PFC pause,
NIC and RCCL bandwidth, ECN and CNP rates, and queue depth. Each host is
shifted to start at 0, because `CLOCK_MONOTONIC` has a different origin on
every host, so lining up hosts in the trace is only approximate.

Useful `report` options:

| Option | Default | Meaning |
|---|---|---|
| `--pfc-prio N` | from the NIC settings or per-priority bytes | PFC priority RCCL traffic uses; needed on NICs that expose neither (AINIC) |
| `--warmup N` | `5` | Skip collectives with sequence number below N |
| `--threshold-pct` | `5` | Pause share of wall time that counts |
| `--threshold-ecn-pct` | `1` | Share of received packets ECN-marked that counts |
| `--threshold-cnps` | `100` | CNPs sent plus received per second that count |
| `--link-busy-pct` | `80` | Link use that counts as busy |
| `--link-other-pct` | `20` | Non-RCCL share of a busy link that counts as shared |
| `--per-nic` | off | One row per NIC instead of only the worst |

## Read the report

The header says which traffic class RCCL sent with, which priorities have PFC
enabled, which PFC priority RCCL's traffic rides and how that was found, and
the thresholds.

### How the report finds RCCL's PFC priority

PFC works per priority, so the report first needs the priority RCCL's packets
use. It never guesses. In this order:

1. `--pfc-prio N`, if you pass it.
2. The NIC's DCB settings, which telemetry reads at startup the way
   `dcb pfc show` and `dcb app show` do, with no root needed. An app entry for
   RCCL's DSCP (traffic class ÷ 4) names the priority; so does one for RoCE's
   UDP port 4791 on Thor2, the only NIC where that was seen to apply. The same
   settings say whether PFC is enabled on it.
3. Per-priority byte counters: the priority that carried the job's bytes.
   These include other traffic, so they overrule the settings only when the
   configured priority carried almost nothing. Either way a disagreement is
   printed as a warning.

If none of these work, the priority is UNKNOWN. Pause is then not judged, and
the header says why and asks for `--pfc-prio`.

What this gives on the NICs tested:

| NIC | DCB settings readable | Per-priority bytes | RCCL's priority found from |
|---|---|---|---|
| mlx5 | yes | yes | settings in DSCP trust mode (as on OCI); bytes in PCP trust, where the app table is empty |
| Broadcom Thor2 | yes | yes | settings (RoCE UDP port entry), checked against bytes |
| AMD AINIC (`ionic`) | no (driver does not support it; `nicctl` needs root) | no | `--pfc-prio` only |

On AINIC, find the priority once per cluster from the administrator, or by
experiment. Send an incast with `ib_write_bw --tclass=<RCCL's traffic class>`
and see which `frames_rx_pri_N` counter moves. If none moves, RCCL's traffic
class is not on a PFC priority.

Then there is one row per rank and (collective, message size):

| Column | Meaning |
|---|---|
| `wall ms`, `busBW` | Time those collectives ran, and their mean bus bandwidth |
| `rxPause` | Switch paused our transmit, as % of wall time (fabric side) |
| `txPause` | Our NIC paused the switch, as % of wall time (host side) |
| `loss` | Out-of-sequence drops, sequence-error NAKs, ACK timeouts |
| `rnr` | Receiver-not-ready / out-of-receive-buffer events |
| `ecn%` | ECN-marked packets as % of packets received |
| `cnp/s` | CNPs sent plus received per second of wall time |
| `link%`, `rccl%` | How busy the link was, and how much of that was this job's RCCL traffic |
| `B/coll` | RCCL bytes sent per collective on that NIC |
| `cmpl p50/p99` | Work-request completion latency percentiles |

Counters are sampled, not continuous, so most values are a range: `a-b`
means at least `a`, at most `b`. A finding is **yes** when its lower bound
crosses the threshold, and **maybe** when only the upper bound does.

Findings:

| Finding | Meaning |
|---|---|
| `FABRIC-CONGESTION` | The switch paused our transmit. Marked `(echo?)` when a NIC in the same collective pushed back, since a receiver that stops can make the switch pause its senders |
| `HOST-BACKPRESSURE` | Our NIC paused the switch: this host could not take data in |
| `FABRIC-LOSS` | Drops, NAKs or timeouts on the wire |
| `HOST-RNR` | The receiver was not ready |
| `LINK-SHARED` | The link was busy, mostly with traffic that is not this job's RCCL |
| `CONGESTION-MARKS` | ECN marks or CNPs above their thresholds |
| `CLEAN` | Every counter below threshold. The time went elsewhere; accl's own classification is shown |
| `INCONCLUSIVE` | Samples are missing, or only an upper bound crosses a threshold |

At the end, **Summary per collective and size** gives one conclusion across
all ranks. A host that paused the switch is checked first, because it can
explain rx pauses everywhere else. When both happen, the summary says
`FABRIC+HOST`. It splits the blame: senders were paused for at least x, and
the receiver pushed back for at most y. The receiver can explain about y of
that, so roughly x − y is left for the fabric. A `?` after `HOST` means the
host part is possible but not proven.

**Notes** at the bottom explain anything that weakens the numbers, such as
overlapping collectives or counters that did not refresh.

### Why CNPs are a rate, not a percent

A switch marks a share of the packets it queues, so ECN is shown as a percent
of packets. CNPs are different: a NIC sends at most one per flow every few
tens of microseconds, however many marked packets arrive. Even a flow that is
slowed the whole time sees CNPs for well under 1% of its packets, so CNPs are
shown per second instead. On a congested mlx5 fabric they ran at several
hundred per second per NIC, and at 0 when the fabric was quiet.

### NICs that count only pause frames

Some firmware never moves its pause-time counters. Broadcom Thor2 on MI355X
nodes reports 0 µs of pause however many pause frames arrive. Frames cannot
be turned into time: on AINIC, which counts both, one frame stood for anywhere
from 3 µs to 53 µs. They can only bound it. One frame pauses the link for at
most 65535 × 512 bit times, which is 84 µs at 400 Gb/s. So on these NICs pause
is shown as `0-x%`, with x = frames × that maximum, and:

- if x is under the threshold, pause is ruled out;
- if x is over it, the finding is only "possible". Frames prove the link was
  paused, not for how long.

## Limitations

- **One collective at a time per NIC.** The NIC counters belong to the NIC,
  not to one collective. With `-g 8` in one process, or several communicators
  running at once, collectives overlap and share traffic. The report marks
  these rows `B/coll*` and adds a note.
- **Only collectives.** AlltoAll and other send/recv operations reach the
  profiler as P2P events, which the accl plugin does not record.
- **Short collectives need fresh counters.** Some drivers refresh some
  counters only about once a second. The report widens its ranges for this,
  but a counter that steps once a second cannot say much about one 200 µs
  collective. Many back-to-back iterations of the same size are measured as
  one busy stretch, which helps.
- **Hosts are lined up by collective and size, not by clock.** Clocks differ
  between hosts, so "who paused whom" across hosts is a rule of thumb.

## Check that the counters show the right values

Before trusting the report on a new NIC type, driver or firmware, check two
things: that the counters refresh often enough, and that the values in the
telemetry match what the NIC itself reports. Do both while a job is running
on the node. On a Slurm cluster, run these inside your allocation, not on the
login node.

### 1. How often do the counters refresh?

`tools/net-counter-probe` reads the same NIC counters as telemetry and shows
how often each one really changes:

```bash
cd projects/rccl/tools/net-counter-probe
make
# while a long, steady rccl-tests job runs on this node:
./net-counter-probe --eth <netdev> --ib <rdma-dev> --seconds 10
```

`rdma link show` lists which netdev belongs to which RDMA device. Each
counter gets a verdict:

- `live` changes at least every 2 ms;
- `near-live` changes at least every 50 ms;
- `batched` refreshes in steps, often about once a second. Its values are
  correct, but only over windows longer than the step.

A pause counter that `did not move` just means nothing paused during the
probe. To test a pause counter, you need traffic that really congests, such
as several nodes sending to one. See the probe's `README.md` for more.

### 2. Does telemetry record the same values as the NIC?

Read one counter yourself with `ethtool -S` while the job runs, and check
that every reading falls between the two telemetry samples around it.
Counters only go up, so a correct sampler can never be outside that range.

Pick the counter. These are the ethtool names behind each `hw_samples`
field, with `N` the PFC priority:

| `hw_samples` field | mlx5 | Broadcom Thor2 | AMD AINIC |
|---|---|---|---|
| `pfc_rx_frames[N]` | `rx_prioN_pause` | `rx_pfc_ena_frames_priN` | `frames_rx_pri_N` |
| `pfc_tx_frames[N]` | `tx_prioN_pause` | `tx_pfc_ena_frames_priN` | `frames_tx_pri_N` |
| `pfc_rx_pause_us[N]` | `rx_prioN_pause_duration` | `pfc_priN_rx_duration_us` | `rx_pripause_N_1us_count` |
| `pfc_tx_pause_us[N]` | `tx_prioN_pause_duration` | `pfc_priN_tx_duration_us` | `tx_pripause_N_1us_count` |

On the node, log the counter with a `CLOCK_MONOTONIC` time before and after
each read. Save this as `log_counter.py`:

```python
import subprocess, sys, time

netdev, key = sys.argv[1], sys.argv[2]
while True:
    t0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC) // 1000
    out = subprocess.run(["ethtool", "-S", netdev], capture_output=True, text=True).stdout
    t1 = time.clock_gettime_ns(time.CLOCK_MONOTONIC) // 1000
    for line in out.splitlines():
        k, _, v = line.strip().partition(":")
        if k == key:
            print(t0, t1, int(v), flush=True)
    time.sleep(0.5)
```

Run it during the job, and stop it with Ctrl-C when the job ends:

```bash
python3 log_counter.py <netdev> rx_pfc_ena_frames_pri3 > ethtool_log.txt
```

Then compare the log with that node's telemetry files. Save this as
`check_counter.py`:

```python
import bisect, glob, json, sys

tel_dir, host, roce, field, log = sys.argv[1:6]
name, _, idx = field.partition("[")
samples = {}
for path in glob.glob(f"{tel_dir}/rccl_telemetry_{host}_*.json"):
    for s in json.load(open(path)).get("hw_samples", []):
        if s.get("roce_device") != roce:
            continue
        v = s[name][int(idx.rstrip("]"))] if idx else s[name]
        if v >= 0:
            samples[s["ts_us"]] = v
ts = sorted(samples)
ok = bad = 0
for line in open(log):
    t0, t1, v = map(int, line.split())
    i = bisect.bisect_right(ts, t0) - 1   # last sample at or before the read started
    j = bisect.bisect_left(ts, t1)        # first sample at or after it ended
    if i < 0 or j >= len(ts):
        continue                          # outside the sampled time
    if samples[ts[i]] <= v <= samples[ts[j]]:
        ok += 1
    else:
        bad += 1
        print(f"MISMATCH at {t0}: ethtool {v}, telemetry {samples[ts[i]]} .. {samples[ts[j]]}")
print(f"{ok} readings inside the telemetry range, {bad} outside")
```

```bash
python3 check_counter.py $OUT $(hostname) <rdma-dev> 'pfc_rx_frames[3]' ethtool_log.txt
```

Every reading should be inside the range. Any mismatch means telemetry is
reading the wrong counter, or a stale value, for that field. The hostname
selects this node's telemetry files; all of its processes are read together,
since each one samples only the NICs it drives.

Two more quick checks:

- **Bytes.** On a quiet fabric, `nic_tx_bytes` should be a little more than
  RCCL's own `tx_bytes` over the same span: about 1.03× for RoCE headers and
  ACKs. Much more means other traffic shares the link, and the report should
  say `LINK-SHARED`.
- **A known cause.** Run a test with a known answer, such as several nodes
  sending to one node (incast) with `ib_write_bw`. The receiving host should
  show tx pause (`HOST-BACKPRESSURE`), and the senders should show rx pause.
