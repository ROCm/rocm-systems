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
  ./all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 500 -w 5
```

Run each message size for at least a few seconds. NIC counters are sampled,
and some refresh slowly: Broadcom Thor2 port counters update only once a
second. A size that runs for less than that leaves `link%` as `stale`, so
LINK-SHARED cannot fire, and most other findings stay INCONCLUSIVE. Raise
`-n` until each size runs for several seconds.

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

This is decided for each NIC on its own, since NICs can be set up differently.
When they disagree, the header lists each NIC's priority and each NIC is judged
on its own.

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

Then there is one row per rank and (collective, message size). Different
communicators, algorithms or protocols at the same size get separate rows; the
summary names them when a size has more than one:

| Column | Meaning |
|---|---|
| `wall ms`, `busBW` | Time those collectives ran, and their mean bus bandwidth |
| `rxPause` | Switch paused our transmit, as % of wall time (fabric side) |
| `txPause` | Our NIC paused the switch, as % of wall time (host side) |
| `oos/nak/to` | Out-of-sequence packets / sequence-error NAKs / ACK timeouts, each on its own; `-` where the NIC does not give one |
| `nobuf/rnr` | Times this NIC had no receive buffer posted / RNR NAKs it got from a peer |
| `ecn%` | ECN-marked packets as % of packets received |
| `cnp/s` | CNPs sent plus received per second of wall time |
| `link%`, `rccl%` | How busy the link was, and how much of that was this job's RCCL traffic |
| `B/coll` | RCCL bytes sent per collective on that NIC |
| `rxwin/coll` | RCCL bytes received per collective on that NIC, inside the network window only. The receive that opened the window was counted before the window began, so it is not here |
| `rx/coll` | RCCL bytes received per collective across all NICs, with the window-opening receive added once. A lower bound: one completion can finish several grouped receives and only the first one's size is recorded. Not used by any verdict |
| `cmpl p50/p99` | Work-request completion latency percentiles |

Counters are sampled, not continuous, so most values are a range: `a-b`
means at least `a`, at most `b`. A finding is **yes** when its lower bound
crosses the threshold, and **maybe** when only the upper bound does.

Findings are observations: what a counter showed, not yet a cause.

| Finding | Meaning |
|---|---|
| `FABRIC-CONGESTION` | The switch paused our transmit. Marked `(echo?)` when a NIC in the same collective pushed back, since a receiver that stops can make the switch pause its senders |
| `HOST-BACKPRESSURE` | Our NIC paused the switch: this host could not take data in |
| `OUT-OF-SEQ` | This NIC received packets out of sequence (receiver side) |
| `SEQ-NAK` | This NIC got sequence-error NAKs back (sender side) |
| `ACK-TIMEOUT` | This NIC's sends timed out waiting for an ACK (sender side) |
| `RX-NO-BUFFER` | This NIC had no receive buffer posted: this host's receiver |
| `PEER-RNR` | A host this rank sends to was not ready to receive. The counter sits on the sender, so the faulty host is one of its peers |
| `CQ-ERROR` | RCCL saw completion errors in these collectives (from accl, no telemetry needed) |
| `LINK-SHARED` | The link was busy, mostly with traffic that is not this job's RCCL |
| `CONGESTION-MARKS` | ECN marks or CNPs above their thresholds |
| `CLEAN` | Every counter below threshold. The time went elsewhere; accl's own classification is shown |
| `INCONCLUSIVE` | Samples are missing, only an upper bound crosses a threshold, the NIC gave no counters for loss, ECN/CNP, link use or pause, or the accl output says it lost data (or the process never finished) |

At the end, **Summary per collective and size** gives one conclusion across
all ranks. A host that paused the switch is checked first, because it can
explain rx pauses everywhere else. When both happen, the summary gives both
sides, as a share of each rank's own time and in milliseconds, and does not
subtract one from the other: a receiver that stops taking data makes the
switch pause the senders behind it, so its pause can cause some or all of
theirs, and these counters cannot say how much. `FABRIC+HOST?` means the
receiver paused about as long as the senders were paused (or above the
threshold); `FABRIC?` means it paused far less, so only part of it can be
backpressure.

Findings the summary does not lead with, proven or possible, are listed after
`also:`, so the order never hides one. A drop diagnosis that points at a host
leads right after pushback.

**Drop diagnosis** comes next; see below. **Notes** at the bottom explain
anything that weakens the numbers, such as overlapping collectives or
counters that did not refresh.

### Drop diagnosis

For each collective and size the report weighs the delivery and recovery
signals against what is known about the endpoints, and puts the result in a
bucket with a confidence: `HOST_LIKELY`, `FABRIC_LIKELY`, `MIXED` or
`INCONCLUSIVE` (`NONE` when no signal was seen and every one was measured).
These are likelihoods from counters, not the failing component. Seeing these
events also does not show that a collective was slow: pass `--baseline
<accl dir of a healthy run>` to compare busBW against a matched run.

**Signals.** Each is shown on its own, never added up: one lost packet can
show as an out-of-sequence packet at its receiver, a sequence-error NAK at its
sender and a retransmission, so a sum is not a count of lost packets. None of
them alone proves a fabric fault. A timeout also follows a lost ACK on the
way back, a long delay, or an endpoint that stopped answering.

| Signal | mlx5 | Thor2 (`bnxt_re`) | AINIC (`ionic`) | Counted at |
|---|---|---|---|---|
| OUT-OF-SEQ | `out_of_sequence` | `res_oos_drop_count` | `resp_rx_outouf_seq` | receiver |
| SEQ-NAK | `packet_seq_err` | `seq_err_naks_rcvd` | `req_rx_pkt_seq_err` | sender |
| ACK-TIMEOUT | `local_ack_timeout_err` | `local_ack_timeout_err` if the driver has it, else `to_retransmits` (retransmits after a timeout, another unit; the report says so where it was used) | `tx_rdma_ack_timeout` | sender |
| RX-NO-BUFFER | `out_of_buffer` | `rx_out_of_buffer` | `resp_rx_outof_buf` | receiver (this host) |
| PEER-RNR | `rnr_nak_retry_err` (RNR NAKs) | `rnr_naks_rcvd` (RNR NAKs) | `req_rx_rnr_retry_err` (by its name, retry errors; unit not verified) | sender (about a peer) |
| CQ-ERROR | RCCL completion errors from accl, per collective; overlapping windows can count one twice | | | rank |

Thor2 names differ between `bnxt_re` versions; telemetry reads the first name
the driver has, so a column can come from either. A counter a NIC does not
give is "not measured", never 0, and the report never lets one column stand
in for another. A rank whose accl records name NICs that telemetry did not
sample is blind, not clean.

"In window" means the lower bound is above 0: the count fell inside the
collectives, given each counter's measured refresh lag. "Possibly in window"
means only the upper bound is: the count may have fallen just before or after.
A counter whose last sample came less than its refresh lag after the
collectives may not have shown their events yet, so its upper bound there is
open: "not measured", never 0, however short that tail. This holds for every counter, so on NICs whose
counters refresh about once a second (Thor2) the last message size of a run
is often `INCONCLUSIVE`; let the job run a few seconds past it. A counter that
never moved takes the slowest refresh of the NIC's byte counters, which can
only widen its ranges.
Being in the window is overlap in time, not proof of cause.

**Endpoint evidence.** Each is "fine" only where it was measured, covered by
samples, and zero (pause: under the threshold, a proven small pause is shown).
No error reported is not the same as healthy.

| Evidence | From | Pressure means |
|---|---|---|
| Receive buffers | `out_of_buffer` on every NIC | that host's receiver had no buffer posted |
| Peer readiness | RNR counter on every NIC | some peer of this sender was not ready (which one is not known) |
| Backpressure | PFC tx pause on RCCL's priority | that host could not take data in. Unknown with PFC off on RCCL's priority (an overloaded receiver drops packets then, and port discards are not sampled) or with the priority unknown |
| NIC progress | port byte counters moved both ways inside the collectives | (never pressure) unknown when they did not provably move. RCCL's own byte counts are posted work, not bytes on the wire |

The report also shows the share of accl's proxy step time spent waiting on
the GPU, as context only: several steps are in flight at once, so it is not
wall time, and healthy runs show about half. Over 80% on some rank is listed
as another explanation for slowness, never as drop evidence.

With PFC off on RCCL's priority (RCCL's default traffic class on the OCI and
TensorWave clusters), or the priority unknown (AINIC without `--pfc-prio`),
backpressure is always unknown. Endpoints then never count as healthy: ACK
timeouts alone stay `INCONCLUSIVE`, and `FABRIC_LIKELY` is at most low
(medium with congestion marks).

**Rules.** "Receiver not ready" is receive buffers or peer readiness under
pressure; those counters are themselves drop and recovery events, and they are
the only host cause the rules use. Pause is flow control: on a lossless
priority it kept the paused receiver from dropping. So in the drop diagnosis
backpressure is never a host cause; it only stops the endpoints from counting
as healthy, and is listed as another explanation. The summary line still
reports it as `HOST-BACKPRESSURE`, a slow receiver.

| Signals in window | Endpoints | Bucket |
|---|---|---|
| none (or only CQ-ERROR) | receiver not ready | `HOST_LIKELY` (medium; low if a symptom is only possibly in the window or not measured) |
| none, some only possibly in window | | `INCONCLUSIVE` |
| none, a signal not measured somewhere | | `INCONCLUSIVE` |
| none, everything measured | | `NONE` (high; medium when backpressure or NIC progress is unknown) |
| only CQ-ERROR | | `INCONCLUSIVE`: completion errors have local causes too, and RCCL does not record which |
| yes | receiver not ready, and congestion (rx pause with no pushback, ECN or CNPs) | `MIXED` (medium) |
| yes | receiver not ready, no congestion | `HOST_LIKELY` (low) |
| only ACK-TIMEOUT | healthy | `FABRIC_LIKELY` (low; medium with congestion) |
| only ACK-TIMEOUT | not established | `INCONCLUSIVE` |
| yes | receiver possibly not ready | `INCONCLUSIVE` (leaning `MIXED`) |
| yes | no receive-buffer or RNR counter on any NIC | `INCONCLUSIVE` (leaning `FABRIC_LIKELY`) |
| yes | healthy | `FABRIC_LIKELY` (medium; high with congestion) |
| yes | no receiver pressure seen, health not established | `FABRIC_LIKELY` (low; medium with congestion) |

Then each kind of data-quality issue lowers the confidence by one step: the
accl file lost data, a counter reset during the collectives, a gap in the
samples, drop counters that refresh later than a fifth of the collectives'
time, overlapping collectives or communicators, a NIC driven by several
processes of the job, or a NIC where less than 80% of the traffic was this
job's RCCL. Below low the bucket is `INCONCLUSIVE`, and the rules' answer is
kept as "leaning". For `NONE` only lost accl data and counter resets count:
a drop counter that stayed at 0 is 0 whoever else shared the NIC, and its
refresh lag is already in its bounds. The signals stay on show either way. A rank with no
telemetry, or missing accl files, leaves its endpoints unknown.

Each diagnosis prints the signals with their bounds, the endpoint evidence
(including what is unknown), what was not measured, why, what else could
explain it, the data-quality issues and the baseline comparison:

From the OCI cluster (mlx5, RCCL on a priority without PFC), with incast
traffic from a third node into one of the two hosts:

```text
  AllReduce 64M: FABRIC_LIKELY (low)
    signals       OUT-OF-SEQ, packets received out of sequence (receiver side): in window 170034 at rank 8 (useocpm2m-097-042 mlx5_0), 170857 at rank 9 (useocpm2m-097-042 mlx5_2), 25216 at rank 10 (useocpm2m-097-042 mlx5_3) and 5 more
                  SEQ-NAK, sequence-error NAKs received (sender side): in window 1835 at rank 0 (useocpm2m-097-030 mlx5_0), 1972 at rank 1 (useocpm2m-097-030 mlx5_2), 7490 at rank 2 (useocpm2m-097-030 mlx5_3) and 5 more
                  ACK-TIMEOUT, ACK timeouts (sender side): none on any NIC
                  these are separate counters: one lost packet can show in several of them, so they are not added up into a count of lost packets
    endpoints     receive buffers: no receiver ran out of buffers
                  peer readiness: no sender got RNR NAKs
                  backpressure: unknown at rank 0 (useocpm2m-097-030 mlx5_0), rank 1 (useocpm2m-097-030 mlx5_2), rank 2 (useocpm2m-097-030 mlx5_3) and 13 more: PFC off on RCCL's priority: an overloaded receiver drops instead, and port discards are not sampled
                  NIC progress: every port moved bytes both ways inside the collectives
                  accl proxy waiting on the GPU, share of its step time (context only): send 28-52%, receive 1-5% across ranks
    why           delivery symptoms (OUT-OF-SEQ, SEQ-NAK) with no receiver short of buffers or RNR seen, but endpoint health not established (backpressure); fabric congestion seen (CONGESTION-MARKS at rank 10 (useocpm2m-097-042 mlx5_3), CONGESTION-MARKS at rank 11 (useocpm2m-097-042 mlx5_4), CONGESTION-MARKS at rank 12 (useocpm2m-097-042 mlx5_5) and 6 more)
    could also be an endpoint fault these counters do not show (port discards, PCIe, NIC firmware)
                  with PFC off on RCCL's priority, a receiver host that cannot keep up drops packets too, and those port discards are not sampled
    data quality  only 70% of the traffic at rank 2 (useocpm2m-097-030 mlx5_3) was this job's RCCL
    slowness      no matched baseline (same collective, size, algorithm, protocol and rank count); seeing these events does not by itself show the collective was slow
```

What these counters cannot tell, and would need new instrumentation:

- Which peer or path a signal belongs to. Proxy steps carry no NIC or QP, and
  the counters are per port.
- Retries that ran out (`max_retry_exceeded`), the extra retransmit counters
  (`roce_adp_retrans*` on mlx5) and the discard counters (AINIC
  `hw_rx_dropped`, Thor2 `rx_discards`). They are read at start and exit but
  not sampled during the run. mlx5 port discards (`rx_discards_phy`), where an
  overloaded host drops packets on a lossy priority, are not read at all.
- Per-NIC receive bytes on a multi-NIC connection. RCCL counts them all on the
  first NIC.

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
