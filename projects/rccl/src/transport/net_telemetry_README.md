# RCCL Network Telemetry

Per device / channel / QP network telemetry for the `IB-CAST` transport,
collected in-tree without a separate plugin or a build flag. Software counters
accumulate over the whole process; hardware counters are reported as deltas
against a baseline taken when the device is registered. One JSON file per rank
is written at process exit.

Telemetry is off by default and has no effect unless `RCCL_TELEMETRY_ENABLE=1`.

## Enabling and configuration

| Variable | Default | Meaning |
|---|---|---|
| `RCCL_TELEMETRY_ENABLE` | `0` | Set to `1` to enable collection. Any other value keeps it off. |
| `RCCL_TELEMETRY_OUTPUT_DIR` | `/tmp` | Directory the JSON file is written to. |
| `RCCL_TELEMETRY_HISTOGRAM_BUCKETS` | `5` | Number of completion-latency buckets emitted per QP (1..16). |
| `RCCL_TELEMETRY_HISTOGRAM_INTERVAL_NS` | `30000` | Width of one latency bucket in nanoseconds. Bucket `b` covers up to `(b+1) * interval` ns. |
| `RCCL_TELEMETRY_LATENCY_SAMPLE` | `1` | Measure the completion latency of one posted WQE in `N`. `1` measures every WQE. Must be a power of two; any other value is rounded **up** to the next one, with a line on stderr saying so. See [Latency sampling](#latency-sampling). |
| `RCCL_TELEMETRY_HW_COUNTERS` | all | Comma-separated allow-list of hardware counter names to collect. Empty means collect every counter the driver exposes. |
| `RCCL_TELEMETRY_SAMPLE_MS` | `0` | If greater than 0, a background thread samples a small set of congestion counters every N ms into a `hw_samples` time series. `0` disables the sampler; hardware counters are then read only at registration and at exit. |
| `RCCL_TELEMETRY_DEBUG` | unset | If set, prints device registration and flush diagnostics to stderr. |

## Where the data lands

One file per rank, named:

```
<RCCL_TELEMETRY_OUTPUT_DIR>/rccl_telemetry_<hostname>_<pid>.json
```

The file is written once, at process exit (via an `atexit` handler). A 2-node
run of 8 ranks each therefore produces 16 files.

## Enabling and running

```bash
export RCCL_TELEMETRY_ENABLE=1
export RCCL_TELEMETRY_OUTPUT_DIR=/path/to/results
export NCCL_NET=IB-CAST
# ... normal launch, e.g. mpirun ... all_reduce_perf -b 8K -e 1M -f 2 -g 1 -n 20
```

After the run, inspect any `rccl_telemetry_*.json` in the output directory.

## JSON structure

```
{
  "version", "host_name", "process_name", "process_id",
  "start_time", "end_time", "transport",
  "latency_sample_interval",   // only when RCCL_TELEMETRY_LATENCY_SAMPLE > 1
  "devices": [
    {
      "device_id", "roce_device", "eth_device", "hw_type",
      "tx_bytes", "rx_bytes", "num_cq_errors", "cq_poll_count",
      "num_channels", "active_channels", "num_qp_untracked",
      "wqe_size_stats": [ {"max_wqe_size", "num_wqe"} ],
      "channels": [
        {
          "id", "num_wqe_sent", "num_recv_wqe", "num_wqe_rcvd",
          "num_wqe_completed", "num_wqe_sampled",   // sampled: only when N > 1
          "num_cts_sent", "num_req_completed",
          "num_data_qp", "num_cts_qp", "num_qp_untracked",
          "queue_pairs": [ { ... per-QP counters ... } ]
        }
      ],
      "hw_counters": { ... driver counters ..., "delta_tx_bytes", ... }
    }
  ],
  "hw_samples": [ ... only if RCCL_TELEMETRY_SAMPLE_MS > 0 ... ]
}
```

The channel WQE/CTS counters are the sum of the counters of the QPs in that
channel, by construction: they are not stored, they are computed from the QP
slots when the JSON is written. Keeping them as a second counter on the data path
made every QP on a channel contend for one cache line, which cost up to 11% on
mid-size collectives. `num_req_completed` is the exception — it counts requests,
not WQEs, so no QP sum produces it and it is still a stored counter.

A QP whose slot could not be allocated is not counted anywhere; its count is
reported in `num_qp_untracked` instead of being dropped silently.

## Data-path cost

Nothing on the data path looks a slot up. Each QP's statistics slot is resolved
once, when the connection is set up, and the resulting pointer is stored on the
QP; the per-WQE hooks take that pointer and do nothing but their counter
updates. Slot addresses are stable for the life of the process — blocks are
appended, never freed or moved — which is what makes this safe. A QP that has no
slot holds a null pointer, and every hook is then a no-op on it, so call sites
still never test anything before calling. Posting one send WQE is a single hook
call that updates both counters it owns. The completion hook reaches its
histogram bucket by a multiply rather than a 64-bit division; the bucket index is
identical to the division's for every input.

## Latency sampling

Every counter above is an increment. The completion **latency** is not: it costs
two `clock_gettime` calls per WQE — one when the WQE is posted, one when its
completion is drained — plus a bucket computation and two compare-and-swap loops
for `min`/`max`. That family is the bulk of the per-WQE cost, and
`RCCL_TELEMETRY_LATENCY_SAMPLE=N` measures only one posted WQE in `N` instead of
all of them.

The decision is taken on the post path, because that is where the first
timestamp would be read, and it is carried to the completion path in the post
timestamp itself, using a sentinel:

| post timestamp | meaning |
|---|---|
| `0` | the completion matched no tracked posting |
| `1` | it matched one, but this WQE was not sampled |
| `> 1` | it matched one and was sampled; a real `CLOCK_MONOTONIC` reading |

A WQE the interval skips therefore reads the clock **zero** times, on both
sides, and still lands in `num_wqe_completed`.

### What sampling does and does not change

Sampling writes to nothing except the three latency fields, so no other counter
can move with `N`. On the 2-node alltoall regression, three runs at `N = 1` and
three at `N = 16` agree bit for bit on `num_wqe_sent`, `num_recv_wqe`,
`num_wqe_rcvd`, `num_wqe_completed`, `num_cts_sent` and its
signalled/unsignalled split, `num_write_wqe`, `num_write_imm_wqe`,
`num_req_completed`, `tx_bytes`, `rx_bytes`, `num_data_qp`, `num_cts_qp` and
`num_qp_untracked`.

`num_slot_miss` is the one counter that is not reproducible, and it is not
reproducible at a *fixed* `N` either: it counts how often a sender found the
CTS FIFO slot not yet published, which is a polling race, and it varied by 33%
across three runs at `N = 1` alone. Do not read a change in it as an effect of
sampling.

Affected by `N`: `wqe_completion_histogram`, `wqe_completion_ns_min` and
`wqe_completion_ns_max`, plus the two new `num_wqe_sampled` /
`latency_sample_interval` keys that describe them.

### Reading a sampled histogram

The counts are **not** scaled by `N` to fake un-sampled totals. What you get is
the truth about a sample, plus enough information to know how large that sample
was:

- `latency_sample_interval` at the top level of the file is the `N` actually in
  effect after the power-of-two round-up, not the value you asked for.
- `num_wqe_sampled`, per QP and per channel, is how many completions the
  histogram is built from. It is exactly the sum of that QP's histogram buckets.

Both keys appear **only when `N > 1`**. At the default the histogram already
covers every matched completion, `num_wqe_sampled` would just repeat
`num_wqe_completed`, and the file is byte-for-byte what it was before sampling
existed.

`min`/`max` become the extremes **of the sample**, so at `N > 1` they understate
the true range — and they understate it asymmetrically, since the rare
long-tail completion is exactly the one most likely to be skipped. A 2-node
alltoall that reported a 78.2 ms maximum with every WQE measured reported
43.5 ms at `N = 16` off the same traffic. Use them as a sampled range, and read
the tail from the top histogram bucket rather than from `max`. The histogram
shape itself is unbiased: selection is by posting position, not by latency.

## Performance

Telemetry is inactive unless `RCCL_TELEMETRY_ENABLE=1`. With it disabled, the cost is within
run-to-run noise of a build that has no telemetry code at all.

With telemetry enabled the cost is about 59 ns per posted WQE. Measured on 2 nodes x 8 ranks
(MI300X, mlx5, IB-CAST), it peaks at roughly +6-8% in the 192K-256K range and falls monotonically
to zero by 1M, with no measurable cost at small (8K-128K) or large (4M-2G) message sizes.

The shape of that curve comes from RCCL's channel-count rule, not from telemetry. With
`nc = clamp(nBytes/65536, 1, 4)` and `30 * nc` WQEs posted per operation, the WQE count per
operation reaches its maximum of 120 exactly at 256K, at the shortest operation time for that
count, so the per-WQE cost is at its most visible there. Each size that first reaches a new
channel count shows the same step, which is why 192K behaves like 256K. Algorithm, protocol and
channel count are identical with and without telemetry.

### What latency sampling buys

Sampling is the one knob that changes this cost, and the honest way to measure it is within one
binary, varying only `RCCL_TELEMETRY_LATENCY_SAMPLE`: two builds that differ only in code layout
measure up to 4% apart at these sizes, which is more than the effect. Medians of 14 reps,
`all_gather` and `reduce_scatter`, 2 nodes x 8 ranks, as a percentage of the `N = 1` run time:

| N | 192K-256K | median over 192K-1M | share of the whole latency family |
|---|---|---|---|
| 4 | -1.0% | -0.9% | ~64% |
| 16 | -1.4% | -1.2% | ~86% |
| 64 | -1.6% | -1.4% | ~98% |

In overhead-against-a-no-telemetry-build terms, the peak at 256K goes from about +8.0% at `N = 1`
to about +6.4% at `N = 16`.

So the whole family — two `clock_gettime` per WQE, the bucket computation and both min/max CAS
loops — is worth about 1.6% of run time at the peak, and `N = 16` recovers essentially all of it.
`N = 64` is within 0.2 points of `N = 16` and measures only 1.6% of WQEs, so it gives up a great
deal of histogram resolution for no further gain. `N = 16` is the point where the curve flattens.

The default `N = 1` costs nothing: it is a load of a never-written global and a
perfectly-predicted branch in front of a clock read that used to be unconditional. Measured
against the same pre-sampling code on two different node pairs, `N = 1` and the pre-sampling tip
peak within 0.3 points of each other once each build's own layout term is removed, which is well
inside the layout sensitivity described below.

What sampling does not do is make an instrumented build indistinguishable from an uninstrumented one.
What remains after sampling is the per-WQE counter work, which sampling does not touch and is not
meant to, plus whatever the code layout of the particular build is worth. A reference build with
the latency instrumentation deleted outright measures 1.3% *faster* than a build with no telemetry
code at all at 928K-1M, where no telemetry cost can exist; that is the scale of the layout term,
and it is why the table above is a within-binary comparison.

Hardware counters are read twice per process, at first use of a device and at exit, and only for
the devices a rank actually uses. Periodic sampling is off unless `RCCL_TELEMETRY_SAMPLE_MS` is
set, and when enabled it runs on a background thread, never on the data path.

## Application-level counters

These are software counters maintained on the data path. They appear both per
channel (aggregated) and per QP.

| Counter | Meaning |
|---|---|
| `num_wqe_sent` | Send WQEs posted on this QP. |
| `num_recv_wqe` | Receive WQEs posted (`ibv_post_recv`). |
| `num_wqe_rcvd` | Completions drained from the CQ, send and receive alike. |
| `num_wqe_completed` | Completions that matched a tracked posting, i.e. those with a recorded post timestamp so latency is computable. |
| `num_cts_sent` | CTS (clear-to-send) messages posted. |
| `num_cts_sent_signalled` / `num_cts_sent_unsignalled` | Split of `num_cts_sent` by whether the WR was signalled. Their sum equals `num_cts_sent`. |
| `num_slot_miss` | CTS FIFO slot misses (no free slot when one was needed). |
| `num_write_wqe` | `IBV_WR_RDMA_WRITE` postings (plain write, no immediate). |
| `num_write_imm_wqe` | `IBV_WR_RDMA_WRITE_WITH_IMM` postings. |
| `num_req_completed` | Network requests completed on this channel. This is **not** a WQE count: one request is striped over one WQE per QP, and one CQE completes every sub-request of a multi-send, so this is neither an upper nor a lower bound on the `num_wqe_*` counters. |
| `num_wqe_sampled` | Completions whose latency was actually measured, i.e. the sample the three fields below are computed from, and exactly the sum of the histogram buckets. Emitted only when `RCCL_TELEMETRY_LATENCY_SAMPLE > 1`; without it, every matched completion is measured. |
| `wqe_completion_ns_min` / `wqe_completion_ns_max` | Min/max completion latency on the QP, over the sampled completions. |
| `wqe_completion_histogram` | Completion-latency histogram, one entry per bucket (see the histogram parameters above), over the sampled completions. |

Device-level counters:

| Counter | Meaning |
|---|---|
| `tx_bytes` / `rx_bytes` | Software byte totals sent / received on the device. |
| `num_cq_errors` | Completions drained with an error status. |
| `cq_poll_count` | Number of `ibv_poll_cq` calls. |
| `num_channels` / `active_channels` | Channels seen / channels that carried traffic. |
| `num_qp_untracked` | QPs on this device that could not be given a stats slot; their traffic is absent from every counter above. |
| `wqe_size_stats` | Distribution of send WQE payload sizes; only non-empty buckets are emitted. |

### Reading the completion counters correctly

Three counters look similar but mean different things, and the difference is the
usual source of confusion:

- `num_recv_wqe` counts receive WQEs **posted**, not completed.
- `num_wqe_rcvd` counts **all** completions drained from the CQ (send and recv).
- `num_wqe_completed` counts only the subset of those completions that were
  matched to a tracked posting.

And `num_req_completed` counts **requests**, not WQEs; do not expect it to equal
any of the WQE counters except in the degenerate one-QP, one-request case.

## Hardware counters

`hw_counters` reports driver counters (mlx5, thor2/`bnxt_re`, ainic/`ionic`) under
a single canonical name vocabulary. A counter a given driver does not expose is
reported as `-1` rather than omitted. The `delta_*` byte/packet fields and the
scalar hardware counters are reported as the value at exit minus a baseline
captured on the device's first use, which still precedes any traffic on it.

By default the hardware counters are read exactly twice per *used* device for the
whole run: once on first use (baseline) and once at exit (final). A rank
registers every NIC it can enumerate but normally drives only one or two, and
each read costs an `ethtool -S` subprocess, so devices the rank never connects
over are not read at all and report `-1`. The periodic `hw_samples` time series
is produced only when `RCCL_TELEMETRY_SAMPLE_MS > 0`, and even then it reads only
IB sysfs files. The data path itself never reads sysfs or spawns a subprocess.

## Known limitations

- On the CTS offload path with optional receive completion, the sender issues a
  plain `IBV_WR_RDMA_WRITE` and the receiver posts no receive WQE and gets no
  completion, so `rx_bytes` cannot be measured and reads 0. This is expected,
  not a lost count. When the receiver does take completions,
  `IBV_WR_RDMA_WRITE_WITH_IMM` is used and `rx_bytes` is populated.
