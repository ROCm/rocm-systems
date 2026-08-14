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
  "devices": [
    {
      "device_id", "roce_device", "eth_device", "hw_type",
      "tx_bytes", "rx_bytes", "num_cq_errors", "cq_poll_count",
      "num_channels", "active_channels", "num_qp_untracked",
      "wqe_size_stats": [ {"max_wqe_size", "num_wqe"} ],
      "channels": [
        {
          "id", "num_wqe_sent", "num_recv_wqe", "num_wqe_rcvd",
          "num_wqe_completed", "num_cts_sent", "num_req_completed",
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

## Performance

Telemetry is inactive unless `RCCL_TELEMETRY_ENABLE=1`. With it disabled, the cost is within
run-to-run noise of a build that has no telemetry code at all.

With telemetry enabled the cost is about 59 ns per posted WQE. Measured on 2 nodes x 8 ranks
(MI300X, mlx5, IB-CAST), it peaks at roughly +6-7% in the 192K-256K range and falls monotonically
to zero by 1M, with no measurable cost at small (8K-128K) or large (4M-2G) message sizes.

The shape of that curve comes from RCCL's channel-count rule, not from telemetry. With
`nc = clamp(nBytes/65536, 1, 4)` and `30 * nc` WQEs posted per operation, the WQE count per
operation reaches its maximum of 120 exactly at 256K, at the shortest operation time for that
count, so the per-WQE cost is at its most visible there. Each size that first reaches a new
channel count shows the same step, which is why 192K behaves like 256K. Algorithm, protocol and
channel count are identical with and without telemetry.

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
| `wqe_completion_ns_min` / `wqe_completion_ns_max` | Min/max observed completion latency on the QP. |
| `wqe_completion_histogram` | Completion-latency histogram, one entry per bucket (see the histogram parameters above). |

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
