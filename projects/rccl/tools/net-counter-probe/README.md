# net-counter-probe

Measures how often a NIC's counters really change, and what one read costs.

The accl fabric report (`plugins/profiler/accl/accl_fabric_report.py`) charges
each collective with the PFC pause time, ECN/CNP and bytes that fell inside it,
using RCCL telemetry's `hw_samples`. That only works if the counters are fresh.
Some drivers refresh some ethtool counters only periodically, around once a
second. A counter that moves in 1 s steps says nothing about a 200 us
collective. Run this once per NIC type before trusting short-collective numbers.

## Build

```bash
make
```

No dependencies beyond a C++14 compiler and the Linux kernel headers.

## Run

Start a long, steady rccl-tests job that drives the NIC (large messages, many
iterations), then on one of its nodes:

```bash
./net-counter-probe --eth <netdev> --ib <rdma-dev> --seconds 10
# e.g. ./net-counter-probe --eth enp12s0np0 --ib mlx5_0
```

`rdma link show` lists which netdev belongs to which RDMA device.

By default it watches:

- ethtool: PFC pause duration counters under each supported driver's naming
  (mlx5 `rx_prioN_pause_duration`, AINIC `rx_pripause_N_1us_count`, Thor2
  `pfc_priN_rx_duration_us`) and the mlx5 RoCE byte counters
  (`tx/rx_vport_rdma_unicast_bytes`).
- IB sysfs (`/sys/class/infiniband/<dev>/ports/<port>/{hw_counters,counters}`):
  every ECN/CNP counter plus `port_xmit_data` / `port_rcv_data`.

Pick your own with `--keys a,b` (ethtool names) and `--sysfs a,b`.

## Reading the output

Illustrative output:

```
ethtool GSTATS (1520 stats): read cost median 38.0 us, p99 120.0 us, max 410.0 us

counter                       source  changes  read us  change every ms (p50/min/max)  verdict
tx_vport_rdma_unicast_bytes   ethtool      10     38.0  1000.10 / 998.70 / 1003.20     batched: use time windows, not per collective
port_xmit_data                sysfs     95000      2.1  0.10 / 0.09 / 0.50             live: usable per collective
rx_prio3_pause_duration       ethtool       0     38.0  -                              did not move: no traffic or no pause on it during the probe
```

- **change every ms**: time between consecutive value changes. Under steady
  traffic, a live counter changes on nearly every read; a driver-refreshed one
  changes in fixed steps.
- **verdict**: `live` (p50 at most 2 ms), `near-live` (at most 50 ms) or
  `batched`. A pause counter that `did not move` just means nothing paused
  during the probe: it says nothing about freshness. To test a pause counter,
  you need traffic that actually congests (e.g. many-to-one).
- **read cost**: what the telemetry sampler pays per device per sample
  (`RCCL_TELEMETRY_SAMPLE_MS`).

Under light or bursty traffic the gaps reflect the traffic, not the driver: a
ping every 10 ms shows up as "changes every 10 ms".
