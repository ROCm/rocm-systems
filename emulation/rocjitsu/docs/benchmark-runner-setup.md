# Set up a benchmark runner

Configure CPU allocation, fixed frequency and idle settings, and Bpfland to
reduce benchmark noise. Stop the Actions runner before changing these settings.

The configuration files are in
[`scripts/benchmark-runner`](../scripts/benchmark-runner/).

## Prerequisites

- A Linux host with systemd, cgroup v2, sched_ext support, Python 3, and
  util-linux (`setarch`).
- **64 physical cores and 128 online logical CPUs**, with SMT sibling pairs
  `N,N+64`. Use the fixed CPU masks only with this layout.
- CPU controls supporting a fixed 3.2 GHz frequency, performance governor/EPP,
  and idle-state disabling.
- A `scx_bpfland` 1.1.3 binary built from revision
  `fead94e15184e957469b346b129f412641691bbe`. The service uses default arguments;
  use the same build across runners.
- An existing GitHub Actions runner service using `runsvc.sh`.

The allocation reserves CPUs `16-63` for benchmarks. Runner control activity
uses `0-15`; background processes use `0-15,64-79`. CPUs `80-127`, the benchmark
cores' SMT siblings, remain excluded from background work.

## Install the host configuration

Disable conflicting scheduler and power-management services. Review existing
CPU restrictions before installing the drop-ins below; preserve any narrower
allocation.

Set `SRC`, `BPFLAND`, `EXPECTED_SHA256`, and `RUNNER_USER` for the installation.
Use the shown destination paths, which match the service files. Run the
installation commands as root:

```bash
set -euo pipefail
SRC=/PATH_TO_ROCM_SYSTEMS/emulation/rocjitsu/scripts/benchmark-runner
BPFLAND=/PATH_TO_BUILD/scx_bpfland
EXPECTED_SHA256=REPLACE_WITH_RECORDED_SHA256
RUNNER_USER=REPLACE_WITH_RUNNER_ACCOUNT
RUNNER_GROUP=$(id -gn "${RUNNER_USER}")
# Account names are rendered into systemd templates below.
[[ "${RUNNER_USER}" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*[$]?$ ]]
[[ "${RUNNER_GROUP}" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*[$]?$ ]]
INSTALL_DIR=/usr/local/libexec/rocjitsu-benchmark-runner
SYSTEMD_DIR=/etc/systemd/system
printf '%s  %s\n' "${EXPECTED_SHA256}" "${BPFLAND}" | sha256sum --check --strict

install -d -o root -g root -m 0755 "${INSTALL_DIR}"
install -o root -g root -m 0644 "${SRC}/host_setup.py" "${SRC}/benchmark_run.py" \
  "${INSTALL_DIR}/"
install -o root -g root -m 0755 "${BPFLAND}" \
  "${INSTALL_DIR}/scx_bpfland"
install -o root -g root -m 0644 "${SRC}/systemd/rocjitsu-bpfland.service" "${SRC}/systemd/benchmark.slice" \
  "${SYSTEMD_DIR}/"
for UNIT in system.slice user.slice machine.slice; do
  install -d -o root -g root -m 0755 "${SYSTEMD_DIR}/${UNIT}.d"
  install -o root -g root -m 0644 "${SRC}/systemd/housekeeping-slice.conf" \
    "${SYSTEMD_DIR}/${UNIT}.d/50-rocjitsu-benchmark.conf"
done
install -d -o root -g root -m 0755 "${SYSTEMD_DIR}/init.scope.d"
install -o root -g root -m 0644 "${SRC}/systemd/housekeeping-scope.conf" \
  "${SYSTEMD_DIR}/init.scope.d/50-rocjitsu-benchmark.conf"
systemctl daemon-reload
systemd-analyze verify "${SYSTEMD_DIR}/rocjitsu-bpfland.service" "${SYSTEMD_DIR}/benchmark.slice"
systemctl enable rocjitsu-bpfland.service
```

`rocjitsu-bpfland.service` runs `host_setup.py` before starting Bpfland. Setup
applies fixed 3.2 GHz min/max limits, performance governor/EPP, boost disabling
on reserved cores and their SMT siblings, and idle-state disabling for exit
latency ≥10 μs. If per-policy boost controls are unavailable, it disables boost
globally. The service then starts Bpfland and waits for scheduler readiness.
CPU setup runs again each time the service starts, including automatic restarts.

The policy stays applied between jobs. Stopping the service doesn't restore
defaults. If setup fails after partial writes, resolve it with the runner
offline before continuing.

## Configure the Actions runner

Install `runner.conf` as a drop-in for the existing runner service. Replace the
unit name below with the actual `actions.runner.*.service` unit. Use the same
variables from the installation shell:

```bash
RUNNER_UNIT=actions.runner.REPLACE_WITH_ACTUAL_UNIT.service
install -d -o root -g root -m 0755 "${SYSTEMD_DIR}/${RUNNER_UNIT}.d"
sed -e "s/@RUNNER_USER@/${RUNNER_USER}/g" \
  -e "s/@RUNNER_GROUP@/${RUNNER_GROUP}/g" "${SRC}/systemd/runner.conf" | \
  install -o root -g root -m 0644 /dev/stdin \
    "${SYSTEMD_DIR}/${RUNNER_UNIT}.d/50-rocjitsu-benchmark.conf"
systemctl daemon-reload
systemctl cat "${RUNNER_UNIT}"
```

The drop-in places the runner in `benchmark.slice` with initial affinity
`0-15`, leaving the wrapper able to select `16-63` for benchmark children. It
also disables swap and waits for Bpfland before starting the runner. If Bpfland
stops, the runner stops too; restart the runner after scheduler maintenance.

Reboot after installing the files so existing background processes pick up the
CPU restrictions. Bpfland starts at boot and applies the CPU policy before the
runner starts.

## Run benchmarks

Run the wrapper inside the Actions runner service context. Set `INSTALL_DIR`
to the installation path in that shell:

```bash
/usr/bin/python3 -I "${INSTALL_DIR}/benchmark_run.py" -- \
  /PATH_TO_RELEASE_BENCHMARK --WORKLOAD_ARGUMENTS
```

The wrapper checks the effective benchmark and background cgroup CPU masks
before and after execution and confirms affinity `16-63` before launching the
command. It also requires Bpfland to be enabled and rejects results if the
scheduler identity or enable sequence changes during execution, including
internal Bpfland restarts. It doesn't serialize runs; schedule one benchmark
job at a time.
