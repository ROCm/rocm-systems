# Set up a benchmark runner

Reduce benchmark noise with CPU reservation, fixed frequency, idle controls,
and Bpfland. Configuration files are in
[`scripts/benchmark-runner`](../scripts/benchmark-runner/).

Provision every runner labeled `rocjitsu-benchmark` before enabling the updated
workflow. Container jobs require the installed scripts, `benchmark.slice`, and
Bpfland; the workflow doesn't install them.

## Prerequisites

- A Linux host with systemd, cgroup v2, sched_ext support, Python 3.10 or newer, and
  util-linux (`setarch`).
- A `scx_bpfland` 1.1.3 binary built from revision
  `fead94e15184e957469b346b129f412641691bbe`. The service uses default arguments;
  use the same build across runners.
- An existing GitHub Actions runner service using `runsvc.sh`.
- For container jobs, Docker with the systemd cgroup driver and cgroup v2, and
  runner account access to its daemon (commonly through the `docker` group).
  Access to a rootful daemon grants administrative control beyond the runner
  service restrictions.

The supplied configuration uses **64 physical cores and 128 online logical CPUs**,
with SMT sibling pairs `N,N+64`: benchmarks use `16-63`, runner control activity
uses `0-15`, and background processes use `0-15,64-79`. The benchmark cores'
SMT siblings (`80-127`) remain excluded.

## Install the host configuration

Stop the Actions runner and disable conflicting scheduler and power-management
services. Preserve any narrower CPU restrictions when installing the drop-ins.

Set the four placeholders below and run as root. Keep the destination paths,
which match the service files:

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
  "${SRC}/cpu_policy.py" \
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

On each start, `rocjitsu-bpfland.service` runs `host_setup.py`, then starts
Bpfland and waits for readiness. Setup validates topology and controls, then
applies and verifies these settings on reserved cores and their SMT siblings:

- Min/max frequency set to the CPPC nominal frequency (`acpi_cppc/nominal_freq`).
- Governor and EPP set to performance; boost disabled (globally if per-policy boost
  controls are unavailable).
- Idle states with exit latency ≥10 μs disabled.

Stopping the service doesn't restore defaults. Resolve any partial setup
failure before restarting the runner.

## Configure the Actions runner

In the same shell, set `RUNNER_UNIT` and install the runner drop-in:

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

The runner uses `benchmark.slice` (CPUs `0-63`, no swap) with initial affinity
`0-15`. It waits for Bpfland and stops if Bpfland stops; restart the runner
after scheduler maintenance.

Reboot after installing the files so existing background processes pick up the
CPU restrictions. Bpfland starts at boot and applies the CPU policy before the
runner starts.

The [benchmark workflow](https://github.com/ROCm/rocm-systems/blob/develop/.github/workflows/rocjitsu-benchmarks.yml)
sets the Docker CPU limits, host mounts for validation, and
`seccomp=unconfined` for `setarch -R`. It mounts the installed script directory
read-only so setup and validation use the same policy version. The container
runs under `benchmark.slice`, inheriting its swap limit; the harness uses CPUs
`0-15` and wrapped workloads use `16-63`.

To deploy script or policy updates:

1. Stop all runners labeled `rocjitsu-benchmark` before replacing installed files.
2. Reinstall the scripts and any changed systemd files, run
   `systemctl daemon-reload`, and restart `rocjitsu-bpfland.service`. Reboot if
   CPU allocation drop-ins changed. Checkout changes take effect after installation.
3. Restart the runners after the matching configuration is installed on every
   runner.

Record when measurement settings change. The dashboard doesn't separate results
by installed host policy, so comparisons across the change can appear as
code-performance changes.

## Run benchmarks

Run the wrapper inside the Actions runner service context. Set `INSTALL_DIR`
to the installation path in that shell:

```bash
/usr/bin/python3 -E -s "${INSTALL_DIR}/benchmark_run.py" -- \
  /PATH_TO_RELEASE_BENCHMARK --WORKLOAD_ARGUMENTS
```

The wrapper selects CPUs `16-63`, disables ASLR, and checks CPU allocation,
CPU policy, and Bpfland before and after execution. It rejects results if the
policy drifts or Bpfland restarts. Schedule one benchmark job at a time;
the wrapper doesn't serialize runs.

The caller must cancel the whole process group, not just the wrapper PID.
The corpus benchmark harness handles this automatically; the wrapper doesn't
forward signals.
