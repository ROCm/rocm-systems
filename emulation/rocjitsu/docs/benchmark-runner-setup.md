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

## Persistent setup storage

The benchmark container requires two pre-provisioned, dedicated host directories:

| Host directory | Container mount | Purpose |
| --- | --- | --- |
| `/srv/rocjitsu-ci/sources` | `/__w/rocm-systems/rocm-systems/ci-sources` | Separate pinned measurement, corpus, TensileLite, full Tensile, llama and nanobind checkouts |
| `/srv/rocjitsu-ci/cache` | `/opt/rocjitsu-cache` | uv packages (`uv/`) and compiler objects (`ccache/`) |

These are bind mounts, not restored Actions cache archives. They must exist before
Docker starts the job; the workflow does not provision the host. The cache root
must be owned by the container's configured UID 0; both mounts must be writable
by that UID, without symbolic-link ancestors. Limit
writers to the trusted benchmark runner/jobs. Same-repository opt-in PR workflows
share this storage; it is not an isolation boundary for untrusted code. Operators
own disk capacity, quotas and pruning, outside active jobs. Run only one benchmark
job at a time on a shared source root.

Checkouts retain immutable revisions, `clean: true` and disabled credential
persistence. TensileLite includes `projects/hipblaslt/tensilelite`, `cmake`,
`shared/origami`, `shared/stinkytofu` and `shared/mxdatagenerator` for the
source-built ROCISA wheel and native adapter; full
Tensile includes `shared/tensile`. Other measurement dependencies retain full
checkouts, including recursive nanobind submodules. Workflow tooling remains an
independent sparse scripts checkout at the workflow revision, not the measurement
revision. Preparation still verifies source revisions and cleanliness.

The pinned setup-uv action installs uv 0.12.21; Python remains 3.12 from the pinned
setup-python action. Preparation uses the action's absolute uv executable output,
not PATH. `uv venv --python 3.12` creates a fresh attempt-owned environment without
seeding pip. Registry installs, package lists and dependency checks use `uv pip`
with that environment's explicit Python. Registry installs retain explicit indexes
and benchmark constraints, with bytecode compilation and copied cache materialization.
`uv build --wheel --no-index --no-build-isolation` builds ROCISA from its pinned
source using preinstalled build requirements, an external build directory and
16 build workers. Exact-wheel installation uses an attempt-local requirements
receipt with the wheel's SHA-256, `--require-hashes --no-index --no-deps`, and no
bytecode compilation. Import checks still verify venv/noneditable URL origin,
actual extension and mapped-library hashes, and exclude `/opt/rocm` mappings;
they do not require an archive hash in uv's `direct_url.json`. Compact inventories
read installed metadata with stdlib `importlib.metadata`; command logs retain
verbose registry resolution information. No pip bootstrap, cache or configuration
is used by benchmark preparation.

Only package setup data and compiler-cache entries persist alongside sources.
Complete builds, venvs, generated artifacts,
Tensile helper caches, Triton caches and measurements remain attempt-local.
Preparation rejects package-cache roots overlapping scratch or source roots and
symbolic paths. Owned-scratch cleanup leaves both host mounts intact. Package
cache reuse does not imply ROCISA wheel reuse or change workload cache policy.

## Provider preparation phases

The workflow retains **Install benchmark dependencies** and **Build rocjitsu**,
then runs one **Prepare benchmark providers** step. Its owned-venv Python commands
run serially: `rocisa`, `tensilelite-build`, `tensilelite-generate`, `tensile-build`,
`tensile-generate`, `llama-build`, `gluon-verify`. The five inventory-consuming
commands starting at `tensilelite-generate` require the selected `--manifest`.
Granular command logs, timings and phase receipts remain in owned scratch. A
failed command stops the grouped step and blocks suite admission. See the
[preparation contract](../website/dashboard/docs/architecture-and-publication.md#isolated-preparation-before-measurement)
for validation and environment-export boundaries. This split adds no wheel or
artifact reuse and does not change measurement or JIT policy.

## Compiler caching

The workflow installs ccache inside the job container. Preparation shares
`/opt/rocjitsu-cache/ccache` across C/C++ builds; no additional runner directory
or mount needs provisioning. Fresh CMake builds initialize both C and C++ compiler
launchers from the helper's environment, including the scikit-build ROCISA wheel,
rocjitsu, the TensileLite adapter/host tests and the full-Tensile adapter.
The pinned corpus llama builder explicitly disables caching and clears launchers;
it remains excluded. HIP kernel generation/assembly and Gluon verification are
not compiler-cache reuse paths.

The helper removes inherited ccache/launcher overrides and writes an attempt-owned
configuration with compiler-content checking and a `5G` shared size limit. A common
source/scratch base directory normalizes changing attempt paths; `/` is not used.
No sloppiness options or ignored headers are enabled. Per-phase ccache statistics
logs, configuration and diagnostic summaries distinguish real hits from compiler
fallbacks. Existing source, package, import/loader, host-test and artifact checks
remain active.

Only compilation results are reused. Configuration, linking, wheel packaging and
validation still run against fresh attempt outputs. The helper does not export
ccache settings or compiler launchers to the measured suite. No completed-artifact,
Tensile helper/JIT, Triton cache or benchmark-result reuse is enabled here.

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
