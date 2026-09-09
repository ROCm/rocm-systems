# Benchmarking rocjitsu

This guide measures rocjitsu simulator and developer-tool performance on the
host. It does not measure GPU hardware performance or kernel algorithm quality.

Use a fixed workload, toolchain, simulator configuration, and launch environment.
Measure each revision in separate Release build directories and alternate their
run order. Small simulator changes are easily hidden by CPU frequency, process
startup, JIT caches, and unrelated host activity.

## Metrics

| Metric | Use |
|---|---|
| End-to-end wall time | `Elapsed (wall clock) time` in `time.txt`. It includes startup, JIT/compiler work, runtime work, and simulation, and is the primary metric for short kernels and test suites. |
| Active-dispatch throughput | From the [throughput plugin](plugins.md#throughput-plugin) completed-dispatch records: `sum(wave_instructions) / sum(wall_seconds) / 1e6`. This excludes gaps between dispatches. Aggregate counts and time before dividing; do not average the per-dispatch `mips` values. |
| `cycles:u` and `instructions:u` | Whole-process counters in `perf.csv`, used to cross-check noisy wall time. Keep the measured process boundary identical. |
| Peak RSS and page faults | `Maximum resident set size` in `time.txt` and `minor-faults`/`major-faults` in `perf.csv`, used for allocation and simulator-concurrency changes. Measure a fresh process. |

Do not substitute one metric for another. For example, lazy allocation may
improve process wall time and RSS without changing instruction-handler
throughput.

## Build and record provenance

Use the same compiler, SDK, CMake options, and workload artifacts for both
revisions. Build before reserving the machine for measurements.

```bash
repo=/path/to/rocm-systems
build=/path/to/build-release
rocm_sdk=/path/to/the-rock-venv/lib/python3.12/site-packages/_rocm_sdk_devel
config=/path/to/rocjitsu-config.json

cmake -S "$repo/emulation/rocjitsu" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DLTO=OFF \
  -DROCM_PATH="$rocm_sdk"
cmake --build "$build" --target \
  rocjitsu_bin rocjitsu_shared rocjitsu_plugin_throughput_so

git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
sha256sum "$config"
```

Retain the revision, build type, compiler and SDK versions, config hash,
workload revision, command line, environment, and raw output with every result.
Run from each revision's build tree; do not install either build into a shared
SDK during the comparison.

## Prepare the host

Prefer a quiet host. If background work cannot be stopped, reserve complete
physical cores with disjoint systemd cgroup cpusets. `taskset` constrains only
the benchmark and does not stop an unrestricted build from using the same CPUs.

```bash
uptime
ps -eo pid,psr,pcpu,pmem,comm,args --sort=-pcpu | head -n 20
lscpu -e=CPU,CORE,SOCKET,ONLINE
cpupower frequency-info
```

Use `lscpu` to identify SMT siblings. CPU `N+core_count` is not universally the
sibling of CPU `N`. A CPU mask is not isolated if another job can use either
thread of the same core.

Keep the governor and energy-performance policy unchanged across the complete
comparison. Choose one fixed physical core, with a consistent SMT-sibling
policy, for a single-threaded simulator-only probe. Size a larger mask to the
parallelism the runner and its compiler/JIT workers can actually use. Avoid
broad masks when the process may migrate but cannot use the extra cores.

### Reserve CPUs with systemd

Use the system service manager on a cgroup-v2 host. Reserve every SMT sibling
of each selected core, remove those logical CPUs from the top-level background
units, and launch the complete benchmark process tree in a sibling slice. The
copy-paste example below requires those units to be unrestricted and fails if
one already has a nonempty `AllowedCPUs`. In that case, use a root coordinator
that applies the intersection of the existing mask and the housekeeping mask;
never broaden an existing policy.

The masks below are examples from a 96-core, 192-thread host. Derive them from
`lscpu` for the benchmark machine. Run the setup and cleanup from Bash:

```bash
set -euo pipefail
grep -qw cpuset /sys/fs/cgroup/cgroup.controllers

reserved=80-95,176-191
housekeeping=0-79,96-175
background_units=()
declare -A previous_allowed=()
old_paranoid=
paranoid_changed=0

for unit in init.scope system.slice user.slice machine.slice; do
  if [[ $(systemctl show "$unit" -p LoadState --value) != not-found ]]; then
    background_units+=("$unit")
    previous_allowed["$unit"]=$(systemctl show "$unit" -p AllowedCPUs --value)
    if [[ -n ${previous_allowed[$unit]} ]]; then
      echo "$unit already has AllowedCPUs=${previous_allowed[$unit]}" >&2
      echo "compute its intersection with $housekeeping before continuing" >&2
      exit 2
    fi
  fi
done

cleanup_host() {
  local unit
  local status=0
  for unit in "${background_units[@]}"; do
    sudo systemctl set-property --runtime "$unit" \
      "AllowedCPUs=${previous_allowed[$unit]}" || status=1
  done
  if (( paranoid_changed )); then
    sudo sysctl -w kernel.perf_event_paranoid="$old_paranoid" || status=1
  fi
  return "$status"
}
trap cleanup_host EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for unit in "${background_units[@]}"; do
  sudo systemctl set-property --runtime "$unit" "AllowedCPUs=$housekeeping"
done

systemctl show "${background_units[@]}" \
  -p Id -p AllowedCPUs -p EffectiveCPUs -p ControlGroup
```

The sample wrapper below places the complete `/usr/bin/time`, `taskset`, `perf`,
RocJITsu, and workload chain in a transient sibling scope, then drops privileges
before measurement. Its pinned CPU pair is a subset of the reservation. Pass
every required runtime input explicitly to `systemd-run`; do not measure only
the client process.

Common alternatives have narrower guarantees:

- `taskset` does not exclude competitors or protect the selected core's other
  SMT sibling.
- A user-level or nested systemd scope can only narrow its parent's effective
  cpuset. It cannot claim CPUs excluded by `user.slice`, and restricting only
  the benchmark still leaves competitors unrestricted.
- Legacy `cset shield` packages may expect the cgroup-v1 `/cpusets` hierarchy.
  Verify cgroup-v2 support rather than treating a successful command as proof;
  always inspect `cpuset.cpus.effective`.
- Boot-time CPU isolation is static and still requires separate IRQ and kernel
  work handling. Prefer reversible runtime cpusets unless residual kernel noise
  is visible in the measurements.

The cleanup trap must restore the exact prior policies even after a failed run.
If the coordinator is killed forcefully, restore the recorded values manually
before starting unrelated work.

Check access to the userspace counters before collecting samples. If the
preflight fails and the policy is greater than 2, use a dedicated shell on a
controlled benchmark host. The trap restores the original policy if that shell
is interrupted. Do not lower the policy below 2 or make the change persistent.
The conditional keeps an expected preflight failure from terminating the
strict reservation shell before recovery runs.

```bash
old_paranoid=$(sysctl -n kernel.perf_event_paranoid)
printf 'kernel.perf_event_paranoid = %s\n' "$old_paranoid"

if perf stat -e cycles:u,instructions:u -- true; then
  :
else
  perf_status=$?
  if (( old_paranoid <= 2 )); then
    echo "perf preflight failed with policy $old_paranoid" >&2
    exit "$perf_status"
  fi

  sudo sysctl -w kernel.perf_event_paranoid=2
  paranoid_changed=1
  if ! declare -F cleanup_host >/dev/null; then
    trap 'sudo sysctl -w kernel.perf_event_paranoid="$old_paranoid"' EXIT
  fi

  if perf stat -e cycles:u,instructions:u -- true; then
    :
  else
    perf_status=$?
    echo "perf preflight still fails after temporary policy change" >&2
    exit "$perf_status"
  fi
fi

# Keep this shell open through all recorded samples.
```

## Run one sample

Add the throughput plugin and a file sink to a copy of the target config:

```bash
results_root=/path/to/results
out="$results_root/sample-01"
base_config=$config
mkdir -p "$results_root"
if ! mkdir "$out"; then
  echo "sample output already exists: $out" >&2
  exit 2
fi

jq --arg out "$out" \
  '.plugins = {"throughput": {}} |
   .sinks = {"types": ["file"], "dir": $out}' \
  "$base_config" >"$out/config.json"

scope_env=(
  --setenv="HOME=$HOME"
  --setenv="PATH=$PATH"
)
direct_env=(
  "HOME=$HOME"
  "PATH=$PATH"
)
# Add matching scope_env and direct_env entries for every required input.
```

### Triton-specific cache control

Skip this subsection unless the workload compiles through Triton. This includes
Gluon workloads that use Triton as their backend. Choose whether the Triton
cache is fresh or warm before the comparison and keep that policy fixed. An
unset `TRITON_CACHE_DIR` below requests a fresh per-sample cache; a preset value
must name a managed, prepopulated directory.

```bash
if [[ -z ${TRITON_CACHE_DIR+x} ]]; then
  sample_triton_cache="$out/triton-cache"
  if ! mkdir "$sample_triton_cache"; then
    echo "fresh Triton cache path already exists: $sample_triton_cache" >&2
    exit 2
  fi
elif [[ ! -d $TRITON_CACHE_DIR ]]; then
  echo "managed Triton cache path is not a directory: $TRITON_CACHE_DIR" >&2
  exit 2
else
  sample_triton_cache=$TRITON_CACHE_DIR
fi

scope_env+=(--setenv="TRITON_CACHE_DIR=$sample_triton_cache")
direct_env+=("TRITON_CACHE_DIR=$sample_triton_cache")
```

Wrap the exact workload command with rocjitsu, affinity, counters, and resource
measurement. This recipe measures local mode: the workload and simulator share
one process tree. Do not add `--daemon` or `--attach`; those modes require a
separate process-boundary protocol. The throughput report is written to
`$out/throughput.log`. Completed-dispatch records are emitted as dispatches
finish. A shutdown summary is not guaranteed in local mode because normal
process teardown may not destroy the in-process VM, so this workflow does not
depend on it.

```bash
workload=(/path/to/workload --its --arguments)
command=(
  "$build/tools/rocjitsu/rocjitsu"
  --config "$out/config.json"
  -- "${workload[@]}"
)

if [[ -v reserved ]]; then
  cpus=${BENCHMARK_CPUS:-80,176} # This pair must be inside $reserved.
else
  cpus=${BENCHMARK_CPUS:-8,104} # Example only: verify the topology.
fi

make_launcher() {
  if [[ -v reserved ]]; then
    # Collection is asynchronous, so every invocation needs a fresh unit.
    unit="rocjitsu-benchmark-$(systemd-id128 new)"
    launcher=(
      sudo systemd-run --scope --quiet --collect
      --unit="$unit" --slice=benchmark.slice
      --property="AllowedCPUs=$reserved"
      --property=KillMode=control-group
      "${scope_env[@]}"
      --
      /usr/bin/setpriv --reuid="$(id -u)" --regid="$(id -g)" --init-groups
      --no-new-privs
    )
    printf 'systemd unit: %s.scope\n' "$unit" >&2
  else
    launcher=(/usr/bin/env "${direct_env[@]}")
  fi
}

make_launcher
events=cycles:u,instructions:u,task-clock,context-switches,cpu-migrations,minor-faults,major-faults
measurement=(
  /usr/bin/time -v -o "$out/time.txt"
  taskset -c "$cpus"
  perf stat "-x," -o "$out/perf.csv"
  -e "$events"
  -- "${command[@]}"
)

if "${launcher[@]}" "${measurement[@]}" \
    >"$out/stdout.log" 2>"$out/stderr.log"; then
  :
else
  measurement_status=$?
  echo "measurement failed with status $measurement_status" >&2
  exit "$measurement_status"
fi

printf 'dispatches\twave_instructions\tdispatch_seconds_sum\tactive_mips\n'
jq -s -e -r '
  [.[] | select(.schema == "rocjitsu.throughput.v2" and
                .record == "dispatch")] as $dispatches |
  if ($dispatches | length) == 0 then
    error("expected at least one completed throughput dispatch")
  else
    ($dispatches | map(.wave_instructions) | add) as $wave_instructions |
    ($dispatches | map(.wall_seconds) | add) as $dispatch_seconds_sum |
    if ($wave_instructions <= 0 or $dispatch_seconds_sum <= 0) then
      error("throughput dispatches have no completed work")
    else
      [($dispatches | length), $wave_instructions, $dispatch_seconds_sum,
       ($wave_instructions / $dispatch_seconds_sum / 1000000)] | @tsv
    end
  end
' "$out/throughput.log"
```

While a long isolated sample is running, inspect its printed unit name from
another shell. The scope and one of its processes must both be confined to the
reserved CPUs, while the background units exclude them. The measured process
must also have the invoking user's UID, zero effective capabilities, and
`NoNewPrivs: 1`.

```bash
unit=rocjitsu-benchmark-12345 # Replace with the printed unit name.
systemctl show "$unit.scope" \
  -p ControlGroup -p AllowedCPUs -p EffectiveCPUs
cgroup=$(systemctl show "$unit.scope" -p ControlGroup --value)
pid=$(head -n 1 "/sys/fs/cgroup$cgroup/cgroup.procs")
grep -E '^(Uid|CapEff|NoNewPrivs|Cpus_allowed_list):' "/proc/$pid/status"
cat "/sys/fs/cgroup$cgroup/cpuset.cpus.effective"
```

Use the same wrapper for every revision. `perf` overhead was below measurement
noise in a warmed short-kernel control, but changing the process boundary or
affinity changed the absolute counts materially.

## Find CPU hotspots

Profile a representative long-running workload with the same Release binary
and affinity used for timing. Sampling adds overhead, so use this run only for
diagnosis.

```bash
make_launcher
"${launcher[@]}" taskset -c "$cpus" \
  perf record -F 199 -e cycles:u -g --call-graph dwarf \
    -o "$out/perf.data" -- "${command[@]}"

perf report --stdio --no-children --sort=dso,symbol \
  -i "$out/perf.data" >"$out/perf-report.txt"

# Inspect one hot function at the instruction level.
perf annotate --stdio -i "$out/perf.data" \
  --symbol='fully::qualified::symbol'
```

After the final timing or profiling run, restore both host policies. Clear the
trap only after successful cleanup; otherwise it remains available to retry on
shell exit.

```bash
if declare -F cleanup_host >/dev/null; then
  cleanup_host && trap - EXIT
elif (( ${paranoid_changed:-0} )); then
  sudo sysctl -w kernel.perf_event_paranoid="$old_paranoid" && trap - EXIT
fi
```

## Sampling protocol

Set every persistent compiler, JIT, or translation cache used by the workload
to an explicit path rather than relying on a default under the user's home
directory. Record each setting. If the workload is already compiled and has no
frontend cache, record that instead. The Triton-specific subsection above shows
how to control that frontend's cache without imposing it on other workloads.

1. Decide whether compiler/JIT and translation caches are warm or fresh before
   running. Never mix policies within a comparison.
2. For warm-cache simulator comparisons, run one unrecorded warm-up for each
   workload and affinity, then reuse that cache. Do not warm a recorded
   fresh-cache sample; create a new empty cache path for every sample.
3. Use an even number of paired rounds so order is balanced:
   `baseline, candidate`, then `candidate, baseline`. Six pairs are a reasonable
   starting point for changes near 1%; use more when an identical-binary control
   has a similar spread.
4. Report per-revision medians, the observed range or median absolute deviation,
   and paired percentage changes. Do not select the fastest run.
5. Require identical pass/skip results and one or more completed dispatch
   records with nonzero aggregate work and time per run. Require identical
   dispatch counts and aggregate `wave_instructions` before interpreting
   timing.
6. Confirm small wall-time results with both cycles and instructions. Re-run a
   representative long kernel and a short-kernel/test-suite aggregate when the
   optimization may affect them differently.

Run the throughput plugin alone when comparing its instruction-family timing;
other execution plugins add observer cost and may change interval boundaries.

## Case study: gfx1250 IREE and Gluon

This case study records how the protocol behaved during recent functional
gfx1250 simulation work on one quiet 96-core/192-thread host. The short workload
was one Gluon kernel; the larger workload combined 26 precompiled IREE cases
with 10 Gluon cases. The results explain the guidance above, but they are not
universal acceptance thresholds.

The historical runner measured `wall_seconds` with `time.monotonic()` around the
selected suite, not GNU `time`. Its `active_dispatch_seconds` is the sum of the
throughput plugin's dispatch `wall_seconds`, matching the active-dispatch
denominator computed above. Active-throughput CV is the CV of each sample's
`wave_instructions / active_dispatch_seconds`, not the CV of the seconds field.
CV below is `100 * sample standard deviation / mean`, using `n - 1` in the
standard deviation.

| Setup | Samples | Runner wall CV | Active-throughput CV | Cycles CV | Instructions CV |
|---|---:|---:|---:|---:|---:|
| One Gluon kernel, unrestricted, fresh cache, `perf` | 7 | 0.86% | 6.90% | 0.28% | 0.07% |
| Same kernel, 40 allowed physical CPUs | 7 | 2.29% | 7.57% | 0.56% | 0.14% |
| Same kernel, one core and its SMT sibling | 7 | 0.37% | 6.71% | 0.31% | 0.17% |
| 26 precompiled IREE plus 10 fresh-cache Gluon cases, 40 physical CPUs | 3 | 0.75% | 0.97% | 0.28% | 0.06% |

A follow-up used ten balanced interleaved pairs while five AFL workers remained
active. One worker saturated logical CPU 0. The affinity-only run used physical
core 0 (`0,96`); the isolated run used physical core 80 (`80,176`) inside a
16-core systemd reservation. An uncontended affinity-only control on core 8
verified that isolation itself did not change wall-time variation.

| Background setup | Samples | Runner wall CV | Active-throughput CV | Cycles CV | Instructions CV |
|---|---:|---:|---:|---:|---:|
| `taskset`, uncontended core 8 | 10 | 0.31% | 6.68% | 0.26% | 0.26% |
| systemd-isolated core 80, uncontended control | 10 | 0.31% | 6.14% | 0.23% | 0.12% |
| `taskset`, contended core 0 | 10 | 1.38% | 12.58% | 2.65% | 0.55% |
| systemd-isolated core 80 | 10 | 0.54% | 5.21% | 0.27% | 0.11% |

Against the contended affinity-only setup, systemd isolation reduced CV by 60%
for runner wall time, 59% for active throughput, 90% for cycles, and 80% for
instructions. The median wall-time difference includes removal of actual CPU
contention and is not a simulator speedup. On the uncontended core, wall CV was
unchanged as expected. This makes CPU reservation useful insurance on a shared
host, not a substitute for the identical-work validity checks above.

The quiet-host short workload reported 24 or 25 dispatches with the same
wave-instruction count, the contended follow-up reported 24 through 27, and the
larger suite reported 200 or 201. The tables include that variability to
characterize the workloads, but none of these datasets would pass the strict
A/B validity gate above.

The sub-second active interval remained noisy even with fixed affinity; the
larger suite reduced that noise substantially. For the Triton-backed Gluon
case, prewarming the Triton cache changed median wall time by 4.82%, cycles by
4.19%, and instructions by 5.89%. Establish the measurement floor with an
identical-binary A/B control, and treat an optimization within that floor as
unresolved until more pairs or a longer workload reproduce it.
