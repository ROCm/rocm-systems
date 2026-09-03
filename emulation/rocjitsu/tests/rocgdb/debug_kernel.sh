#!/usr/bin/env bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# rocgdb wave-debugging demo + CI harness for the rocjitsu emulator.
#
# Compiles the demo HIP kernel with device debug info, then drives real ROCgdb
# through `mirage run` to: set a breakpoint on the GPU kernel `add_one`, run,
# stop at the wave, read PC/EXEC and the instruction at PC, single-step, and
# continue the kernel to completion. It asserts the expected markers so it can
# double as a CI smoke test for the full mirage + rocjitsu + rocm-dbgapi + ROCgdb
# stack.
#
# Exit codes:
#   0   success (all debug markers observed) OR a required tool is missing and
#       $ROCGDB_DEMO_REQUIRE is not set (skipped)
#   77  skipped (missing tool) when $ROCGDB_DEMO_REQUIRE=1 uses exit 1 instead
#   1   failure (a marker was missing or the run errored)
#
# Environment:
#   MIRAGE_BIN            path to the mirage binary (default: search PATH and the
#                        repo's target/{debug,release})
#   ROCJITSU_LIB         path to librocjitsu.so (default: mirage auto-discovers)
#   MIRAGE_PROFILE       mirage profile / GPU target (default: mi350x = gfx950)
#   OFFLOAD_ARCH         hipcc --offload-arch (default: gfx950, must match profile)
#   ROCGDB_DEMO_REQUIRE  if set, missing tools fail (exit 1) instead of skipping
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${MIRAGE_PROFILE:-mi350x}"
arch="${OFFLOAD_ARCH:-gfx950}"
if [[ "$arch" == gfx1250 ]]; then
  exec_regex='^exec +0x(00000000)?ffffffff'
  expected_wave_positions=(0 1 2 3)
  expected_wave_locals=(3 227 451 675)
else
  exec_regex='^exec +0xffffffffffffffff'
  expected_wave_positions=(0 1)
  expected_wave_locals=(3 451)
fi

# CTest matches the whole output against SKIP_REGULAR_EXPRESSION "SKIP:" (see
# the RocgdbDebug.KernelBreakpoint registration in tests/CMakeLists.txt), and a
# match reports the entire seven-scenario run skipped regardless of the exit
# status. So the literal string "SKIP:" must be emitted only from here, where it
# always means a test-wide prerequisite is missing and the run stops
# immediately. A mid-run scenario that cannot proceed sets fail=1 instead --
# printing the token from there would hide every result the run had already
# produced. Anchoring the CMake regex is not an alternative: cmsys anchors to
# the start of the whole output blob, not per line, which would break this
# function instead.
skip() {
  echo "SKIP: $1"
  [[ -n "${ROCGDB_DEMO_REQUIRE:-}" ]] && exit 1
  exit 0
}

# --- Locate tools -----------------------------------------------------------
# MIRAGE_BIN is authoritative when set, never advisory. ctest points it at the
# staged <ROCM_HOME>/bin/mirage precisely so mirage resolves *this* build's
# librocjitsu.so; falling through to PATH when the staged copy is missing or not
# executable would run some other mirage against some other library and report
# the result as this build's, which is the failure the staging exists to prevent.
find_mirage() {
  if [[ -n "${MIRAGE_BIN:-}" ]]; then echo "${MIRAGE_BIN}"; return; fi
  if command -v mirage >/dev/null 2>&1; then command -v mirage; return; fi
  for c in "$here/../../../mirage/target/debug/mirage" \
           "$here/../../../mirage/target/release/mirage"; do
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
}

# Checked here rather than inside find_mirage: that runs in a command
# substitution, where `exit` would leave only the subshell and the empty result
# would be reported as a skip.
if [[ -n "${MIRAGE_BIN:-}" && ! -x "${MIRAGE_BIN}" ]]; then
  echo "FAIL: MIRAGE_BIN=${MIRAGE_BIN} is not an executable file" >&2
  exit 1
fi
mirage_bin="$(find_mirage)"
[[ -z "$mirage_bin" ]] && skip "mirage binary not found (set MIRAGE_BIN)"
command -v hipcc  >/dev/null 2>&1 || skip "hipcc not found"
command -v rocgdb >/dev/null 2>&1 || skip "rocgdb not found"

# Python ROCm SDK environments keep runtime and development libraries in
# sibling wheel directories rather than a conventional ROCM_HOME/lib. Pass
# those directories through Mirage so the HIP workload can resolve
# libamdhip64 and the HSA runtime without relying on the caller's environment.
runtime_library_path="${LD_LIBRARY_PATH:-}"
venv_prefix="$(cd "$(dirname "$(command -v rocgdb)")/.." 2>/dev/null && pwd || true)"
for sdk_lib in \
  "$venv_prefix/lib/python"*/site-packages/_rocm_sdk_core/lib \
  "$venv_prefix/lib/python"*/site-packages/_rocm_sdk_devel/lib; do
  [[ -d "$sdk_lib" ]] || continue
  runtime_library_path="${runtime_library_path:+$runtime_library_path:}$sdk_lib"
done
mirage_runtime_args=()
if [[ -n "$runtime_library_path" ]]; then
  mirage_runtime_args+=(--env "LD_LIBRARY_PATH=$runtime_library_path")
fi
# The nightly gfx1250 HIP/ROCr stack enables its code-object rewrite path by
# default. A debugger qualification must observe the code object that hipcc
# emitted, not a replacement held in anonymous memory, so disable that path in
# both CLR and ROCr for every scenario below. The first ROCgdb run also prints
# the inherited value and loaded libraries; the assertions below make this
# fail closed if the variable is lost or the rewrite library is loaded anyway.
mirage_runtime_args+=(--env "HSA_HOTSWAP_DISABLE=1")

# --- Build the demo kernel --------------------------------------------------
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
app="$workdir/add_one"
echo "building $here/add_one.hip (--offload-arch=$arch -g -O0)"
if ! hipcc --offload-arch="$arch" -g -O0 -o "$app" "$here/add_one.hip" 2>"$workdir/build.log"; then
  cat "$workdir/build.log" >&2
  skip "hipcc could not build for $arch"
fi

# --- Drive ROCgdb through mirage --------------------------------------------
echo "running rocgdb under: $mirage_bin run --profile $profile"
# The gating flow exercises the full core: stop at the kernel, inspect wave
# state, single-step three instructions (asserting the PC advances each time),
# and continue to a correct result. Single-stepping is also covered
# deterministically by the engine unit test
# WaveDebugTest.SingleStepExecutesOneInstructionThenReports.
#
# The `print/x $pc` before and after each stepi lets the assertions below verify
# the wave advanced by real instruction boundaries rather than running away.
#
# Output is captured to a file rather than $(...) command substitution: `mirage
# run` execs under a PTY whose forwarded fds keep a bash command substitution
# blocked waiting for EOF, so a file redirect is used instead. stdin is taken
# from /dev/null so the PTY setup does not block when run non-interactively
# (e.g. under CI); rocgdb --batch needs no input.
outfile="$workdir/rocgdb.out"
timeout 180 "$mirage_bin" run --profile "$profile" "${mirage_runtime_args[@]}" -- \
  rocgdb --batch \
    -ex 'show environment HSA_HOTSWAP_DISABLE' \
    -ex 'set breakpoint pending on' \
    -ex 'break add_one' \
    -ex 'run' \
    -ex 'disable 1' \
    -ex 'info sharedlibrary' \
    -ex 'info registers pc exec' \
    -ex 'x/i $pc' \
    -ex 'print/x $pc' \
    -ex 'stepi' \
    -ex 'print/x $pc' \
    -ex 'stepi' \
    -ex 'print/x $pc' \
    -ex 'stepi' \
    -ex 'print/x $pc' \
    -ex 'continue' \
    "$app" </dev/null >"$outfile" 2>&1
status=$?
out="$(cat "$outfile")"
echo "--------------------------------------------------------------------"
echo "$out"
echo "--------------------------------------------------------------------"

if [[ $status -ne 0 ]]; then
  echo "FAIL: rocgdb run exited with status $status" >&2
  exit 1
fi

# --- Assert the debug markers ----------------------------------------------
fail=0

# The two scenarios whose inferior faults on purpose (SIGILL, SIGSEGV) are not
# held to `exit 0` the way the clean scenarios are -- rocgdb's batch status for a
# run that ends on a fatal GPU signal is not something this harness pins down,
# and guessing it would make the test fail on a healthy stack. What must never
# pass silently is the run never producing a whole log, so reject everything from
# 124 up.
#
# The status is mirage's, not rocgdb's: every scenario runs `timeout N mirage run
# -- rocgdb ...`, and mirage forwards the guest's status masked to 8 bits
# (ctl/src/lib.rs), reporting 128+signo for a signalled guest and 127 when it
# could not spawn the command at all. On top of that `timeout` uses 124 for a
# kill and 125 for its own failure, and the shell uses 126/127 for a command it
# could not execute. rocgdb itself only ever exits 0 or 1, so nothing legitimate
# reaches 124 and everything at or above it means the markers below would be
# matched against a truncated log.
check_not_killed() { # <status> <scenario description>
  if [[ $1 -ge 124 ]]; then
    echo "FAIL: $2 rocgdb run did not complete (status $1)" >&2
    fail=1
  fi
}

check() { # <regex> <description>
  if grep -qaE "$1" <<<"$out"; then
    echo "  ok: $2"
  else
    echo "  MISSING: $2 (/$1/)" >&2
    fail=1
  fi
}

check 'hit Breakpoint 1, .*add_one .*at .*:[0-9]+' 'stopped at the GPU kernel breakpoint'
check '^HSA_HOTSWAP_DISABLE = 1' 'disabled the gfx1250 code-object rewrite path'
if grep -qaE 'libhsa_hotswap_rocjitsu|HotSwap: forwarding' <<<"$out"; then
  echo "  FAIL: a HotSwap implementation was active during ROCgdb qualification" >&2
  fail=1
else
  echo "  ok: no HotSwap implementation was loaded"
fi
check '^pc +0x[0-9a-f]+' 'read the wave PC register'
check "$exec_regex" "read the wave EXEC mask (all lanes for $arch)"
check '=> 0x[0-9a-f]+ <.*add_one.*>:' 'disassembled the instruction at PC'
check 'add_one done: host\[0\]=1 host\[63\]=1' 'kernel produced the correct result after continue'
check 'Inferior 1 .*exited normally' 'inferior exited normally'

# Single-step assertion: the four `print/x $pc` values (at the breakpoint and
# after each of three stepi) must be strictly increasing. A runaway wave would
# either overshoot (fewer than four values, having exited) or repeat a PC.
mapfile -t pcs < <(grep -aoE '\$[0-9]+ = 0x[0-9a-f]+' <<<"$out" | grep -aoE '0x[0-9a-f]+')
if [[ ${#pcs[@]} -lt 4 ]]; then
  echo "  MISSING: four PC samples across three stepi (got ${#pcs[@]}: ${pcs[*]:-none})" >&2
  fail=1
else
  strictly_increasing=1
  for ((i = 1; i < 4; i++)); do
    # 64-bit compare via bash arithmetic (PCs fit in a signed 64-bit range here).
    if (( $((pcs[i])) <= $((pcs[i - 1])) )); then
      strictly_increasing=0
      break
    fi
  done
  if [[ $strictly_increasing -eq 1 ]]; then
    echo "  ok: PC advanced across three single-steps (${pcs[0]} -> ${pcs[1]} -> ${pcs[2]} -> ${pcs[3]})"
  else
    echo "  MISSING: strictly increasing PC across single-steps (got ${pcs[*]})" >&2
    fail=1
  fi
fi

# The interior breakpoint, the displaced-step check and the watchpoint check all
# want the line that actually stores to data[i]. Derive it rather than spell it
# out three times: adding a licence header to add_one.hip moves it, and two of
# the scenarios below would then fail on a line number instead of on behaviour.
store_line="$(grep -nE 'data\[i\] \+= 1' "$here/add_one.hip" | head -1 | cut -d: -f1)"
[[ -z "$store_line" ]] && store_line=18

# --- Second scenario: interior breakpoint + continue (displaced stepping) -----
# Continuing past a breakpoint whose original instruction dbgapi cannot simulate
# forces amd_dbgapi_displaced_stepping: dbgapi copies the instruction into the
# per-queue debugger memory reserved in the CWSR header, steps it there, and
# resumes. This asserts that reserved region exists and the emulator executes
# from it (otherwise dbgapi aborts: "Per-queue memory reserved for the debugger
# is missing").
echo "running rocgdb (interior breakpoint + continue) ..."
outfile2="$workdir/rocgdb2.out"
timeout 180 "$mirage_bin" run --profile "$profile" "${mirage_runtime_args[@]}" -- \
  rocgdb --batch \
    -ex 'set breakpoint pending on' \
    -ex "break add_one.hip:${store_line}" \
    -ex 'run' \
    -ex 'disable 1' \
    -ex 'info registers pc' \
    -ex 'continue' \
    "$app" </dev/null >"$outfile2" 2>&1
status2=$?
out2="$(cat "$outfile2")"
echo "--------------------------------------------------------------------"
echo "$out2"
echo "--------------------------------------------------------------------"
if [[ $status2 -ne 0 ]]; then
  echo "FAIL: interior-breakpoint rocgdb run exited with status $status2" >&2
  fail=1
fi
check2() { # <regex> <description>  (checks the second run's output)
  if grep -qaE "$1" <<<"$out2"; then
    echo "  ok: $2"
  else
    echo "  MISSING: $2 (/$1/)" >&2
    fail=1
  fi
}
check2 "hit Breakpoint 1, .*add_one .*at .*:${store_line}" 'stopped at the interior line breakpoint'
check2 'add_one done: host\[0\]=1 host\[63\]=1' 'kernel completed after displaced-stepping the breakpoint'
check2 'Inferior 1 .*exited normally' 'inferior exited normally after interior breakpoint'
if grep -qaE 'Per-queue memory reserved for the debugger is missing' <<<"$out2"; then
  echo "  FAIL: dbgapi could not find the reserved debugger memory" >&2
  fail=1
fi

# --- Third scenario: GPU address watchpoint -----------------------------------
# Capture the device buffer address at a host breakpoint on the launch line,
# then set a hardware watchpoint on it from GPU context (at the kernel
# breakpoint) and continue. The kernel's `data[i] += 1` store must trip the
# watchpoint, which exercises SET_NODE_ADDRESS_WATCH + the memory-pipeline watch
# check + the addr_watch TRAPSTS reporting. Both breakpoints are set before
# `run` so the pending kernel breakpoint resolves as the code object loads and
# the host breakpoint (on the launch line, before the launch executes) lets us
# read the device pointer `d` — no hard-coded VA.
launch_line="$(grep -nE 'add_one<<<' "$here/add_one.hip" | head -1 | cut -d: -f1)"
[[ -z "$launch_line" ]] && launch_line=30
echo "running rocgdb (GPU address watchpoint, launch line $launch_line) ..."
outfile3="$workdir/rocgdb3.out"
timeout 180 "$mirage_bin" run --profile "$profile" "${mirage_runtime_args[@]}" -- \
  rocgdb --batch \
    -ex 'set breakpoint pending on' \
    -ex "break add_one.hip:${launch_line}" \
    -ex 'break add_one' \
    -ex 'run' \
    -ex 'set $waddr = (unsigned long)d' \
    -ex 'continue' \
    -ex 'disable 2' \
    -ex 'watch *(int*)$waddr' \
    -ex 'continue' \
    "$app" </dev/null >"$outfile3" 2>&1
status3=$?
out3="$(cat "$outfile3")"
echo "--------------------------------------------------------------------"
echo "$out3"
echo "--------------------------------------------------------------------"
if [[ $status3 -ne 0 ]]; then
  echo "FAIL: watchpoint rocgdb run exited with status $status3" >&2
  fail=1
fi
check3() { # <regex> <description>  (checks the third run's output)
  if grep -qaE "$1" <<<"$out3"; then
    echo "  ok: $2"
  else
    echo "  MISSING: $2 (/$1/)" >&2
    fail=1
  fi
}
check3 'hit (Hardware )?watchpoint [0-9]+' 'the GPU address watchpoint triggered'
check3 'Old value = 0' 'watchpoint captured the pre-write value'
check3 'New value = 1' 'watchpoint captured the post-write value'
check3 "add_one .*at .*:${store_line}" 'stopped at the store that wrote the watched address'

# --- Fourth scenario: illegal instruction -------------------------------------
# Single-step once past the entry breakpoint to a clean instruction boundary,
# overwrite that instruction with an undecodable opcode, and continue. The wave
# fetching it must raise an illegal-instruction exception (SIGILL) reported at
# that PC, exercising the CU's illegal-instruction path + EC_QUEUE_WAVE_ILLEGAL_
# INSTRUCTION + TRAPSTS.illegal_inst.
echo "running rocgdb (illegal instruction) ..."
outfile4="$workdir/rocgdb4.out"
timeout 180 "$mirage_bin" run --profile "$profile" "${mirage_runtime_args[@]}" -- \
  rocgdb --batch \
    -ex 'set breakpoint pending on' \
    -ex 'break add_one' \
    -ex 'run' \
    -ex 'disable 1' \
    -ex 'stepi' \
    -ex 'set {unsigned int}$pc = 0xffffffff' \
    -ex 'continue' \
    -ex 'info registers pc' \
    "$app" </dev/null >"$outfile4" 2>&1
status4=$?
check_not_killed "$status4" "illegal-instruction"
out4="$(cat "$outfile4")"
echo "--------------------------------------------------------------------"
echo "$out4"
echo "--------------------------------------------------------------------"
check4() { # <regex> <description>  (checks the fourth run's output)
  if grep -qaE "$1" <<<"$out4"; then
    echo "  ok: $2"
  else
    echo "  MISSING: $2 (/$1/)" >&2
    fail=1
  fi
}
check4 'SIGILL, Illegal instruction' 'the illegal instruction raised SIGILL'
check4 '^pc +0x[0-9a-f]+ +0x[0-9a-f]+ <.*add_one' 'stopped at the faulting GPU instruction'
if grep -qaE 'misaligned pc|corrupted state' <<<"$out4"; then
  echo "  FAIL: the reported wave state was corrupted" >&2
  fail=1
fi

# --- Fifth scenario: memory violation -----------------------------------------
# Run a kernel that stores through a wild (never-mapped) device pointer. The
# emulator must report the fault to the debugger as a memory violation (SIGSEGV),
# exercising the CU's unmapped-access detection + EC_QUEUE_WAVE_MEMORY_VIOLATION +
# TRAPSTS.xnack_error. Uses a second demo kernel (bad_access.hip).
badapp="$workdir/bad_access"
if hipcc --offload-arch="$arch" -g -O0 -o "$badapp" "$here/bad_access.hip" 2>"$workdir/badbuild.log"; then
  echo "running rocgdb (memory violation) ..."
  outfile5="$workdir/rocgdb5.out"
  timeout 180 "$mirage_bin" run --profile "$profile" "${mirage_runtime_args[@]}" -- \
    rocgdb --batch \
      -ex 'set breakpoint pending on' \
      -ex 'break bad_access' \
      -ex 'run' \
      -ex 'disable 1' \
      -ex 'continue' \
      "$badapp" </dev/null >"$outfile5" 2>&1
  status5=$?
  check_not_killed "$status5" "memory-violation"
  out5="$(cat "$outfile5")"
  echo "--------------------------------------------------------------------"
  echo "$out5"
  echo "--------------------------------------------------------------------"
  check5() { # <regex> <description>  (checks the fifth run's output)
    if grep -qaE "$1" <<<"$out5"; then
      echo "  ok: $2"
    else
      echo "  MISSING: $2 (/$1/)" >&2
      fail=1
    fi
  }
  check5 'SIGSEGV, Segmentation fault|MEMORY_VIOLATION|memory violation' \
    'the wild store raised a memory violation'
  check5 'bad_access .*at .*:[0-9]+' 'stopped in the faulting kernel'
else
  cat "$workdir/badbuild.log" >&2
  echo "FAIL: could not build bad_access.hip for $arch" >&2
  fail=1
fi

# --- Sixth scenario: private/scratch variable reads ---------------------------
# At the kernel breakpoint, read scratch-resident kernel arguments and locals
# (`data`, `n`) and a dereferenced device value (`data[0]`). This exercises the
# full private-memory path: the emulator stores scratch in the hardware
# dword-interleaved layout, publishes scratch_backing_memory_location +
# COMPUTE_TMPRING_SIZE and the flat_scratch register into the CWSR, and
# rocm-dbgapi resolves the DWARF private_lane locations against it. A regression
# shows up as "Cannot access memory at address private_lane#..." or the dbgapi
# warning "flat_scratch may be corrupted, private memory access is disabled".
echo "running rocgdb (private/scratch variable reads) ..."
outfile6="$workdir/rocgdb6.out"
timeout 180 "$mirage_bin" run --profile "$profile" "${mirage_runtime_args[@]}" -- \
  rocgdb --batch \
    -ex 'set breakpoint pending on' \
    -ex 'break add_one' \
    -ex 'run' \
    -ex 'disable 1' \
    -ex 'print data' \
    -ex 'print n' \
    -ex 'print data[0]' \
    -ex 'info args' \
    -ex 'continue' \
    "$app" </dev/null >"$outfile6" 2>&1
status6=$?
out6="$(cat "$outfile6")"
echo "--------------------------------------------------------------------"
echo "$out6"
echo "--------------------------------------------------------------------"
# Tested after the log is echoed, the way scenarios 2, 3 and 7 do it, so the
# diagnosis is not printed above the output that explains it.
if [[ $status6 -ne 0 ]]; then
  echo "FAIL: private/scratch rocgdb run exited with status $status6" >&2
  fail=1
fi
check6() { # <regex> <description>  (checks the sixth run's output)
  if grep -qaE "$1" <<<"$out6"; then
    echo "  ok: $2"
  else
    echo "  MISSING: $2 (/$1/)" >&2
    fail=1
  fi
}
check6 '\$1 = \(int \*\) 0x[0-9a-f]+' 'scratch-resident pointer arg `data` resolves'
check6 '\$2 = 64' 'scratch-resident scalar arg `n` resolves to 64'
check6 '\$3 = 0' 'dereferenced device value data[0] reads back 0'
check6 'n = 64' '`info args` reports the scratch-resident argument'
# A wave that faults or runs away during the final `continue` still lets rocgdb
# exit 0, so the status check above cannot stand in for these.
check6 'add_one done: host\[0\]=1 host\[63\]=1' 'kernel completed after the scratch reads'
check6 'Inferior 1 .*exited normally' 'inferior exited normally after the scratch reads'
if grep -qaE 'Cannot access memory at address private_lane|flat_scratch may be corrupted' <<<"$out6"; then
  echo "  FAIL: scratch/private memory was not readable" >&2
  fail=1
fi

# --- Seventh scenario: multi-wave workgroup correlation -----------------------
# Launch one workgroup of 128 threads (two wave64 or four wave32 wavefronts) and
# break inside the kernel. Every wave of the workgroup traps; the emulator serializes them
# together (atomic capture once the queue quiesces) so rocm-dbgapi decodes a
# stable control stack and correlates each wave to workgroup (0,0,0) at its
# position. Reading the scratch-resident `local` in each wave must return that
# wave's own value, which
# exercises the per-wave scratch scoreboard mapping. A regression shows up as a
# dbgapi fatal ("not in the same workgroup as the group_leader" /
# "os_queue_packet_id ... not within"), a crash, or identical/again-corrupted
# per-wave values.
mwapp="$workdir/multi_wave"
if hipcc --offload-arch="$arch" -g -O0 -o "$mwapp" "$here/multi_wave.hip" 2>"$workdir/mwbuild.log"; then
  echo "running rocgdb (multi-wave workgroup correlation) ..."
  # Break on the store line so both waves have computed `i` and `local`.
  mw_line="$(grep -nE 'data\[i\] = local' "$here/multi_wave.hip" | head -1 | cut -d: -f1)"
  [[ -z "$mw_line" ]] && mw_line=19
  outfile7="$workdir/rocgdb7.out"
  timeout 180 "$mirage_bin" run --profile "$profile" "${mirage_runtime_args[@]}" -- \
    rocgdb --batch \
      -ex 'set breakpoint pending on' \
      -ex "break multi_wave.hip:${mw_line}" \
      -ex 'run' \
      -ex 'info threads' \
      -ex 'thread apply all print local' \
      -ex 'delete breakpoints' \
      -ex 'continue' \
      "$mwapp" </dev/null >"$outfile7" 2>&1
  status7=$?
  out7="$(cat "$outfile7")"
  echo "--------------------------------------------------------------------"
  echo "$out7"
  echo "--------------------------------------------------------------------"
  if [[ $status7 -ne 0 ]]; then
    echo "FAIL: multi-wave rocgdb run exited with status $status7" >&2
    fail=1
  fi
  check7() { # <regex> <description>  (checks the seventh run's output)
    if grep -qaE "$1" <<<"$out7"; then
      echo "  ok: $2"
    else
      echo "  MISSING: $2 (/$1/)" >&2
      fail=1
    fi
  }
  check7_wave_local() { # <wave-position> <local-value>
    local wave_position="$1"
    local local_value="$2"
    if awk -v position="$wave_position" -v value="$local_value" '
      /^[[:space:]]*Thread [0-9]+ .*AMDGPU Wave / {
        current = index($0, "(0,0,0)/" position) != 0
        next
      }
      current && $1 ~ /^\$[0-9]+$/ && $2 == "=" && $3 == value { found = 1 }
      END { exit found ? 0 : 1 }
    ' <<<"$out7"; then
      echo "  ok: wave ${wave_position} scratch read includes local == ${local_value}"
    else
      echo "  MISSING: wave ${wave_position} did not report local == ${local_value}" >&2
      fail=1
    fi
  }
  check7 "hit Breakpoint 1, .*multi_wave .*at .*:${mw_line}" 'both waves stop at the kernel breakpoint'
  for index in "${!expected_wave_positions[@]}"; do
    check7_wave_local "${expected_wave_positions[index]}" "${expected_wave_locals[index]}"
  done
  check7 'multi_wave done: h\[0\]=3 h\[64\]=451' 'kernel produced the correct per-thread results'
  check7 'Inferior 1 .*exited normally' 'inferior exited normally after multi-wave debug'
  if grep -qaE 'not in the same workgroup as the group_leader|os_queue_packet_id .* is not within|Segmentation fault' <<<"$out7"; then
    echo "  FAIL: multi-wave correlation aborted (dbgapi grouping/packet-id fault or crash)" >&2
    fail=1
  fi
else
  cat "$workdir/mwbuild.log" >&2
  echo "FAIL: could not build multi_wave.hip for $arch" >&2
  fail=1
fi

if [[ $fail -ne 0 ]]; then
  echo "FAIL: one or more debug markers were missing" >&2
  exit 1
fi
echo "PASS: rocgdb debugged the emulated GPU kernel end to end"
