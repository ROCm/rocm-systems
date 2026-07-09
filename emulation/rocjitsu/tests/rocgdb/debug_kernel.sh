#!/usr/bin/env bash
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

skip() {
  echo "SKIP: $1"
  [[ -n "${ROCGDB_DEMO_REQUIRE:-}" ]] && exit 1
  exit 0
}

# --- Locate tools -----------------------------------------------------------
find_mirage() {
  if [[ -n "${MIRAGE_BIN:-}" && -x "${MIRAGE_BIN}" ]]; then echo "${MIRAGE_BIN}"; return; fi
  if command -v mirage >/dev/null 2>&1; then command -v mirage; return; fi
  for c in "$here/../../../mirage/target/debug/mirage" \
           "$here/../../../mirage/target/release/mirage"; do
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
}

mirage_bin="$(find_mirage)"
[[ -z "$mirage_bin" ]] && skip "mirage binary not found (set MIRAGE_BIN)"
command -v hipcc  >/dev/null 2>&1 || skip "hipcc not found"
command -v rocgdb >/dev/null 2>&1 || skip "rocgdb not found"

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
timeout 180 "$mirage_bin" run --profile "$profile" -- \
  rocgdb --batch \
    -ex 'set breakpoint pending on' \
    -ex 'break add_one' \
    -ex 'run' \
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
check() { # <regex> <description>
  if grep -qaE "$1" <<<"$out"; then
    echo "  ok: $2"
  else
    echo "  MISSING: $2 (/$1/)" >&2
    fail=1
  fi
}

check 'hit Breakpoint 1, .*add_one .*at .*:[0-9]+' 'stopped at the GPU kernel breakpoint'
check '^pc +0x[0-9a-f]+' 'read the wave PC register'
check '^exec +0xffffffffffffffff' 'read the wave EXEC mask (all 64 lanes)'
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

# --- Second scenario: interior breakpoint + continue (displaced stepping) -----
# Continuing past a breakpoint whose original instruction dbgapi cannot simulate
# forces amd_dbgapi_displaced_stepping: dbgapi copies the instruction into the
# per-queue debugger memory reserved in the CWSR header, steps it there, and
# resumes. This asserts that reserved region exists and the emulator executes
# from it (otherwise dbgapi aborts: "Per-queue memory reserved for the debugger
# is missing").
echo "running rocgdb (interior breakpoint + continue) ..."
outfile2="$workdir/rocgdb2.out"
timeout 180 "$mirage_bin" run --profile "$profile" -- \
  rocgdb --batch \
    -ex 'set breakpoint pending on' \
    -ex 'break add_one.hip:15' \
    -ex 'run' \
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
check2 'hit Breakpoint 1, .*add_one .*at .*:15' 'stopped at the interior line breakpoint'
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
[[ -z "$launch_line" ]] && launch_line=27
echo "running rocgdb (GPU address watchpoint, launch line $launch_line) ..."
outfile3="$workdir/rocgdb3.out"
timeout 180 "$mirage_bin" run --profile "$profile" -- \
  rocgdb --batch \
    -ex 'set breakpoint pending on' \
    -ex "break add_one.hip:${launch_line}" \
    -ex 'break add_one' \
    -ex 'run' \
    -ex 'set $waddr = (unsigned long)d' \
    -ex 'continue' \
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
check3 'add_one .*at .*:15' 'stopped at the store that wrote the watched address'

# --- Fourth scenario: illegal instruction -------------------------------------
# Single-step once past the entry breakpoint to a clean instruction boundary,
# overwrite that instruction with an undecodable opcode, and continue. The wave
# fetching it must raise an illegal-instruction exception (SIGILL) reported at
# that PC, exercising the CU's illegal-instruction path + EC_QUEUE_WAVE_ILLEGAL_
# INSTRUCTION + TRAPSTS.illegal_inst.
echo "running rocgdb (illegal instruction) ..."
outfile4="$workdir/rocgdb4.out"
timeout 180 "$mirage_bin" run --profile "$profile" -- \
  rocgdb --batch \
    -ex 'set breakpoint pending on' \
    -ex 'break add_one' \
    -ex 'run' \
    -ex 'stepi' \
    -ex 'set {unsigned int}$pc = 0xffffffff' \
    -ex 'continue' \
    -ex 'info registers pc' \
    "$app" </dev/null >"$outfile4" 2>&1
status4=$?
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
  timeout 180 "$mirage_bin" run --profile "$profile" -- \
    rocgdb --batch \
      -ex 'set breakpoint pending on' \
      -ex 'break bad_access' \
      -ex 'run' \
      -ex 'continue' \
      "$badapp" </dev/null >"$outfile5" 2>&1
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
  echo "  SKIP: could not build bad_access.hip for $arch"
fi

if [[ $fail -ne 0 ]]; then
  echo "FAIL: one or more debug markers were missing" >&2
  exit 1
fi
echo "PASS: rocgdb debugged the emulated GPU kernel end to end"
