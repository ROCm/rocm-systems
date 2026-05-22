# Race Detector

A standalone library for detecting synchronization hazards in AMD GPU
workgroups. It tracks in-flight memory events (loads, stores) and
reports races when a register or LDS byte is accessed before the
operation that writes it has been properly synchronized.

This library is in the process of being integrated into rocjitsu's
plugin system.

This project currently only detects intra-workgroup races involving
reads for locations where the value is undeterminable.

## What is a race condition?

A race occurs when the value read from a register or LDS byte is
ambiguous due to unsynchronized access. On AMD GPUs, correct use of
`s_waitcnt` (to wait for a wave's own memory operations to complete)
and `s_barrier` (to synchronize waves within a workgroup) is required
to avoid races. Some examples:

1. A thread issues a load from global memory to a register, but does
   not wait for the load to complete before using the register.
2. Two threads in different waves write to the same LDS address without
   a barrier to specify their relative order. A subsequent read of that
   address produces an ambiguous value.

## What it detects

- **VGPR races**: a vector register is read before a pending global or
  LDS load has completed (`s_waitcnt vmcnt` / `s_waitcnt lgkmcnt`
  insufficient).
- **SGPR races**: a scalar register is read before a pending scalar load
  has completed (`s_waitcnt lgkmcnt` insufficient).
- **LDS races**: an LDS byte is read or written by one wave while
  another wave has an outstanding write to the same byte, without an
  intervening `s_barrier`.

Detection is at byte granularity: D16 (half-register) loads only flag
races on the affected bytes, and LDS races are tracked per byte.

## How it works

Each in-flight memory operation creates an **event** with a lifecycle:

1. **ACTIVE** — the operation is in flight. Accessing the destination
   register or LDS byte is a race.
2. **WAVE_COMPLETE** — `s_waitcnt` has retired the event for the owning
   wave. Safe for the owning wave, but still a race if another wave
   accesses the same LDS byte.
3. **RETIRED** — `s_barrier` has synchronized all waves. The event is
   fully retired and safe for everyone.

**LDS race detection** uses per-byte counters for fast-path checks, with
interval-based overlap scanning as a fallback. Live events are split by
direction so that RAW and WAR hazards are checked independently.

**VGPR race detection** tracks events per register, using the stored
exec mask to determine which lanes are affected. Tracking is at byte
granularity within each 32-bit VGPR so that D16 instructions do not
cause false positives when the other half is accessed independently.

## Tests

The library has its own standalone test suite. Tests drive
`RaceDetector` and
`WaveRaceState` directly via `RaceTestBuilder`, a lightweight helper
that issues memory events and sync operations programmatically.

```bash
ninja -C build race_detector_tests
ctest --test-dir build -R RaceDetector
```

## History

The race detection logic was originally developed as part of
**race-emulator**, a standalone CPU-side GPU assembly emulator that
parsed `.s` assembly text files.

The detection logic in this project is independent of any particular
emulation approach — it operates on abstract memory events (register
loads, LDS accesses, waitcnt, barrier) regardless of how those events are
produced. This library contains only the detection core and its tests.

The emulation part of race-emulator is no longer under development, this
project will use rocjitsu for emulation.
