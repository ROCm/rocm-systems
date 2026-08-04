# Rocjitsu Checkpoint Restore Review Fixes

## Scope

Address the four actionable review findings on PR 9370 without changing the
checkpoint's public C API:

1. Reject malformed FlatBuffers before dereferencing them.
2. Preserve JSON-loader-derived architecture behavior and execution mode.
3. Restore resident wavefronts to their recorded hardware slots.
4. Resume restored resident work after the simulation engine is attached.

## Design

### Checkpoint validation

Construct a FlatBuffers verifier over the complete file buffer and call
`VerifySimulationCheckpointBuffer` before `GetSimulationCheckpoint`. Invalid
buffers throw from the internal restore function; `rj_vm_restore_checkpoint`
continues mapping those failures to `ROCJITSU_STATUS_INVALID_FILE`.

### Configuration parity

Move command-processor architecture policy (VGPR encoding granularity, packed
TIDs, and SDMA packet dialect) behind one shared initialization function. Both
the declarative JSON factory and the legacy XCD constructor call it, preventing
checkpoint restore from carrying a stale duplicate of that policy.

Serialize the source execution mode into the embedded `SimulationConfig` and
apply it to both the reconstructed `SoC::Config` and returned `LoadedConfig`.
This ensures restored compute units and links use the original FUNCTIONAL or
CLOCKED mode.

### Wavefront identity

Refactor compute-unit dispatch so the existing first-free-slot API and a new
explicit-slot API share the same allocation and initialization implementation.
Checkpoint restore uses the serialized `wf_id` as the requested slot. It
rejects out-of-range, already-occupied, or otherwise unallocatable slots rather
than silently changing wave identity.

### Startup lifecycle

After `SimulationEngine::create()` attaches components to their partitions,
`create_from_loaded` asks each non-idle CU on every reconstructed SoC to
schedule work. The existing `schedule_work` guards keep idle or already-running
CUs unchanged. This is safe for ordinary JSON-created VMs and makes restored
resident waves runnable.

## Tests and CI

Add focused regressions that prove:

- junk and truncated checkpoint files return `ROCJITSU_STATUS_INVALID_FILE`;
- gfx942 packed-TID behavior, an architecture-specific SDMA dialect, and
  CLOCKED execution mode survive a round trip;
- a resident wave above a halted lower slot retains its `wf_id` and PC;
- a C API-restored resident wave executes an instruction and terminates.

Run formatting/pre-commit plus the full Release test suite. Mirror the current
rocjitsu workflow with Clang ASan+UBSan, TSan, and GCC ASan+UBSan builds/tests,
then push and verify the resulting GitHub checks.

## Compatibility

The FlatBuffer schema shape and public C API remain unchanged. Existing valid
checkpoints continue to load; malformed files now fail cleanly. The execution
mode field was already part of `SimulationConfig`, so older checkpoints with no
value retain the existing FUNCTIONAL default.
