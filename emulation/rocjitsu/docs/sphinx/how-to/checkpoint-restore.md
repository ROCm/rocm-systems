---
myst:
    html_meta:
        "description": "How to save and restore a rocJITsu virtual machine checkpoint for resuming simulations or reproducing failures."
        "keywords": "rocJITsu, ROCm, checkpoint, restore, simulation, rj_vm_save_checkpoint, rj_vm_restore_checkpoint, FlatBuffers"
---
# Save and restore a simulation checkpoint

Long-running GPU simulations can take hours or days to reach a point of
interest. Checkpointing lets you persist supported virtual machine
state to disk at a specific simulation tick, then restore it later to
resume execution or reproduce a failure without re-running the
simulation from the beginning.

Saving an AMDGPU wave inside a trap handler is rejected because the checkpoint format omits the interrupted execution state.

Use checkpoints when you need to:

-   Resume a simulation after an interruption without replaying from
    tick zero.
-   Capture the exact state at a known tick so you can investigate a
    failure deterministically.
-   Branch a simulation from a single checkpoint to explore different
    scenarios.

## When to use checkpoints

Checkpoints are most valuable with the `RJ_VM_MODE_DEFAULT` standalone
simulation mode, where you drive the simulation through `rj_vm_step` or
`rj_vm_run`. In this mode, the caller controls the simulation loop and
can choose when to save state.

The checkpoint file uses a FlatBuffers schema located at
`schemas/checkpoint.fbs` in the rocJITsu source tree. For details on the
JSON configuration that creates the VM in the first place, see
[JSON topology configuration](../conceptual/json-configuration.md).

## Save a checkpoint

Call `rj_vm_save_checkpoint` to serialize the VM state to a file. You
must supply the VM handle, a file path, and the current simulation tick.

1.  Run the simulation to the tick you want to capture. The stepping example
    uses a configuration with one engine partition.

    ``` c
    rj_vm_t *vm = NULL;
    rj_status_t status = rj_vm_create("configs/gfx1100_w7900.json",
                                       RJ_VM_MODE_DEFAULT, &vm);
    if (status != ROCJITSU_STATUS_SUCCESS)
        return 1;

    uint64_t tick = 0;
    int active = 1;
    while (active && tick < 50000) {
        status = rj_vm_step(vm, &active);
        if (status != ROCJITSU_STATUS_SUCCESS) {
            rj_vm_destroy(vm);
            return 1;
        }
        tick++;
    }
    ```

2.  Save the checkpoint to disk:

    ``` c
    status = rj_vm_save_checkpoint(vm, "/tmp/my_sim.ckpt", tick);
    if (status != ROCJITSU_STATUS_SUCCESS) {
        fprintf(stderr, "Checkpoint failed: %d\n", status);
    }
    ```

`rj_vm_save_checkpoint` returns `ROCJITSU_STATUS_SUCCESS` on success. It
returns `ROCJITSU_STATUS_INVALID_ARGUMENT` if either the VM handle or
the path is `NULL`. A save can return `ROCJITSU_STATUS_ERROR` when the current
state cannot be restored or when serialization or file I/O fails.
When you are finished with the original VM, call `rj_vm_destroy(vm)`.

On GFX10 (RDNA1 and RDNA2), choose a tick when outstanding wave memory
operations and wait dependencies have drained. If those are the reason a save
fails, keep stepping the VM and retry at a quiescent tick. Other restrictions
require a different checkpoint boundary. For example, save before using
process-owned GDS memory or its ordered-count state. A standalone checkpoint
also requires the process-owned wave resources and custom frontend bindings to
be released or absent. The save diagnostic identifies the state that prevented
the checkpoint.

RDNA2 checkpoints can preserve an ordinary workgroup barrier, the LDS owned by a
CU, and per-wave apertures. Complete externally backed scratch allocations and
shared WGP reservations before saving. Older active RDNA2 records need enough
information to identify the owning process and its register allocations.

On all AMDGPU profiles, allow each admitted AQL dispatch to launch every
assigned workgroup and publish its requested completion signal before saving.
If either is pending, step the VM and retry the save. If the runtime paused the
queue, resume the queue before stepping. Wait for resident AQL waves that use
private scratch to finish before saving. The checkpoint preserves their completed
memory writes. Let waves using a KFD process address space finish before saving.
A standalone restore can reconstruct direct memory and the VM's built-in
identity mapping. A process ID alone cannot reconstruct a process page table.

On RDNA4 and CDNA5, let workgroups with allocated named barriers finish before saving.
The checkpoint records the workgroup and trap barriers used by other resident waves.

Active RDNA1 waves require a resource record containing the logical VGPR allocation and
LDS window. Older RDNA2 records can restore an active wave when they identify its process,
LDS and scratch allocation state, logical register counts, and TRAPSTS. A physical VGPR
dump alone cannot identify a live wave's allocation bounds. Files with no active GFX10
waves remain usable.

## Restore from a checkpoint

Call `rj_vm_restore_checkpoint` to reconstruct a VM from a previously
saved checkpoint file. The restored VM is ready to continue execution.

1.  Restore the VM:

    ``` c
    rj_vm_t *restored_vm = NULL;
    rj_status_t status = rj_vm_restore_checkpoint("/tmp/my_sim.ckpt",
                                                  &restored_vm);
    if (status != ROCJITSU_STATUS_SUCCESS) {
        fprintf(stderr, "Restore failed: %d\n", status);
        return 1;
    }
    ```

2.  Continue running the simulation from where it left off:

    ``` c
    uint64_t additional_ticks = 0;
    status = rj_vm_run(restored_vm, &additional_ticks);
    ```

3.  Destroy the restored VM when finished. A call to `rj_vm_release` is
    needed only to balance an explicit `rj_vm_retain`.

    ``` c
    rj_vm_destroy(restored_vm);
    ```

`rj_vm_restore_checkpoint` returns `ROCJITSU_STATUS_INVALID_FILE` if the
checkpoint file cannot be opened, read, or deserialized, or contains invalid
execution-thread allocation settings. For RDNA2, restore also requires every
active wave to have a recorded owning process, LDS and scratch allocation state,
logical register counts, and sticky exception flags. Older RDNA2 files with no
active waves remain usable. The call returns
`ROCJITSU_STATUS_ERROR` if the restored configuration does not contain a SoC.

### Restoring older AMDGPU checkpoints

On CDNA1 through CDNA5, RDNA3, RDNA3.5 and RDNA4, an active wave requires its
saved LDS window and the CU's saved LDS contents. A zero-length window and an
empty backing record represent a wave with no LDS allocation. Earlier writers
could produce identical files for an empty window and a live allocation. Restore
returns `ROCJITSU_STATUS_INVALID_FILE` for an active wave missing these records.
Idle checkpoints from these writers remain usable.

## Save periodic checkpoints during a long simulation

In a directory dedicated to this VM, write each candidate to a separate file.
After the save succeeds, rename the candidate over the latest checkpoint in
the same directory. An interrupted or rejected save leaves the previous
successful checkpoint available. The time between successful saves can exceed
the requested interval.

``` c
const uint64_t CKPT_INTERVAL = 10000;
const char *candidate = "sim.next.ckpt";
const char *latest = "sim.ckpt";
uint64_t tick = 0;
int active = 1;

while (active) {
    rj_status_t status = rj_vm_step(vm, &active);
    if (status != ROCJITSU_STATUS_SUCCESS) {
        fprintf(stderr, "VM step failed (%d)\n", status);
        break;
    }
    ++tick;
    if (active && tick % CKPT_INTERVAL != 0)
        continue;

    status = rj_vm_save_checkpoint(vm, candidate, tick);
    if (status != ROCJITSU_STATUS_SUCCESS) {
        fprintf(stderr, "Checkpoint at tick %" PRIu64 " failed (%d)\n", tick, status);
        remove(candidate);
        continue;
    }
    if (rename(candidate, latest) != 0) {
        perror("Publishing checkpoint");
        remove(candidate);
    }
}
```

## Status codes

Both checkpoint functions return `rj_status_t`. The relevant codes are:

| Code | Meaning |
|------|---------|
| `ROCJITSU_STATUS_SUCCESS` | Operation completed successfully. |
| `ROCJITSU_STATUS_INVALID_ARGUMENT` | A required argument is `NULL`. |
| `ROCJITSU_STATUS_INVALID_FILE` | Restore could not read or deserialize the checkpoint, found invalid execution-thread allocation settings, or found an active wave missing required LDS records. An active RDNA2 wave also requires its owner, allocation state, and registers. |
| `ROCJITSU_STATUS_ERROR` | The current VM state cannot be restored, serialization or file I/O failed during save, or the restored configuration has no SoC. |


For the full list of status codes, see
[API reference: status codes](../reference/api-status.md). For the complete
VM API reference, see [API reference: virtual machine](../reference/api-vm.md).
