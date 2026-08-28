---
myst:
    html_meta:
        "description": "How to save and restore a rocJITsu virtual machine checkpoint for resuming simulations or reproducing failures."
        "keywords": "rocJITsu, ROCm, checkpoint, restore, simulation, rj_vm_save_checkpoint, rj_vm_restore_checkpoint, FlatBuffers"
---
# Save and restore a simulation checkpoint

Long-running GPU simulations can take hours or days to reach a point of
interest. Checkpointing lets you persist the entire virtual machine
state to disk at a specific simulation tick, then restore it later to
resume execution or reproduce a failure without re-running the
simulation from the beginning.

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

1.  Run the simulation to the tick you want to capture:

    ``` c
    rj_vm_t *vm = NULL;
    rj_status_t status = rj_vm_create("configs/amdgpu_cdna4.json",
                                       RJ_VM_MODE_DEFAULT, &vm);

    uint64_t tick = 0;
    int active = 1;
    while (active && tick < 50000) {
        rj_vm_step(vm, &active);
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
the path is `NULL`, and `ROCJITSU_STATUS_ERROR` if serialization or file
I/O fails.

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
    }
    ```

2.  Continue running the simulation from where it left off:

    ``` c
    uint64_t additional_ticks = 0;
    status = rj_vm_run(restored_vm, &additional_ticks);
    ```

3.  Destroy the VM when finished:

    ``` c
    rj_vm_destroy(restored_vm);
    rj_vm_release(restored_vm);
    ```

`rj_vm_restore_checkpoint` returns `ROCJITSU_STATUS_INVALID_FILE` if the
checkpoint file cannot be opened and `ROCJITSU_STATUS_ERROR` if
deserialization fails.

## Save periodic checkpoints during a long simulation

For simulations that run for many ticks, save checkpoints at regular
intervals so that a crash or interruption loses at most one interval's
worth of work.

``` c
const uint64_t CKPT_INTERVAL = 10000;
uint64_t tick = 0;
int active = 1;
char path[256];

while (active) {
    rj_vm_step(vm, &active);
    tick++;
    if (tick % CKPT_INTERVAL == 0) {
        snprintf(path, sizeof(path), "/tmp/sim_%06" PRIu64 ".ckpt", tick);
        rj_vm_save_checkpoint(vm, path, tick);
    }
}
```

## Status codes

Both checkpoint functions return `rj_status_t`. The relevant codes are:

| Code | Meaning |
|------|---------|
| `ROCJITSU_STATUS_SUCCESS` | Operation completed successfully. |
| `ROCJITSU_STATUS_INVALID_ARGUMENT` | A required argument is `NULL`. |
| `ROCJITSU_STATUS_INVALID_FILE` | The checkpoint file could not be opened or read (restore only). |
| `ROCJITSU_STATUS_ERROR` | Serialization, deserialization, or I/O failed. |


For the full list of status codes, see
[API reference: status codes](../reference/api-status.md). For the complete
VM API reference, see [API reference: virtual machine](../reference/api-vm.md).
