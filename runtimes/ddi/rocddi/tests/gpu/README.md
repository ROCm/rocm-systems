<!-- SPDX-License-Identifier: MIT -->

# Queue creation fault probes

After the [CMake build](../../README.md#build-and-validation), run
`./run_queue_faults.sh /tmp/rocddi-cmake/lib` on Linux x86-64 with an
accessible GFX1201 GPU, `/dev/kfd`, a DRM render node, a C compiler, and
GNU `timeout`. The script compiles C ABI probes and a KFD `ioctl`
interposer, checks that each probe loads the staged image, and runs four
fresh processes with 30-second watchdogs. The interposer is intended
only for these probe processes, whose `ioctl` calls pass a third argument.

HSA tests an ambiguous CREATE result and a CU-mask failure followed by failed
rollback DESTROY. It reads the inactive-signal and scratch addresses from the
AQL control buffer passed to KFD, finds their separate KFD allocations, and
checks that neither allocation was freed. AMDF tests an ambiguous CREATE result
and an invalid returned doorbell offset followed by failed rollback DESTROY.
It checks that destruction of the public scratch memory reports `BUSY` after
both failures. All four calls must return an error without a public queue.
Each process exits after its check; process teardown then releases the
intentionally retained native resources.
