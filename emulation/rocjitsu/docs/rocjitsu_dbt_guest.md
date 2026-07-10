# rocjitsu DBT Guest Mode

A rocjitsu-only path where an unmodified ROCm process can see a
guest GPU, but translate kernels and execute on a real host GPU.

The design has three main components with their own responsibilities:

1. **GuestKfd Interposer**: Shows applications a guest GPU device that looks
   like a real KFD/DRM device.
2. **HSA Hooks**: Intercepts HSA Runtime calls for the guest GPU and forwards
   them to the host GPU, translating kernels as needed.
3. **DBT**: Translates guest GPU kernels to host GPU kernels.

## High Level Flow

User:
```bash
rocjitsu --config configs/guest_gfx950_on_gfx942.json -- ./app
```

Underneath:
```
  LD_PRELOAD=librocjitsu.so
      open/read sysfs topology -> host topology plus synthetic gfx950 node
      open/ioctl /dev/kfd      -> real KFD for host nodes, GuestKfd for guest discovery/startup

  HSA_TOOLS_LIB=librocjitsu_hooks.so
      ROCR internally sees     -> selected host agent plus synthetic guest agent
      hsa_iterate_agents       -> public host slot is replaced by the guest agent
      hsa_agent_get_info       -> guest still reports gfx950 identity
      hsa_queue_create         -> guest agent is replaced with host agent
      hsa_executable_load...   -> guest ELF translated to host ELF, then loaded on host
      hsa_executable_get...    -> guest symbol lookup is replaced with host symbol lookup
      AMD extension calls      -> guest agents are replaced with host agents where needed
```

## Design Principles

1. ROCR must internally discover both the real host GPU and the synthetic guest
   GPU.

   The selected host must stay available for queues, allocations, and code
   loading. The synthetic guest must also exist so ROCR can build the guest HSA
   agent. Public HSA iteration is then shadowed: rocjitsu emits the guest agent
   in the selected host's ordinal slot and suppresses the guest's own slot. This
   keeps applications that choose the first GPU on the guest path while leaving
   the host agent alive for execution.

   If `ROCR_VISIBLE_DEVICES` is set, the launcher expands it so the selected
   host ordinal and appended guest ordinal both remain visible to ROCR before
   the HSA hook applies this public shadowing.

2. Guest-facing discovery stays guest-shaped.

   Calls that applications use to decide which kernel image to load, such as
   agent name and ISA iteration, should report the synthetic guest agent. If we
   mapped those calls to the host, the application would choose host kernels
   and skip DBT.

3. Execution-facing calls map guest handles to host handles.

   Queue creation, code-object load, symbol lookup, memory access, and AMD
   extension operations that take an agent are redirected from the guest agent
   to the selected host agent.

4. KFD emulation stops at guest discovery.

   `GuestKfd` should not run queues, allocate real guest VRAM, or process AQL
   packets. It does handle enough startup plumbing for ROCR discovery: process
   apertures, clock counters, VM acquisition, available-memory queries, memory
   policy setup, synthetic guest memory handles, and guest-to-host gpu_id
   rewrites for map/unmap requests. If a guest execution ioctl is reached after
   the HSA hooks are in place, that is a missed HSA interception and should be
   visible in logs.

5. Code objects are loaded against the host ROCR agent.

   ROCR validates code-object ISA against the load agent using ELF header
   fields. The hook translates the ELF to host ISA and calls the original
   `hsa_executable_load_agent_code_object()` with the host agent, not the guest
   agent. Symbol queries using the guest agent are later remapped to the host
   agent so application code still works.

## Architecture

rocjitsu DBT guest mode has two layers:

1. A Linux KMD/driver interposer that makes the process believe a synthetic GPU
   exists.
2. An HSA tools hook that forwards synthetic-GPU work to the selected real host
   GPU and translates guest code objects to host code objects.

These layers solve different problems and should stay separate.

### Execution Backends

The guest/HSA layers do not require the execution agent to be backed by a
physical GPU. `dbt_guest.execution_backend` selects one of two implementations
of the same execution contract:

| Backend | Execution topology | KFD execution | Code-object load |
| --- | --- | --- | --- |
| `hardware` | Real KFD topology plus the synthetic guest | Forwarded to real `/dev/kfd` | Translated ELF loaded on the selected physical host agent |
| `simulator` | Simulated target topology plus the synthetic guest | Forwarded to `SimulatedKfd` | Translated ELF loaded by ROCR on the simulated target agent |

In simulator mode, `simulator_config` names a normal RocJITsu VM config. A
relative path is resolved relative to the DBT guest config, which keeps the
guest/translation policy separate from the reusable target-machine model:

```json
{
  "dbt_guest": {
    "enabled": true,
    "guest_isa": "gfx1250",
    "host_isa": "gfx1201",
    "execution_backend": "simulator",
    "simulator_config": "gfx1201_functional.json",
    "guest_device": {
      "gpu_id": 1250,
      "gfx_target_version": 120500
    }
  }
}
```

The simulator backend is composition rather than a separate DBT runtime:

1. The interposer creates the target VM from `simulator_config`. This creates
   the normal `SimulatedKfd`, target topology, GPU memory, command processor,
   and simulation engine.
2. `GuestKfd` wraps that `SimulatedKfd`. It copies the simulated target
   topology, appends the guest node, and delegates host-target ioctls, mmaps,
   DRM metadata, doorbells, and unmaps to the wrapped driver.
3. ROCR discovers both agents internally. The HSA hook presents the guest in
   the target agent's public slot and maps execution-facing calls back to the
   target agent exactly as it does for hardware.
4. The shared DBT hook translates the guest ELF once. Normal ROCR queue and
   code-object operations then reach `SimulatedKfd`, so dispatch, memory,
   signals, faults, and completion use the production simulator path.

This preserves the central identity invariant for both backends:

```text
public identity:    configured guest agent and ISA
execution identity: selected physical or simulated target agent and ISA
```

The gfx1250 example selects `gfx1201_functional.json`, a reduced correctness
profile that retains the R9700 KFD/HSA identity while reducing the execution
topology to one real RDNA4 WGP (two sibling CUs). Keeping functional E2E on a
single WGP avoids multiplying event-loop overhead by every idle CU in the
hardware-shaped `gfx1201_r9700.json` model; it does not increase or bypass any
architectural resource or narrow the code objects ROCR may load. Each physical
RDNA4 CU contributes 64 KiB of LDS. A
translated descriptor with
`COMPUTE_PGM_RSRC1.WGP_MODE=1` is placed on an adjacent CU pair and sees the
combined 128 KiB WGP capacity, matching LLVM's GFX10+ local-memory model. CU
mode remains limited to one CU's 64 KiB. Requests larger than the selected
mode's capacity, or WGP requests on a topology without a sibling pair, fail at
dispatch with an explicit capacity diagnostic instead of remaining queued.
Focused scheduler tests also exercise the same rules across the full R9700
topology.

`GuestKfd` owns one application-facing open-reference domain and keeps the
wrapped driver's primary fd private. The first application open retains the
simulator process; duplicate KFD fds retain only the wrapper reference; the
last wrapper close releases the simulator reference. After `fork()`, the child
drops inherited wrapper/VM state and creates a fresh VM on its next launch so
it never reuses an engine thread that existed only in the parent.

Backend selection is explicit and never falls back. `simulator` requires a
non-empty `simulator_config`; `hardware` rejects one. Invalid configuration or
failure to construct the selected backend stops discovery rather than silently
using the other path.

The tree includes complete example configurations for both supported
simulator-backed development paths:

- `configs/guest_gfx1250_on_simulated_gfx1201.json`
- `configs/guest_gfx950_on_simulated_gfx942.json`

The peer hardware-backed gfx1250 configuration is
`configs/guest_gfx1250_on_gfx1201.json`; omitting `execution_backend` selects
the backward-compatible `hardware` default.

Use the ROCm SDK from the workspace's TheRock virtual environment, not a system
ROCm installation. A launch from the workspace root looks like:

```bash
source venv/bin/activate
sdk_root=$(rocm-sdk path --root)
export ROCM_PATH="$sdk_root"
export HIP_PATH="$sdk_root"
export PATH="$PWD/venv/bin:$sdk_root/bin:$sdk_root/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$sdk_root/lib:$sdk_root/lib64:$sdk_root/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

build/tools/rocjitsu/rocjitsu \
  --config rocm-systems/emulation/rocjitsu/configs/guest_gfx1250_on_simulated_gfx1201.json \
  -- rocminfo
```

`rocminfo` should report the configured guest publicly. The target simulator
agent remains internal to the hook mapping. Application launches use the same
command shape after `--`; no target GPU is required.

The functional simulator is much slower than hardware, especially for large
matmul modules. A long-running dispatch that continues to retire instructions
is not treated as a backend fallback or semantic success: the local validation
helper uses a bounded 1,800-second per-module timeout and still requires the
application's numerical checks to pass. The default RocJITsu regression command
remains:

```bash
ctest --test-dir build -j4 --output-on-failure -E '^(Rocblas|Rccl)'
```

CDNA4-to-CDNA3 dynamic-copy dispatches are registered by default for both
direct simulated execution and DBT guest execution. The larger Triton cases
remain opt-in with `RJ_ENABLE_EXPERIMENTAL_CDNA3_SIM_DBT_TESTS=ON` because they
exercise separately tracked CDNA3 simulator/semantic gaps.

### KMD Driver Interposer

Applications do not always use HSA as their only GPU discovery path. Some
frameworks and tools inspect `/dev/kfd`, KFD topology sysfs, DRM render nodes,
or AMDGPU device metadata directly. If rocjitsu only hooks HSA, those clients
may never believe a guest GPU exists.

The KMD layer therefore appends a synthetic GPU to KFD topology and exposes a
matching synthetic DRM render node. Its job ends at discovery and startup
plumbing. It should not execute guest queues or emulate packet dispatch for this
DBT path.

For host-facing operations, `GuestKfd` forwards to the real `/dev/kfd`. For
guest-facing startup operations, it either answers from the synthetic metadata
or rewrites the request to the configured host GPU. Guest doorbell mappings and
unsupported guest execution ioctls fail visibly.

### HSA Hooks

Applications generally dispatch through HSA. HSA API calls are mostly target
independent, but AQL packets, queue resources, code-object ISA validation, and
some extension calls carry target-specific meaning. Patching every packet and
every target-dependent field after queue creation is too fragile.

Instead, ROCR builds two HSA agents:

- the selected host agent, backed by the real GPU and real KFD execution path;
- the synthetic guest agent, backed by the KFD overlay and guest metadata.

The HSA hook presents one public replacement agent: the guest agent in the
selected host's enumeration slot. Execution-facing guest handles are then mapped
back to the host agent. When a guest code object is loaded, the hook translates
the code object to the host ISA and calls ROCR's original load API with the host
agent. Kernel dispatch then uses normal host queues and normal host ROCR
execution machinery.

The important invariant is:

```text
public identity:   guest agent, guest ISA, guest name, guest memory pools
execution identity: host agent, host queues, host memory backing, host code object
```

## Known Limits and Follow-Ups

- File-backed HSA code-object readers are not translated in the MVP. Add a way
  to capture stable bytes from `hsa_file_t` if an application needs this path.
  Today the hook prints a warning when such a reader is created, and guest loads
  without registered memory bytes fail rather than retrying an incompatible
  original ELF.
- `HSA_TOOLS_DISABLE_REGISTER=1` is a workaround. The better design is a
  rocprofiler-register API-table interposer that applies the same shadowing
  before rocprofiler validates HSA agents.
- KFD guest queue execution is intentionally unsupported. If a guest execution
  ioctl is reached, add or fix an HSA forwarding hook rather than teaching
  `GuestKfd` to execute packets.
- `signal_backtrace` is an opt-in best-effort diagnostic. rocjitsu prewarms the
  unwinder before installing the handler, but fatal-signal stack unwinding can
  still hang if unwinder or loader state is corrupted.
- DBT guest mode is local-launch only right now. Daemon and attach modes still
  belong to the simulation path.
- IPC and advanced multi-process memory sharing are out of scope for the MVP.
