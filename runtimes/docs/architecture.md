<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Runtime ownership and component boundaries

Compatibility belongs to the selected public C/C++ APIs, binary interfaces,
and their observable behavior. Private Rust APIs, implementation classes,
module names, and source layouts do not need to reproduce ROCr or any other
implementation. Keep a boundary when it protects ownership, a native build
dependency, or independently testable behavior.

## Components

The root package links peer AMDF and HSA frontends with one copy of the private
rocddi core. Their public handles and frontend lifecycles remain separate.
The `ddi`, `loader`, and `image` directories group components; they introduce
no additional workspace or native library boundary.

```text
AMDF C adapter --------------------------> rocddi
HSA C adapter -> HSA object operations --> rocddi
                    |                         |
                    +--> rocloader -----------+
                    |      |
                    |      +--> private, safe ELF byte operations
                    |
                    +--> rocaddr3 -> selected AddrLib source + C++ bridge

rocddi -> private driver/resource contracts -> Linux KFD/DRM
```

`rocddi` owns passive topology, activated devices, backing and mappings, native
queues, and native retirement. Its checked GPU view exposes GPU capabilities
without imposing them on every device family. An activated GPU owns shared
copy resources. Public HSA statuses, AMDF tables, HSA handle encodings, callback
policy, and initialization reference counts remain with the respective adapter.

## HSA operations and the C boundary

`libhsa::api` owns foreign argument conversion, exports, tool dispatch, public
admission, and output/error rules. Resource modules own the operations and
records. Internal dispatch and image/copy operations call those resource
operations directly rather than routing through exported C functions. This
direction is enforced on the private dispatch and synchronous copy paths;
remaining public adapters can migrate as their owning operations are changed.

Use explicit requests, retained owners, and existing operation/generation
leases. Do not add a parallel framework of service traits or another global
registry. Callback and native work must leave lifecycle locks before blocking;
the operation lease still prevents release of the retained HSA session.

## Frontend sessions

`libhsa::session::HsaSession` and `libamdf::session::AmdfSession` extend
`rocddi::session::Session` through composition. Each frontend owner holds its own native
session and adds the objects and policy required by its public API. The AMDF
owner remains exposed as `amdf_instance_t`; HSA publishes one session under its
process lifecycle coordinator. Generation numbers identify successive HSA
session lifetimes and remain part of admission and stale-work checks.

The native session already owns driver selection, admission, process identity,
resource dependencies, and checked retirement. Allocator-backed owner storage
and native signal-event services also live in rocddi. Both frontend sessions
borrow those services directly. Shared process driver backing sits below the
independent native sessions; retiring one frontend never retires another's
session.

`rocddi::memory::storage::RetainOnDrop` provides the common explicit-retention
mechanism used by both frontends and by `SessionRetirement`. It suppresses the
owner's destructor on scope exit or unwind; the caller explicitly releases it
after cleanup proves destruction is safe. The guard does not pin storage,
extend borrowed lifetimes, perform cleanup, or choose the public API's error
policy.

Frontend lifecycle policy stays with its API. AMDF destruction checks public
children before mutation and returns `BUSY` while they remain; creation is inert
and cached metadata queries keep their no-allocation/no-lock contract. HSA
closes public admission before draining calls and preparing tools. Its native
session stays open until tool cleanup finishes. These guarantees require
separate frontend transitions over the common native mechanisms.

## HSA session ownership

HSA initialization references describe the lifetime of public HSA objects and
tool callbacks. A rocddi `Session` describes native admission, resource
retirement, and attachment to a driver context. These lifetimes differ: final
HSA shutdown must exclude ordinary callers while tools still use native services
to stop profiling and release their operands. AMDF owns its own sessions and
does not participate in HSA initialization references.

The HSA source layout follows those owners:

| Source | Responsibility |
| --- | --- |
| `api/lifecycle.rs` | C entry points and panic boundaries. |
| `lifecycle.rs`, `lifecycle/operations.rs` | One process coordinator, initialization references, HSA session publication, and exclusive transition execution outside the registry lock. State variants own the HSA session or native retirement object needed for their next action. |
| `session.rs` | The HSA owner graph: public object registries, agents, configuration, events, control, and a private rocddi session. |
| `session/startup.rs` | Construction and rollback of unpublished native attachments and HSA state. |
| `session/retirement.rs` | Detached callback shutdown and ordered release of a prepared HSA session. |
| `session/control.rs`, `admission.rs` | Session health, ordinary and cleanup cancellation, counted leases, and scoped thread-local execution. |
| `agent.rs`, `config.rs`, `events.rs` | HSA agent projections and activated GPU attachments, sampled policy, and system-event registrations and worker ownership. |
| `queue.rs`, `signal.rs`, `memory.rs`, `loader.rs`, `image.rs`, `pc_sampling.rs` | The corresponding HSA object operations and registries. |

The HSA session composes these owners; it does not provide a second native
device abstraction. Agent queries stay with `Agents`, system-event state stays
with `SystemEvents`, and native mechanisms stay with rocddi. Clock, logging, and
status conversion have their own focused modules. There is no catch-all
`runtime.rs` or parallel runtime service hierarchy.

The final HSA reference closes public admission and cancels ordinary work. The
coordinator first drains admitted calls, then grants tools a scoped cleanup
delegation window while the native session remains open. Successful preparation
closes delegation and advances to a second drain for its admitted descendants.
Either drain can return `RESOURCE_BUSY` after five seconds while keeping its
session and checkpoint. A later shutdown resumes there without replaying
completed preparation. Existing leases still admit nested calls; an idle,
closed gate cannot be reopened by stale thread-local state.

After both drains, shutdown stops and joins callback producers, checks native
queue inactivation, and finishes tool retirement and unload under one
synchronous cleanup lease. It then closes native session admission and
detaches a prepared HSA session for ordered resource release. When only native
retirement remains, the lifecycle holds the exact
`SessionRetirement` owner for a later shutdown attempt. Uncertain frontend
cleanup instead retains the remaining owner graph and prohibits replacement.

Startup uses the same distinction: `InitRollback` preserves either completion,
a retryable `SessionRetirement`, or uncertainty. The next initialization retries
native rollback before constructing another HSA session and commits its first
reference only after publication and tool loading complete. Transition guards
retain incomplete ownership on an error or unwind. Transferable lease tokens
are separate from closure-scoped thread-local execution, so callback reentry
and worker handoff cannot extend a thread-local scope by dropping guards out of
order.

## Code objects and loaded images

`rocloader::elf` is private and forbids unsafe code. It checks ELF tables,
segments, symbols, load extents, and AMDGPU relocations using borrowed input and
caller-owned output bytes. It selects no device and owns no native allocation.

`CodeObject` interprets immutable AMD HSA input and checks its version, target,
features, and symbols. `LoadedImage` owns one relocated instance, either with
allocated device storage or an explicit borrow of an application destination.
`Executable` owns loaded objects, symbol scopes, and load/freeze transactions.
HSA `CodeObjectRecord` and `ExecutableRecord` add public handles and metadata.
One code object may produce several loaded instances.

Target-specific executable fixups belong in `rocloader::target`, separate from
ELF byte mechanics. GFX1250 entry trampolines retain their existing behavior in
this refactor. Whether a compiler/device tuple requires that workaround needs
separate qualification; resemblance to another runtime is not proof.

`ImagePublication` owns the work to publish code writes and invalidate caches
for consuming devices. Public freeze and private kernel dispatch use the same
mechanism. Pending, active, completed, and uncertain publication have different
release/retry rules. An error result alone does not prove that operands can be
released. The HSA owner quarantines the HSA session and retains the image after
uncertain native retirement.

## Surface images

`libhsa::image` owns HSA image and sampler records, descriptor translation,
creation, mip views, and transfers. The application owns pixel backing;
libhsa owns descriptor allocations and retains backing leases during internal
work. The C extension table lives with `api::image`.

Hardware descriptor and sampler word encoding belongs in
`rocddi::device::gpu::image`. HSA enum validation, version adaptation, and
anisotropy normalization remain in the HSA adapter. Encoder inputs use native
concepts rather than HSA C structs or status codes.

`rocaddr3` supplies layout calculations through a narrow native bridge. It owns
no GPU allocation or HSA handle. The selected AddrLib source must match its
headers and be validated as a dependency update. Builds default to the pinned
copy and can explicitly select a shared source tree. No ROCr shared library is
loaded to obtain those algorithms.

Internal image transfers retain the original descriptor, a backing lease,
private views, staging, arguments, and loaded kernel code through checked queue
retirement. Imported image views must preserve producer tiling and compression.
An ordinary Rust scope exit is not a substitute for proving the last GPU user
has stopped.

## Source and verification rules

Use named module roots (`image.rs` and `image/`) for reorganized code. Keep
private helpers with their owner, and split files for coherent responsibilities
rather than to mirror another implementation or a parallel API directory.

Host tests cover byte parsing, target matching, encoders, transactional state,
and injected failures. Native C tests and symbol/layout checks cover the actual
public ABI and production admission path. Real GPU tests are separately needed
for code publication, queue retirement, imported/tiled transfers, and device
compatibility. Parser support does not imply execution support or qualification.
