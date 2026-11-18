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
the operation lease still prevents release of the retained generation.

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
released. The HSA owner quarantines the generation and retains the image after
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
