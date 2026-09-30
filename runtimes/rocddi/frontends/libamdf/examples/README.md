# GFX1201 GPU execution examples

`sdma-copy.c`, `sdma-local-round-trip.c`, `pm4-local-round-trip.c`,
`sdma-registered-copy.c`, `aql-copy-add.c`, and `device-producer.c` are
execution-qualified AMDF
consumers. They use only the imported AMDF core-v3 and GPU-v1 tables, create fresh queues, publish directly
to mapped rings, distinguish application completion from queue consumption,
validate results, and release every public child before its parent.
`sdma-dmabuf-copy.c` exercises the qualified SYSTEM DMA-BUF import capability.

The unmodified upstream `Pm4QueueTest` qualifies PM4 SYSTEM copies.
`pm4-local-round-trip.c` qualifies LOCAL memory using the same type-3
COPY_DATA, WRITE_DATA, EVENT_WRITE, ACQUIRE_MEM, and NOP packet forms through
a fixed 4 KiB native KFD COMPUTE ring.

These are standalone C sources. Build `libamdf`, then compile each example
against `../api-headers/include` and the shared or static library. Run them
only with native GPU access. `sdma-readonly-fault.c` deliberately triggers a
fault and should be run separately from the non-faulting examples.

That test deliberately makes one valid SDMA copy target a GPU-read-only SYSTEM
allocation. KFD should report the resulting memory exception, and AMDF must
publish a sticky `DEVICE_LOST` queue state and advance the device reset epoch.
The process retains all potentially referenced objects until exit. A healthy
SDMA workload runs after each fault process to verify system-wide availability.

Every publication
has a five-second deadline. A timeout retains the queue and all potentially
referenced memory until process teardown; the examples do not infer retirement
from failure or attempt a device reset.

## SDMA copy

`sdma-copy.c` performs 64 checked SYSTEM-memory copies with lengths from 1
through 65,535 bytes and wraps its 1 KiB ring eight times. The command sequence
follows ROCr's GFX12 `BlitSdmaV5` path at
`rocr-runtime@e3f270e247ab104874e395cac42aa1c8fbcc2c4e`:

1. `SDMA_PKT_GCR` with the acquire fields used before a system-memory read.
2. `SDMA_PKT_COPY_LINEAR` with CPV clear.
3. `SDMA_PKT_GCR` with the release fields used after a system-memory write.
4. `SDMA_PKT_FENCE` to a separate uncached/coherent SYSTEM allocation.

This path is advertised only on Linux x86-64 when discovery reports GFX1201 and
SDMA IP 7.0.1. Each SDMA example requires the reported GCR and explicit-system
FENCE format features, then verifies that the same exact mask propagates from
the selected family through queue and producer-mapping information. The fixed
stream does not claim compatibility with a different feature combination.

## SDMA LOCAL round trip

`sdma-local-round-trip.c` accepts `private` or `public` and `process` or
`instance`; the defaults are private and PROCESS. From `runtimes/rocddi`:

```sh
cc -O2 -std=c11 -Wall -Wextra -Werror -I../api-headers/include \
  frontends/libamdf/examples/sdma-local-round-trip.c \
  -Ltarget/release -lamdf -o /tmp/sdma-local-round-trip
LD_LIBRARY_PATH=target/release /tmp/sdma-local-round-trip private process
LD_LIBRARY_PATH=target/release /tmp/sdma-local-round-trip private instance
LD_LIBRARY_PATH=target/release /tmp/sdma-local-round-trip public process
LD_LIBRARY_PATH=target/release /tmp/sdma-local-round-trip public instance
```

Both modes require exact GPU read/write access and verify LOCAL placement,
addressability, completion, consumed indices, results, untouched tails, active
queue status, and cleanup. Private LOCAL rejects host mapping while preserving
the caller's output pointer. Public LOCAL requires the advertised host-visible
profile, verifies its WC host mapping and both prospective and concrete
host/device cache-pair recipes, and uses explicit host cache transitions. Each
of 64 iterations checks CPU reads of SDMA-written VRAM and SDMA reads of
CPU-written VRAM. The private path wraps its 1 KiB ring eight times; the
public path wraps it 16 times. All four modes passed on the current GFX1201
source. The cache-pair claim covers one owner and this SDMA family; LOCAL peers
need separate qualification. AQL and PM4 LOCAL execution are covered below.

## PM4 LOCAL round trip

`pm4-local-round-trip.c` accepts `private` or `public` and `process` or
`instance`. From `runtimes/rocddi`:

```sh
cc -O2 -std=c11 -Wall -Wextra -Werror -I../api-headers/include \
  frontends/libamdf/examples/pm4-local-round-trip.c \
  -Ltarget/release -lamdf -o /tmp/pm4-local-round-trip
LD_LIBRARY_PATH=target/release /tmp/pm4-local-round-trip private process
LD_LIBRARY_PATH=target/release /tmp/pm4-local-round-trip private instance
LD_LIBRARY_PATH=target/release /tmp/pm4-local-round-trip public process
LD_LIBRARY_PATH=target/release /tmp/pm4-local-round-trip public instance
```

The private path verifies GPU copies through unmappable LOCAL storage. The
public path also verifies prospective and concrete directional cache pairs,
reads PM4-written VRAM through a WC host mapping, and copies CPU-written VRAM
with PM4. Both use explicit host cache transitions and conservative PM4 GCR
barriers. Sixteen checked iterations and four or eight ring wraps passed in
each native lifetime. Queue consumption and application completion are checked
separately. This qualifies only one owner on GFX1201; LOCAL peers remain
unqualified on this host.

## SDMA registered-host copy

`sdma-registered-copy.c` allocates caller-owned, page-aligned host storage and
registers three deliberately subpage-offset logical ranges for exact GPU access.
Run it with no argument for PROCESS or with `instance` for INSTANCE. It verifies
the selected lifetime's feature reporting, REGISTER profile limits,
logical and page-cover lengths, source offsets, common host/GPU alignment,
unknown physical identity, preserved host pointers, independent GPU addresses,
write-back mappings, and directional SYSTEM-memory cache pairs.

The workload performs 64 checked copies with lengths from 1 through 65,535
bytes and wraps its 1 KiB ring eight times. Source, target, and completion ranges
use distinct offsets; the completion offset remains dword aligned for the SDMA
fence write. Every iteration observes completion and consumption independently,
verifies the copied range and untouched tail, and keeps the registered pages
live. After destroying all AMDF views and memory objects, it writes the caller
allocations again before freeing them, proving that registration never took
ownership or replaced their host mappings.

## SDMA DMA-BUF copy

`sdma-dmabuf-copy.c` selects a SYSTEM CREATE/IMPORT/EXPORT/HOST_MAP profile with
page-aligned DMA-BUF source offsets and cross-process transport. It exports the
middle page of a three-page shareable allocation, imports it with exact READ
GPU permission, and checks physical identity, native extent, host mapping, and
move-on-success descriptor release. Invalid offsets, descriptors, and backing
identity preserve the external value and output. A child process imports the
same exported page through a fresh AMDF instance and checks every byte. Run
without arguments for PROCESS lifetime or with `instance` for INSTANCE.

The parent destroys the original allocation while its imported attachment
remains usable, re-exports independent descriptors, and copies the imported
page through SDMA into another SYSTEM allocation. It checks completion,
consumption, and result bytes. The workload passed on GFX1201 under both
PROCESS and INSTANCE lifetime. Foreign API DMA-BUFs, LOCAL transport,
distinct-GPU attachment, and arbitrary subpage offsets remain unsupported.

## AQL copy-add

`aql-copy-add.c` performs 128 single-producer scratch-backed compute dispatches
on each of fresh normal-, low-, and high-priority queues, then 128 dispatches on
a fresh normal-priority multiple-producer queue. Each 1 KiB queue wraps eight
times. The example allocates executable code, kernarg, source, target,
completion, and fixed scratch storage in coherent SYSTEM memory. The generated
descriptor declares 36 private bytes per workitem; queue construction rounds
its accepted maximum to 40 bytes and supplies 5,120 bytes for one wave on each
of four shader engines. It also verifies that queue ownership blocks early
scratch-memory destruction and that queue information reports the requested
producer mode and priority. The selected family, queue, and mapping all report
the AQL format-v1 baseline feature mask of zero.

Each standard 64-byte AQL packet uses system acquire/release scopes and declares
the descriptor's private-segment size. One work-item adds an iteration-specific
value through dynamically indexed private storage, performs a system release,
and writes a separate completion token. The consumer independently checks that
token, the queue's consumed packet index, every result, and the untouched output
tail.

The multiple-producer phase creates a fresh queue and launches four host threads.
Each thread atomically reserves packets and has a separate 64-byte kernarg slot,
source, target, completion word, and token. The first four reservations are held
INVALID while the producer frontier advances to four. Reservations one through
three are then release-published and doorbelled while reservation zero remains
INVALID; the consumer must remain at zero. Publishing reservation zero recovers
the queue. All 128 reservations must be unique and cover the complete frontier.
On x86-64 an `SFENCE` orders packet writes before each MMIO doorbell store. Each
producer observes its kernel completion before reserving its next packet, which
bounds this fixed-scratch qualification to four concurrent dispatches while
retaining real producer contention and eight ring wraps.

The example also submits two ordering sequences on a fresh normal-priority
queue. In the control sequence, a following dispatch completes while its
predecessor waits on a host-released gate, proving that the queue can overlap
unbarriered dispatches. The second sequence inserts a zero-dependency AQL
barrier-AND packet with the barrier bit between the same roles. The follower
remains blocked for a 10 ms observation interval, then both dispatches complete
after the host releases the predecessor. This qualifies same-queue barrier-bit
ordering without requiring a signal-allocation service. Barrier-OR and nonzero
dependency signals remain outside this example.

`aql-copy-add.c` requires a generated include containing the freestanding
OpenCL kernel descriptor and text bytes for GFX1201 code-object v4. That
artifact must be checked for relocations, section bounds, a nonzero private
segment declaration, and scratch instructions before use. The generator performs
those checks and writes the include to the requested path. From `runtimes/rocddi`
with ROCm LLVM installed at `/opt/rocm/llvm/bin`:

```sh
cargo build --release -p libamdf --locked
python3 frontends/libamdf/examples/generate-aql-copy-add-gfx1201.py \
  /tmp/aql-copy-add-gfx1201.inc
cc -O2 -std=c11 -Wall -Wextra -Werror \
  -I../api-headers/include -I/tmp \
  frontends/libamdf/examples/aql-copy-add.c \
  -Ltarget/release -lamdf -pthread -o /tmp/aql-copy-add
LD_LIBRARY_PATH=target/release /tmp/aql-copy-add
LD_LIBRARY_PATH=target/release /tmp/aql-copy-add instance
LD_LIBRARY_PATH=target/release /tmp/aql-copy-add local
LD_LIBRARY_PATH=target/release /tmp/aql-copy-add local instance
```

The generated include keeps this example independent of a runtime code-object
loader while it exercises executable AMDF allocation, kernarg access, kernel
loads/stores, scratch loads/stores, and completion visibility. The native AQL
queue owns executable EOP and context-save backing as required by KFD. On the
2026-09-24 source, 50 fresh dynamic SYSTEM runs in each lifetime passed all
phases. The LOCAL mode passed one fresh run in each lifetime with CPU-mapped
WC source and target VRAM, explicit host cache transitions, prospective and
concrete cache-pair checks, and 516 completed dispatches. INSTANCE uses
coherent GTT ring backing because secondary KFD VMs reject USERPTR allocation.

This path is advertised only on Linux x86-64 GFX1201. It qualifies low, normal,
and high single-host-producer queues plus a normal-priority four-thread multiple-
producer queue with fixed caller-supplied scratch. This confirms queue creation,
reported priority, execution, completion, and cleanup; it does not claim a
relative scheduling rate, fairness, or latency bound. Dependency-bearing
barriers, barrier-OR, and dynamic scratch growth need separate support or
qualification. Isolated SDMA fault observation is described below.

## GPU-produced AQL and SDMA queues

`device-producer.c` uses a host-produced AQL dispatch to run a GPU kernel
that writes a packet into a second queue's device-mapped ring, advances its
write index, and rings its device-mapped doorbell. The target is either SDMA,
which copies SYSTEM memory and writes a completion token, or AQL, which runs a
second kernel and writes its own token. The producer writes an AQL packet
header last; release fences order ring, index, doorbell, and completion
stores. The example checks both queue-consumption indices, application
completion, SDMA result bytes, and target ring reuse before cleanup. A
timeout retains all potentially referenced GPU resources until process exit.

Two freestanding OpenCL kernels are compiled as GFX1201 code-object v4
objects. The generator verifies their descriptors, ELF sections, symbols,
relocations, and zero scratch requirement, then emits C includes. From
`runtimes/rocddi`, with ROCm LLVM at `/opt/rocm/llvm/bin`:

```sh
cargo build --release -p libamdf --locked
python3 frontends/libamdf/examples/generate-device-producer-gfx1201.py \
  publisher /tmp/device-producer-publisher-gfx1201.inc
python3 frontends/libamdf/examples/generate-device-producer-gfx1201.py \
  target /tmp/device-producer-target-gfx1201.inc
cc -O2 -std=c11 -Wall -Wextra -Werror \
  -I../api-headers/include -I/tmp \
  frontends/libamdf/examples/device-producer.c \
  -Ltarget/release -lamdf -o /tmp/device-producer
LD_LIBRARY_PATH=target/release /tmp/device-producer sdma process
LD_LIBRARY_PATH=target/release /tmp/device-producer aql process
LD_LIBRARY_PATH=target/release /tmp/device-producer sdma instance
LD_LIBRARY_PATH=target/release /tmp/device-producer aql instance
```

Each mode passed five fresh runs of 64 published packets on the current
GFX1201 source. SDMA reused its target ring eight times and AQL four times.
This qualifies same-device device production in both native lifetimes;
peer-device publication still needs a second GPU and a qualified route.

## Fault observation

`sdma-readonly-fault.c` deliberately submits a valid SDMA copy targeting
GPU read-only SYSTEM memory. Run it in a fresh process when GPU fault testing
is authorized, then run a healthy workload in another process. It leaves
faulted resources to process teardown because retirement was not proved.

```sh
cc -O2 -std=c11 -Wall -Wextra -Werror -I../api-headers/include \
  frontends/libamdf/examples/sdma-readonly-fault.c \
  -Ltarget/release -lamdf -o /tmp/sdma-readonly-fault
LD_LIBRARY_PATH=target/release /tmp/sdma-readonly-fault
LD_LIBRARY_PATH=target/release /tmp/device-producer sdma process
LD_LIBRARY_PATH=target/release /tmp/sdma-readonly-fault instance
LD_LIBRARY_PATH=target/release /tmp/device-producer sdma instance
```

On the current GFX1201 source, both fault processes reported sticky
`DEVICE_LOST` and reset epoch 2; the healthy workloads completed afterward.
This is a fault-event observation, not a live reset or hot-unplug test.

## Qualification boundary

The observed native tuple is Linux 6.17.0-23-generic, KFD UAPI 1.23, GPU ID
33844, PCI device ID `0x7550`, render minor 128, and SDMA IP 7.0.1. The
supporting kernel source used to explain runtime activation is
`amdgpu@48f2f4486a8dc6122195fa8755b6a1a6ee2ad3e2`; its exact compatibility with
the running distribution kernel has not been established.

Packet construction remains caller-owned. AMDF advertises queue formats and
semantic release/acquire operations; the provider does not expose packet
builders or silently insert cache commands. The PM4 LOCAL example adds WC
VRAM coverage to the upstream SYSTEM test.
These examples do not qualify multi-device SYSTEM or
LOCAL/VRAM peer execution, foreign or LOCAL external memory, reset or
hot-unplug recovery, AArch64, additional GPU targets, or a latency/jitter
guarantee. PM4 and SDMA kernel queues are qualified separately. The source
implementation admits LOCAL peers from qualified topology routes and
peer-device queue producers through KFD GPUVM mapping, but this one-GPU host
cannot exercise those routes. Reported SDMA times are observed correctness-run
samples, not a performance contract.
