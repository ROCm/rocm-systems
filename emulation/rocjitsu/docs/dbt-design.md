# DBT Design

Rocjitsu's dynamic binary translator (DBT) turns an AMDGPU code object for a
guest ISA into a loadable code object for a host ISA.

The Dynamic Binary Translation (DBT) system translates AMDGPU code objects compiled for one ISA to execute on a different ISA. The initial target pairs are CDNA4 (GFX950) → RDNA4 (GFX1200/1201) and CDNA4 (GFX950) → CDNA3 (GFX942), but the architecture is designed to support any directional ISA pair.

The central design choice is:

> Translate one kernel scope at a time, then relocate that scope as a unit.

DBT does **not** edit `.text` as one flat stream. A code object can contain
several kernels, shared helpers, PC-relative control flow, and more than one
runtime variant of a kernel. Treating each kernel as a relocation unit makes
those relationships explicit.

## The Short Version

```text
 guest ELF
    |
    v
+---------------------+
| Read kernel          |  kernel descriptors identify roots and ABI state
| descriptors          |
+---------------------+
    |
    v
+---------------------+
| Build source CFG     |  direct edges + recovered static indirect edges
+---------------------+
    |
    v
+---------------------+
| Form kernel scopes   |  reachable blocks for each entry and variant
+---------------------+
    |
    v
+---------------------+
| Translate each scope |  semantic expansion, LDS lowering, re-encoding
| into a local buffer  |  and resource accounting
+---------------------+
    |
    v
+---------------------+
| Place and relocate   |  entry stubs and control-flow PC builders
+---------------------+
    |
    v
+---------------------+
| Commit the ELF       |  descriptors, .text, sidecars, metadata, e_flags
+---------------------+
    |
    v
 host ELF
```

### Runtime host configuration

DBT guest mode keeps the synthetic guest identity separate from the target that
runs translated code. `DbtGuestConfig` describes the advertised guest ISA and
device, while its `DbtHostConfig` describes the host ISA, topology GPU ID, and
execution backend. The backend is an enum with two modes:

- `hardware` forwards execution-facing operations to a real host GPU.
- `simulator` delegates host KFD execution to `SimulatedKfd` and runs the
  translated code on a RocJITsu VM.

A simulator-backed DBT config supports two layouts. When `simulator_config` is
set, it names an external host simulator config; relative paths are resolved
beside the DBT guest config, and that external file overrides any simulator
VM/topology in the DBT guest file. This is the normal composition form and lets
`guest_gfx950_on_simulated_gfx942.json` reuse the golden
`gfx942_cdna3_kmd.json` config. When `simulator_config` is omitted, the DBT
guest block and simulator VM/topology are read from the same file. The
self-contained form is useful for tests and generated temporary configs.

Both layouts use the same translation and validation path. Before discovery,
the runtime rejects guest execution limits that the selected simulator device
cannot provide, including LDS size, scratch slots, waves per SIMD, and wavefront
size.

---

There are three different address spaces to keep straight:

```text
source .text offsets  ->  kernel-local buffer offsets  ->  final .text offsets
```

DBT records mappings between these spaces instead of assuming that every
instruction moves by one constant delta.

## A Small Relocation Example

Suppose a source kernel has three blocks:

```text
source .text

0x100  block A: conditional branch to block C
0x104  block B: guest_op
       ...
0x120  block C: s_endpgm
```

On the host, `guest_op` needs a longer instruction sequence. The translated
layout might look like this (offsets are illustrative):

```text
kernel-local buffer

0x00   block A: [branch patch window]
0x08   block B: host_op_1
                 host_op_2
                 host_op_3
                 ...
0x34   block C: s_endpgm
```

The source branch targeted `0x120`; it must now target local offset `0x34` and,
after final placement, the corresponding final `.text` address. DBT therefore
records the branch while emitting block A and patches it only after every block
has a known location.

This example captures the reason for most of the design:

- instruction sizes can change;
- fallthrough should remain adjacent;
- explicit branches must be relocated; and
- descriptors must point at the final entry, not the temporary buffer.

## Input and Output Shape

A source code object is approximately:

```text
AMDGPU ELF
  ELF header
    e_machine = EM_AMDGPU
    e_flags   = guest machine id + AMDGPU feature bits

  LOAD segment
    .text
      kernel bodies and helpers

    descriptor storage
      kernel_a.kd  ----+
      kernel_b.kd  ----+----> entries in .text
```

An AMDHSA dispatch packet's `kernel_object` is the loaded address of a kernel
descriptor. The entry PC is computed from that descriptor and its signed
`KERNEL_CODE_ENTRY_BYTE_OFFSET`:

```text
entry_pc = descriptor_address + kernel_code_entry_byte_offset
```

This is why moving code and moving descriptors cannot be handled independently.

The translated object is approximately:

```text
Translated AMDGPU ELF
  ELF header
    e_flags = host machine id + preserved upper feature bits

  LOAD segment
    .text
      [kernel_a entry stub, when needed]
      [kernel_a translated body]
      [kernel_a sidecar entry stub and body, when needed]
      [kernel_b entry stub, when needed]
      [kernel_b translated body]
      ...

    existing descriptor storage
      kernel_a.kd  ----> translated kernel_a entry
      kernel_b.kd  ----> translated kernel_b entry

    loaded descriptor tail, when sidecars exist
      kernel_a virtual-LDS descriptor  ----> sidecar entry

  non-ALLOC sections
    .rocjitsu.lds   runtime sidecar metadata, when needed
```

Existing application-visible descriptors are reused and patched. Sidecar
descriptors are loaded data appended after the translated text storage; they are
not exported as ordinary application-visible kernel symbols.

## Translation Phases

### 1. Build Descriptor Plans

Kernel descriptor symbols identify the source kernel entries. For each
descriptor, `KernelDescriptorTranslator` builds a `KdTranslation` plan that
records:

- the source descriptor and entry locations;
- target wave and register resource fields;
- LDS, private-memory, and kernarg sizes;
- descriptor-controlled initial SGPR state;
- any target entry prologue; and
- whether this is a normal or sidecar variant.

Descriptor translation happens early because CFG construction needs the kernel
roots. Some final resource values are intentionally deferred until instruction
lowering has chosen its scratch registers.

Wave-size changes are not emulated by the instruction stream. If the guest wave
size cannot be preserved on the host, descriptor translation reports the kernel
as unsupported.

### 2. Build the Source CFG

`BasicBlock::build` decodes source text and creates block leaders at:

- the first decoded instruction;
- every descriptor entry;
- every firmware preload entry;
- every direct branch or call target;
- every recovered indirect target; and
- the instruction after a terminator, when more text follows.

It then adds ordinary CFG edges for branches and fallthrough, and separate call
edges for validated calls.

#### Why Calls Have Their Own Edge Type

A call has two relevant destinations:

```text
             +------------+
call site -->| callee body |
     |       +------------+
     |
     +------> continuation after the callee returns
```

The callee is included in the kernel scope, but a return through
`s_setpc_b64` is valid only for the continuation and saved return-SGPR pair of
that call. Modeling calls separately avoids connecting every callee return to
every possible continuation.

Direct `s_call_b64` and recovered `s_swappc_b64` transfers become call edges
only when DBT can validate a matching return. Otherwise, a recovered concrete
target is treated conservatively as ordinary control flow.

#### Static Indirect Branch Recovery

Indirect recovery recognizes common sequences shaped like:

```text
s_getpc_b64  s[target:target+1]
... scalar arithmetic adjusts the PC ...
s_setpc_b64  s[target:target+1]       ; jump
```

or:

```text
s_getpc_b64  s[target:target+1]
... scalar arithmetic adjusts the PC ...
s_swappc_b64 s[return:return+1], s[target:target+1]  ; call
```

The analysis first builds a temporary direct CFG. It gathers local PC-builder
facts, propagates them across simple block boundaries with bounded dataflow, and
accepts only small concrete target sets.

Unknown writes, conflicting builders, or saturated target sets are not guessed.
If an unrecovered indirect transfer is reachable in a relocated kernel, normal
translation fails closed.

### 3. Form Kernel Scopes

A kernel scope contains the source blocks reachable from one descriptor entry.
For a kernarg-preload kernel, the firmware entry at `entry + 256` is an
additional root of the same scope.

Scope traversal follows:

- branch and fallthrough successors; and
- validated call edges into callees.

It stops at another kernel's hardware entry. Shared helpers are therefore
duplicated when multiple kernels reach them:

```text
source CFG                    translated .text

kernel A --+                 [kernel A body]
           +--> helper         [copy of helper]
kernel B --+                 [kernel B body]
                               [copy of helper]
```

Duplication is deliberate. Every branch, call, and return in a translated body
can resolve through that body's own source-to-target map.

Normal and sidecar variants are separate scopes even when they start from the
same source entry. They can have different descriptors, prologues, LDS lowering,
scratch requirements, and final entries.

### 4. Translate One Scope

Blocks are emitted into a kernel-local byte buffer in source-address order. That
preserves ordinary fallthrough without adding branches between adjacent blocks.

For an ordinary instruction, the decision order is:

```text
source instruction
       |
       v
handwritten semantic EXPAND rule matched?
       | yes                         | no
       v                             v
emit replacement sequence     virtual-LDS lowering matched?
                                     | yes                 | no
                                     v                     v
                              emit LDS replacement    legalization says EXPAND?
                                                           | yes        | no
                                                           v            v
                                                          fail     generated
                                                                   re-encoding
```

Control-flow instructions take a related but separate path: DBT emits or
reserves a relocation window and records a fixup instead of trusting the source
PC-relative value.

#### Semantic Equivalence and Generated Translation

The ISA XML describes structure—encodings, operands, opcodes, and fields—not
execution semantics. DBT's equivalence contract is:

- the same mnemonic on two ISAs has the same semantics; and
- a different mnemonic is equivalent only when it is explicitly registered as
  an alias or reviewed rename. An XML `AliasedInstructionNames` entry is valid
  evidence for adding such an alias to legalization.

Structural similarity alone never proves semantic equivalence.

The generator uses that equivalence information and the structural XML to
choose a legalization action:

- **Identity**: the target form is encoding-compatible;
- **Substitute**: the encoding is compatible but the opcode changes;
- **Lower**: the semantic equivalent has a different encoding layout and needs
  generated decode/re-encode work; or
- **Expand**: no single generated target encoding is sufficient.

Identity, Substitute, and Lower flow through the generated encoding translator.
An Expand entry must be handled by a handwritten semantic rule; otherwise DBT
rejects the kernel.

Semantic rules may emit several instructions and explicit waits. They request
scratch registers, private spill storage, or descriptor growth through
`TranslationContext`. Virtual-LDS mode is also visible to the rules so special
operations such as load-to-LDS and transposed LDS reads can use pair-specific
lowerings.

There is no global instruction scheduler or optimization pass. Each source
instruction is handled when the append-only cursor reaches it.

### 5. Place the Body and Repair Control Flow

After local emission, `append_relocated_kernel_text` chooses the body's final
position in output `.text`. It:

- places the descriptor-visible entry at the source entry's residue modulo 256;
- emits descriptor-visible entry stubs when required;
- rebases block placements and fixups into final `.text` coordinates; and
- records the final descriptor entry and relocated body entry separately.

Only then does DBT repair control flow.

#### Direct Branches

A direct branch gets a patch window chosen from its source distance and the
resources available to the kernel. DBT does not reserve the largest possible
window for every branch because that would significantly bloat large kernels.

The final repair choices are:

```text
target in SOPP range
  -> patch the target ISA branch immediate

target out of range + long window + descriptor-backed SGPR pair
  -> build target PC in the scratch pair, then setpc/swappc

eligible branch target out of range + no long-branch SGPR pair
  -> route through skipped branch-island pools inserted in the body

supported out-of-range conditional branch
  -> invert the condition, skip over an unconditional long transfer
```

Far source branches reserve a long window when a legal scratch pair is
available. Kernels that cannot reserve such a pair receive skipped island pools
during emission. Direct calls cannot use branch islands; they need an encodable
direct form or a reserved long form. A late range increase still fails if
neither the reserved window nor an eligible island chain can represent it.

Unused words in a branch window are filled with target NOPs.

#### Recovered Indirect Transfers

For one effective recovered target, DBT reserves a fixed transfer window at the
`setpc` or `swappc` consumer. Final repair emits either:

- a direct branch or call when the target is close; or
- a canonical get-PC, add-delta, set-PC/swap-PC sequence using the original
  target SGPR pair.

If one dynamic consumer has multiple recovered targets, one direct window cannot
preserve it. DBT keeps the indirect consumer and rewrites each recovered
source-side PC-builder range to compute the relocated target instead.

### 6. Feed Resources Back into the Descriptor

Semantic expansion can discover requirements that were unknown during the
initial descriptor pass:

```text
initial descriptor resources
          |
          v
  instruction lowering
     |       |       |
     v       v       v
   VGPRs   SGPRs   private spill bytes
     \       |       /
      +------v------+
      recompute this variant's descriptor
```

Scratch allocation prefers a liveness-dead VGPR window. Using a register beyond
the original allocation requests descriptor growth. Selected semantic rules can
instead borrow an allowed live VGPR window and save/restore it through per-lane
private memory.

Transient spill slots are reusable between independent instruction expansions.
Persistent slots are separate and hold values that must survive across multiple
translated sequences, such as a spilled virtual-LDS base pointer.

SGPR allocation is stricter. DBT can use descriptor-backed growth or a
liveness-proven dead pair for specific rules, but it has no general SGPR spill
stack.

After body emission and branch repair, DBT recomputes the affected descriptor if
the required VGPR count, SGPR count, or private segment size grew. The recomputed
entry prologue must be byte-for-byte identical to the one already emitted. A
change would require a second layout pass, so the current implementation rejects
it.

### 7. Commit the Code Object

Translation is built on a private `CodeObjectPatcher` copy. The completed plan is
committed only after all kernel scopes succeed (or are replaced by allowed skip
stubs).

The commit step:

1. Pads translated `.text` to at least its original size.
2. Applies normal descriptor resource and entry patches.
3. Replaces `.text`.
4. Appends loaded sidecar descriptors, when present.
5. Appends `.rocjitsu.lds` as non-ALLOC metadata, when present.
6. Replaces only the low AMDGPU machine-id bits in `e_flags`.

If `.text` or the descriptor tail grows, the ELF patcher also updates affected:

- section file offsets and allocated virtual addresses;
- LOAD segment offsets, sizes, and virtual addresses;
- symbol values;
- relocation offsets; and
- descriptor entry offsets when descriptor storage itself moves.

Inserted file padding preserves the required
`p_offset % p_align == p_vaddr % p_align` congruence for LOAD segments. Upper
AMDGPU feature bits in `e_flags`, such as XNACK and SRAMECC state, are preserved.

Any commit failure discards the private patcher copy and returns the original
code object with diagnostics.

## Entry Stubs and Kernarg Preload

Most kernels have one hardware-visible entry. If descriptor translation needs a
prologue, DBT places a small stub before the relocated body:

```text
descriptor entry
  [target ABI prologue]
  s_branch relocated body

relocated body
  [translated source entry block]
  ...
```

A nonzero kernarg-preload specification creates two hardware-visible entries
exactly 256 bytes apart:

```text
target entry
  [prologue]
  s_branch translated source entry

target entry + 256
  [same prologue]
  s_branch translated block for source entry + 256

translated body
  [blocks reachable from either source entry]
```

Both source addresses are CFG leaders and scope roots. The normal stub must fit
without overlapping the `entry + 256` slot, and both stub branches must reach
their body entries.

The descriptor's preload range can extend beyond its declared `kernarg_size`.
DBT preserves the larger of `kernarg_size` and the exact declared preload
extent. Any alignment needed for DBT-owned wrapper fields is padding after that
preserved prefix; DBT does not read undeclared source bytes merely to align it.

## Kernarg Extension

Some translated variants need DBT-owned dispatch data. DBT preserves guest
kernarg offsets by placing a byte-for-byte guest prefix at the start of a wrapper:

```text
kernarg wrapper

  offset 0
  +----------------------------------+
  | preserved guest kernarg bytes    |  guest offsets remain unchanged
  +----------------------------------+
  | padding to 8-byte alignment      |
  +----------------------------------+
  | original kernarg pointer (u64)   |
  +----------------------------------+
  | aligned DBT payload 0            |
  +----------------------------------+
  | aligned DBT payload 1 ...        |
  +----------------------------------+
```

The entry prologue reads DBT payloads from the wrapper, then restores the
guest-visible kernarg pointer before entering translated guest code when the
source ABI supplied that pointer.

If the source descriptor did not request a kernarg pointer SGPR, a sidecar may
add a target-only pair when the 16 initialized user-SGPR limit permits it. The
prologue then repairs any source-visible initialized SGPRs whose target
positions shifted.

## Sidecars and Virtual LDS

A sidecar is an additional implementation of a kernel, selected by the rocjitsu
runtime rather than by normal application symbol lookup.

```text
application-visible kernel object
             |
             v
      normal descriptor ----> normal translated body
             |
             | .rocjitsu.lds maps normal to sidecar
             v
      sidecar descriptor ---> virtual-LDS translated body
```

The metadata stores ELF virtual addresses. After loading, the HSA hook derives
the image load base from the normal kernel object and resolves the loaded
sidecar descriptor address.

### Why Virtualize LDS?

CDNA4 can request more LDS for a workgroup than a CDNA3 host can provide. DBT
therefore creates a virtual-LDS sidecar when the static requirement exceeds the
host limit or the scope may use LDS and could exceed the limit because of the
dispatch-time group-segment request.

The normal body keeps hardware LDS. The sidecar:

- advertises zero fixed hardware LDS;
- uses a global-memory allocation as per-workgroup LDS storage; and
- receives the allocation description through a kernarg wrapper.

At dispatch time, the hook uses:

```text
requested_lds = max(descriptor_static_lds, packet.group_segment_size)
```

The AQL `group_segment_size` is the total group allocation request, not an extra
amount to add to the descriptor's static value.

```text
requested_lds <= host limit
  -> keep normal kernel object and packet

requested_lds > host limit
  -> allocate GPU-visible backing storage
  -> build a kernarg wrapper and VirtualLdsDispatchState
  -> switch kernel_object to the sidecar descriptor
  -> set packet group_segment_size to zero
  -> use the sidecar private_segment_size
```

The 24-byte runtime payload is:

```text
VirtualLdsDispatchState
  uint64 backing_base
  uint32 stride_x
  uint32 stride_y
  uint32 stride_z
  uint32 reserved
```

The sidecar entry prologue computes a private slice for the current workgroup:

```text
virtual_lds_base = backing_base
                 + workgroup_id_x * stride_x
                 + workgroup_id_y * stride_y
                 + workgroup_id_z * stride_z
```

A singleton grid dimension gets a zero runtime stride. A dimension with more
than one workgroup requires the corresponding descriptor-initialized workgroup
ID SGPR.

Every sidecar instruction that accesses LDS storage must be translated away
from hardware LDS. Pair-specific semantic rules handle special cases such as
MUBUF load-to-LDS and transposed reads; the virtual-LDS layer handles supported
ordinary DS loads and stores. Cross-lane DS operations that do not access LDS
storage may remain native. An unsupported storage operation rejects the sidecar
instead of silently using hardware LDS with a zero-LDS descriptor.

Lowered accesses preserve address/data aliasing with temporaries or private
spills and emit conservative waits where the source LDS completion behavior
requires them.

DBT prefers a permanent descriptor-backed SGPR pair for `virtual_lds_base`. If
that is impossible, spill-per-use mode saves the pointer in persistent private
memory and borrows/restores an SGPR pair around each lowered access.
