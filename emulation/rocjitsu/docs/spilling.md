# AMDGPU Register Spilling for DBI

RocJitsu provides a reusable register-spilling backend for post-allocation
dynamic binary instrumentation (DBI). It reserves per-lane private memory,
emits save and restore sequences for borrowed VGPRs, and keeps the kernel
descriptor consistent with any added private storage.

The backend is deliberately smaller than a compiler register allocator. A DBI
client remains responsible for deciding which registers may be borrowed, where
the sequence belongs, and which kernels can reach the patched instruction.

## Credit and provenance

This implementation builds on Kunwar Grover's text-relocation work in
`users/Groverkss/text-relocation-land`. That work established the direction of
requesting semantic scratch registers, preferring liveness-proven registers,
assigning stable per-lane spill storage, and bracketing transformed code with
save and restore sequences.

RocJitsu shares the `PrivateSegmentCursor` range-allocation abstraction with
that work. The DBI backend adds transactional multi-register reservations,
target-specific save and restore emission, kernel-descriptor updates, and
focused tests. Kunwar's broader DBT and CDNA3 text-relocation implementation is
not a dependency of this backend. In particular, both implementations
deliberately avoid rewriting AMDGPU MessagePack notes for private-storage
growth.

## Responsibilities and boundaries

The reusable implementation provides:

- a monotonic allocator for ranges in a kernel private segment;
- stable, idempotent register-to-slot assignment;
- transactional reservation of a multi-register window;
- fixed-offset save and restore sequences for ordinary VGPRs on gfx942,
  gfx950, gfx1201, and gfx1250;
- site-local dynamic-stack frame sequences on those targets;
- kernel-descriptor private-size updates.

It does not provide:

- register liveness or victim selection;
- ownership analysis for text shared by multiple kernels;
- SGPR or AccVGPR save and restore sequences;
- code-cave placement or control-flow relocation;
- dispatch-packet rewriting; or
- a general calling convention for instrumentation bodies.

Callers must fail closed when they cannot prove those surrounding properties.

## Private-segment allocation

`PrivateSegmentCursor` is the policy-free range allocator. Given a byte count,
power-of-two alignment, and exclusive upper limit, `preview()` computes the
next allocation without changing state and `allocate()` commits it. A failed
request does not advance the high-water mark.

`SpillManager` layers stable register identity on that cursor. It appends a
DBI-owned spill zone after the compiler-owned fixed private segment:

```text
0                                      original_private_bytes
+-----------------------------------------------------------+
| guest/compiler private segment                            |
+-----------------------------------------------------------+
                         align up to 16 bytes
                         |
                         v
                         +--------+--------+--------+
DBI spill zone           | slot 0 | slot 1 |  ...   |
                         +--------+--------+--------+
                           4 B      4 B
```

Each 32-bit register lane receives one four-byte slot per work-item. A
multi-lane register window receives consecutive slots. Reallocating the same
register returns the original offset. `reserve()` and `allocate_slots()` check
the complete request before committing, so capacity or encoding failure cannot
leave a partially advanced layout.

For text reachable from multiple kernels, the client must construct one
compatible layout. A sound policy starts the shared zone at
`align_up(max(original_private_bytes), 16)` and grows every owning descriptor to
the resulting common extent. `SpillManager` does not discover those owners.

## Fixed-offset sequences

`build_vgpr_spill_sequence()` supports ordinary B32 VGPR windows on
gfx942, gfx950, gfx1201, and gfx1250. It reserves stable slots and returns
separate target-native save and restore word sequences plus the resulting
private-segment size.

Conceptually, the emitted sequence is:

```text
wait until outstanding guest loads complete
scratch_store_b32 vN, private_offset_N
...
wait until scratch stores complete

instrumentation may clobber vN...

scratch_load_b32 vN, private_offset_N
...
wait until scratch loads complete
resume guest code
```

The waits are intentionally conservative. The leading wait prevents a late
guest load from overwriting a saved victim after the save point. The store
wait ensures the value reaches private memory before instrumentation clobbers
it, and the load wait ensures restoration completes before guest code consumes
it.

The instructions execute under the caller's current `EXEC` mask and do not
change it. Instrumentation that narrows `EXEC` must restore the guest mask
before the VGPR window is returned.

Address-free scratch instructions impose a target-specific encodable
private-offset limit and private-size normalization. Requests outside that
limit return `std::nullopt` without changing `SpillManager`.

## Dynamic-stack sequences

`build_dynamic_stack_vgpr_spill_sequence()` emits a site-local frame for a
kernel whose compiler-generated scratch accesses use a runtime stack. The
caller supplies the stack-top SGPR, frame-base SGPR, and scalar save registers.
The sequence:

1. preserves SCC and the incoming frame base;
2. assigns the frame base from the current stack top;
3. stores borrowed VGPRs at frame-relative offsets;
4. advances the stack top by the temporary frame size;
5. restores the VGPRs; and
6. restores the stack top, frame base, and SCC state.

This helper encodes a mechanism, not an ABI discovery policy. A client may use
it only after proving the target kernel follows the supplied convention and
that the scalar save registers are safe. Mixed conventions or unknown owners
must be rejected.

Dynamic-stack accesses remain frame-relative, but the runtime still needs
enough backing storage for the maximum additional depth. The descriptor update
therefore records the compiler maximum plus the instrumentation frame depth.

## Descriptor and runtime updates

`update_kernel_descriptor_for_spills()` grows one named kernel descriptor's
private requirement and enables private-segment support when a kernel grows
from zero private bytes. It preserves unrelated descriptor fields and does not
shrink an existing allocation.

ROCR derives the loaded kernel symbol's private size from
`kernel_descriptor_t::private_segment_fixed_size`; it does not use the
duplicated `.private_segment_fixed_size` value in the AMDGPU MessagePack note.
That note is not a runtime authority and may be absent, stale, contradictory,
or malformed. The spilling backend therefore neither reads nor rewrites it.

The descriptor helper mutates the supplied image. A DBI client should perform
register, layout, encoding, descriptor, and text-placement validation before
committing its final code-object image. If text growth can move later ELF
contents, resolve descriptors by stable kernel identity rather than retaining
stale file offsets across mutations.

The HSA dispatch packet also carries a private-segment size. Updating it is a
runtime integration responsibility because the static spill backend does not
intercept kernel loads or dispatches.

## Supported boundary

| Capability | Current state |
|---|---|
| Stable private slots | SGPR, VGPR, and AccVGPR identities can be assigned storage. |
| Fixed-offset save/restore | Ordinary B32 VGPR windows on gfx942, gfx950, gfx1201, and gfx1250. |
| Dynamic-stack save/restore | Ordinary B32 VGPR windows on gfx942, gfx950, gfx1201, and gfx1250, with a caller-proven stack convention and scalar saves. |
| Descriptor growth | Fixed and dynamic private backing, including zero-to-nonzero growth. |
| AMDGPU metadata notes | Deliberately untouched; they are not a ROCR runtime authority. |
| SGPR save/restore | Not implemented. |
| AccVGPR save/restore | Not implemented. |
| Other GPU targets | Not implemented by this backend. |

Slot assignment for a register class does not imply a save and restore emitter
exists for that class.

## Failure contract

The backend reports failure instead of emitting a partial sequence for:

- an unsupported architecture or register class;
- an invalid or overflowing register window;
- private-size or scratch-offset overflow;
- exhaustion of the caller-provided per-lane limit;
- an invalid kernel descriptor;
- an invalid dynamic-stack register recipe.

Callers should preserve this distinction through their own diagnostics. A
backend rejection is different from successfully instrumenting a kernel and
observing no event.

## Source and tests

The implementation is in:

- `lib/rocjitsu/src/rocjitsu/code/patch/spill_manager.h`;
- `lib/rocjitsu/src/rocjitsu/code/patch/spill_manager.cpp`;
- `lib/rocjitsu/src/rocjitsu/code/patch/cdna3_instrumentation_builder.h`;
- `lib/rocjitsu/src/rocjitsu/code/patch/cdna4_instrumentation_builder.h`;
- `lib/rocjitsu/src/rocjitsu/code/patch/rdna4_instrumentation_builder.h`.

Focused tests are in `tests/patch/spill_manager_test.cpp`, with instruction
encoding coverage in the target-specific `*_instrumentation_builder_test.cpp`
files and integration coverage in `tests/patch/trampoline_builder_test.cpp`.

Run the focused host tests with:

```sh
"$ROCJITSU_BUILD_DIR/tests/rocjitsu_tests" \
  --gtest_filter='PrivateSegmentCursor.*:SpillManager.*:InstructionBuilder.*'
```

See [DBI design](dbi-design.md) for the surrounding patch pipeline. Individual
DBI clients should document their victim-selection, ownership, dispatch, and
runtime policies separately.
