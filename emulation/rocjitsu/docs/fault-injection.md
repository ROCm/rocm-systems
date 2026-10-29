# Fault injection

The fault-injection library creates deliberately modified AMDGPU code objects
for experiments with synchronization failures. It runs offline on the host:
callers supply an image, select a qualified operation, and receive a mutant
image or a typed error. It neither initializes HSA nor dispatches GPU work.

The current interface is an internal C++ library. A mutation CLI, durable
selection files, and replay bundles are planned but not implemented. ConSan is
an optional consumer of the output; the library builds and runs its host tests
with the ConSan compiler, runtime, and analyzer disabled.

## Architecture and lifetime

This flow shows the current complete-barrier Drop path and its shared DBI
transaction boundary:

```mermaid
flowchart TD
    Source[Immutable source context] --> Facts[Concrete decode and all-entry direct CFG ownership]
    Source --> Structural[Cached full-text structural facts]
    Facts --> Inventory[Exact-kernel inventory of complete barriers]
    Inventory --> Selection[Exact handle or logical ordinal]
    Selection --> Recipe[Drop recipe: ordered member edits]
    Recipe --> Transaction[Shared checked-edit transaction]
    Structural --> Transaction
    Instrumentor[Instrumentor: original-text anchor edits] --> Transaction
    Transaction --> Image[Private edited image]
    Image --> Mutant[Drop mutant]
```

[`FaultMutationContext`](../lib/rocjitsu/src/rocjitsu/code/patch/fault_injection/drop_barrier.h)
owns one immutable copy of the input. Retain it in the input/session owner until
all discovery, selection, and apply operations finish. One context can serve
many requests for that original image; each apply produces an independent
mutant. A session handling several inputs keeps a separate context for each.
To analyze or mutate an emitted image, create a new context from its bytes.

The context supplies target, section, and owner facts. It caches the fixed
all-entry ownership analysis and per-kernel qualification once. Full-text
structural facts are cached separately: references in unreachable code still
matter to byte editing. These facts are reusable for the same immutable source
and analysis profile, not arbitrary new roots or broader helper ownership.
Selections expire with their source context and cannot be replayed in another
context, even if its input bytes are identical. Their kind, owner, and barrier
facts are value snapshots exposed through const accessors; they remain readable
after expiry but cannot authorize another mutation. The in-process API does not
require callers to supply a target or content digest.

The library reuses the ISA registry and concrete decoder, `BasicBlock` and
`KernelScopeSpec`, descriptor scanning, instruction builders, and
[`CodeObjectPatcher`](../lib/rocjitsu/src/rocjitsu/code/patch/code_object_patcher.h).
Concrete identity is preserved, including the differing gfx1250/gfx1251 decoder
features. The host-only tests link the existing static-analysis composition and
model ISA providers; simulator execution support is not an admission condition.
See [ISA target providers](isa-target-providers.md) and the
[test target wiring](../tests/CMakeLists.txt).

Generic mechanics and recipe semantics have separate responsibilities:

- [`CheckedTextEditContext` and `apply_checked_text_edits`](../lib/rocjitsu/src/rocjitsu/code/patch/checked_text_edits.h)
  check source structure, complete instruction extents, expected bytes,
  references, clauses, and replacement encoding before changing a private
  image. They do not decide which barrier should be removed or prove ownership.
- [`qualify_static_barriers`](../lib/rocjitsu/src/rocjitsu/code/analysis/static_barriers.h)
  and the [Drop frontend](../lib/rocjitsu/src/rocjitsu/code/patch/fault_injection/drop_barrier.cpp)
  establish admitted logical members and choose the corresponding NOP edits.
  Qualification uses the same full-text instruction boundaries and clause facts
  as the checked-edit transaction, including facts from unreachable text.
  Recipe tests verify output semantics; emission does not run a second domain
  validator or rebuild the mutant's CFG.
- [`Instrumentor`](../lib/rocjitsu/src/rocjitsu/code/patch/instrumentor.cpp)
  also uses checked edits for original-text anchor splicing, reusing its own
  full-section instruction index. It retains responsibility for appended probe
  bodies, entry stubs, trampolines, and resource metadata. A KernelEntry callback
  that changes only a stub branch does not become an original-text anchor edit.

## Features and status

| Feature | Status and main boundary | References |
| --- | --- | --- |
| Complete barrier Drop | Implemented for the target/wave forms below. Removes one logical barrier: an ordered two-member signal/wait pair or one true full-barrier instruction. A half-pair is never a singleton. | [`StaticBarrier` / family policy](../lib/rocjitsu/src/rocjitsu/code/analysis/static_barriers.h), [qualification](../lib/rocjitsu/src/rocjitsu/code/analysis/static_barriers.cpp), [family and RDNA3 regressions](../tests/patch/fault_drop_barrier_test.cpp) |
| Exact and logical-ordinal selection | Implemented. Exact kernel name chooses one inventory; an exact context-bound handle takes precedence over the zero-based logical ordinal. Invalid exact requests fail without fallback. Default ordinal zero deterministically chooses the first barrier in first-member source order; there is no race ranking or legacy physical-index adapter. | [`DropBarrierSelector`](../lib/rocjitsu/src/rocjitsu/code/patch/fault_injection/drop_barrier.h), [`select_drop_barrier`](../lib/rocjitsu/src/rocjitsu/code/patch/fault_injection/drop_barrier.cpp), [selection regressions](../tests/patch/fault_drop_barrier_test.cpp) |
| Multiple kernels in one text section | Limited support for independent sibling kernels. Discovery checks all entries before exposing one exact kernel's inventory; shared blocks, calls, aliased entries, and unresolved owners are rejected. | [`FaultMutationContext`](../lib/rocjitsu/src/rocjitsu/code/patch/fault_injection/drop_barrier.h), [ownership analysis](../lib/rocjitsu/src/rocjitsu/code/patch/fault_injection/drop_barrier.cpp), [sibling preservation tests](../tests/patch/fault_drop_barrier_test.cpp) |
| Shared checked text edits | Implemented. One transaction handles nonoverlapping, equal-length replacements of complete 4/8/12-byte instructions. This does not widen Instrumentor's trampoline-anchor admission. | [contract](../lib/rocjitsu/src/rocjitsu/code/patch/checked_text_edits.h), [implementation](../lib/rocjitsu/src/rocjitsu/code/patch/checked_text_edits.cpp), [edit tests](../tests/patch/checked_text_edits_test.cpp), [Instrumentor tests](../tests/patch/instrumentor_test.cpp) |
| CLI, durable replay and artifact publication | Planned; no executable, persistent selection format, or cross-context replay is provided by this library. | — |
| Helper, indirect-flow and firmware-preload ownership | Planned extensions; these inputs are not admitted by the current ownership profile. | — |
| Other mutation recipes and composition | Planned. Barrier movement, address/order/scope changes, grouped or destructive partial Drop, and multi-mutation composition are not implemented here. | — |
| Multiple text sections or dynamic occurrence selection | Unsupported. | — |

A caller uses `discover_drop_barriers` with an exact kernel name to obtain an
inventory, `select_drop_barrier` to choose a complete logical barrier, and
`apply_drop_barrier` to emit it. Each operation returns `FailureOr` and a typed
`FaultMutationError`; the [API header](../lib/rocjitsu/src/rocjitsu/code/patch/fault_injection/drop_barrier.h)
defines the arguments, lifetime contract, and result fields.

## Target coverage and qualification

`fault_mutation_capabilities()` lists transformation admission from the linked
default ISA registry and family policy used by discovery and apply. The current
bindings cover seven concrete
targets and ten target/wave combinations:

| Family | Concrete target | Wave size | Complete logical barrier |
| --- | --- | --- | --- |
| RDNA3 | gfx1100 | 32 | Full `s_barrier` singleton, workgroup |
| RDNA3 | gfx1100 | 64 | Full `s_barrier` singleton, workgroup |
| RDNA4 | gfx1200 | 32 | Builtin signal/wait pair, workgroup ID -1 |
| RDNA4 | gfx1200 | 64 | Builtin signal/wait pair, workgroup ID -1 |
| RDNA4 | gfx1201 | 32 | Builtin signal/wait pair, workgroup ID -1 |
| RDNA4 | gfx1201 | 64 | Builtin signal/wait pair, workgroup ID -1 |
| CDNA5 | gfx1250 | 32 | Builtin signal/wait pair, ID -1 workgroup or -3 cluster |
| CDNA5 | gfx1251 | 32 | Builtin signal/wait pair, ID -1 workgroup or -3 cluster |
| CDNA3 | gfx942 | 64 | Full `s_barrier` singleton, workgroup |
| CDNA4 | gfx950 | 64 | Full `s_barrier` singleton, workgroup |

The [host regression matrix](../tests/patch/fault_drop_barrier_test.cpp) covers
discovery, exact/ordinal selection, emission, concrete decoding, and preservation
for these rows. RDNA3 has its own instruction encodings and descriptor-selected
wave modes; the shared singleton representation does not imply CDNA encodings
or wave64-only admission. CDNA1/CDNA2 and unregistered targets are not admitted.
A different provider composition may expose fewer bindings.

Trap-handler IDs -2 (RDNA4/CDNA5) and -4 (CDNA5) are rejected, including when
they appear before an otherwise admitted user barrier. Their operations are
inert outside privileged mode and must not consume a user barrier's logical
ordinal. See the [RDNA4 ISA, section 5.6 and Table 24](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content)
and [CDNA5 ISA, section 5.6 and Table 21](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf).
CDNA5 ID -3 is admitted as a static cluster barrier, but its operations are
inert for nonclustered dispatches and single-workgroup clusters.
Selecting it records that dispatch assumption in `DropBarrierResult`; offline
qualification does not establish that the dispatch activates cluster semantics.

These are host transformation capabilities. Neither the listing nor a successful
mutation establishes GPU execution or detector qualification.
`DropBarrierResult` leaves those evidence states `Unverified`. A removed barrier
may be redundant, unreachable in a particular run, or insufficient to cause a
race; successful execution alone also does not establish a detector finding.

## Current limits

Inputs must be little-endian AMDHSA ET_DYN ELF64 images with one executable
`.text`. ABI admission follows the [shared ELF reader](../lib/rocjitsu/src/rocjitsu/code/amdgpu_code_object.cpp)
and [DBI patcher](../lib/rocjitsu/src/rocjitsu/code/patch/code_object_patcher.cpp).
Text and descriptors require unambiguous file-backed load mappings. The ownership
profile admits closed direct CFGs with independent kernel entries. It rejects
shared helpers/calls, indirect or unresolved flow,
firmware-preload entries, and declared executable owners outside the analyzed
text. Non-aliasing unselected sections are preserved; their executable flag alone
does not imply another text section. Detailed admission and diagnostics are in
the [frontend](../lib/rocjitsu/src/rocjitsu/code/patch/fault_injection/drop_barrier.cpp)
and [checked-edit contract](../lib/rocjitsu/src/rocjitsu/code/patch/checked_text_edits.h).

A split pair must complete in the same basic block, although ordinary
instructions may separate its members. Partial, interleaved, cross-block,
named/dynamic/lifecycle and `signal_isfirst` forms, as well as barriers inside
clauses, are rejected. A direct loop can contain a qualified static barrier;
removal affects **every executed occurrence** of that position. There is no
selection of the Nth dynamic occurrence or proof of convergence.

For ConSan use, follow the [fresh-prepare integration contract](consan.md#fault-injection).
ConSan applies its own input admission during fresh prepare. Mutation admission
and detector admission are separate decisions.

## Optional compiled corpus regression

`RJ_FAULT_INJECTION_CORPUS_OBJECT` is empty by default. Without it, the
`FaultDropBarrier.CompiledCorpusCompleteDrop` test is not registered; the
synthetic host regressions remain available. Supplying an existing device ELF
enables the corpus test, but CMake does not verify its provenance or digest.
The test expects the compiled `fa_barrier_epoch` kernel and checks complete
Drop, emitted instructions, and preservation. It does not run the GPU kernel.

To reproduce the reference fixture, use `fa_barrier_epoch.hip` and `utils.hpp`
from [rocjitsu-test-corpus revision c5e3c38464d7440ef7dc977ed95419d7544afe50](https://github.com/ROCm/rocjitsu-test-corpus/tree/c5e3c38464d7440ef7dc977ed95419d7544afe50/corpus/race/kernels).
The recorded build used TheRock SDK `10.1.0a20260812`, AMD clang `23.0.0git`
(LLVM revision `a01cdbd92d1f931738a4897502747a68ae36eccf`). Prepare this
version in a per-worktree SDK environment, then run from the monorepo root.
See [building](building.md) for general build prerequisites.

```bash
source .venv-therock/bin/activate
fi_corpus_dir="${PWD}/artifacts/fi-corpus"
mkdir -p "${fi_corpus_dir}"
fi_corpus_source=https://raw.githubusercontent.com/ROCm/rocjitsu-test-corpus/c5e3c38464d7440ef7dc977ed95419d7544afe50/corpus/race/kernels
curl --fail --location "${fi_corpus_source}/fa_barrier_epoch.hip" --output "${fi_corpus_dir}/fa_barrier_epoch.hip"
curl --fail --location "${fi_corpus_source}/utils.hpp" --output "${fi_corpus_dir}/utils.hpp"
rocjitsu_rocm_root=$(rocm-sdk path --root)
"${rocjitsu_rocm_root}/bin/amdclang++" -x hip --offload-arch=gfx1250 --cuda-device-only -O2 -mcode-object-version=5 -c "${fi_corpus_dir}/fa_barrier_epoch.hip" -o "${fi_corpus_dir}/fa_barrier_epoch.bundle"
"${rocjitsu_rocm_root}/lib/llvm/bin/clang-offload-bundler" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 --input="${fi_corpus_dir}/fa_barrier_epoch.bundle" --output="${fi_corpus_dir}/fa_barrier_epoch.hsaco"
sha256sum "${fi_corpus_dir}/fa_barrier_epoch.hip" "${fi_corpus_dir}/utils.hpp" "${fi_corpus_dir}/fa_barrier_epoch.hsaco"
```

Recorded SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `fa_barrier_epoch.hip` | `86f020b8d16279ae950764b2cf377018560ab0082a6e7c9b0ce6063dd9dec5f1` |
| `utils.hpp` | `4c97b572ffaf6da56323be29e408c911c9c7c33f6c6ee454a76aa37175563966` |
| `fa_barrier_epoch.hsaco` | `d6802574e78ebe0acdd68e3c855099949b21cd74a7df361a3d2016d7ee458ef2` |

Add `-DRJ_FAULT_INJECTION_CORPUS_OBJECT="${fi_corpus_dir}/fa_barrier_epoch.hsaco"`
to this checkout's CMake configuration, preserving its compiler, sanitizer,
ROCm, and other options. For an existing build, the corresponding commands are:

```bash
cmake -S emulation/rocjitsu -B emulation/rocjitsu/build -DRJ_FAULT_INJECTION_CORPUS_OBJECT="${fi_corpus_dir}/fa_barrier_epoch.hsaco"
cmake --build emulation/rocjitsu/build --target rocjitsu_fault_injection_tests
ctest --test-dir emulation/rocjitsu/build -R '^FaultDropBarrier\.CompiledCorpusCompleteDrop$' --output-on-failure --no-tests=error
```

These digests document fixture provenance; they are not a production mutation
request requirement or evidence of physical GPU or detector qualification.
