# IR Test Case: RCCL Device API Inline vs Bitcode Paths (Bucket A, 2-GPU)

> **Audience:** RCCL developers, downstream device-API consumers, reviewers
> **Scope:** Functional + linkage validation of the bucket A device API
> (`ncclGetPeerPointer` / `ncclGetPeerPointerTeam`) across the two
> packaging paths RCCL exposes.
> **Location:** `bindings/ir/test/bucketA2/` in the RCCL tree.
> **Status:** Implemented and passing on gfx942 (MI300X), `ROCm 6.x` toolchain.

---

## 1. Background and scope

A *device API* in this document is a function that runs **inside a GPU
kernel** — e.g. `ncclGetPeerPointer`, `ncclCoopAny::size()`. Consumers
write their own `__global__` kernels and invoke those functions. RCCL
ships two parallel delivery modes for the same APIs:

| Path | Header surface | Body lives in | Resolved at |
|---|---|---|---|
| **Inline** | `<nccl_device.h>` — full `__device__` source | every consumer TU that includes it | clang **frontend** of the consumer (inlined) |
| **Bitcode** | `<nccl_device_wrapper.h>` — `extern "C" __device__` decls only | `librccl_device.bc` (AOT LLVM IR module) | AMDGPU **device LTO linker** (lld) of the consumer |

The two paths must produce identical user-visible behavior. The
bucketA2 test is the smallest possible end-to-end demonstration of
that, exercising a real cross-GPU peer write through bucket A's
pointer-arithmetic API.

### Why bucket A is the right target for the first 2-GPU smoke

`ncclGetPeerPointer` does pure arithmetic in a *symmetric* virtual
address space:

```
i      = lsaRank + (peer - tm.rank) * tm.stride
result = add4G(lsaFlatBase, i * stride4G) + offset
```

That math is only meaningful if rank N's memory lives at
`base + N * stride4G * 4 GiB`. In production, RCCL's symmetric heap
allocator builds exactly that layout (xGMI / IPC / multi-cast
plumbing). For a standalone two-GPU smoke we fake the layout with a
single `hipMallocManaged` region big enough to span two 4-GiB-spaced
slots — a shared virtual address space where the computed pointer is
genuinely dereferenceable from both GPUs.

This is the **minimum** test that exercises both halves of bucket A:
the arithmetic *and* the dereference contract on top of it.

---

## 2. Test objectives

1. **Functional equivalence** — inline and bitcode paths produce identical
   results (same value written, same value read back) on two real GPUs.
2. **Linkage correctness** —
   - Inline path: no `ncclGetPeerPointer` symbol survives in the final
     device ELF; the body is fully inlined into the calling kernel.
   - Bitcode path: `ncclGetPeerPointerTeam` is resolved from
     `librccl_device.bc` at offload-LTO link time. Whether the call
     survives or gets inlined again is a link-flag decision, not an
     ABI break.
3. **ABI stability** — the `extern "C" __device__` thunk signatures in
   `nccl_device_wrapper.h` are the only contract a downstream consumer
   sees. C++ name-mangling of `ncclGetPeerPointer` is *not* part of the
   bitcode ABI.
4. **Cross-GPU dereferenceability** — the address returned by
   `ncclGetPeerPointer(peer=N)` from rank M, when M ≠ N, is a real
   address that GPU M can write to and GPU N can read from.

---

## 3. Test matrix

| # | Dimension | Inline path | Bitcode path |
|---|---|---|---|
| 1 | Header surface | Full `__device__` source in `<nccl_device.h>` | Only `extern "C" __device__` decls in `<nccl_device_wrapper.h>` |
| 2 | Where body lives | In each consumer TU | In `librccl_device.bc` (AOT LLVM IR) |
| 3 | Resolution stage | Consumer frontend (clang) inlines into kernel | Device LTO linker (lld) merges bitcode, may inline or call |
| 4 | ELF symbol | None (inlined away) | Thunk may persist or be inlined depending on flags |
| 5 | ABI surface | C++ inline (rebuild on change) | `extern "C"` thunks (ABI-stable) |
| 6 | Codegen quality | Max inlining + const-prop | Near-parity via LTO; depends on flags |
| 7 | Number of GPUs touched | 2 (write + read on different devices) | 2 (write + read on different devices) |

---

## 4. Test inputs and kernels

Both tests build the same symmetric layout on top of one
`hipMallocManaged` allocation and use single-threaded kernels for
determinism. They differ in **which** bucket A entry point they call.

### 4.1 Symmetric layout setup (host, identical in both tests)

```cpp
constexpr uint64_t SLOT_STRIDE = (uint64_t)1 << 32;          // 4 GiB
constexpr size_t   ALLOC_SIZE  = 2 * SLOT_STRIDE + 4096;     // 8 GiB + pad

char* raw = nullptr;
hipMallocManaged((void**)&raw, ALLOC_SIZE, hipMemAttachGlobal);
uintptr_t aligned = ((uintptr_t)raw + SLOT_STRIDE - 1) & ~(SLOT_STRIDE - 1);
char* base = (char*)aligned;
//  slot 0 = base + 0
//  slot 1 = base + 4 GiB
```

Only the two touched cachelines back to physical memory; the rest of
the 8 GiB stays as unbacked virtual range.

### 4.2 Inline-path kernels (`peer2_inline_test.cpp`)

`ncclGetPeerPointer` comes from `<nccl_device.h>` (full body inlined
into the kernel at frontend time):

```cpp
__global__ void k_write(char* base, uint32_t stride4G, uint64_t* wrote_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ncclWindow_vidmem w{};
    w.lsaFlatBase = base;  w.lsaRank = 0;  w.worldRank = 0;
    w.stride4G    = stride4G;
    ncclTeam tm{ /*nRanks=*/2, /*rank=*/0, /*stride=*/1 };

    uint64_t* peer = (uint64_t*)ncclGetPeerPointer(&w, /*offset=*/0, tm,
                                                    /*peer=*/1);
    *peer      = MAGIC;     // peer-1's slot
    *wrote_out = MAGIC;     // echo for host-side print
  }
}

__global__ void k_read(char* base, uint32_t stride4G, uint64_t* read_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ncclWindow_vidmem w{};
    w.lsaFlatBase = base;  w.lsaRank = 1;  w.worldRank = 1;
    w.stride4G    = stride4G;
    ncclTeam tm{ 2, 1, 1 };

    uint64_t* mine = (uint64_t*)ncclGetPeerPointer(&w, 0, tm, /*peer=*/1);
    *read_out = *mine;      // my own slot, also the target of the write above
  }
}
```

### 4.3 Bitcode-path wrapper and kernels (`peer2_bitcode_test.cpp`)

The producer-side thunk in `bindings/ir/nccl_device_wrapper__impl.h` republishes
the inline function under a C-ABI name:

```cpp
NCCL_IR_EXPORT                              // extern "C" __device__ used,retain,visibility(default)
void* ncclGetPeerPointerTeam(ncclWindow_t w, size_t offset,
                             ncclTeam tm, int peer) {
  return ncclGetPeerPointer(w, offset, tm, peer);
}
```

The consumer's kernels are mechanical translations of the inline
versions — same logic, the only delta is the function name:

```cpp
__global__ void k_write(char* base, uint32_t stride4G, uint64_t* wrote_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ncclWindow_vidmem w{};
    w.lsaFlatBase = base; w.lsaRank = 0; w.worldRank = 0;
    w.stride4G    = stride4G;
    ncclTeam tm{2, 0, 1};
    uint64_t* peer = (uint64_t*)ncclGetPeerPointerTeam(&w, 0, tm, 1);
    *peer      = MAGIC;
    *wrote_out = MAGIC;
  }
}

__global__ void k_read(char* base, uint32_t stride4G, uint64_t* read_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ncclWindow_vidmem w{};
    w.lsaFlatBase = base; w.lsaRank = 1; w.worldRank = 1;
    w.stride4G    = stride4G;
    ncclTeam tm{2, 1, 1};
    uint64_t* mine = (uint64_t*)ncclGetPeerPointerTeam(&w, 0, tm, 1);
    *read_out = *mine;
  }
}
```

### 4.4 End-to-end flow (both tests)

```text
  host                GPU 0 (rank 0)                  GPU 1 (rank 1)
  ----                --------------                  --------------
  allocate 8 GiB managed
  align base
                      k_write<<<1,1>>>:
                        ptr = ncclGetPeerPointer*(
                                w, 0, {2,0,1}, peer=1)
                        *ptr      = MAGIC
                        *wrote_out= MAGIC
                      hipDeviceSynchronize()
                                                      k_read<<<1,1>>>:
                                                        ptr = ncclGetPeerPointer*(
                                                                w, 0, {2,1,1}, peer=1)
                                                        *read_out = *ptr
                                                      hipDeviceSynchronize()
  print + assert(*wrote_out == *read_out == MAGIC)
```

For GPU 0:  `i = 0 + (1 - 0) * 1 = 1` → address `base + 4 GiB` (slot 1).
For GPU 1:  `i = 1 + (1 - 1) * 1 = 1` → address `base + 4 GiB` (same slot).
Both arrive at the same address through the bucket A math, validating
the symmetric-layout contract from both ranks' perspectives.

---

## 5. Build and link procedures

> The runner scripts `run_peer2_inline.sh` and `run_peer2_bitcode.sh`
> encapsulate everything below. The recipes here are what those
> scripts run, with the actual flags used in this tree.

### 5.1 Inline path (consumer-only)

```bash
hipcc --offload-arch=gfx942 -O2 \
  -D__HIP_PLATFORM_AMD__=1 \
  -I$BUILD/hipify/src/include \
  -I$BUILD/hipify/src/include/nccl_device \
  -I$BUILD/include \
  bindings/ir/test/bucketA2/peer2_inline_test.cpp \
  -o peer2_inline.exe
```

Notes:

- `__HIP_PLATFORM_AMD__=1` must have a **value**; `utility.h` does
  `#if __HIP_PLATFORM_AMD__` and treats an empty macro as a parse
  error.
- We point at the *hipified* copy of `src/include/` first so the
  consumer's view of the headers matches the bitcode build's view
  byte-for-byte.

### 5.2 Bitcode path

**Producer side** (run once when RCCL is built; CMake target
`llvm_ir`):

1. Compile `bindings/ir/nccl_device_wrapper__impl.h` to AMDGPU LLVM IR:

   ```bash
   clang++ -x hip --offload-device-only --offload-arch=gfx942 \
     -D__clang_llvm_bitcode_lib__ -D__HIP_PLATFORM_AMD__=1 \
     -emit-llvm -O1 -c bindings/ir/nccl_device_wrapper__impl.h \
     -o librccl_device.bc.unoptimized
   ```

2. Internalize all symbols except the exported thunks. We pass an
   **explicit symbol list**, not a glob — `opt`'s
   `-internalize-public-api-list` does not support globbing:

   ```bash
   opt -passes=internalize \
       -internalize-public-api-list=ncclGetPeerPointerTeam,ncclCoopAnyInitThread,…  \
       -O1 librccl_device.bc.unoptimized -o librccl_device.bc
   ```

   The resulting `librccl_device.bc` is the shipped artifact (27 KiB in
   the current tree).

**Consumer side** (linked at AMDGPU offload-LTO):

```bash
hipcc --offload-arch=gfx942 -O2 \
  -D__HIP_PLATFORM_AMD__=1 \
  -I$BUILD/hipify/src/include \
  -I$BUILD/hipify/src/include/nccl_device \
  -I$BUILD/include \
  bindings/ir/test/bucketA2/peer2_bitcode_test.cpp \
  -Xoffload-linker $BUILD/lib/librccl_device.bc \
  -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false \
  -o peer2_bitcode.exe
```

Flag rationale:

- `-Xoffload-linker <bc>` — feeds the bitcode module to lld so its
  symbols are in scope for the device-side LTO link.
- `-plugin-opt=-amdgpu-internalize-symbols=false` — AMDGPU LTO
  normally re-internalizes everything unreferenced from the kernel,
  which would drop the thunks even when the kernel calls them, because
  by that point the call has already been resolved. Disabling
  internalization keeps the thunk visible **even when the kernel call
  site is itself inlined**. Without this flag, downstream consumers
  that load the bitcode by other means lose access to the API.

Flipping this flag changes whether the thunk survives as a call or
gets inlined into the kernel; either outcome is functionally
equivalent and the test is designed to pass under both.

---

## 6. Assertions and expected outcomes

### 6.1 Functional correctness

- `*wrote_out` (GPU 0's echo of what it wrote) == `MAGIC` (`0xDEADBEEFCAFEBABE`).
- `*read_out` (GPU 1's readback of slot 1) == `MAGIC`.
- The two values are equal, proving GPU 0's write through
  `ncclGetPeerPointer*` actually landed in GPU 1's slot and was
  observable from GPU 1.

### 6.2 Symbol-level expectations

| Build | Expected on the device side |
|---|---|
| `peer2_inline.exe` | No `ncclGetPeerPointer` symbol in the device ELF. Pointer arithmetic is fused into `k_write` / `k_read`. Verify with `llvm-objdump --offloading | llvm-objdump -d` and check for absence of an external call. |
| `peer2_bitcode.exe` | `ncclGetPeerPointerTeam` is resolved from `librccl_device.bc`. With `-amdgpu-internalize-symbols=false`, the thunk symbol is retained; with the default flag, the body is inlined and the thunk symbol may be dropped. Either is acceptable. |

### 6.3 Optimization characteristics

- Constants from the caller propagate through `nccl::utility::loadConst`
  and `nccl::utility::add4G`.
- Dead branches inside `ncclGetPeerPointer`'s template specializations
  fold when `tm.stride`, `stride4G`, etc. are compile-time constants.
- Final computed addresses **must** agree between the inline and
  bitcode paths for the same inputs. The bucketA2 test exercises this
  with concrete values (`stride4G=1`, `peer=1`, etc.).

---

## 7. Test procedure

### 7.1 Prerequisites

- ROCm toolchain with `hipcc`, `clang`, `lld`, `opt`, `llvm-objdump`,
  `llvm-dis`, `llvm-nm`.
- Target architecture, e.g. `gfx942` (MI300X).
- RCCL build tree with hipified headers and (for bitcode) the
  `llvm_ir` CMake target built:

  ```bash
  cmake -DEMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=gfx942 ...
  cmake --build . --target llvm_ir
  ```

- **Two visible GPUs** (the test asserts `hipGetDeviceCount() >= 2`).

### 7.2 Run the inline path

```bash
GPUS=0,1 ARCH=gfx942 bindings/ir/test/bucketA2/run_peer2_inline.sh
```

Expected output:

```text
[peer2-inline] arch=gfx942  GPUS=0,1
[peer2-inline] built: /tmp/peer2_inline/peer2_inline.exe
[peer2-inline] running...
[peer2-inline] devices=2
[peer2-inline] managed alloc=0x7fa3fd49b000 aligned base=0x7fa400000000 \
               (slot0=0x7fa400000000, slot1=0x7fa500000000)
GPU0 wrote: 0xdeadbeefcafebabe
GPU1 read:  0xdeadbeefcafebabe
[peer2-inline] [OK]
```

Note `slot1 - slot0 == 0x100000000` (= 4 GiB) — the symmetric layout
assumption is satisfied.

### 7.3 Run the bitcode path

```bash
GPUS=0,1 ARCH=gfx942 bindings/ir/test/bucketA2/run_peer2_bitcode.sh
```

Expected output:

```text
[peer2-bitcode] arch=gfx942  GPUS=0,1  bc=…/librccl_device.bc (27332 bytes)
[peer2-bitcode] built: /tmp/peer2_bitcode/peer2_bitcode.exe
[peer2-bitcode] running...
[peer2-bitcode] devices=2
[peer2-bitcode] managed alloc=0x7effc549b000 aligned base=0x7f0000000000 \
               (slot0=0x7f0000000000, slot1=0x7f0100000000)
GPU0 wrote: 0xdeadbeefcafebabe
GPU1 read:  0xdeadbeefcafebabe
[peer2-bitcode] [OK]
```

### 7.4 Inspection commands (optional)

```bash
# Bitcode artifact as text
llvm-dis $BUILD/lib/librccl_device.bc -o /tmp/librccl_device.ll
grep -n 'define.*ncclGetPeerPointerTeam' /tmp/librccl_device.ll

# Symbols actually exported by the bitcode
llvm-nm --defined-only $BUILD/lib/librccl_device.bc

# Disassemble device ELF
llvm-objdump --offloading peer2_bitcode.exe
llvm-objdump -d peer2_bitcode.exe.0.hipv4-amdgcn-amd-amdhsa--gfx942
```

---

## 8. Pass / fail criteria

### Pass

- Both binaries print `[OK]` and exit 0.
- `*wrote_out == *read_out == 0xDEADBEEFCAFEBABE`.
- `slot1 - slot0 == 4 GiB` (symmetric layout invariant).
- Inline binary's device ELF contains no `ncclGetPeerPointer*` symbol.
- Bitcode binary's device ELF resolves `ncclGetPeerPointerTeam` from
  `librccl_device.bc` (or has it inlined, depending on the
  `-amdgpu-internalize-symbols` flag).

### Fail

- Mismatched values between `*wrote_out` and `*read_out` (e.g. one is
  zero) — points at a broken peer-pointer arithmetic, a missing
  cross-GPU synchronization, or a misaligned slot.
- `hipDeviceCount() < 2` — environment issue, not a test failure
  (test exits with code 2 and a clear message).
- Unresolved `ncclGetPeerPointerTeam` at device link time — the
  bitcode wasn't fed to the offload linker, or `-amdgpu-internalize`
  killed the symbol upstream in the bitcode build.
- Symbol retention contradicts the link flags — investigate
  `NCCL_IR_EXPORT` (must include `used`, `retain`,
  `visibility("default")`).

---

## 9. Design rationale

- **Inline path** maximizes compile-time visibility for in-tree
  builds (RCCL itself, RCCL's tests). The compiler sees the full body,
  inlines aggressively, and propagates constants through the
  pointer-arithmetic helpers (`loadConst`, `add4G`).

- **Bitcode path** lets out-of-tree consumers (apps, DSLs, JITs) call
  RCCL device APIs **without** depending on RCCL's source headers.
  They only need:
  1. `<nccl_device_wrapper.h>` (a small, ABI-stable declarations file)
  2. `librccl_device.bc` (the AOT-compiled IR module)

  This mirrors the ROCm device-libs distribution model while
  preserving a C ABI through `extern "C"` thunks. The bitcode is
  retargetable to any AMDGPU architecture the consumer is building
  for, because LLVM IR is lowered to ISA at the consumer's link step.

- **The 2-GPU smoke** is deliberately the smallest possible meaningful
  test: one write, one read, on two different GPUs, with the symmetric
  layout faked by `hipMallocManaged`. Larger NCCL integration tests
  rely on the symmetric heap to set up the layout; this test bypasses
  that machinery so a bucket A regression can be caught without a
  full NCCL bring-up.

---

## 10. Files in this test

| File | Purpose |
|---|---|
| `peer2_inline_test.cpp` | Inline-path consumer; calls `ncclGetPeerPointer` from `<nccl_device.h>`. |
| `peer2_bitcode_test.cpp` | Bitcode-path consumer; calls `ncclGetPeerPointerTeam` from `<nccl_device_wrapper.h>`, body resolved from `librccl_device.bc`. |
| `run_peer2_inline.sh` | Build + run the inline test. Env: `ARCH`, `ROCM_PATH`, `GPUS`, `OUTDIR`. |
| `run_peer2_bitcode.sh` | Build + run the bitcode test. Same env plus `BC` (path to `librccl_device.bc`). |
| `CONFLUENCE.md` | This document. |

## 11. Related tests in the same tree

| Path | Scope |
|---|---|
| `bindings/ir/test/bucketA/` | Single-GPU arithmetic test for `ncclGetPeerPointer` across many synthetic windows; verifies the pointer math but not cross-GPU dereferenceability. |
| `bindings/ir/test/bucketB/` | Cooperative-thread API (`ncclCoopAny`) inline/bitcode equivalence. |
| `bindings/ir/test/bucketA+B/` | Combined bucket A and bucket B in a single kernel. |
| `bindings/ir/test/smoke.cpp` | Original single-GPU smoke that established the bitcode link recipe. |
