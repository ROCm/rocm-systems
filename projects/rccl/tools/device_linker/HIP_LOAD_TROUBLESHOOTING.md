# Why HIP might not load merged_device.elf

When you see **HSA_STATUS_ERROR_INVALID_ISA** or "invalid kernel file" at runtime, the code object we embed is being rejected by the ROCm loader (COMGR/HSA). Below are the main things that can cause that and how to check them.

**Tested:** Omitting DWARF was tried (rebuild device linker, then `RCCL_DEVICE_LINKER_OMIT_DWARF=1 ./install.sh -l --device-linker`). Debug sections are confirmed absent from `merged_device.elf`. The smoke test **still fails with HSA_STATUS_ERROR_INVALID_ISA**, so the load failure is **not** caused by DWARF; the cause is elsewhere. Rebuild the device linker binary (in `tools/device_linker/`, with LLVM) for `--omit-dwarf` to take effect. See the “Debugging HIP load / COMGR” section for env vars and the “omit-dwarf” workaround.

## What we know so far

- **Table sizes**: 859 entries (6872 bytes), matching `merged_device_funcid_names.h`; 843 `.rela.dyn` entries (16 slots are `nullptr` from arch guards). OK.
- **ELF header**: `e_flags` = 0x54c → gfx942, xnack, sramecc. OK for MI300. A minimal `hipcc` device object has the same: **Flags: 0x54c, gfx942, xnack, sramecc** — so header flags match.

## Things to verify

### 1. ELF type: ET_DYN vs ET_EXEC

The device linker emits **ET_DYN** (position‑independent executable style). Some loaders only accept **ET_EXEC** for GPU code objects.

**Check:**

```bash
# Our merged code object
readelf -h merged_device.elf
# Compare to a code object produced by normal HIP compile:
hipcc -c -x hip --offload-arch=gfx942 /dev/null -o /tmp/empty.hip.o 2>/dev/null || true
readelf -h /tmp/empty.hip.o   # or extract the amdgcn object from it and readelf -h that
```

If the working code object is ET_EXEC and ours is ET_DYN, the loader may be rejecting based on type.

### 2. Bundler target vs runtime expectation (features)

A leading hypothesis is that **target features** (e.g. xnack/sramecc spelling or order in the triple or note) don’t match what the runtime expects. Building a **minimal HIP kernel** with normal `hipcc` and comparing its code object (e_flags, .note, feature string) to `merged_device.elf` is a good way to check. A normal `hipcc` device bundle is named **`hip_kernel.0.hipv4-amdgcn-amd-amdhsa--gfx942`** (no `:sramecc`/`:xnack` in the bundle name; features may be in the ELF note instead).

We pass to `clang-offload-bundler`:

- `-targets=host-x86_64-...,hipv4-amdgcn-amd-amdhsa--${GPU_TARGET}` or with `:${GPU_TARGET_FEATURES}`.

The runtime picks a code object by matching the current GPU’s triple/ISA. If the **triple or feature string** in the bundle doesn’t match what the driver reports (e.g. different order or spelling of xnack/sramecc), the loader might not select our blob or might reject it.

**Check:** Inspect what’s inside the fatbin (e.g. with `extract_fatbin` or by unbundling) and confirm the embedded device blob’s ELF `e_flags` and any triple in the bundle match the GPU you’re running on.

### 3. .note section (ISA metadata)

The merged ELF’s **.note** is taken from the **dispatcher** only and then **patched** (e.g. `private_segment_fixed_size`, VGPR/SGPR). COMGR uses note metadata (including ISA) to validate and load the code object. If the note is corrupted or the layout/format doesn’t match what COMGR expects, you can get INVALID_ISA or “invalid kernel file”.

**Check:**

```bash
# Dump note contents
readelf -n merged_device.elf
# Compare to a known-good amdgcn code object from a normal hipcc build
```

Also compare note size and alignment with a working code object. In a minimal `hipcc` device object, the note arch is **`amdhsa.target: amdgcn-amd-amdhsa--gfx942`** (no `:sramecc`/`:xnack` in the target string).

### 3b. Kernel descriptors (KDs)

There are **18 KDs** in `.rodata` (copied from the dispatcher):

- **3** – Generic kernels (one per unroll factor): `ncclDevKernel_Generic_1`, `_2`, `_4`. These are the main dispatch kernels.
- **3** – “Debug” (profiling) kernels: `ncclDevKernelDebug_Generic_1`, `_2`, `_4`. They enable profiling/collTrace when `ENABLE_COLLTRACE` is used; the host may launch them when profiling is enabled.
- **12** – One-rank (oneRankReduce PreMulSum) kernels. Special kernels used for single-rank reduce.

We **patch the first 6** (Generic at 0, 64, 128 and Debug/profiling at 192, 256, 320): LDS, stack, VGPR/SGPR, and `kernel_code_entry_byte_offset` for the new layout. The **12 oneRank** KDs are left as in the dispatcher (oneRank kernels are separately compiled and live in dispatcher .text; their layout relative to .rodata is unchanged).

**Generic vs Debug (profiling):** In `src/device/common.cu`, Generic kernels call `ncclKernelMain<..., false, Unroll>` and Debug kernels call `ncclKernelMain<..., true, Unroll>` (template parameter `COLLTRACE`). Trace calls in `common.h` are guarded by `if (COLLTRACE && ...)`, so only the Debug instantiations should run the profiling code. If the six kernels look the same in the binary (e.g. profiling code present in all), possible causes: the compiler may not be fully eliminating the trace path when `COLLTRACE` is false, or the trace expansion is shared and the only difference is the runtime branch. To have truly separate no-profiling vs profiling behavior we would need **two dispatchers** (e.g. one built without `ENABLE_COLLTRACE` with only the 3 Generic kernels, and one with `ENABLE_COLLTRACE` with all 6); today we use a single dispatcher that contains both.

Things that can go wrong:

- **KD ↔ note consistency**: The loader may expect the **order and count** of KDs in `.rodata` to match the kernel list in the note (or PAL metadata). If the note lists kernels in a different order, or the loader assumes one KD per “kernel” symbol in `.dynsym`, a mismatch could cause rejection.
- **kernel_code_entry_byte_offset** (KD offset 0x10): Must be the **signed offset from the KD’s load address to the kernel’s first instruction**. We patch it for the three dispatcher KDs using the layout delta. If the loader uses a different base (e.g. segment vs section) or expects a different ABI, the value could be wrong.
- **KD layout / code object version**: The 64-byte KD layout (e.g. 0x00 LDS/stack, 0x10 entry offset, 0x2c reserved, 0x30/0x34 RSRC, 0x3c props) may differ by code object version. If the runtime expects a newer layout, our patching might leave required fields wrong or in the wrong place.
- **.rodata writable**: We set `.rodata` to `SHF_WRITE` so relocations can fill the function tables. Some loaders may require read-only `.rodata` or may validate segment flags and reject writable “rodata”.

**Check:** Compare the first 192 bytes (3 × 64) of `merged_device.elf`’s `.rodata` with a normal `hipcc` code object that has 3 kernels: KD layout, entry offsets, and segment/section flags.

### 3c. .kd symbol visibility

The **.kd symbols** in `.symtab` (and `.dynsym` if present) point to the 64-byte kernel descriptors in `.rodata`. Observed in our merged object: all .kd symbols are **PROTECTED** (STV_PROTECTED), e.g.:

```
000000000000c780 g     O .rodata	0000000000000040 .protected _Z23ncclDevKernel_Generic_224ncclDevKernelArgsStorageILm5120EE.kd
```

(g = STB_GLOBAL, O = STT_OBJECT, size 0x40 = 64 bytes; readelf shows visibility as `.protected`.) The **12 oneRank** .kd symbols look like this (no `.protected` in readelf output, so likely STV_DEFAULT):

```
000000000000c8c0 g     O .rodata	0000000000000040 _ZN12_GLOBAL__N_113oneRankReduceI13FuncPreMulSumIaEEEvPvS3_mmb.kd
```

So we have a mix: the 6 Generic/Debug .kd show PROTECTED; the 12 oneRank .kd show default visibility. The device linker only forces **ncclDevFuncTable_*** to STB_LOCAL STV_HIDDEN; it does **not** change binding/visibility for .kd symbols (patchSymbol only updates st_value and st_shndx), so .kd symbols keep whatever the compiler emitted. IFC uses HIDDEN for similar symbols. **Unknown whether PROTECTED vs HIDDEN matters** for the loader; if INVALID_ISA persists after other checks, trying STB_LOCAL STV_HIDDEN for all .kd in the device linker is an option.

**Inspecting .kd with llvm-objdump:** Use **`-D`** (disassemble all) together with **`--disassemble-symbols=<exact_name>`** and the full symbol name. Example:

```bash
llvm-objdump -D --disassemble-symbols=_Z23ncclDevKernel_Generic_224ncclDevKernelArgsStorageILm5120EE.kd merged_device.elf
```

Without `-D`, objdump may report "missing" because .kd live in .rodata. Fallback: `llvm-objdump -s -j .rodata merged_device.elf` or `readelf -s` + `xxd -s <offset> -l 64 merged_device.elf`.

**Full resource audit:** See **`RESOURCE_AUDIT.md`** for a step-by-step checklist: (1) input values from each object (including that the **dispatcher** is not in the max reduction), (2) max reduction correctness, (3) consistency of .KDs, ABS symbols, and .note, (4) bit-level encoding (RSRC1 truncation, msgpack, etc.). Fixes should be evaluated against that audit.

**KD next_free_vgpr / RSRC1 may be wrong:** We patch RSRC1 (offset 0x30) with granulated values: `vgpr_g = (max_vgpr_ + 3) / 4 - 1`, `sgpr_g = (max_sgpr_ + 7) / 8 - 1`, then `rsrc1 = (rsrc1 & ~0x3FF) | (vgpr_g & 0x3F) | ((sgpr_g & 0xF) << 6)`. The VGPR field is only **6 bits** (0x3F), so the maximum representable is 63 granulated → 256 VGPRs. If **max_vgpr_** from the note (e.g. vgpr_count 280) is &gt; 255, we **truncate** (e.g. 69 & 0x3F = 5) and under-allocate. The note’s `.vgpr_count` might also use a different convention than the KD’s “next free” encoding. So the KD’s next_free_vgpr (and our patched value) may be incorrect.

**Example: disassembled Generic_2 .kd** (reference for layout; we patch group/private segment and resource fields):

```
.amdhsa_kernel _Z23ncclDevKernel_Generic_224ncclDevKernelArgsStorageILm5120EE
	.amdhsa_group_segment_fixed_size 32832
	.amdhsa_private_segment_fixed_size 360
	.amdhsa_kernarg_size 5376
	.amdhsa_accum_offset 128
	.amdhsa_tg_split 0
	.amdhsa_next_free_vgpr 48
	.amdhsa_reserve_vcc 0
	.amdhsa_reserve_xnack_mask 0
	.amdhsa_next_free_sgpr 112
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_fp16_overflow 0
	.amdhsa_enable_private_segment 1
	.amdhsa_system_sgpr_workgroup_id_x 1
	.amdhsa_system_sgpr_workgroup_id_y 1
	.amdhsa_system_sgpr_workgroup_id_z 1
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 2
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
	.amdhsa_user_sgpr_dispatch_ptr 1
	.amdhsa_user_sgpr_queue_ptr 1
	.amdhsa_user_sgpr_kernarg_segment_ptr 1
	.amdhsa_user_sgpr_dispatch_id 1
	.amdhsa_user_sgpr_private_segment_size 0
	.amdhsa_uses_dynamic_stack 1
```

**Example: note metadata for the same kernel** (YAML-style kernel metadata in the .note; loader/COMGR use this together with the .kd). Compare `group_segment_fixed_size` / `private_segment_fixed_size` here with the .kd and with what we patch.

```
.name:           _Z23ncclDevKernel_Generic_224ncclDevKernelArgsStorageILm5120EE
.private_segment_fixed_size: 360
.sgpr_count:     106
.sgpr_spill_count: 0
.symbol:         _Z23ncclDevKernel_Generic_224ncclDevKernelArgsStorageILm5120EE.kd
.uniform_work_group_size: 1
.uses_dynamic_stack: true
.vgpr_count:     280
.vgpr_spill_count: 0
.wavefront_size: 64
.agpr_count:     0
.args: [ ... offsets 0-5320, value_kind by_value / hidden_* ... ]
.group_segment_fixed_size: 4976
.kernarg_segment_align: 16
.kernarg_segment_size: 5376
.language:       OpenCL C
.max_flat_workgroup_size: 256
```

(Full `.args` list: by_value 5120, hidden_block_count_x/y/z, hidden_group_size_x/y/z, hidden_remainder_x/y/z, hidden_global_offset_x/y/z, hidden_grid_dims, hidden_hostcall_buffer, hidden_multigrid_sync_arg, hidden_heap_v1, hidden_default_queue, hidden_completion_action, hidden_dynamic_lds_size, hidden_queue_ptr.)

**Do they match?** Partially. Same: **private_segment_fixed_size 360**, **kernarg 5376**. **Mismatch: group_segment_fixed_size** — .kd has **32832**, note has **4976**. We patch the KD (LDS/stack/resources) but the note may still contain the dispatcher’s original group_segment value; if the loader compares or allocates using the note, that inconsistency could cause INVALID_ISA. SGPR/VGPR: .kd has next_free_sgpr 112, next_free_vgpr 48; note has sgpr_count 106, vgpr_count 280 (different conventions; compare with care).

**Where did the patch value (32832) in the .kd come from?** In the device linker we do not copy the dispatcher note value into the KD. We set **max_lds_** = max of (1) every kernel’s `.group_segment_fixed_size` from their **.note** (dispatcher + specialized), and (2) **requiredLDS** from **`calculateRequiredLDS()`** — an RCCL formula (per-warp scratch × num warps; for gfx942 that yields 32832). The value written into the KD at offset 0 is that **max_lds_**. The .note for Generic_2 still has 4976 because we only patch the **KD** and (for some fields) the note’s **private_segment_fixed_size**; we do **not** patch the note’s **group_segment_fixed_size** to match.

**What is updated in the .note?** In **`patchNote()`** we update: **(1) private_segment_fixed_size** → **max_stack_** (scratch for specialized callees); **(2) .vgpr_count** → **max_vgpr_**; **(3) .sgpr_count** → **max_sgpr_**. We also fix the note header **descsz** and padding after edits. We do **not** change **uses_dynamic_stack** (kept as-is to match IFC) and we do **not** patch **group_segment_fixed_size** in the note — hence the KD (32832) vs note (4976) mismatch. That omission has been there since note patching was added; the note has never had **group_segment_fixed_size** updated to match the patched KD. (Single-GPU tests that were reported as “passing” never actually load a kernel — with one GPU everything is a reduction that degenerates to memcpy, so no kernel is launched; they do not validate code object load. The one exception is the 12 oneRank kernels (e.g. oneRankReduce PreMulSum, “post multiply” style), which can run on single-GPU; those would trigger a real load. The first real load in typical tests is when a kernel is launched, e.g. multi-GPU or collectives, and that is when INVALID_ISA appears.)

### 4. Code object / metadata version

ROCm and COMGR expect a certain **code object version** and **metadata (e.g. PAL) version**. If the dispatcher was built with a different LLVM/ROCm than the system runtime, the note format or version might be incompatible.

**Check:** Build the dispatcher with the same ROCm/LLVM you run with; compare `readelf -n` and, if available, any COMGR/loader logs that mention version or metadata.

### 5. How the fatbin is embedded

We use **-fcuda-include-gpubinary** with the bundler output. The resulting **.hip_fatbin** section must be in a format the ROCm runtime understands (e.g. CLANG_OFFLOAD_BUNDLE with the expected triples and blob layout). If the host object is built with a different clang than the system’s HIP runtime expects, the embedding format might differ and the loader might not find or accept the code object.

**Check:** Run `extract_fatbin` on `librccl.so`, confirm the bundle format and that the extracted device ELF is exactly `merged_device.elf` (or equivalent). Confirm with the same ROCm version that a trivial `hipcc`-built library loads correctly.

---

## Debugging HIP load / COMGR

Use these environment variables to see what the runtime and COMGR are doing. They can help track down truncated code objects, load failures, and KD/kernel enumeration.

**Observed so far:** With `GPU_DUMP_CODE_OBJECT=1`, two code objects were dumped; both match the one we produce (no truncation). With `AMD_COMGR_EMIT_VERBOSE_LOGS`, COMGR logs show building from bitcode with **-mcpu=gfx942:sramecc+:xnack-** (and related target flags). `AMD_LOG_LEVEL` / `AMD_LOG_MASK` did not show anything useful for this failure. **Next step when stuck:** Inspect the HSA (and/or COMGR) code to see where **HSA_STATUS_ERROR_INVALID_ISA** is returned and what is actually validated (e.g. note vs KD, ISA flags, metadata layout). **Note:** In HSA source, the only use of INVALID_ISA found was in “agent” code (GPU has invalid ISA), not in the code-object load path — so the failure may be a different status (e.g. INVALID_ISA_NAME) or may come from COMGR/HIP rather than core HSA. Confirm the **exact** error code/string from the runtime (what the smoke test or logs actually print) and search for that. Stepping through in a debugger would help, but HIP/COMGR runtime libraries are typically **stripped**, so you may need debug builds or symbol packages to get useful stack traces. **COMGR** is a good place to look for the code-object load/validation path that produces this error; the **HIP** runtime is another possibility. HIP appears in the mono-repo but may be marked retired / a pointer; the current HIP source might live in internal GitHub.

### Code object dump and COMGR

| Variable | Effect |
|----------|--------|
| **`GPU_DUMP_CODE_OBJECT=1`** | Dump code objects (often creates `.co` files). If those files are **truncated**, the failure may be in the dump path or a size limit when writing. |
| **`AMD_COMGR_SAVE_TEMPS`** | Non-zero: keep COMGR temporary files (in platform temp dir). |
| **`AMD_COMGR_EMIT_VERBOSE_LOGS`** | Non-zero: extra COMGR logging. |
| **`AMD_COMGR_REDIRECT_LOGS`** | Redirect COMGR logs (e.g. to stdout/stderr). |

### HIP / runtime logging

| Variable | Effect |
|----------|--------|
| **`AMD_LOG_LEVEL`** | `0`–`5` (higher = more detail); e.g. `4` or `5` for debug. |
| **`AMD_LOG_LEVEL_FILE`** | File path for log output (default stderr). |
| **`AMD_LOG_MASK`** | Bitmask. `0x100000` = Comgr path information; `0x4000` = code creation debug; `0x800` = init/shutdown; `0xFFFFFFFF` = log all. |

### Scratch / runtime debug

- **`HSA_ENABLE_DEBUG=1`** – extra HSA validation and logging.
- **`HSA_NO_SCRATCH_RECLAIM`** – affects scratch allocation (can matter at load time).

### KD count vs what you see

There are **18 KDs** in `.rodata` (3 Generic + 3 Debug/profiling + 12 oneRank); we **patch** the first 6 (Generic + Debug). If the runtime reports 6 Generic + 12 oneRank, that matches the 18-KD layout (3 Generic + 3 Debug + 12 oneRank). If those reported kernels don’t match the note or metadata, or if the dumped `.co` is truncated, the loader may be parsing partial data. Correlate with `GPU_DUMP_CODE_OBJECT=1` and COMGR verbose logs to see where the .co is written and whether it’s complete.

---

## Unit test: fails standalone, passes under rocprofv3

**Observation:** `UT_MIN_GPUS=2 UT_MAX_GPUS=2 ./rccl-UnitTests --gtest_filter=AllGather.OutOfPlace` **fails** when run directly, but **passes** (4 SP + 4 MP tests) when run under rocprofv3:

```bash
UT_MAX_GPUS=2 /opt/rocm/bin/rocprofv3 --kernel-trace --hsa-trace --stats -f csv -o rocp -- ./rccl-UnitTests --gtest_filter=AllGather.OutOfPlace
```

That suggests environment or library loading differs between the two runs. rocprofv3 may change `LD_LIBRARY_PATH` or inject libraries; the test binary’s **RPATH** should still prefer the built `librccl.so`, but it’s worth confirming.

**LD_DEBUG=libs result (confirmed):**

- **Direct run (fails):** Loader resolves `librccl.so.1` from **build/release** first (RUNPATH has build/release before /opt/rocm). So `calling init: .../build/release/librccl.so.1` — the **device-linker** build is used, and the test fails.

- **Under rocprofv3 (passes):** Loader resolves `librccl.so.1` from **/opt/rocm-7.0.0/lib** first: `trying file=/opt/rocm-7.0.0/lib/librccl.so.1` and `calling init: /opt/rocm-7.0.0/lib/librccl.so.1`. So when run under rocprofv3 the test is actually using the **system** (non–device-linker) RCCL, which passes. The profiler (or the way it launches the child) changes the dynamic linker search order so the system lib is found before the build dir.

So the “pass under rocprofv3” does **not** mean the device-linker build passes when profiled; it means the test is accidentally running against the system RCCL under rocprofv3. To actually test the device-linker build under the profiler, ensure the child sees the same RUNPATH/LD_LIBRARY_PATH as the direct run (e.g. run from build/release and/or set `LD_LIBRARY_PATH=.../build/release` when invoking rocprofv3 so build/release is searched first).

---

## Getting visibility when the unit test fails (no LD_LIBRARY_PATH tricks)

The failure shows up as `TestBed.cpp:196: Failure` and "Child %d reports failure" with no other message. The **child** process is the one that fails (during `InitComms` → `ncclCommInitRank` or similar); it writes `TEST_FAIL` (1) back to the parent and may print `ERROR(...)` to stdout. To see what is actually failing:

### 1. Environment variables (RCCL + test harness)

Run from **build/release** so the device-linker `librccl.so` is used (same as direct run). Capture all output; child stdout can be buffered when redirected, so use `stdbuf` to force line unbuffering if you don't see the child's ERROR line.

**RCCL logging (in both parent and child):**

| Variable | Effect |
|----------|--------|
| **`NCCL_DEBUG=INFO`** | RCCL prints init, bootstrap, and collective messages (INFO level). |
| **`NCCL_DEBUG=WARN`** | Less noise than INFO; still shows warnings. |
| **`NCCL_DEBUG_SUBSYS=ALL`** | Enable all debug subsystems (optional, use with INFO/TRACE). |
| **`NCCL_DEBUG_FILE`** | If set, RCCL writes logs to this file (helps if child stdout is lost). |

**Test harness:**

| Variable | Effect |
|----------|--------|
| **`UT_VERBOSE=1`** | TestBed and child print INFO (e.g. "Child N finishes InitComms() [FAIL]"). |
| **`UT_DEBUG_PAUSE=1`** | Parent prints child PIDs and waits for Enter; use this to attach **rocgdb** to a child. |

**AMD / ROCm runtime (HIP, COMGR, HSA):**

These can help when the failure is in HIP init, kernel load, or the ROCr/HSA stack. See also the full **"Debugging HIP load / COMGR"** section earlier in this doc.

| Variable | Effect |
|----------|--------|
| **`AMD_LOG_LEVEL`** | `0`–`5` (higher = more detail); e.g. `4` or `5` for runtime debug. |
| **`AMD_LOG_LEVEL_FILE`** | File path for AMD log output (default stderr). |
| **`AMD_LOG_MASK`** | Bitmask: `0x100000` = Comgr path; `0x4000` = code creation; `0x800` = init/shutdown; `0xFFFFFFFF` = all. |
| **`GPU_DUMP_CODE_OBJECT=1`** | Dump code objects (e.g. `.co` files); useful to confirm what is being loaded. |
| **`AMD_COMGR_EMIT_VERBOSE_LOGS`** | Non-zero: extra COMGR logging. |
| **`AMD_COMGR_REDIRECT_LOGS`** | Redirect COMGR logs to stdout/stderr. |
| **`HSA_ENABLE_DEBUG=1`** | Extra HSA validation and logging. |

**Example (from repo root):**

```bash
cd build/release
UT_MIN_GPUS=2 UT_MAX_GPUS=2 NCCL_DEBUG=INFO UT_VERBOSE=1 \
  ./test/rccl-UnitTests --gtest_filter=AllGather.OutOfPlace 2>&1 | tee out.log
```

If the child's ERROR line still doesn't appear (e.g. buffered), try unbuffered stdout:

```bash
stdbuf -o 0 -e 0 env UT_MIN_GPUS=2 UT_MAX_GPUS=2 NCCL_DEBUG=INFO UT_VERBOSE=1 \
  ./test/rccl-UnitTests --gtest_filter=AllGather.OutOfPlace 2>&1 | tee out.log
```

### 2. rocgdb (see where the child fails)

- **Attach to child:** Run the test with `UT_DEBUG_PAUSE=1`. The parent will print child PIDs and wait. In another terminal, attach to the **child** process (the one that runs ranks), then set breakpoints and continue.

  ```bash
  cd build/release
  UT_MIN_GPUS=2 UT_MAX_GPUS=2 UT_DEBUG_PAUSE=1 ./test/rccl-UnitTests --gtest_filter=AllGather.OutOfPlace
  # When it pauses, note the child PID (e.g. Child 00: processID: 12345), then:
  rocgdb -p <child_pid>
  (rocgdb) break TestBedChild.cpp:266    # before ncclCommInitRank failure path
  (rocgdb) break ncclCommInitRank
  (rocgdb) continue
  ```

- **Run test under rocgdb:** Run the **test binary** under rocgdb (no `UT_DEBUG_PAUSE`). Break in the parent when the child has already failed (e.g. `TestBed.cpp` around the `PIPE_CHECK` that triggers the failure) to confirm which child and that `response == 1` (TEST_FAIL). Then re-run with `UT_DEBUG_PAUSE=1` and attach to that child to see the real failure (e.g. `ncclCommInitRank` return value, or HIP/init errors).

- **Ensure device-linker lib is used:** Run from `build/release` (or set `LD_LIBRARY_PATH` so `build/release` is first). In rocgdb you can check which `librccl.so` is loaded, e.g. `info sharedlibrary` or `maintenance info sections`.

### 3. What the child actually reports

The parent only checks that the pipe response is `TEST_SUCCESS` (0). The child can fail in several places in `InitComms` (e.g. `hipSetDevice`, `hipStreamCreate`, `ncclCommInitRank`, `ncclGroupEnd`). Each failure path calls `ERROR(...)` and then returns `TEST_FAIL`. So with `NCCL_DEBUG=INFO` and `UT_VERBOSE=1` (and unbuffered stdout if needed), you should see either an RCCL message or the child's ERROR line indicating which call failed and, for NCCL, the `ncclResult_t` code.

### 4. Result with device-linker build

Running with `NCCL_DEBUG=INFO` and `UT_VERBOSE=1` from `build/release` produced (log showed `cudaArch 940`). To see the real architecture on the node, use e.g. **`rocminfo | grep gfx`**, **`rocm_agent_enumerator`** (used by the RCCL build for `GPU_TARGETS` when using `-l`), or whatever method your build uses.

- **NCCL WARN:** `cudaArch 940 ncclMaxSharedMem 32832 exceeds device/fn maxSharedMem 32704`
- **Failure:** `Child process 0 fails NCCL call ncclGroupEnd with code 2`

So the problem **without** any LD_LIBRARY_PATH/rocprofv3 is that **we request 32832 bytes of LDS (group segment) but the device reports max 32704** (over by 128 bytes).

**Root cause (why this error is reported):**

The LDS size that triggers the error is the one **in the code object**: the **KD** (kernel descriptor) and the **.note** section. The device linker writes that value; the host then checks it against the device limit.

1. **What’s in the KD and .note (merged code object):** The device linker sets **max_lds_** = max of (a) every kernel’s `.group_segment_fixed_size` from their **.note** (dispatcher + specialized), and (b) **requiredLDS** from `calculateRequiredLDS()` (same formula as RCCL host). It then:
   - **KD:** Patches the first 6 KDs (Generic 1/2/4 + Debug 1/2/4) in `.rodata`: at offset 0 of each KD it writes **max_lds_** (4 bytes) → **group_segment_fixed_size** in the KD = **32832**.
   - **.note:** Patches the dispatcher **.note** so every `.group_segment_fixed_size` field in the msgpack metadata is set to **max_lds_** (32832), so the note matches the KD.
   So in the merged binary, both the **KD** and the **.note** declare **32832** bytes LDS (group segment).

2. **What the host does:** In `src/init.cc`, init gets `hipDeviceGetAttribute(&maxSharedMem, hipDeviceAttributeMaxSharedMemoryPerBlockOptin, cudaDev)` → the HIP/driver reports **32704** as the max shared memory per block. In `src/enqueue.cc`, `ncclInitKernelsForDevice()` sets `ncclMaxSharedMem = rcclShmemDynamicSize(cudaArch, WarpSize)` (same formula as `calculateRequiredLDS()`) → **32832**. For each loaded kernel it checks `if (ncclMaxSharedMem > (maxSharedMem - attr.sharedSizeBytes))`; **32832 > 32704**, so it WARNs and returns `ncclSystemError`. So the **canonical** “how much we’re asking for” is what’s in the **KD and .note** (32832); the host formula matches that and performs the check against the device’s 32704.

3. **Why 32704:** Hardware has 64 KiB LDS per block; the **API** `hipDeviceAttributeMaxSharedMemoryPerBlockOptin` returns a lower limit (32704) on this device/driver, so the runtime correctly rejects a request of 32832.

**Fix direction:** Cap LDS to the API-reported maximum: (1) In the **device linker**, cap **max_lds_** to the device max (e.g. 32704) when known at build time, or have the host pass the device max into the pipeline so the KD and .note never exceed it. (2) In **RCCL host**, use `min(rcclShmemDynamicSize(...), maxSharedMem)` so we never request more than the device allows.

**Why did the smoke tests run (and pass)?** The smoke tests use **ncclCommInitAll** (blocking init, single process); the unit test uses **ncclCommInitRank** (and for some cases non-blocking init). Both paths call the same **ncclInitKernelsForDevice**, which does the LDS check and returns `ncclSystemError` when 32832 > device max. So the likely explanation is **which GPU** each run used. The failure log showed **cudaArch 940** and **device maxSharedMem 32704**. If the **smoke test** was run on a node where the API reports a higher per-block limit (≥32832), it would pass; the **unit test** run on the node that reports 32704 fails. Run the smoke test on the **same** node as the unit test (or set `HIP_VISIBLE_DEVICES` to the same devices) to confirm: you should see the same LDS WARN and failure when the device limit is 32704.

---

## Minimal comparison test

1. Build a minimal shared library with **normal** HIP (no device linker): one empty or trivial kernel, link into a `.so`, run a test that loads and launches.
2. Build the same with the **device linker** pipeline and the same `--offload-arch`/target.
3. Compare the two code objects (the one inside the normal-build `.so` vs `merged_device.elf`):
   - `readelf -h` (type, machine, flags)
   - `readelf -n` (notes)
   - `readelf -S` (sections)
   - `readelf -l` (segments)

Any structural difference (type, missing/extra segments or notes, different note content) is a candidate for why HIP loads one and not the other.
