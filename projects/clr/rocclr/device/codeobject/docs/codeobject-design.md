# Code object loading without comgr

How CLR reads HIP and OpenCL code objects (offload bundles, compressed bundles, ELF and the
AMDGPU metadata notes) without comgr, why, what still uses comgr, and how it is tested. The
container formats live in `rocclr/device/codeobject` (no HIP, OpenCL or device dependency
beyond `amd::Isa`); the callers are `hipamd/src/hip_fatbin.cpp`, `device/devprogram`,
`device/devkernel` and the ROCm blit program (`device/blit_kernels`, `device/rocm/rocblitelf`).

---

## 1 · Summary

- **Native code objects load without comgr.** comgr, and with it libLLVM and libclang-cpp, is
  loaded only on first use of SPIR-V, hipRTC, hipLink, OpenCL compilation, or a compressed
  bundle in a format newer than this code reads.
- **Device init doesn't load comgr.** The ISA values init used to query from comgr are columns
  of the `amd::Isa` table, and the ROCm blit kernels are built at build time.
- **Selection follows the code object, not the triple spelling.** Each bundle entry is named by
  its own ELF header, the way the loader names it, so LLVM's `amdgpuN.M-...` triples need no
  parsing.
- **Same observable behavior.** Kernel names, every argument and every attribute of the 9,773
  kernels in 155 installed code objects are identical before and after (section 10).
- **Measured** (gfx1100, release, three alternating rounds, pinned; before → after): device init
  65 → 31 ms, RSS after init 50 → 16 MB, rocBLAS 555-kernel code object cold load 145 → 67 ms
  and warm load 102 → 48 ms, first device-to-device copy 34 → 17-24 ms.

---

## 2 · Why not comgr on the load path

| Cost with comgr | Where |
|---|---|
| Any comgr call maps libamd_comgr, libLLVM and libclang-cpp: about 34 MB of RSS and half of device init | device init (ISA queries), first load |
| `set_data` copies the whole fat binary into comgr; the unbundle action copies each selected entry; `get_data` copies it out again | compressed bundles |
| Entries are requested by bundle-id strings and come back as file names the caller parses (`hipfatbin-hipv4-<isa>.<ext>`) | unbundle |
| Metadata is read through comgr nodes: a lookup and a string copy per key, per field, per kernel | `devkernel` / `devprogram` |
| The first blit compiles the blit program from OpenCL C | blit manager |

Interpreting the metadata (which key means what) was always the runtime's job: CLR maps every
key through its own `kArgFieldMapV3` / `kKernelFieldMapV3` tables. comgr only turned msgpack
into nodes. The fields and their meaning are unchanged; only the decoding moved.

---

## 3 · Layout

| Layer | Files | Responsibility |
|---|---|---|
| Container formats | `codeobject/offload_bundle.{hpp,cpp}` | `IsOffloadBundle`, `OffloadBundleSize`, `LookupCodeObjects`, `IsCompatibleTarget`, `TargetToGeneric` |
| | `codeobject/compressed_bundle.{hpp,cpp}` | `IsCompressedBundle`, `IsKnownCompressedBundle`, `CompressedBundleSize`, `DecompressBundle` (zlib, zstd via `Os::loadLibrary`) |
| | `codeobject/elf_header.{hpp,cpp}` | `IsElf`, `ElfImageSize`, `GetIsaFromHeader` (ROCr's mach table), `ForEachNote`, `DynamicSymbolNames` |
| Metadata | `codeobject/msgpack.{hpp,cpp}` | `MsgPackReader`: bounds-checked, zero-copy reader |
| | `codeobject/metadata_notes.{hpp,cpp}` | `MetadataNotes`: walks and merges the `NT_AMDGPU_METADATA` notes |
| Runtime | `device/devprogram.{hpp,cpp}` | `createKernelMetadataMap`: ISA check, notes, printf strings, `.dynsym` names |
| | `device/devkernel.{hpp,cpp}` | `GetAttrCodePropMetadata`: one kernel map, read in place |
| HIP | `hipamd/src/hip_fatbin.cpp` | container dispatch, per-device selection (native, generic, SPIR-V), the comgr fallback |
| Blit | `device/blit_kernels`, `device/rocm/rocblitelf.{hpp,cpp}` | offline blit program per target, selection at runtime |

The ELF code reads headers, section headers and notes in place; it has no ELF parser and doesn't
use `amd::Elf` (ELFIO's type header only). `amd::Elf` remains for OpenCL's `ClBinary` container.
ROCr's loader still parses each code object itself when it loads it.

---

## 4 · The load path

1. `FatBinaryInfo::ExtractFatBinaryUsingCOMGR` (name kept; comgr is only used for newer CCOB
   formats and SPIR-V) gets the image and its readable bound: the file size for file loads,
   `kUnknownSize` for a pointer (`hipModuleLoadData`, registered fat binaries).
2. The container decides the next step:
   - **raw ELF**: used as it is, sized with `ElfImageSize`;
   - **offload bundle**: `LookupCodeObjects` finds an entry per target; the result is offsets
     into the image, nothing is copied;
   - **CCOB v1-v3**: `DecompressBundle` once for all devices, then `LookupCodeObjects` on the
     result; only the selected entries are copied out;
   - **CCOB in a newer format**: comgr unbundles it (`UnbundleWithComgr`, section 6).
3. `AddDevProgram` per device takes the native entry, else the device's generic target
   (`TargetToGeneric`), else SPIR-V (compiled by comgr).
4. The backend's `createKernels` calls `createKernelMetadataMap`: the ISA from the ELF header
   (`GetIsaFromHeader`) is checked against the device, `MetadataNotes` reads every metadata
   note, keeping the printf strings, the code object version and each kernel's map as a view
   into the code object, and `DynamicSymbolNames` collects the `.dynsym` functions and
   variables (`getGlobalFuncFromCodeObj`, `getGlobalVarFromCodeObj`).
5. Each backend kernel (`roc::Kernel`, `pal::Kernel`) reads its map in place in `init()`
   (`GetAttrCodePropMetadata`); the views are dropped once the kernels exist.
6. A program built from a finished code object creates its kernels in `linkImpl` before any
   comgr call.

---

## 5 · Selecting a bundle entry

| Rule | Detail |
|---|---|
| Offload kinds | only `hip`, `hipv4` and `hcc` entries; `host-*`, `openmp-*` are skipped |
| Naming an entry | an entry holding an AMDGPU ELF is named by its header: `GetIsaFromHeader` turns `e_flags` (mach, xnack, sramecc) and the ABI version into `amdgcn-amd-amdhsa--<processor>[:sramecc±][:xnack±]`, the name ROCr's loader uses. Other entries (SPIR-V) are named by their id |
| Compatibility | comgr's `isCompatibleIsaName`: same processor; a feature the code object sets (+ or -) must match the device; one it leaves unset (any) matches both. Processor and feature validity come from `amd::Isa` |
| Order | entries in file order; each entry fills the first query still empty that it is compatible with (comgr's `lookup_code_object`) |
| Per device | native entry first, then the device's generic target (`TargetToGeneric`), then SPIR-V |
| Skipped | an ELF whose ISA the loader doesn't know (unknown mach, code object v2, 32-bit): it couldn't load anyway |

The mach table in `elf_header.cpp` is ROCr's `GetMachInfo` with its numeric values, kept local
because the copies of `amd_hsa_elf.h` in the PAL build lack the newer processors. A unit test
checks that every processor it names is in `amd::Isa`.

---

## 6 · Compressed bundles (CCOB)

The format is defined by LLVM (`llvm/lib/Object/OffloadBundle.cpp`):

| Version | Header | Introduced |
|---|---|---|
| 1 | magic, version, method, 32-bit uncompressed size, hash (20 bytes); no total size | 2023 |
| 2 | adds a 32-bit total size (24 bytes) | 2024 |
| 3 | 64-bit total and uncompressed sizes (32 bytes), for bundles over 4 GB | January 2025 |

- **Methods:** zlib (0) and zstd (1), each loaded with `Os::loadLibrary` on first use
  (`librocm_sysdeps_*` first, then the system library).
- **v1 has no size.** zstd input is cut at the frame end (`ZSTD_findFrameCompressedSize`); zlib
  stops at its stream end. The input copy `hipLibraryLoadData` keeps can't be sized, so a v1
  bundle loads through `hipModuleLoad*` and from files, but not through `hipLibraryLoadData`.
- **Sizes are checked:** a recorded total size larger than the readable bound is rejected. The
  uncompressed size is checked before it is allocated: for zstd it must equal the frame's own
  content size, for zlib it must fit deflate's maximum ratio (1032:1).
- **Newer formats go to comgr.** A version above 3 or an unknown method is unbundled by comgr,
  as CLR always did; its length is read from the 32-bit total size of the v2 header
  (`UnknownCompressedBundleSize`). No compiler writes such a bundle yet: the hip-test for it (a
  version patched to 4) shows the path is taken and fails cleanly.
- The stored hash is not checked, as in clang's own bundler.

---

## 7 · Metadata

- **Notes.** `ForEachNote` visits every `NT_AMDGPU_METADATA` note (type 32, name `AMDGPU`) in
  every `SHT_NOTE` section. A linked code object can hold one note per input object: rocBLAS's
  Tensile libraries carry one per kernel. `MetadataNotes` merges them like comgr's
  `mergeNoteRecords`: the `amdhsa.kernels` lists are concatenated and `amdhsa.printf` is taken
  from the first note that has it. A note whose `amdhsa.version` or printf list differs from an
  earlier one fails the load (comgr dropped it, leaving printf IDs that don't match); a kernel
  listed twice keeps its first map.
- **In place.** `MsgPackReader` returns `string_view`s into the note; each kernel's map is kept
  as a view and read once by its kernel's `init()`. No node tree, no copies.
- **Field handling** is comgr's `iterate_map_metadata`, which the runtime used before: every
  scalar is accepted as text, integer or boolean; an unknown field or a value that can't be used
  is ignored (comgr ignored the callback's status); only malformed msgpack fails the kernel.
- **Version.** `amdhsa.version` 1.x is code object v(3 + x); code object v2 (YAML metadata) is
  not supported.
- **Validation:** malformed msgpack, conflicting notes, a missing note, a missing
  `amdhsa.version` and a kernel map without `.name` fail the load. Counts read from the note
  (printf strings, arguments) never size an allocation beyond the bytes left in the note.

---

## 8 · Blit kernels built offline

The ROCm blit program (copies, fills, images, stream ops, heap init, GWS init, the OpenCL device
enqueue scheduler) used to be compiled from OpenCL C through comgr on the first blit.

- **One source.** `gen_blit_cl` writes the `.cl` file from the strings the runtime compiles
  (`blitcl.cpp`, `rocblitcl.cpp`), so the offline and runtime kernels can't drift.
- **Targets:** gfx9, gfx9-4, gfx10-1, gfx10-3, gfx11, gfx11-7, gfx12 and gfx12-5 generic, plus
  native gfx908 and gfx90a, which have no generic target. Each is compiled with the options
  `Program::compileImpl` uses for that target's default settings (wave size, CU mode, kernarg
  preload, GWS init, image kernels); built that way for gfx1100, the code object is identical in
  disassembly and metadata to the runtime's own build. gfx125x leaves out the image kernels: it
  has no image support and the compiler can't build them for it.
- **Selection** (`FindOfflineBlitProgram`): the device's processor, else its generic target,
  checked with `IsCompatibleTarget` against each embedded code object's header.
- **Fallback:** the kernels are compiled at runtime as before when no code object matches, the
  device settings differ from the ones it was built for (`GPU_ENABLE_WAVE32_MODE`,
  `GPU_ENABLE_WGP_MODE`, `DEBUG_CLR_BLIT_KERNARG_OPT`), with ASan, or with
  `GPU_DUMP_BLIT_KERNELS`. The PAL backend always compiles them.
- **Build:** `CLR_OFFLINE_BLIT_KERNELS` (on by default) needs amdclang and the device libraries
  under `ROCM_PATH`; without them it turns off. The embedded code objects add about 0.5 MB to
  each runtime library.

---

## 9 · What still uses comgr

| Use | When |
|---|---|
| SPIR-V code objects (`amdgcnspirv`) | a fat binary with no native or generic entry for the device |
| hipRTC, hipLink | API use |
| OpenCL compilation | from source, LLVM IR, SPIR or SPIR-V |
| Compressed bundles newer than v3 | section 6 |
| Blit kernels | the fallback cases in section 8, and PAL |

comgr is loaded on first use (`Comgr::Load`) and resolves 22 entry points; if it can't be
loaded, they return `AMD_COMGR_STATUS_ERROR`, and OpenCL compile, link and build report
`CL_COMPILER_NOT_AVAILABLE` or `CL_LINKER_NOT_AVAILABLE`. Device init no longer loads it: the
register limits, trap handler and image support that `roc::Device` read from comgr's ISA
metadata are columns of the `amd::Isa` table. The CLR profiler demangles kernel names with
`Os::CxaDemangle`, like the rest of CLR.

---

## 10 · Tests

| Layer | Where | Needs | What |
|---|---|---|---|
| Offline unit tests | `codeobject/tests` (`codeobject_test`, `-DCODEOBJECT_BUILD_TESTS=ON`, Catch2 via FetchContent) | nothing for the format tests; comgr for the ISA check | 52 cases: bundle selection by `e_flags` including `amdgpu12.50` ids, generic targets, xnack/sramecc, SPIR-V, bounds, malformed headers; target-id compatibility; `TargetToGeneric`; CCOB v1-v3 x zlib/zstd and malformed sizes; ELF size, notes, `.dynsym`; the mach table against `amd::Isa`; note merging; msgpack in every width; truncation at every byte; the `amd::Isa` ISA columns against comgr's ISA metadata |
| GPU tests | hip-tests `unit/module/hipCodeObjectMetadata.cc` in `ModuleTest` | a GPU, hipcc | 7 cases: uncompressed and compressed (v2, v3) fat binaries, generic-only, native next to generic, raw ELF, two notes linked from separately assembled objects, each from a file and from memory; a CCOB patched to version 4. Checks names, argument offsets and sizes, LDS, launch bounds, scratch, argument values, buffered printf |
| Fuzzing | `codeobject/tests/fuzz_codeobject.cpp` (`codeobject_fuzz`) | clang | bundle lookup, CCOB, ELF size, notes, `.dynsym`, msgpack |
| Comparison | `co_compare` (CRUX tools) | a second runtime | names, parameters and attributes of every kernel in the installed Tensile libraries and hip-tests code objects: 9,773 kernels, no differences |

Compressed and generic fat binaries registered by an executable are covered by `unit/compiler`
(`SimpleCompressedCodeObjectTest`, `hipSquareGenericTarget*`).

---

## 11 · Limits and open items

- **The comgr fallback's success path has never run:** no compiler writes a CCOB newer than v3.
- **v1 CCOB through `hipLibraryLoadData`** is rejected (section 6).
- **Unknown bound.** For a pointer load the image size is unknown, so entry offsets in a bundle
  header are trusted, as before; the header of every hip entry is read.
- **gfx906 with sramecc** has no generic-target match (`amd::Isa` lists no sramecc for
  `gfx9-generic`), in the bundle lookup as before, and its blit kernels are compiled at runtime.
- **Code object v2** is not supported.
- **The profiler on Windows** reports mangled kernel names: `Os::CxaDemangle` doesn't demangle
  there, as for the rest of CLR.
- **Two parses.** CLR reads a code object's headers and notes in place, and ROCr's loader
  parses it again to load it; handing one parse to the loader needs a ROCr API.
- **Builds:** Linux release is tested; Windows and PAL have not been built, and the fuzz harness
  has not been built (it needs a clang host build).
