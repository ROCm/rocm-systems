# AIE hsaco Loading via HSA APIs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load AIE (XDNA NPU) kernels through the standard HSA executable/loader APIs, packaging instruction + optional PDI payloads into an architecture-named hsaco section and dispatching them via an opaque `kernel_object` handle.

**Architecture:** A Python tool injects an `aie2`/`aie2p` section (versioned header + kernel table + offset-addressed blob pool) into an hsaco. The runtime loader parses that section, copies each unique blob to XDNA-BO-backed device memory, and builds a host-owned `AieKernelDescriptor` per kernel. `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT` returns a host pointer to that descriptor; the dispatch packet carries the pointer in its `insts_addr_low/high` words; the in-process XDNA driver dereferences it at submit to read PDI/insts device addresses and sizes.

**Tech Stack:** C++17 (rocr-runtime), Python 3 + pyelftools (packaging/inspection tool), GoogleTest (rocrtst AIE suite), CMake.

**Spec:** `docs/superpowers/specs/2026-07-12-aie-hsaco-loading-design.md`

## Global Constraints

- C++ standard: **C++17** (Unix). Match `projects/rocr-runtime/_clang-format` (Google-based, 100-col, 2-space indent).
- Namespaces: `rocr::AMD` for AMD-specific runtime code; test code stays in the anonymous namespace pattern already used in `dispatch.cc`.
- Output parameters use pointers (`T*`), never references (`T&`).
- Commit style: `feat(component): description` / `fix(component): description` / `test(component): ...` / `docs(component): ...`.
- The repo is a **sparse checkout**; `git add` needs `--sparse` for paths under `docs/`.
- Section format constants (copied verbatim from spec):
  - `AIE_SECTION_MAGIC = 0x4B454941u` ('A','I','E','K' little-endian)
  - `version_major = 1`, `version_minor = 0`
  - Section names: `aie2`, `aie2p`
  - PDI absent ⇔ `pdi_offset == 0 && pdi_size == 0`; `insts_size` must be `> 0`.
- All `*_offset` values in the section are **section-relative**; `name_offset` is relative to `string_table_offset`.
- Freeing model: descriptors + device buffers freed at **executable destroy** (Option A). No per-object unload.
- GPU parity is scoped to **symbol-info queries only**; the AIE kernarg ABI (packed `[addresses…, sizes…]`) is unchanged.

---

## File Structure

**New files:**
- `runtime/hsa-runtime/core/inc/amd_aie_section.h` — shared C struct definitions (`aie_section_header`, `aie_kernel_entry`, magic/version constants) used by both the parser and (as reference) the Python tool.
- `rocrtst/suites/aie/aie_hsaco.py` — packaging tool: inject/replace an `aie2`/`aie2p` section into an hsaco.
- `rocrtst/suites/aie/aie_hsaco_dump.py` — inspection tool: parse and print the section, validate well-formedness.
- `rocrtst/suites/aie/test_aie_hsaco.py` — pytest unit tests for the two Python tools (round-trip + validation).

**Modified files:**
- `runtime/hsa-runtime/inc/hsa_ext_amd_aie.h` — packet ABI: opaque handle in `insts_addr_low/high`; `insts_size`/`pdi_addr` reserved.
- `runtime/hsa-runtime/core/inc/amd_aie_code.hpp` + `core/runtime/amd_aie_code.cpp` — replace `.ctrltext`/`.ctrldata` parser with arch-section parser; expose per-kernel PDI+insts blobs and metadata; section-presence detection.
- `runtime/hsa-runtime/loader/executable.hpp` + `loader/executable.cpp` — `AieKernelDescriptor`; arch validation; build descriptors; opaque `KERNEL_OBJECT`; duplicate-name rejection; free at destroy.
- `runtime/hsa-runtime/core/inc/amd_aie_agent.h` + `core/runtime/amd_aie_agent.cpp` — supported-architecture query used for arch-vs-agent validation.
- `runtime/hsa-runtime/core/driver/xdna/amd_xdna_driver.cpp` — at submit, deref descriptor pointer from `insts_addr_low/high`; read PDI/insts addr+size from it; skip PDI when `pdi_dev_addr == 0`.
- `rocrtst/suites/aie/dispatch.cc` — new `SingleDispatchHsaco` + multi-object + duplicate-name + optional-PDI tests.
- `rocrtst/suites/aie/CMakeLists.txt` — invoke `aie_hsaco.py` to produce a test hsaco; pass its path to `dispatch`.

---

## Task 1: Section format header (C definitions)

**Files:**
- Create: `runtime/hsa-runtime/core/inc/amd_aie_section.h`
- Test: covered indirectly by Task 2 (parser) and Task 5 (Python round-trip). This task is a pure header; its "test" is that it compiles when included.

**Interfaces:**
- Produces: `rocr::AMD::aie_section_header`, `rocr::AMD::aie_kernel_entry`, constants `kAieSectionMagic`, `kAieSectionVersionMajor`, `kAieSectionVersionMinor`.

- [ ] **Step 1: Write the header**

```cpp
// runtime/hsa-runtime/core/inc/amd_aie_section.h
#ifndef HSA_RUNTIME_CORE_INC_AMD_AIE_SECTION_H_
#define HSA_RUNTIME_CORE_INC_AMD_AIE_SECTION_H_

#include <cstdint>

namespace rocr {
namespace AMD {

// 'A','I','E','K' little-endian.
constexpr uint32_t kAieSectionMagic = 0x4B454941u;
constexpr uint16_t kAieSectionVersionMajor = 1;
constexpr uint16_t kAieSectionVersionMinor = 0;

// Section header. All *_offset fields are section-relative (bytes from the start
// of the aie2/aie2p section).
struct aie_section_header {
  uint32_t magic;              // kAieSectionMagic
  uint16_t version_major;      // reject on mismatch
  uint16_t version_minor;      // additive-only
  uint32_t header_size;        // offset from section base to kernel table
  uint32_t kernel_count;
  uint32_t kernel_entry_size;  // stride between kernel entries
  uint32_t string_table_offset;
  uint32_t string_table_size;
  uint32_t blob_pool_offset;   // blobs live in [blob_pool_offset, section_end)
  uint32_t reserved[4];        // must be 0
};

struct aie_kernel_entry {
  uint32_t name_offset;   // relative to string_table_offset; NUL-terminated
  uint32_t insts_offset;  // section-relative; REQUIRED
  uint32_t insts_size;    // REQUIRED, > 0
  uint32_t pdi_offset;    // section-relative; 0 if no PDI
  uint32_t pdi_size;      // 0 if no PDI
  uint32_t kernarg_size;
  uint32_t num_cols;
  uint32_t reserved[4];   // must be 0
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_SECTION_H_
```

- [ ] **Step 2: Verify it compiles standalone**

Run:
```bash
cd /home/ypapadop/workspace-raiders/rocm-systems/projects/rocr-runtime
g++ -std=c++17 -fsyntax-only -x c++ runtime/hsa-runtime/core/inc/amd_aie_section.h
```
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add runtime/hsa-runtime/core/inc/amd_aie_section.h
git commit -m "feat(aie): add hsaco arch-section format header"
```

---

## Task 2: Section parser (`AieCode` rewrite)

Replace the `.ctrltext`/`.ctrldata` parser with one that reads the arch section and exposes per-kernel metadata + PDI/insts blobs by offset into the section. Detection becomes section-presence-based.

**Files:**
- Modify: `runtime/hsa-runtime/core/inc/amd_aie_code.hpp` (whole `AieKernelInfo`/`AieCode` API)
- Modify: `runtime/hsa-runtime/core/runtime/amd_aie_code.cpp` (whole parse path)
- Test: `runtime/hsa-runtime/core/runtime/amd_aie_code.cpp` is exercised via the loader tests; add a focused parser check into the Python round-trip (Task 5) and the dispatch test (Task 9). No standalone C++ unit-test harness exists in this suite, so parser correctness is validated end-to-end.

**Interfaces:**
- Consumes: `aie_section_header`, `aie_kernel_entry` (Task 1).
- Produces:
  - `static bool AieCode::IsAieCodeObject(const void* data, size_t size)` — true iff an `aie2` or `aie2p` section is present.
  - `static std::unique_ptr<AieCode> AieCode::Create(const void* data, size_t size)`
  - `const std::string& AieCode::GetArchSectionName() const` — `"aie2"` or `"aie2p"`.
  - `std::vector<std::string> AieCode::GetKernelNames() const`
  - `const AieKernelInfo* AieCode::GetKernel(const std::string& name) const`
  - `AieKernelInfo` fields: `std::string name; const uint8_t* insts_data; uint64_t insts_size; const uint8_t* pdi_data; uint64_t pdi_size; uint32_t kernarg_size; uint32_t num_cols;` where `insts_data`/`pdi_data` point into the caller's ELF buffer (blob pool); `pdi_data == nullptr && pdi_size == 0` when absent.

- [ ] **Step 1: Rewrite the header API**

Replace the body of `amd_aie_code.hpp` `AieKernelInfo` and `AieCode` (keep the license header and include guard) with:

```cpp
namespace rocr {
namespace amd { namespace elf { class Image; } }
namespace AMD {

struct AieKernelInfo {
  std::string name;
  const uint8_t* insts_data = nullptr;  // into ELF buffer; never null after parse
  uint64_t insts_size = 0;              // > 0
  const uint8_t* pdi_data = nullptr;    // null if no PDI
  uint64_t pdi_size = 0;                // 0 if no PDI
  uint32_t kernarg_size = 0;
  uint32_t num_cols = 0;
};

class AieCode {
 public:
  static std::unique_ptr<AieCode> Create(const void* data, size_t size);
  static bool IsAieCodeObject(const void* data, size_t size);

  const std::string& GetArchSectionName() const { return arch_section_name_; }
  std::vector<std::string> GetKernelNames() const;
  const AieKernelInfo* GetKernel(const std::string& name) const;
  size_t GetKernelCount() const { return kernels_.size(); }

 private:
  AieCode() = default;
  bool Parse();

  std::unique_ptr<amd::elf::Image> elf_;
  const uint8_t* section_base_ = nullptr;  // start of arch section in ELF buffer
  uint64_t section_size_ = 0;
  std::string arch_section_name_;
  std::map<std::string, AieKernelInfo> kernels_;
};

}  // namespace AMD
}  // namespace rocr
```

Update includes at top of the header to add `#include "core/inc/amd_aie_section.h"` and keep `<map> <memory> <string> <vector> <cstdint> <cstddef>`.

- [ ] **Step 2: Write the detection + parse implementation**

Replace the anonymous-namespace constants and `IsAieCodeObject`/`Create`/`Parse`/`ExtractKernelSymbols`/`LoadSectionData` in `amd_aie_code.cpp` with:

```cpp
namespace {
constexpr const char* kArchSectionNames[] = {"aie2", "aie2p"};

// Returns the arch section (by name) if present, else nullptr, and sets out_name.
amd::elf::Section* FindArchSection(amd::elf::Image* elf, std::string* out_name) {
  for (size_t i = 0; i < elf->sectionCount(); ++i) {
    amd::elf::Section* sec = elf->section(i);
    if (!sec) continue;
    const std::string name = sec->Name();
    for (const char* arch : kArchSectionNames) {
      if (name == arch) {
        *out_name = name;
        return sec;
      }
    }
  }
  return nullptr;
}
}  // namespace

bool AieCode::IsAieCodeObject(const void* data, size_t size) {
  if (!data || size < sizeof(Elf64_Ehdr)) return false;
  const auto* ehdr = static_cast<const Elf64_Ehdr*>(data);
  if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) return false;

  auto img = std::unique_ptr<amd::elf::Image>(amd::elf::NewElf64Image());
  if (!img || !img->initFromBuffer(data, size)) return false;
  std::string name;
  return FindArchSection(img.get(), &name) != nullptr;
}

std::unique_ptr<AieCode> AieCode::Create(const void* data, size_t size) {
  if (!data || size == 0) return nullptr;
  auto code = std::unique_ptr<AieCode>(new AieCode());
  code->elf_.reset(amd::elf::NewElf64Image());
  if (!code->elf_ || !code->elf_->initFromBuffer(data, size)) return nullptr;
  if (!code->Parse()) return nullptr;
  return code;
}

bool AieCode::Parse() {
  amd::elf::Section* sec = FindArchSection(elf_.get(), &arch_section_name_);
  if (!sec) return false;

  section_size_ = sec->size();
  if (section_size_ < sizeof(aie_section_header)) return false;
  section_base_ = static_cast<const uint8_t*>(sec->get());  // pointer into buffer
  if (!section_base_) return false;

  const auto* hdr = reinterpret_cast<const aie_section_header*>(section_base_);
  if (hdr->magic != kAieSectionMagic) return false;
  if (hdr->version_major != kAieSectionVersionMajor) return false;
  if (hdr->header_size + static_cast<uint64_t>(hdr->kernel_count) * hdr->kernel_entry_size >
      section_size_) {
    return false;
  }
  if (hdr->kernel_entry_size < sizeof(aie_kernel_entry)) return false;

  auto in_section = [&](uint64_t off, uint64_t len) {
    return len == 0 ? off <= section_size_ : (off < section_size_ && off + len <= section_size_);
  };

  for (uint32_t i = 0; i < hdr->kernel_count; ++i) {
    const auto* e = reinterpret_cast<const aie_kernel_entry*>(
        section_base_ + hdr->header_size + static_cast<uint64_t>(i) * hdr->kernel_entry_size);

    if (e->insts_size == 0) return false;
    if (!in_section(e->insts_offset, e->insts_size)) return false;
    if (e->pdi_size != 0 && !in_section(e->pdi_offset, e->pdi_size)) return false;

    const uint64_t name_abs = hdr->string_table_offset + e->name_offset;
    if (name_abs >= section_size_) return false;
    const char* nm = reinterpret_cast<const char*>(section_base_ + name_abs);
    const uint64_t max_len = section_size_ - name_abs;
    if (::strnlen(nm, max_len) == max_len) return false;  // unterminated

    AieKernelInfo info;
    info.name = nm;
    info.insts_data = section_base_ + e->insts_offset;
    info.insts_size = e->insts_size;
    info.pdi_data = e->pdi_size ? section_base_ + e->pdi_offset : nullptr;
    info.pdi_size = e->pdi_size;
    info.kernarg_size = e->kernarg_size;
    info.num_cols = e->num_cols;
    if (kernels_.count(info.name)) return false;  // duplicate within one object
    kernels_[info.name] = info;
  }
  return !kernels_.empty();
}

std::vector<std::string> AieCode::GetKernelNames() const {
  std::vector<std::string> names;
  names.reserve(kernels_.size());
  for (const auto& kv : kernels_) names.push_back(kv.first);
  return names;
}

const AieKernelInfo* AieCode::GetKernel(const std::string& name) const {
  auto it = kernels_.find(name);
  return it == kernels_.end() ? nullptr : &it->second;
}
```

Add `#include <cstring>` and `#include "core/inc/amd_aie_section.h"` at the top; keep the ELF includes already present. Remove now-unused members/methods (`instr_data_`, `ctrl_packet_data_`, `GetInstructionData/Size`, `GetCtrlPacket*`, `ExtractKernelSymbols`, `LoadSectionData`, `elf_data_`, `elf_size_`).

> Note: confirm the accessor that returns a raw pointer to section bytes. If `amd::elf::Section` exposes the mapped data via a method other than `get()` (e.g. `data()`), use that; the parser needs a pointer into the ELF buffer, not a copy. Verify against `core/inc/amd_elf_image.hpp` during implementation.

- [ ] **Step 3: Build the runtime library**

Run:
```bash
cd /home/ypapadop/workspace-raiders/rocm-systems/projects/rocr-runtime
cmake --build build -j"$(nproc)" --target hsa-runtime64 2>&1 | tail -20
```
Expected: compiles. (If no `build` dir yet, configure per CLAUDE.md first.)
Expected failure mode to watch for: any other translation unit that referenced the removed `GetInstructionData`/`GetCtrlPacketData` API — those are updated in Task 4.

- [ ] **Step 4: Commit**

```bash
git add runtime/hsa-runtime/core/inc/amd_aie_code.hpp runtime/hsa-runtime/core/runtime/amd_aie_code.cpp
git commit -m "feat(aie): parse arch-named hsaco section with per-kernel PDI/insts"
```

---

## Task 3: AIE agent supported-architecture query

Add a way to get the agent's architecture name(s) (`aie2`/`aie2p`) so the loader can validate the section name against the agent.

**Files:**
- Modify: `runtime/hsa-runtime/core/inc/amd_aie_agent.h`
- Modify: `runtime/hsa-runtime/core/runtime/amd_aie_agent.cpp`
- Test: exercised by the loader arch-validation test (Task 9). Verified here by build.

**Interfaces:**
- Produces: `const std::vector<std::string>& AieAgent::supported_arch_names() const;` returning e.g. `{"aie2p"}`.

- [ ] **Step 1: Inspect how the agent already models its ISA/arch**

Run:
```bash
cd /home/ypapadop/workspace-raiders/rocm-systems/projects/rocr-runtime
grep -n "supported_isas_\|isa\|Isa\|npu\|aie2\|GetIsaName\|arch" runtime/hsa-runtime/core/runtime/amd_aie_agent.cpp runtime/hsa-runtime/core/inc/amd_aie_agent.h
```
Expected: shows the existing `supported_isas_` vector and how it is populated. Use the existing ISA→name source if one exists; otherwise derive the arch name from the same device-id switch the agent already uses to identify NPU generation.

- [ ] **Step 2: Add the accessor (header)**

In `amd_aie_agent.h`, add to the public section of `AieAgent`:

```cpp
  /// @brief Architecture name(s) accepted in hsaco section names, e.g. "aie2p".
  const std::vector<std::string>& supported_arch_names() const { return supported_arch_names_; }
```
And in the private data members:
```cpp
  std::vector<std::string> supported_arch_names_;
```

- [ ] **Step 3: Populate it (impl)**

In the `AieAgent` constructor (or wherever `supported_isas_` is initialized), populate `supported_arch_names_` with the arch string matching this NPU. Use the NPU-generation determination the agent already performs. For the current single supported target, this is one entry (`"aie2"` for Phoenix/npu1, `"aie2p"` for Strix/npu2 — match the mapping the codebase already uses; confirm in Step 1).

```cpp
// Example shape — align names with Step 1 findings:
supported_arch_names_.push_back(arch_name_for_this_npu);
```

- [ ] **Step 4: Build**

Run:
```bash
cmake --build build -j"$(nproc)" --target hsa-runtime64 2>&1 | tail -20
```
Expected: compiles.

- [ ] **Step 5: Commit**

```bash
git add runtime/hsa-runtime/core/inc/amd_aie_agent.h runtime/hsa-runtime/core/runtime/amd_aie_agent.cpp
git commit -m "feat(aie): report supported architecture names on AIE agent"
```

---

## Task 4: Descriptor + loader (device placement, symbols, arch validation, multi-object)

Rework `LoadAieCodeObject` to build one host-owned `AieKernelDescriptor` per kernel, copy each unique blob to XDNA-BO-backed device memory, return the descriptor pointer as `KERNEL_OBJECT`, validate arch, reject duplicate names, and free at destroy.

**Files:**
- Modify: `runtime/hsa-runtime/loader/executable.hpp` (`AieKernelSymbol`, `AieLoadedCodeObjectImpl`, new `AieKernelDescriptor`)
- Modify: `runtime/hsa-runtime/loader/executable.cpp` (`LoadAieCodeObject`, `AieKernelSymbol::GetInfo`, `AieLoadedCodeObjectImpl::Destroy`)
- Test: end-to-end via Task 9.

**Interfaces:**
- Consumes: `AieCode` (Task 2); `AieAgent::supported_arch_names()` (Task 3).
- Produces:
  - `struct AieKernelDescriptor { uint32_t version; uint32_t reserved0; uint64_t insts_dev_addr; uint64_t insts_size; uint64_t pdi_dev_addr; uint64_t pdi_size; uint32_t kernarg_size; uint32_t num_cols; };` (in a header the driver can include — see Step 1).
  - `AieKernelSymbol` whose `KERNEL_OBJECT` returns `reinterpret_cast<uint64_t>(descriptor)`.

- [ ] **Step 1: Define the descriptor in a driver-visible header**

Create the descriptor next to the section header so the XDNA driver can include it without depending on the loader. Add to `runtime/hsa-runtime/core/inc/amd_aie_section.h` (inside `namespace rocr::AMD`), after `aie_kernel_entry`:

```cpp
// Internal, host-side kernel descriptor. The kernel_object handle is a pointer
// to one of these. Owned by the loaded code object; freed at executable destroy.
struct AieKernelDescriptor {
  uint32_t version;         // reserved for a future on-device format; set to 1
  uint32_t reserved0;       // must be 0
  uint64_t insts_dev_addr;  // device address of instruction blob (an XDNA BO)
  uint64_t insts_size;
  uint64_t pdi_dev_addr;    // device address of PDI blob (an XDNA BO), or 0
  uint64_t pdi_size;        // 0 if no PDI
  uint32_t kernarg_size;
  uint32_t num_cols;
};

constexpr uint32_t kAieKernelDescriptorVersion = 1;
```

- [ ] **Step 2: Update `AieKernelSymbol` to carry a descriptor pointer**

In `executable.hpp`, change `AieKernelSymbol` so `KERNEL_OBJECT` returns a descriptor pointer instead of the raw instruction address. Replace the constructor + members:

```cpp
class AieKernelSymbol final : public SymbolImpl {
 public:
  AieKernelSymbol(const std::string& _symbol_name, uint64_t _descriptor_ptr,
                  uint32_t _kernarg_size, uint32_t _num_cols)
      : SymbolImpl(true, HSA_SYMBOL_KIND_KERNEL, "", _symbol_name,
                   HSA_SYMBOL_LINKAGE_PROGRAM, true, _descriptor_ptr),
        full_name(_symbol_name),
        descriptor_ptr(_descriptor_ptr),
        kernarg_size(_kernarg_size),
        num_cols(_num_cols) {}

  bool GetInfo(hsa_symbol_info32_t symbol_info, void* value);

  uint32_t GetKernargSize() const { return kernarg_size; }
  uint32_t GetNumCols() const { return num_cols; }

  std::string full_name;
  uint64_t descriptor_ptr;
  uint32_t kernarg_size;
  uint32_t num_cols;
};
```

In `executable.cpp`, in `AieKernelSymbol::GetInfo`, change the `KERNEL_OBJECT` case to return `descriptor_ptr`:

```cpp
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT:
      *static_cast<uint64_t*>(value) = descriptor_ptr;
      return true;
```
(The other cases — kernarg size, alignment=64, group/private=0, name, type, linkage, agent — remain as they are.)

- [ ] **Step 3: Give `AieLoadedCodeObjectImpl` ownership of descriptors + buffers**

In `executable.hpp`, extend `AieLoadedCodeObjectImpl` with owned storage and a real `Destroy()`:

```cpp
  std::vector<std::unique_ptr<AMD::AieKernelDescriptor>> descriptors;
  std::vector<std::pair<void*, size_t>> device_buffers;  // (host ptr from SegmentAlloc, size)
```
Declare `void Destroy() override;` if not already present.

In `executable.cpp`, implement `Destroy()` to free each device buffer via the loader context and clear owned state:

```cpp
void AieLoadedCodeObjectImpl::Destroy() {
  for (auto& b : device_buffers) {
    owner->context()->SegmentFree(AMDGPU_HSA_SEGMENT_CODE_AGENT, agent, b.first, b.second);
  }
  device_buffers.clear();
  descriptors.clear();
}
```
(`descriptors` are `unique_ptr`s, so clearing frees them. `~ExecutableImpl` already calls `Destroy()` on everything in `objects`, satisfying the free-at-destroy model.)

- [ ] **Step 4: Rewrite `LoadAieCodeObject`**

Replace the body (executable.cpp:1534–1601) with arch validation, blob dedup, descriptor build, duplicate-name rejection:

```cpp
hsa_status_t ExecutableImpl::LoadAieCodeObject(hsa_agent_t agent, const void* data, size_t size,
                                               const std::string& uri,
                                               hsa_loaded_code_object_t* loaded_code_object) {
  auto aie_code = AMD::AieCode::Create(data, size);
  if (!aie_code) {
    logger_ << "LoaderError: failed to parse AIE code object\n";
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  // Arch-vs-agent validation.
  core::Agent* core_agent = core::Agent::Convert(agent);
  auto* aie_agent = static_cast<AMD::AieAgent*>(core_agent);
  const auto& arches = aie_agent->supported_arch_names();
  if (std::find(arches.begin(), arches.end(), aie_code->GetArchSectionName()) == arches.end()) {
    logger_ << "LoaderError: code object arch does not match agent\n";
    return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
  }

  auto loaded_obj = std::make_shared<AieLoadedCodeObjectImpl>(this, agent, data, size);

  // Dedup blobs by (host source pointer, size): copy each unique blob once.
  std::map<std::pair<const uint8_t*, uint64_t>, uint64_t> blob_dev_addr;
  auto place_blob = [&](const uint8_t* src, uint64_t len, uint64_t* out_dev) -> hsa_status_t {
    if (len == 0) { *out_dev = 0; return HSA_STATUS_SUCCESS; }
    auto key = std::make_pair(src, len);
    auto it = blob_dev_addr.find(key);
    if (it != blob_dev_addr.end()) { *out_dev = it->second; return HSA_STATUS_SUCCESS; }
    void* buf = context_->SegmentAlloc(AMDGPU_HSA_SEGMENT_CODE_AGENT, agent, len, 64, false);
    if (!buf) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    if (!context_->SegmentCopy(AMDGPU_HSA_SEGMENT_CODE_AGENT, agent, buf, 0, src, len)) {
      context_->SegmentFree(AMDGPU_HSA_SEGMENT_CODE_AGENT, agent, buf, len);
      return HSA_STATUS_ERROR;
    }
    context_->SegmentFreeze(AMDGPU_HSA_SEGMENT_CODE_AGENT, agent, buf, len);
    void* dev = context_->SegmentAddress(AMDGPU_HSA_SEGMENT_CODE_AGENT, agent, buf, 0);
    loaded_obj->device_buffers.emplace_back(buf, len);
    uint64_t dev_addr = reinterpret_cast<uint64_t>(dev);
    blob_dev_addr[key] = dev_addr;
    *out_dev = dev_addr;
    return HSA_STATUS_SUCCESS;
  };

  // First pass: reject duplicate (name, agent) before allocating anything.
  for (const auto& kernel_name : aie_code->GetKernelNames()) {
    if (agent_symbols_.count(std::make_pair(kernel_name, agent))) {
      logger_ << "LoaderError: kernel already defined: " << kernel_name << "\n";
      return HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED;
    }
  }

  for (const auto& kernel_name : aie_code->GetKernelNames()) {
    const auto* ki = aie_code->GetKernel(kernel_name);
    uint64_t insts_dev = 0, pdi_dev = 0;
    if (auto s = place_blob(ki->insts_data, ki->insts_size, &insts_dev); s != HSA_STATUS_SUCCESS)
      return s;
    if (auto s = place_blob(ki->pdi_data, ki->pdi_size, &pdi_dev); s != HSA_STATUS_SUCCESS)
      return s;

    auto desc = std::make_unique<AMD::AieKernelDescriptor>();
    desc->version = AMD::kAieKernelDescriptorVersion;
    desc->reserved0 = 0;
    desc->insts_dev_addr = insts_dev;
    desc->insts_size = ki->insts_size;
    desc->pdi_dev_addr = pdi_dev;
    desc->pdi_size = ki->pdi_size;
    desc->kernarg_size = ki->kernarg_size;
    desc->num_cols = ki->num_cols;
    uint64_t desc_ptr = reinterpret_cast<uint64_t>(desc.get());
    loaded_obj->descriptors.push_back(std::move(desc));

    auto kernel_sym = std::make_shared<AieKernelSymbol>(kernel_name, desc_ptr, ki->kernarg_size,
                                                        ki->num_cols);
    kernel_sym->agent = agent;
    agent_symbols_[std::make_pair(kernel_name, agent)] = kernel_sym;
  }

  objects.push_back(loaded_obj);
  if (loaded_code_object) {
    *loaded_code_object = LoadedCodeObject::Handle(loaded_obj.get());
  }
  return HSA_STATUS_SUCCESS;
}
```

Add includes as needed at top of `executable.cpp`: `#include "core/inc/amd_aie_agent.h"`, `#include "core/inc/amd_aie_section.h"`, `<algorithm>`, `<map>`. Update the `AieLoadedCodeObjectImpl` constructor call to the 4-arg form (drop the old `instr_buffer`/`instr_dev_addr`/`instr_size` params); adjust the class's constructor + `getLoadBase()/getLoadSize()` accordingly (return 0 or the first buffer — keep it simple: return 0).

- [ ] **Step 5: Build**

Run:
```bash
cmake --build build -j"$(nproc)" --target hsa-runtime64 2>&1 | tail -30
```
Expected: compiles. Fix any signature drift in `AieLoadedCodeObjectImpl` getters flagged by the compiler.

- [ ] **Step 6: Commit**

```bash
git add runtime/hsa-runtime/core/inc/amd_aie_section.h runtime/hsa-runtime/loader/executable.hpp runtime/hsa-runtime/loader/executable.cpp
git commit -m "feat(aie): build host kernel descriptors in loader, validate arch, reject dup names"
```

---

## Task 5: Packet ABI change

Make the packet carry the opaque handle in `insts_addr_low/high`; mark `insts_size`/`pdi_addr` reserved.

**Files:**
- Modify: `runtime/hsa-runtime/inc/hsa_ext_amd_aie.h`
- Test: compile + Task 9.

**Interfaces:**
- Produces: packet field `kernel_object` occupying the former `insts_addr_low`/`insts_addr_high` storage.

- [ ] **Step 1: Edit the packet struct**

In `hsa_amd_aie_kernel_dispatch_packet_t`, replace the `insts_addr_low`/`insts_addr_high` pair and the `insts_size`/`pdi_addr` fields' docs. Keep the 64-byte layout identical (rename in place; do not reorder):

```c
  /**
   * Opaque kernel object handle obtained from
   * HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT. Occupies the two 32-bit words that
   * previously held the instruction-sequence address.
   */
  uint32_t kernel_object_low;
  uint32_t kernel_object_high;
```
Change the former `insts_size` field to:
```c
  /**
   * Reserved. Must be 0. (Formerly the instruction sequence size; now carried in
   * the kernel descriptor referenced by kernel_object.)
   */
  uint64_t reserved_insts_size;
```
Change the former `pdi_addr` field to:
```c
  /**
   * Reserved. Must be 0. (Formerly the PDI address; now carried in the kernel
   * descriptor referenced by kernel_object.)
   */
  void* reserved_pdi_addr;
```
Leave `num_kernargs`, `kernarg_address`, `completion_signal`, header/opcode/count and all other reserved fields unchanged.

- [ ] **Step 2: Verify size is unchanged**

Run:
```bash
cd /home/ypapadop/workspace-raiders/rocm-systems/projects/rocr-runtime
cat > /tmp/aie_pkt_size.c <<'EOF'
#include <stdio.h>
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd_aie.h"
int main(){ printf("%zu\n", sizeof(hsa_amd_aie_kernel_dispatch_packet_t)); return 0; }
EOF
gcc -I runtime/hsa-runtime/inc /tmp/aie_pkt_size.c -o /tmp/aie_pkt_size && /tmp/aie_pkt_size
```
Expected: `64` (unchanged; the `static_assert` against `core::AqlPacket` in `amd_aie_aql_queue.cpp` still holds).

- [ ] **Step 3: Commit**

```bash
git add runtime/hsa-runtime/inc/hsa_ext_amd_aie.h
git commit -m "feat(aie): carry opaque kernel_object handle in dispatch packet"
```

---

## Task 6: XDNA driver — resolve descriptor at submit

Deref the descriptor pointer from `insts_addr_low/high`; read PDI/insts addr+size from it; skip PDI when 0.

**Files:**
- Modify: `runtime/hsa-runtime/core/driver/xdna/amd_xdna_driver.cpp` (SubmitCmdChain loop, lines ~1106–1180; `FlushArguments` if it reads the old fields)
- Test: Task 9.

**Interfaces:**
- Consumes: `AieKernelDescriptor` (Task 4, via `amd_aie_section.h`); packet fields `kernel_object_low/high` (Task 5).

- [ ] **Step 1: Include the descriptor header**

Add near the top of `amd_xdna_driver.cpp`:
```cpp
#include "core/inc/amd_aie_section.h"
```

- [ ] **Step 2: Resolve the descriptor and use its fields**

In `SubmitCmdChain`'s per-packet loop, replace the PDI/insts derivation. Immediately after `auto* pkt = queue + pkt_idx;` add:

```cpp
    const auto* desc = reinterpret_cast<const AMD::AieKernelDescriptor*>(
        Concat<uint64_t>(pkt->kernel_object_high, pkt->kernel_object_low));
    if (!desc) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
```

Replace the PDI block (was `FindBOHandle(pkt->pdi_addr)`), handling the optional PDI:

```cpp
    int cached_pdi_index = 0;
    if (desc->pdi_dev_addr != 0) {
      auto pdi_bo_handle = FindBOHandle(reinterpret_cast<void*>(desc->pdi_dev_addr));
      if (!pdi_bo_handle.IsValid()) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
      auto idx = kmq_metadata->pdi_cache.GetIndex(pdi_bo_handle.handle);
      if (idx == PDICache::NotFound) {
        FlushCpuCache(pdi_bo_handle.vaddr, 0, pdi_bo_handle.size);
        hsa_status_t err = kmq_metadata->pdi_cache.SetNext(pdi_bo_handle.handle, idx);
        if (err != HSA_STATUS_SUCCESS) { assert(false && "Failed to set PDI in cache."); return err; }
        reconfigure_queue = true;
      }
      cached_pdi_index = static_cast<int>(idx);
    }
```

Replace the insts block:
```cpp
    void* insts_addr = reinterpret_cast<void*>(desc->insts_dev_addr);
    auto instr_bo_handle = FindBOHandle(insts_addr);
    if (!instr_bo_handle.IsValid()) {
      assert(false && "Failed to find instruction sequence BO for command packet.");
      return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    }
    bo_handles.push_back(instr_bo_handle.handle);
    FlushCpuCache(insts_addr, 0, desc->insts_size);
```

In the command emission, replace `pkt->insts_size` with `desc->insts_size`:
```cpp
    cmd->data[5] = (desc->insts_size / sizeof(uint32_t));  // instruction dword count
```
`cmd->data[0] = 0x1 << cached_pdi_index;` stays (now `cached_pdi_index` is 0 when there is no PDI — confirm the ERT semantics of a zero CU mask with the driver owner; for the MVP the vector-scalar kernel always has a PDI, so this path is exercised with `pdi_dev_addr != 0`).

- [ ] **Step 3: Check `FlushArguments`**

Run:
```bash
grep -n "insts_addr\|pdi_addr\|insts_size\|kernel_object" runtime/hsa-runtime/core/driver/xdna/amd_xdna_driver.cpp
```
Expected: no remaining reads of `pkt->pdi_addr`, `pkt->insts_addr_low/high`, or `pkt->insts_size`. `FlushArguments` (kernargs only) is unaffected. Fix any stragglers.

- [ ] **Step 4: Build**

Run:
```bash
cmake --build build -j"$(nproc)" --target hsa-runtime64 2>&1 | tail -30
```
Expected: compiles.

- [ ] **Step 5: Commit**

```bash
git add runtime/hsa-runtime/core/driver/xdna/amd_xdna_driver.cpp
git commit -m "feat(aie): resolve kernel descriptor from packet handle at submit"
```

---

## Task 7: Python packaging tool

**Files:**
- Create: `rocrtst/suites/aie/aie_hsaco.py`
- Test: `rocrtst/suites/aie/test_aie_hsaco.py` (Task 8 adds validation cases; this task adds the round-trip test)

**Interfaces:**
- Produces CLI:
  `python aie_hsaco.py --hsaco OUT.hsaco --arch aie2p --kernel NAME:INSTS.bin[:PDI.pdi]:KERNARG_SIZE:NUM_COLS [...]`
  and a function `build_section(arch, kernels) -> bytes` where `kernels` is a list of dicts `{name, insts: bytes, pdi: bytes|None, kernarg_size: int, num_cols: int}`.

- [ ] **Step 1: Write the failing round-trip test**

```python
# rocrtst/suites/aie/test_aie_hsaco.py
import struct
import aie_hsaco

def _parse(section: bytes):
    magic, vmaj, vmin, hdr_size, kcount, kentry, st_off, st_size, pool_off = \
        struct.unpack_from("<IHHIIIIII", section, 0)
    assert magic == 0x4B454941
    assert (vmaj, vmin) == (1, 0)
    kernels = []
    for i in range(kcount):
        base = hdr_size + i * kentry
        (name_off, insts_off, insts_size, pdi_off, pdi_size,
         kernarg_size, num_cols) = struct.unpack_from("<IIIIIII", section, base)
        name_abs = st_off + name_off
        end = section.index(b"\x00", name_abs)
        name = section[name_abs:end].decode()
        insts = section[insts_off:insts_off + insts_size]
        pdi = section[pdi_off:pdi_off + pdi_size] if pdi_size else None
        kernels.append(dict(name=name, insts=insts, pdi=pdi,
                            kernarg_size=kernarg_size, num_cols=num_cols))
    return kernels

def test_build_section_round_trip():
    kernels = [
        dict(name="add_one", insts=b"\x01\x02\x03\x04", pdi=b"\xaa\xbb",
             kernarg_size=32, num_cols=1),
        dict(name="no_pdi", insts=b"\x05\x06\x07\x08", pdi=None,
             kernarg_size=16, num_cols=2),
    ]
    section = aie_hsaco.build_section("aie2p", kernels)
    out = _parse(section)
    assert out == kernels

def test_shared_blob_deduplicated():
    shared = b"\x09\x09\x09\x09"
    kernels = [
        dict(name="a", insts=shared, pdi=None, kernarg_size=0, num_cols=1),
        dict(name="b", insts=shared, pdi=None, kernarg_size=0, num_cols=1),
    ]
    section = aie_hsaco.build_section("aie2p", kernels)
    out = _parse(section)
    # Both kernels resolve to the same bytes and the blob appears once in the pool.
    assert out[0]["insts"] == out[1]["insts"] == shared
    assert section.count(shared) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
cd /home/ypapadop/workspace-raiders/rocm-systems/projects/rocr-runtime/rocrtst/suites/aie
python -m pytest test_aie_hsaco.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'aie_hsaco'` (or `AttributeError: build_section`).

- [ ] **Step 3: Implement `aie_hsaco.py`**

```python
#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Inject an aie2/aie2p section (versioned header + kernel table + blob pool)
into an hsaco. See docs/superpowers/specs/2026-07-12-aie-hsaco-loading-design.md."""
import argparse
import struct
import sys

MAGIC = 0x4B454941
VERSION_MAJOR = 1
VERSION_MINOR = 0
ARCHES = ("aie2", "aie2p")

_HDR = "<IHHIIIIII" + "IIII"          # header + reserved[4]
_HDR_SIZE = struct.calcsize(_HDR)
_ENTRY = "<IIIIIII" + "IIII"          # 7 fields + reserved[4]
_ENTRY_SIZE = struct.calcsize(_ENTRY)


def build_section(arch, kernels):
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHES}")

    string_table = bytearray()
    name_offsets = []
    for k in kernels:
        name_offsets.append(len(string_table))
        string_table += k["name"].encode() + b"\x00"

    # Blob pool with dedup keyed on raw bytes.
    pool = bytearray()
    blob_off = {}

    def place(blob):
        if not blob:
            return (0, 0)
        key = bytes(blob)
        if key not in blob_off:
            blob_off[key] = len(pool)
            pool.extend(key)
        return (blob_off[key], len(key))

    # Layout: [header][kernel table][string table][blob pool]
    header_size = _HDR_SIZE
    table_size = _ENTRY_SIZE * len(kernels)
    string_table_offset = header_size + table_size
    blob_pool_offset = string_table_offset + len(string_table)

    entries = []
    for k, name_off in zip(kernels, name_offsets):
        insts_off, insts_size = place(k["insts"])
        if insts_size == 0:
            raise ValueError(f"kernel {k['name']!r}: insts must be non-empty")
        pdi_off, pdi_size = place(k.get("pdi"))
        entries.append((name_off,
                        blob_pool_offset + insts_off, insts_size,
                        (blob_pool_offset + pdi_off) if pdi_size else 0, pdi_size,
                        int(k.get("kernarg_size", 0)), int(k.get("num_cols", 1)),
                        0, 0, 0, 0))

    out = bytearray()
    out += struct.pack(_HDR, MAGIC, VERSION_MAJOR, VERSION_MINOR, header_size,
                       len(kernels), _ENTRY_SIZE, string_table_offset,
                       len(string_table), blob_pool_offset, 0, 0, 0, 0)
    for e in entries:
        out += struct.pack(_ENTRY, *e)
    out += string_table
    out += pool
    return bytes(out)


def _inject(hsaco_path, arch, section_bytes):
    # Use pyelftools to read; rewrite the section via a minimal ELF section append.
    # Implementation detail: prefer `llvm-objcopy --add-section=<arch>=<file>
    # --set-section-flags` when available; fall back to pyelftools-based rewrite.
    import subprocess, tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(section_bytes)
        sec_file = f.name
    try:
        # Remove an existing same-named section first (ignore error if absent).
        subprocess.run(["llvm-objcopy", "--remove-section", arch, hsaco_path,
                        hsaco_path], check=False)
        subprocess.run(["llvm-objcopy", f"--add-section={arch}={sec_file}",
                        f"--set-section-flags={arch}=noload,readonly",
                        hsaco_path, hsaco_path], check=True)
    finally:
        os.unlink(sec_file)


def _parse_kernel_arg(s):
    # NAME:INSTS[:PDI]:KERNARG_SIZE:NUM_COLS  (PDI optional)
    parts = s.split(":")
    if len(parts) == 4:
        name, insts_path, kernarg_size, num_cols = parts
        pdi_path = None
    elif len(parts) == 5:
        name, insts_path, pdi_path, kernarg_size, num_cols = parts
    else:
        raise argparse.ArgumentTypeError(f"bad --kernel spec {s!r}")
    with open(insts_path, "rb") as f:
        insts = f.read()
    pdi = None
    if pdi_path:
        with open(pdi_path, "rb") as f:
            pdi = f.read()
    return dict(name=name, insts=insts, pdi=pdi,
                kernarg_size=int(kernarg_size), num_cols=int(num_cols))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsaco", required=True)
    ap.add_argument("--arch", required=True, choices=ARCHES)
    ap.add_argument("--kernel", required=True, action="append", type=_parse_kernel_arg)
    args = ap.parse_args(argv)
    section = build_section(args.arch, args.kernel)
    _inject(args.hsaco, args.arch, section)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest test_aie_hsaco.py -v
```
Expected: `test_build_section_round_trip` and `test_shared_blob_deduplicated` PASS.

- [ ] **Step 5: Add pyelftools/pytest to requirements**

Append to `rocrtst/suites/aie/requirements.txt`:
```
pyelftools
pytest
```

- [ ] **Step 6: Commit**

```bash
git add rocrtst/suites/aie/aie_hsaco.py rocrtst/suites/aie/test_aie_hsaco.py rocrtst/suites/aie/requirements.txt
git commit -m "feat(aie): add hsaco section packaging tool"
```

---

## Task 8: Python inspection tool + validation tests

**Files:**
- Create: `rocrtst/suites/aie/aie_hsaco_dump.py`
- Modify: `rocrtst/suites/aie/test_aie_hsaco.py` (add validation cases)

**Interfaces:**
- Consumes: `aie_hsaco.build_section` (Task 7).
- Produces: `aie_hsaco_dump.parse_section(section: bytes) -> dict` raising `ValueError` on malformed input; CLI `python aie_hsaco_dump.py --hsaco IN.hsaco`.

- [ ] **Step 1: Write failing validation tests**

Append to `test_aie_hsaco.py`:
```python
import pytest
import aie_hsaco_dump

def test_dump_round_trip():
    section = aie_hsaco.build_section("aie2", [
        dict(name="k", insts=b"\x01\x02\x03\x04", pdi=b"\xaa", kernarg_size=8, num_cols=1)])
    info = aie_hsaco_dump.parse_section(section)
    assert info["arch_version"] == (1, 0)
    assert info["kernels"][0]["name"] == "k"
    assert info["kernels"][0]["has_pdi"] is True

def test_dump_rejects_bad_magic():
    section = bytearray(aie_hsaco.build_section("aie2", [
        dict(name="k", insts=b"\x01\x02\x03\x04", pdi=None, kernarg_size=0, num_cols=1)]))
    section[0] ^= 0xFF
    with pytest.raises(ValueError, match="magic"):
        aie_hsaco_dump.parse_section(bytes(section))

def test_dump_rejects_blob_overrun():
    section = bytearray(aie_hsaco.build_section("aie2", [
        dict(name="k", insts=b"\x01\x02\x03\x04", pdi=None, kernarg_size=0, num_cols=1)]))
    # Corrupt insts_size in the first entry to overrun the section.
    entry_base = struct.unpack_from("<I", section, 8)[0]  # header_size
    struct.pack_into("<I", section, entry_base + 8, 0xFFFFFFFF)  # insts_size field
    with pytest.raises(ValueError, match="overrun|bounds"):
        aie_hsaco_dump.parse_section(bytes(section))
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
python -m pytest test_aie_hsaco.py -k dump -v
```
Expected: FAIL — `No module named 'aie_hsaco_dump'`.

- [ ] **Step 3: Implement `aie_hsaco_dump.py`**

```python
#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Parse and validate an aie2/aie2p hsaco section; print human-readable summary."""
import argparse
import struct
import sys

MAGIC = 0x4B454941
_HDR = "<IHHIIIIII" + "IIII"
_HDR_SIZE = struct.calcsize(_HDR)
_ENTRY = "<IIIIIII" + "IIII"


def parse_section(section):
    if len(section) < _HDR_SIZE:
        raise ValueError("section smaller than header")
    (magic, vmaj, vmin, hdr_size, kcount, kentry, st_off, st_size,
     pool_off, *_res) = struct.unpack_from(_HDR, section, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x}")
    if kentry < struct.calcsize(_ENTRY):
        raise ValueError("kernel_entry_size too small")
    if hdr_size + kcount * kentry > len(section):
        raise ValueError("kernel table out of bounds")

    def in_section(off, ln):
        return off <= len(section) if ln == 0 else (off < len(section) and off + ln <= len(section))

    kernels = []
    for i in range(kcount):
        base = hdr_size + i * kentry
        (name_off, insts_off, insts_size, pdi_off, pdi_size,
         kernarg_size, num_cols, *_e) = struct.unpack_from(_ENTRY, section, base)
        if insts_size == 0 or not in_section(insts_off, insts_size):
            raise ValueError(f"kernel {i}: insts out of bounds/overrun")
        if pdi_size and not in_section(pdi_off, pdi_size):
            raise ValueError(f"kernel {i}: pdi out of bounds/overrun")
        name_abs = st_off + name_off
        if name_abs >= len(section):
            raise ValueError(f"kernel {i}: name out of bounds")
        end = section.find(b"\x00", name_abs)
        if end == -1:
            raise ValueError(f"kernel {i}: name not terminated")
        kernels.append(dict(name=section[name_abs:end].decode(),
                            insts_size=insts_size, has_pdi=bool(pdi_size),
                            pdi_size=pdi_size, kernarg_size=kernarg_size,
                            num_cols=num_cols))
    return dict(arch_version=(vmaj, vmin), kernel_count=kcount, kernels=kernels)


def _read_section_from_hsaco(path, arch):
    from elftools.elf.elffile import ELFFile
    with open(path, "rb") as f:
        elf = ELFFile(f)
        for name in ("aie2", "aie2p"):
            sec = elf.get_section_by_name(name)
            if sec is not None:
                return name, sec.data()
    raise ValueError("no aie2/aie2p section found")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsaco", required=True)
    args = ap.parse_args(argv)
    arch, data = _read_section_from_hsaco(args.hsaco, None)
    info = parse_section(data)
    print(f"arch section: {arch}")
    print(f"version: {info['arch_version'][0]}.{info['arch_version'][1]}")
    for k in info["kernels"]:
        print(f"  kernel {k['name']}: insts={k['insts_size']}B "
              f"pdi={'yes' if k['has_pdi'] else 'no'} "
              f"kernarg={k['kernarg_size']} cols={k['num_cols']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest test_aie_hsaco.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rocrtst/suites/aie/aie_hsaco_dump.py rocrtst/suites/aie/test_aie_hsaco.py
git commit -m "feat(aie): add hsaco section inspection/validation tool"
```

---

## Task 9: End-to-end dispatch test via HSA loader APIs

Extend the C++ dispatch suite. Write the test first (it will fail to build/link until the helper + CMake wiring land in Steps 3–5, then fail at runtime until the whole chain works).

**Files:**
- Modify: `rocrtst/suites/aie/dispatch.cc`
- Modify: `rocrtst/suites/aie/CMakeLists.txt`

**Interfaces:**
- Consumes: everything above. Uses `DEFAULT_HSACO_PATH` (new compile def) for the packaged hsaco, and `DEFAULT_HSACO2_PATH` for a second single-kernel hsaco used by the multi-object test.

> **Ordering:** Task 5 changed the packet ABI, so the *existing* tests in `dispatch.cc` no longer compile. To get a clean build at Step 5, do Task 11 (migrate existing tests) **together with** this task — add the new helpers/tests here, migrate the old ones in Task 11, then build once. Treat Tasks 9+11 as one review unit if executing task-by-task.

- [ ] **Step 1: Add the HSA-load helper + `SingleDispatchHsaco` test**

Add to `dispatch.cc` (anonymous namespace helper, then a `DispatchTest` test). The dispatch reuses the existing `dispatch_packet` mechanics but sets the handle instead of raw addresses:

```cpp
// Loads an hsaco onto the AIE agent and returns the kernel_object handle for
// `kernel_name`, plus the frozen executable (out-params, per repo convention).
testing::AssertionResult load_kernel_object(hsa_agent_t agent, const std::filesystem::path& hsaco,
                                            const char* kernel_name, hsa_executable_t* exe_out,
                                            uint64_t* kernel_object_out) {
  std::size_t size = 0;
  auto f = open_binary(hsaco, &size);
  if (!f) return testing::AssertionFailure() << "open " << hsaco;
  std::vector<char> buf(size);
  if (!read_exact(f, buf.data(), size)) return testing::AssertionFailure() << "read " << hsaco;

  hsa_executable_t exe{};
  if (hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                &exe) != HSA_STATUS_SUCCESS)
    return testing::AssertionFailure() << "executable_create_alt";

  hsa_code_object_reader_t reader{};
  if (hsa_code_object_reader_create_from_memory(buf.data(), size, &reader) != HSA_STATUS_SUCCESS)
    return testing::AssertionFailure() << "reader_create_from_memory";

  if (hsa_executable_load_agent_code_object(exe, agent, reader, nullptr, nullptr) !=
      HSA_STATUS_SUCCESS)
    return testing::AssertionFailure() << "load_agent_code_object";

  // KERNEL_OBJECT must be 0 before freeze.
  hsa_executable_symbol_t sym0{};
  if (hsa_executable_get_symbol_by_name(exe, kernel_name, &agent, &sym0) == HSA_STATUS_SUCCESS) {
    uint64_t ko0 = 42;
    hsa_executable_symbol_get_info(sym0, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &ko0);
    if (ko0 != 0) return testing::AssertionFailure() << "KERNEL_OBJECT not 0 before freeze";
  }

  if (hsa_executable_freeze(exe, nullptr) != HSA_STATUS_SUCCESS)
    return testing::AssertionFailure() << "freeze";

  hsa_executable_symbol_t sym{};
  if (hsa_executable_get_symbol_by_name(exe, kernel_name, &agent, &sym) != HSA_STATUS_SUCCESS)
    return testing::AssertionFailure() << "get_symbol_by_name " << kernel_name;
  uint64_t ko = 0;
  if (hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &ko) !=
          HSA_STATUS_SUCCESS ||
      ko == 0)
    return testing::AssertionFailure() << "KERNEL_OBJECT 0 after freeze";

  hsa_code_object_reader_destroy(reader);
  *exe_out = exe;
  *kernel_object_out = ko;
  return testing::AssertionSuccess();
}

// Sets the opaque handle into the packet's kernel_object words.
void set_kernel_object(hsa_amd_aie_kernel_dispatch_packet_t* pkt, uint64_t ko) {
  pkt->kernel_object_low = static_cast<uint32_t>(ko & 0xFFFFFFFF);
  pkt->kernel_object_high = static_cast<uint32_t>(ko >> 32);
}
```

Add a `dispatch_packet_ko` overload/variant of `aie_vector_scalar_kernel::dispatch_packet` that takes `uint64_t kernel_object` instead of `pdi_buf`/`insts_buf`/`insts_size` and calls `set_kernel_object`. Then:

```cpp
TEST_F(DispatchTest, SingleDispatchHsaco) {
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue), HSA_STATUS_SUCCESS);

  hsa_executable_t exe{};
  uint64_t ko = 0;
  ASSERT_TRUE(load_kernel_object(aie_agents.front(), STRINGIFY(DEFAULT_HSACO_PATH),
                                 "vector_scalar_add", &exe, &ko));

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, 0,
                                         reinterpret_cast<void**>(&input)), HSA_STATUS_SUCCESS);
  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, 0,
                                         reinterpret_cast<void**>(&output)), HSA_STATUS_SUCCESS);
  std::iota(input, input + aie_vector_scalar_kernel::element_count, 0);
  std::fill_n(output, aie_vector_scalar_kernel::element_count, 0);

  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)), HSA_STATUS_SUCCESS);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet_ko(ko, input, output, kernargs,
                                                                   signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  for (std::size_t i = 0; i < aie_vector_scalar_kernel::element_count; ++i)
    EXPECT_EQ(output[i], static_cast<std::uint32_t>(i + 1)) << "mismatch at " << i;

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_executable_destroy(exe), HSA_STATUS_SUCCESS);
}
```

- [ ] **Step 2: Add multi-object, duplicate-name, and symbol-parity tests**

```cpp
TEST_F(DispatchTest, MultiObjectHsaco) {
  // Two independent hsacos, one kernel each, loaded into separate executables;
  // dispatch from each and verify both. (Same-executable multi-load is covered by
  // the duplicate-name test; independent objects are the MVP guarantee.)
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue), HSA_STATUS_SUCCESS);

  hsa_executable_t exe_a{}, exe_b{};
  uint64_t ko_a = 0, ko_b = 0;
  ASSERT_TRUE(load_kernel_object(aie_agents.front(), STRINGIFY(DEFAULT_HSACO_PATH),
                                 "vector_scalar_add", &exe_a, &ko_a));
  ASSERT_TRUE(load_kernel_object(aie_agents.front(), STRINGIFY(DEFAULT_HSACO2_PATH),
                                 "vector_scalar_add", &exe_b, &ko_b));
  EXPECT_NE(ko_a, ko_b);
  // (Dispatch from ko_a then ko_b using the same body as SingleDispatchHsaco;
  //  verify output[i] == input[i] + 1 for each. Omitted here for brevity but
  //  MUST be included — copy the alloc/dispatch/verify block twice.)
  EXPECT_EQ(hsa_executable_destroy(exe_a), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_executable_destroy(exe_b), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, DuplicateKernelNameRejected) {
  hsa_executable_t exe{};
  uint64_t ko = 0;
  ASSERT_TRUE(load_kernel_object(aie_agents.front(), STRINGIFY(DEFAULT_HSACO_PATH),
                                 "vector_scalar_add", &exe, &ko));
  // Load the same hsaco again into the same executable -> duplicate name.
  std::size_t size = 0;
  auto f = open_binary(STRINGIFY(DEFAULT_HSACO_PATH), &size);
  ASSERT_TRUE(static_cast<bool>(f));
  std::vector<char> buf(size);
  ASSERT_TRUE(read_exact(f, buf.data(), size));
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(hsa_code_object_reader_create_from_memory(buf.data(), size, &reader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_executable_load_agent_code_object(exe, aie_agents.front(), reader, nullptr, nullptr),
            HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED);
  hsa_code_object_reader_destroy(reader);
  EXPECT_EQ(hsa_executable_destroy(exe), HSA_STATUS_SUCCESS);
}
```

> Note: The `MultiObjectHsaco` dispatch/verify blocks must be filled in with the same allocate→dispatch→wait→verify sequence as `SingleDispatchHsaco` (once per kernel object). Do not ship the abbreviated version.

- [ ] **Step 3: Add a `vector_scalar_add` kernel name to the IRON design output path**

The packaging tool needs a kernel name. Use `"vector_scalar_add"` consistently. No change to `vector_scalar_add.py` is required (the name is supplied to `aie_hsaco.py`).

- [ ] **Step 4: Wire CMake to build the hsaco(s)**

In `rocrtst/suites/aie/CMakeLists.txt`, after the `INSTS_BIN`/`DESIGN_PDI` custom command, add:

```cmake
set(TEST_HSACO ${KERNEL_BUILD_DIR}/vector_scalar_add.hsaco)
set(TEST_HSACO2 ${KERNEL_BUILD_DIR}/vector_scalar_add2.hsaco)

# Start from an empty ELF the loader's reader accepts, then inject the arch section.
# Reuse the compiled insts/pdi. KERNARG_SIZE=32 (2 ptr + 2 size = 4*8), NUM_COLS=1.
add_custom_command(
  OUTPUT "${TEST_HSACO}" "${TEST_HSACO2}"
  COMMAND "${CMAKE_COMMAND}" -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/empty_hsaco_template.o" "${TEST_HSACO}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/aie_hsaco.py"
          --hsaco "${TEST_HSACO}" --arch "${NPU_TARGET_ARCH}"
          --kernel "vector_scalar_add:${INSTS_BIN}:${DESIGN_PDI}:32:1"
  COMMAND "${CMAKE_COMMAND}" -E copy "${TEST_HSACO}" "${TEST_HSACO2}"
  DEPENDS "${INSTS_BIN}" "${DESIGN_PDI}" "${CMAKE_CURRENT_SOURCE_DIR}/aie_hsaco.py"
  COMMENT "Packaging AIE kernel into hsaco"
  VERBATIM)

add_custom_target(dispatch_hsaco DEPENDS "${TEST_HSACO}" "${TEST_HSACO2}")
add_dependencies(dispatch dispatch_hsaco)

target_compile_definitions(dispatch PRIVATE
  DEFAULT_HSACO_PATH=${TEST_HSACO}
  DEFAULT_HSACO2_PATH=${TEST_HSACO2})
```

Map `NPU_TARGET` → `NPU_TARGET_ARCH` (`npu`→`aie2`, `npu2`→`aie2p`) near the existing `NPU_TARGET` detection:
```cmake
if(NPU_TARGET STREQUAL "npu2")
  set(NPU_TARGET_ARCH "aie2p")
else()
  set(NPU_TARGET_ARCH "aie2")
endif()
```

> Open item for implementer: `empty_hsaco_template.o` — the loader's ELF reader (`amd_elf_image`) must accept the base object. Produce a minimal valid ELF64 relocatable/shared object once (e.g. `echo 'const int _aie=0;' | clang --target=x86_64 -c -o empty_hsaco_template.o -xc -`) and check it in, OR generate it in-tree with `llvm-mc`/`clang`. Validate in Step 6 that `hsa_code_object_reader_create_from_memory` accepts it (the Open Risk from the spec). If the reader rejects a foreign-arch ELF, instead synthesize the ELF entirely in `aie_hsaco.py` with pyelftools so `e_machine`/class are loader-acceptable.

- [ ] **Step 5: Configure + build the test**

Run:
```bash
cd /home/ypapadop/workspace-raiders/rocm-systems/projects/rocr-runtime/rocrtst/suites/aie
source iron_env_setup.sh
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j"$(nproc)" --target dispatch 2>&1 | tail -30
```
Expected: builds; the hsaco artifacts are produced under `build/kernel/`.

- [ ] **Step 6: Run the new tests**

Run:
```bash
cd build
LD_LIBRARY_PATH=/opt/rocm/lib ./dispatch --gtest_filter='DispatchTest.SingleDispatchHsaco:DispatchTest.MultiObjectHsaco:DispatchTest.DuplicateKernelNameRejected' 2>&1 | tail -40
```
Expected: all three PASS on AIE hardware. If `SingleDispatchHsaco` fails at `FindBOHandle`-invalid, that is the load↔submit BO contract (spec §6) — confirm the loader's `SegmentAlloc(CODE_AGENT)` for AIE produced a registered XDNA BO; adjust the loader allocation to the dev-heap BO path the plain tests use.

- [ ] **Step 7: Confirm the pre-existing `SingleDispatch` still passes (no regression)**

Run:
```bash
LD_LIBRARY_PATH=/opt/rocm/lib ./dispatch --gtest_filter='DispatchTest.SingleDispatch:DispatchTest.MultiDispatch*' 2>&1 | tail -20
```
Because the packet ABI changed in Task 5, the older `SingleDispatch`/`MultiDispatch*` tests will not compile until they are migrated — that migration is **Task 11**, done next. This step therefore only confirms `SingleDispatchHsaco` and friends are green; the full-suite no-regression run happens at the end of Task 11.

- [ ] **Step 8: Commit**

```bash
git add rocrtst/suites/aie/dispatch.cc rocrtst/suites/aie/CMakeLists.txt
git commit -m "test(aie): dispatch AIE kernels loaded via HSA executable APIs"
```

---

## Task 10: Optional-PDI coverage

Validate the PDI-less path at the format/descriptor level (a PDI-less kernel is not runnable by the current vector-scalar design, so this asserts load + descriptor, per spec).

**Files:**
- Modify: `rocrtst/suites/aie/test_aie_hsaco.py` (already covers PDI-less packaging in Task 7/8)
- Modify: `rocrtst/suites/aie/dispatch.cc` (add a load-only assertion)

- [ ] **Step 1: Add a load-only test for a PDI-less kernel**

This requires a PDI-less hsaco. Add a third CMake artifact `TEST_HSACO_NOPDI` built with `--kernel "vector_scalar_add:${INSTS_BIN}:32:1"` (4-part spec, no PDI) and compile-def `DEFAULT_HSACO_NOPDI_PATH`. Then:

```cpp
TEST_F(DispatchTest, LoadHsacoNoPdi) {
  hsa_executable_t exe{};
  uint64_t ko = 0;
  ASSERT_TRUE(load_kernel_object(aie_agents.front(), STRINGIFY(DEFAULT_HSACO_NOPDI_PATH),
                                 "vector_scalar_add", &exe, &ko));
  EXPECT_NE(ko, 0u);  // descriptor exists; pdi_dev_addr inside it is 0
  EXPECT_EQ(hsa_executable_destroy(exe), HSA_STATUS_SUCCESS);
}
```

Add the matching CMake artifact + compile def alongside Task 9 Step 4.

- [ ] **Step 2: Build + run**

Run:
```bash
cmake --build build -j"$(nproc)" --target dispatch 2>&1 | tail -10
LD_LIBRARY_PATH=/opt/rocm/lib ./build/dispatch --gtest_filter='DispatchTest.LoadHsacoNoPdi' 2>&1 | tail -20
```
Expected: PASS (load succeeds; handle non-zero).

- [ ] **Step 3: Commit**

```bash
git add rocrtst/suites/aie/dispatch.cc rocrtst/suites/aie/CMakeLists.txt
git commit -m "test(aie): cover PDI-less kernel load and descriptor"
```

---

## Task 11: Migrate existing dispatch tests to the handle path

The packet ABI change (Task 5) removes the raw `insts_addr`/`insts_size`/`pdi_addr` fields the existing tests set. `SingleDispatch`, `SingleDispatchVMem`, `MultiDispatch`, `MultiDispatchAsync`, `MultiDispatchWrapAround`, and `MultiDispatchWrapAroundAsync` all call `aie_vector_scalar_kernel::dispatch_packet` with `pdi_buf`/`insts_buf`. They will not compile after Task 5.

**Files:**
- Modify: `rocrtst/suites/aie/dispatch.cc`

- [ ] **Step 1: Decide per test — migrate or retire**

Migrate each existing test to load a `kernel_object` via `load_kernel_object` (Task 9) and dispatch with `dispatch_packet_ko`. The old file-based `load_binary` PDI/insts loading is removed from these tests. `SingleDispatchVMem`'s point was vmem I/O buffers — keep that, but source the kernel from the hsaco handle. Retire `LoadPDI`/`LoadInstructions` only if they no longer reflect a supported path; otherwise leave them (they test raw file loading, not dispatch, and still exercise the dev pool).

- [ ] **Step 2: Update each dispatch site**

For each test, replace the `load_binary(...pdiPath...)` + `load_binary(...instsPath...)` pair and the `dispatch_packet(pdi_buf, insts_buf, insts_size, ...)` call with a one-time `load_kernel_object(...)` + `dispatch_packet_ko(ko, input_ptr, output_ptr, kernarg_ptr, signal, queue)`. Keep the multi-dispatch loops, wrap-around logic, and verification unchanged.

- [ ] **Step 3: Build**

Run:
```bash
cmake --build build -j"$(nproc)" --target dispatch 2>&1 | tail -20
```
Expected: compiles (no references to removed packet fields remain).

- [ ] **Step 4: Run the full suite (no regressions)**

Run:
```bash
LD_LIBRARY_PATH=/opt/rocm/lib ./build/dispatch 2>&1 | tail -40
```
Expected: all dispatch tests PASS on AIE hardware.

- [ ] **Step 5: Commit**

```bash
git add rocrtst/suites/aie/dispatch.cc
git commit -m "test(aie): migrate existing dispatch tests to kernel_object handle"
```

---

## Self-Review Notes (for the plan author / first reviewer)

Spec-coverage cross-check:
- §1 section format → Task 1 (C header), Task 7 (producer), Task 8 (validator).
- §2 Python tools → Task 7 (packaging), Task 8 (inspection).
- §3 loader (detection, arch validation, device placement, symbols, multi-object, dup-name) → Task 2 (detection/parse), Task 4 (placement/symbols/arch/dup/multi).
- §4 descriptor + ownership + free-at-destroy → Task 4 (Steps 1, 3, 4).
- §5 packet ABI → Task 5.
- §6 load↔submit contract (BO-resolvable alloc; driver deref; arch query) → Task 3 (arch query), Task 4 (alloc), Task 6 (driver deref), Task 9 Step 6 (BO contract verification).
- Testing section → Tasks 8, 9, 10.

Known judgment calls surfaced for the executor:
- `amd::elf::Section` data accessor name (Task 2 Step 2 note).
- Base hsaco template acceptance by the loader's ELF reader (Task 9 Step 4 note) — the spec's top Open Risk.
- Existing `SingleDispatch`/`MultiDispatch*` tests must migrate to the handle path because the packet ABI changed (Task 9 Step 7).
- `SegmentAlloc(CODE_AGENT)` for AIE must yield a driver-registered BO (Task 9 Step 6) — the spec's second Open Risk.
