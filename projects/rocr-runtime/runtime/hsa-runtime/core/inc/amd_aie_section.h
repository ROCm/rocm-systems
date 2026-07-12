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

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_SECTION_H_
