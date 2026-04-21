// htl_record.hpp — on-disk binary format for hip-trace-lite.
#pragma once

#include <cstdint>
#include <cstring>

namespace htl {

inline constexpr char     kFileMagic[4]   = {'H', 'T', 'L', '0'};
inline constexpr uint32_t kFileVersion    = 1;
inline constexpr uint32_t kHeaderSize     = 64;

// Sentinel for record_t.kernel_name_off when the record has no name.
// Using UINT64_MAX avoids ambiguity with the legitimate offset 0 (which
// is the byte offset of the first string in the string section).
inline constexpr uint64_t kNoStringOffset = static_cast<uint64_t>(-1);

#pragma pack(push, 1)
struct file_header_t {
    char     magic[4];        // "HTL0"
    uint32_t version;         // kFileVersion
    uint64_t start_ns;        // CLOCK_MONOTONIC ns at file open
    uint64_t pid;
    uint32_t record_size;     // sizeof(record_t)
    uint32_t header_size;     // kHeaderSize
    uint64_t string_section_offset;  // 0 until shutdown; filled at close
    uint64_t string_section_size;    // 0 until shutdown
    uint8_t  reserved[16];
};
static_assert(sizeof(file_header_t) == 64, "file_header_t must be 64 bytes");

struct record_t {
    uint8_t  domain;          // activity_domain_t
    uint8_t  op;              // hip_op_id_t (or HIP_API id)
    uint16_t flags;
    uint32_t reserved0;       // padding for 8-byte alignment of correlation_id
    uint64_t correlation_id;  // matches CLR's activity_correlation_id_t (u64)
    uint64_t begin_ns;
    uint64_t end_ns;
    uint32_t process_id;      // valid for HIP_API records
    uint32_t thread_id;       // valid for HIP_API records (or cached gettid for HIP_OPS)
    int32_t  device_id;       // valid for HIP_OPS records (-1 otherwise)
    uint32_t queue_id;        // valid for HIP_OPS records; truncated low 32 bits of CLR's uint64_t
    uint64_t bytes;           // valid for HIP_OPS COPY ops; 0 otherwise
    uint64_t kernel_name_off; // byte offset into trailing string section; 0 if none
};
static_assert(sizeof(record_t) == 64, "record_t must stay 64 bytes");
#pragma pack(pop)

// Footer (written after the string section; lets the decoder report drops):
struct file_footer_t {
    uint64_t records_written;
    uint64_t records_dropped;
    uint8_t  reserved[16];
};

}  // namespace htl
