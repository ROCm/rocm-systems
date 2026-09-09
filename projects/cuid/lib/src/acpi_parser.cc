// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "src/acpi_parser.h"

#include <errno.h>
#include <sys/stat.h>

#include <cstring>
#include <fstream>
#include <type_traits>

// MADT entry type constants
constexpr uint8_t MADT_TYPE_LOCAL_APIC = 0;
constexpr uint8_t MADT_TYPE_LOCAL_X2APIC = 9;

// MADT flags
constexpr uint32_t MADT_FLAG_ENABLED = 0x00000001;

// ACPI table path
constexpr const char* ACPI_TABLES_PATH = "/sys/firmware/acpi/tables/";

amdcuid_status_t AcpiParser::read_acpi_table(const char* table_name, std::vector<uint8_t>& data) {
  std::string path = std::string(ACPI_TABLES_PATH) + table_name;

  // Check if file exists and get size
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    if (errno == ENOENT) {
      return AMDCUID_STATUS_ACPI_ERROR;
    } else if (errno == EACCES) {
      return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    return AMDCUID_STATUS_FILE_ERROR;
  }

  // Read table data
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    if (errno == EACCES) {
      return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    return AMDCUID_STATUS_FILE_ERROR;
  }

  data.resize(st.st_size);
  file.read(reinterpret_cast<char*>(data.data()), st.st_size);

  if (!file) {
    return AMDCUID_STATUS_FILE_ERROR;
  }

  return AMDCUID_STATUS_SUCCESS;
}

bool AcpiParser::validate_checksum(const uint8_t* data, size_t length) {
  if (!data || length < sizeof(AcpiTableHeader)) {
    return false;
  }

  uint8_t sum = 0;
  for (size_t i = 0; i < length; i++) {
    sum += data[i];
  }

  return sum == 0;
}

namespace {

// Copy a packed firmware structure out of a byte buffer. Taking a copy (rather
// than reinterpret_cast'ing in place) keeps the read inside `src` and keeps the
// access well-defined regardless of the buffer's provenance or alignment.
// Returns false when the buffer does not hold a whole T.
template <typename T>
bool read_struct(const uint8_t* src, size_t available, T& out) {
  static_assert(std::is_trivially_copyable<T>::value,
                "read_struct requires a trivially copyable firmware struct");
  if (src == nullptr || available < sizeof(T)) {
    return false;
  }
  std::memcpy(&out, src, sizeof(T));
  return true;
}

}  // namespace

bool AcpiParser::parse_madt_entry(const uint8_t* entry, size_t entry_len, AcpiCpuInfo& cpu_info) {
  MadtEntryHeader header{};
  if (!read_struct(entry, entry_len, header)) {
    return false;
  }

  if (header.type == MADT_TYPE_LOCAL_APIC) {
    MadtLocalApic apic{};
    // header.length is what the firmware claims; entry_len is what the
    // caller proved is actually present. Both must cover the struct.
    if (header.length < sizeof(MadtLocalApic) || !read_struct(entry, entry_len, apic)) {
      return false;
    }

    cpu_info.apic_id = apic.apic_id;
    cpu_info.processor_uid = apic.acpi_processor_uid;
    cpu_info.enabled = (apic.flags & MADT_FLAG_ENABLED) != 0;
    cpu_info.is_x2apic = false;

    return true;
  } else if (header.type == MADT_TYPE_LOCAL_X2APIC) {
    MadtLocalX2Apic x2apic{};
    if (header.length < sizeof(MadtLocalX2Apic) || !read_struct(entry, entry_len, x2apic)) {
      return false;
    }

    cpu_info.apic_id = x2apic.x2apic_id;
    cpu_info.processor_uid = x2apic.acpi_processor_uid;
    cpu_info.enabled = (x2apic.flags & MADT_FLAG_ENABLED) != 0;
    cpu_info.is_x2apic = true;

    return true;
  }

  return false;
}

amdcuid_status_t AcpiParser::parse_madt_buffer(const uint8_t* data, size_t size,
                                               std::vector<AcpiCpuInfo>& cpu_info) {
  cpu_info.clear();

  // Validate minimum size
  MadtHeader madt{};
  if (!read_struct(data, size, madt)) {
    return AMDCUID_STATUS_INVALID_FORMAT;
  }

  // Validate signature
  if (std::memcmp(madt.header.signature, "APIC", 4) != 0) {
    return AMDCUID_STATUS_INVALID_FORMAT;
  }

  // Validate table length
  if (madt.header.length != size) {
    return AMDCUID_STATUS_INVALID_FORMAT;
  }

  // Validate checksum
  if (!validate_checksum(data, size)) {
    return AMDCUID_STATUS_INVALID_FORMAT;
  }

  // Walk the interrupt controller structures using offsets rather than
  // pointers, so no out-of-range pointer is ever formed and the arithmetic
  // cannot wrap.
  size_t offset = sizeof(MadtHeader);

  while (offset < size) {
    // A 2-byte entry header must be fully present before its `length`
    // field can be read. The previous code tested only `entry < end`, so a
    // table whose last byte started an entry read length one byte past the
    // buffer.
    MadtEntryHeader header{};
    if (!read_struct(data + offset, size - offset, header)) {
      return AMDCUID_STATUS_INVALID_FORMAT;
    }

    // Validate the entry neither overflows the table nor fails to advance.
    if (header.length < sizeof(MadtEntryHeader) || header.length > size - offset) {
      return AMDCUID_STATUS_INVALID_FORMAT;
    }

    // Parse Local APIC or x2APIC entries
    AcpiCpuInfo info{};
    if (parse_madt_entry(data + offset, header.length, info)) {
      cpu_info.push_back(info);
    }

    offset += header.length;
  }

  // Should have found at least one CPU
  if (cpu_info.empty()) {
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
  }

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t AcpiParser::parse_madt(std::vector<AcpiCpuInfo>& cpu_info) {
  cpu_info.clear();

  // Read MADT/APIC table
  std::vector<uint8_t> table_data;
  amdcuid_status_t status = read_acpi_table("APIC", table_data);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  return parse_madt_buffer(table_data.data(), table_data.size(), cpu_info);
}

amdcuid_status_t AcpiParser::get_cpu_count(uint32_t& count) {
  std::vector<AcpiCpuInfo> cpu_info;
  amdcuid_status_t status = parse_madt(cpu_info);

  if (status != AMDCUID_STATUS_SUCCESS) {
    count = 0;
    return status;
  }

  // Count only enabled CPUs
  count = 0;
  for (const auto& info : cpu_info) {
    if (info.enabled) {
      count++;
    }
  }

  return AMDCUID_STATUS_SUCCESS;
}
