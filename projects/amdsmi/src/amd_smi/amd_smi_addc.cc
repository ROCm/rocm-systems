// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/amd_smi_addc.h"

#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <sstream>

#include "addc/lib_c.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace {

struct ContextDeleter_t {
  auto operator()(addc_context_t* context) const -> void { addc_context_destroy(context); }
};
using Context_t = std::unique_ptr<addc_context_t, ContextDeleter_t>;

constexpr size_t kInitialAfidCapacity = 16;

// Strict mode rejects a record that would need repair, so a malformed record
// never yields an AFID it did not carry. The context is read-only after
// creation and safe to share across threads.
auto get_context() -> addc_context_t* {
  static const auto context = []() {
    auto options = addc_context_options_t{};
    options.struct_size = sizeof(options);
    options.parse_mode = ADDC_PARSE_MODE_STRICT;
    return Context_t{addc_context_create_with_options(&options)};
  }();
  return context.get();
}

auto make_error_info() -> addc_error_info_t {
  auto info = addc_error_info_t{};
  info.struct_size = sizeof(info);
  return info;
}

auto to_amdsmi_status(addc_status_t status) -> amdsmi_status_t {
  switch (status) {
    case ADDC_STATUS_SUCCESS:
      return AMDSMI_STATUS_SUCCESS;
    case ADDC_STATUS_INVALID_ARGUMENT:
      return AMDSMI_STATUS_INVAL;
    case ADDC_STATUS_PARSE_ERROR:
    case ADDC_STATUS_DECODE_ERROR:
      return AMDSMI_STATUS_UNEXPECTED_DATA;
    case ADDC_STATUS_INSUFFICIENT_SIZE:
      return AMDSMI_STATUS_INSUFFICIENT_SIZE;
    case ADDC_STATUS_OUT_OF_RESOURCES:
      return AMDSMI_STATUS_OUT_OF_RESOURCES;
    case ADDC_STATUS_INTERNAL_ERROR:
      return AMDSMI_STATUS_INTERNAL_EXCEPTION;
    case ADDC_STATUS_UNSUPPORTED:
      return AMDSMI_STATUS_NOT_SUPPORTED;
    case ADDC_STATUS_IO_ERROR:
      return AMDSMI_STATUS_IO;
  }
  return AMDSMI_STATUS_UNKNOWN_ERROR;
}

auto log_failure(const char* operation, addc_status_t status, const addc_error_info_t& info)
    -> void {
  auto ss = std::ostringstream{};
  ss << "[CPER] " << operation << " failed: " << addc_status_string(status)
     << ", stage: " << info.source;
  if ((info.flags & ADDC_ERROR_INFO_HAS_OFFSET) != 0) {
    ss << ", byte offset: " << info.byte_offset;
  }
  ss << ", " << info.message << "\n";
  LOG_ERROR(ss);
}

// addc reports a size problem as a parse error; the public API reports UNEXPECTED_SIZE.
auto check_record_size(const void* cper, size_t size) -> amdsmi_status_t {
  auto ss = std::ostringstream{};
  if ((cper == nullptr) || (size == 0)) {
    ss << "[CPER] record buffer must be non-null and non-empty\n";
    LOG_ERROR(ss);
    return AMDSMI_STATUS_INVAL;
  }
  if (size < sizeof(amdsmi_cper_hdr_t)) {
    ss << "[CPER] buffer size " << size << " is smaller than the header ("
       << sizeof(amdsmi_cper_hdr_t) << ")\n";
    LOG_ERROR(ss);
    return AMDSMI_STATUS_UNEXPECTED_SIZE;
  }
  const auto* header = static_cast<const amdsmi_cper_hdr_t*>(cper);
  if ((header->record_length < sizeof(amdsmi_cper_hdr_t)) || (header->record_length > size)) {
    ss << "[CPER] record length " << header->record_length << " does not fit the buffer size "
       << size << "\n";
    LOG_ERROR(ss);
    return AMDSMI_STATUS_UNEXPECTED_SIZE;
  }
  return AMDSMI_STATUS_SUCCESS;
}

}  // namespace

auto cper_get_afids(const void* cper, size_t size, std::vector<int>* afids) -> amdsmi_status_t {
  afids->clear();
  const auto record_status = check_record_size(cper, size);
  if (record_status != AMDSMI_STATUS_SUCCESS) {
    return record_status;
  }
  auto* context = get_context();
  if (context == nullptr) {
    return AMDSMI_STATUS_OUT_OF_RESOURCES;
  }

  const auto* bytes = static_cast<const uint8_t*>(cper);
  try {
    // Most records yield a few AFIDs; grow only when addc reports a larger count.
    auto entries = std::vector<addc_cper_summary_entry_t>(kInitialAfidCapacity);
    auto count = entries.size();
    auto info = make_error_info();
    auto status =
        addc_cper_get_summary_ex(context, bytes, size, nullptr, entries.data(), &count, &info);
    if (status == ADDC_STATUS_INSUFFICIENT_SIZE) {
      entries.assign(count, addc_cper_summary_entry_t{});
      info = make_error_info();
      status =
          addc_cper_get_summary_ex(context, bytes, size, nullptr, entries.data(), &count, &info);
      // The record is the same on both calls, so a second size request means addc misreported
      // the count.
      if (status == ADDC_STATUS_INSUFFICIENT_SIZE) {
        log_failure("AFID decode", status, info);
        return AMDSMI_STATUS_INTERNAL_EXCEPTION;
      }
    }
    if (status != ADDC_STATUS_SUCCESS) {
      log_failure("AFID decode", status, info);
      return to_amdsmi_status(status);
    }
    // Reserve first so an allocation failure cannot leave the list half filled.
    afids->reserve(count);
    for (auto i = size_t{0}; i < count; ++i) {
      afids->push_back(entries[i].afid);
    }
  } catch (const std::bad_alloc&) {
    // Not logged: formatting a message would allocate again.
    return AMDSMI_STATUS_OUT_OF_RESOURCES;
  } catch (const std::exception& error) {
    auto ss = std::ostringstream{};
    ss << "[CPER] AFID decode raised an exception: " << error.what() << "\n";
    LOG_ERROR(ss);
    return AMDSMI_STATUS_INTERNAL_EXCEPTION;
  }
  return AMDSMI_STATUS_SUCCESS;
}

auto cper_get_report_json(const void* cper, size_t size, char* json, uint32_t* json_size)
    -> amdsmi_status_t {
  if (json_size == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  const auto record_status = check_record_size(cper, size);
  if (record_status != AMDSMI_STATUS_SUCCESS) {
    return record_status;
  }
  auto* context = get_context();
  if (context == nullptr) {
    return AMDSMI_STATUS_OUT_OF_RESOURCES;
  }

  auto capacity = static_cast<size_t>(*json_size);
  auto info = make_error_info();
  const auto status = addc_cper_decode_json_ex(context, static_cast<const uint8_t*>(cper), size,
                                               nullptr, json, &capacity, &info);
  // A too-small buffer is the expected first step of the two-call contract, not a failure.
  const auto is_size_reported =
      ((status == ADDC_STATUS_SUCCESS) || (status == ADDC_STATUS_INSUFFICIENT_SIZE));
  if (!is_size_reported) {
    log_failure("JSON report", status, info);
    return to_amdsmi_status(status);
  }
  if (capacity > std::numeric_limits<uint32_t>::max()) {
    return AMDSMI_STATUS_OUT_OF_RESOURCES;
  }
  *json_size = static_cast<uint32_t>(capacity);
  return to_amdsmi_status(status);
}
