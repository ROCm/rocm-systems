/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "amd_smi/impl/amd_smi_fabric_ualink.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "amd_smi/impl/amd_smi_common.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "rocm_smi/rocm_smi_logger.h"
#include "rocm_smi/rocm_smi_utils.h"

namespace amd::smi {

namespace fabric_ualink {

namespace {

constexpr auto kPpodValidMask =
    (AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID | AMDSMI_FABRIC_PPOD_FIELD_PPOD_ID |
     AMDSMI_FABRIC_PPOD_FIELD_PPOD_SIZE | AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS |
     AMDSMI_FABRIC_PPOD_FIELD_BANDWIDTH | AMDSMI_FABRIC_PPOD_FIELD_LATENCY);

constexpr auto kVpodValidMask =
    (AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID | AMDSMI_FABRIC_VPOD_FIELD_VPOD_SIZE |
     AMDSMI_FABRIC_VPOD_FIELD_VPOD_ACTIVE_ACCELS | AMDSMI_FABRIC_VPOD_FIELD_ADDR_MODE);

constexpr auto kStationValidMask =
    (AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS | AMDSMI_FABRIC_DF_FIELD_LANE_EN_BITMAP |
     AMDSMI_FABRIC_DF_FIELD_NUM_STATIONS);

// Driver commit payload (confirm with kernel if this changes).
constexpr auto kFabricCommitPayload = "1";

auto get_ualink_root(const AMDSmiGPUDevice& device) -> std::string {
  return (std::string(kUALOE_BASE_PATH) + device.get_gpu_path() +
          std::string(kUALOE_UALINK_DIRECTORY));
}

auto subdirectory_exists(const std::string& ualink_root, std::string_view subdir) -> bool {
  const auto path = (std::filesystem::path(ualink_root) / subdir);
  auto ec = std::error_code{};
  return std::filesystem::is_directory(path, ec);
}

auto write_sysfs_string(const std::string& sysfs_path, const std::string& payload)
    -> amdsmi_status_t {
  auto outstream = std::ostringstream();
  outstream << __func__ << " | path: " << sysfs_path << " | payload: " << payload;
  LOG_DEBUG(outstream);

  /**
   *    std::ofstream does not guarantee errno on a failed open, so clear it first
   *    to avoid mapping a stale value, and fall back to a generic file error when
   *    the open fails without setting errno.
   */
  errno = 0;
  auto sysfs_file_stream = std::ofstream(sysfs_path);
  if (!sysfs_file_stream.good()) {
    const auto saved_errno = errno;
    errno = 0;
    auto err = std::ostringstream();
    err << __func__ << " | open failed | path: " << sysfs_path << " | errno: " << saved_errno;
    LOG_ERROR(err);
    return ((saved_errno != 0) ? rsmi_to_amdsmi_status(ErrnoToRsmiStatus(saved_errno))
                               : amdsmi_status_t::AMDSMI_STATUS_FILE_ERROR);
  }

  // Buffered writes surface a driver rejection at flush/close, not open. Clear
  // errno first so the logged value reflects this write and not a stale one.
  errno = 0;
  sysfs_file_stream << payload << "\n";
  sysfs_file_stream.flush();
  if (!sysfs_file_stream.good()) {
    auto err = std::ostringstream();
    err << __func__ << " | flush failed | path: " << sysfs_path << " | errno: " << errno;
    LOG_ERROR(err);
    return amdsmi_status_t::AMDSMI_STATUS_FILE_ERROR;
  }

  errno = 0;
  sysfs_file_stream.close();
  if (!sysfs_file_stream.good()) {
    auto err = std::ostringstream();
    err << __func__ << " | close failed | path: " << sysfs_path << " | errno: " << errno;
    LOG_ERROR(err);
    return amdsmi_status_t::AMDSMI_STATUS_FILE_ERROR;
  }

  return amdsmi_status_t::AMDSMI_STATUS_SUCCESS;
}

auto join_subdirectory_file(std::string_view subdir, std::string_view filename) -> std::string {
  return (std::filesystem::path(subdir) / filename).string();
}

auto format_u32_decimal(uint32_t value) -> std::string { return (std::to_string(value)); }

auto format_ppod_id(const uint8_t* ppod_id, std::size_t len) -> std::string {
  auto outstream = std::ostringstream();
  outstream << "0x";
  for (auto idx = std::size_t(0); idx < len; ++idx) {
    outstream << std::hex << std::setfill('0') << std::setw(2)
              << static_cast<std::uint32_t>(ppod_id[idx]);
  }

  return outstream.str();
}

auto format_accel_list_space_separated(const uint32_t* accels, uint32_t count) -> std::string {
  auto outstream = std::ostringstream();
  for (auto idx = uint32_t(0); idx < count; ++idx) {
    if (idx > 0) {
      outstream << ' ';
    }
    outstream << accels[idx];
  }
  return outstream.str();
}

// Accelerator IDs are documented as 0..1023 (see amdsmi_fabric_ppod_data_t).
constexpr auto kMaxAcceleratorId = uint32_t(1023);

// Length of the leading accelerator-ID run, ending at the first UNSET sentinel.
// Shared by the serializer and the range validator so both agree on the run.
auto vpod_active_run_length(const uint32_t* accels, std::size_t capacity) -> std::size_t {
  auto count = std::size_t(0);
  while ((count < capacity) && (accels[count] != std::numeric_limits<uint32_t>::max())) {
    ++count;
  }
  return count;
}

// True when every ID in [accels, accels + count) is within the valid range.
auto accel_ids_in_range(const uint32_t* accels, std::size_t count) -> bool {
  for (auto idx = std::size_t(0); idx < count; ++idx) {
    if (accels[idx] > kMaxAcceleratorId) {
      return false;
    }
  }
  return true;
}

// Guard byte marking an unset ppod_id; the readback fill uses the same value so
// a caller echoing back a fully-unset readback can be rejected on write.
constexpr auto kPpodIdUnsetByte = uint8_t(0x99);

// True when every byte of the physical pod ID equals `value`.
auto ppod_id_all_bytes(const uint8_t* ppod_id, std::size_t len, uint8_t value) -> bool {
  for (auto idx = std::size_t(0); idx < len; ++idx) {
    if (ppod_id[idx] != value) {
      return false;
    }
  }
  return true;
}

/**
 *  Serializes vpod_active_accelerators as a space-separated decimal ID list, the
 *  inverse of the read path. There is no count field: unused slots hold the UNSET
 *  sentinel (UINT32_MAX), so emit only the leading run up to the first sentinel.
 *  0 is a valid accelerator ID and must not terminate the run.
 */
auto format_vpod_active_accelerators(const uint32_t* accels, std::size_t capacity) -> std::string {
  const auto count = static_cast<uint32_t>(vpod_active_run_length(accels, capacity));
  return format_accel_list_space_separated(accels, count);
}

auto format_addr_mode(amdsmi_fabric_npa_address_mode_t mode) -> std::optional<std::string> {
  switch (mode) {
    case AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_ALIASING:
      return std::string("aliasing");
    case AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_IDENTIFICATION:
      return std::string("identification");
    default:
      return std::nullopt;
  }
}

auto format_lane_en_bitmap(const uint8_t* bytes, std::size_t count) -> std::string {
  auto outstream = std::ostringstream();
  outstream << "0x";
  for (auto idx = std::size_t(0); idx < count; ++idx) {
    outstream << std::hex << std::setfill('0') << std::setw(2) << static_cast<uint32_t>(bytes[idx]);
  }
  return outstream.str();
}

auto validate_request_common(uint32_t version, uint32_t expected_version, uint32_t mask,
                             uint32_t valid_mask) -> amdsmi_status_t {
  if (version != expected_version) {
    return amdsmi_status_t::AMDSMI_STATUS_INVAL;
  }

  /**
   *  A commit with nothing freshly staged would flush whatever a prior partial write left
   *  behind, so every request must name at least one field
   */
  if (mask == 0) {
    return amdsmi_status_t::AMDSMI_STATUS_INVAL;
  }
  if ((mask & ~valid_mask) != 0) {
    return amdsmi_status_t::AMDSMI_STATUS_INVAL;
  }

  return amdsmi_status_t::AMDSMI_STATUS_SUCCESS;
}

auto write_field(const std::string& ualink_root, std::string_view subdir, std::string_view filename,
                 const std::string& payload) -> amdsmi_status_t {
  const auto relative_path = join_subdirectory_file(subdir, filename);
  const auto full_path = (ualink_root + "/" + relative_path);

  return write_sysfs_string(full_path, payload);
}

auto write_commit(const std::string& ualink_root, std::string_view subdir) -> amdsmi_status_t {
  return write_field(ualink_root, subdir, kUALOE_UALINK_COMMIT_FILE, kFabricCommitPayload);
}

/**
 *  A single deferred sysfs write produced during the build phase
 *      - Formatting and input validation happen while the plan is assembled
 *      - Nothing reaches sysfs until ::flush_writes iterates the completed list
 */
struct WriteRequest_t {
  std::string_view m_subdir;
  std::string_view m_file;
  std::string m_payload;
};
using WriteRequestList_t = std::vector<WriteRequest_t>;

/**
 *  Accumulates masked sysfs writes into one subtree's request list
 *      - Binds the list, the request mask, and the target @p subdir once
 *      - ::add_if appends only when the field's mask bit is set, keeping the
 *        per-field call sites down to (bit, file, payload)
 */
struct WriteRequestBuilder_t {
  WriteRequestList_t& m_list;
  uint32_t m_mask;
  std::string_view m_subdir;

  auto add_if(uint32_t bit, std::string_view file, std::string payload) -> void {
    if (((m_mask & bit) != 0)) {
      m_list.push_back(WriteRequest_t{m_subdir, file, std::move(payload)});
    }
  }
};

/**
 *  Flushes a built write data request, then writes the subtree commit
 *      - All entries share one subtree, so commit targets that same @p subdir
 *      - A mid-flush hardware failure could still leave partial state; the
 *        build/flush split only removes avoidable partial state from bad input
 *      - The driver ignores the staging files until a commit follows, so a non-committing
 *        request stops at validation instead of leaving values for the next commit to apply
 */
auto flush_writes(const std::string& ualink_root, const WriteRequestList_t& request_list,
                  std::string_view subdir, bool commit) -> amdsmi_status_t {
  if (!commit) {
    return amdsmi_status_t::AMDSMI_STATUS_SUCCESS;
  }

  for (const auto& request : request_list) {
    if (auto status = write_field(ualink_root, request.m_subdir, request.m_file, request.m_payload);
        status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
      return status;
    }
  }

  return write_commit(ualink_root, subdir);
}

/**
 *  Reads a single sysfs attribute, returning its trimmed content
 *      - Multi-line files (e.g. accelerator lists) are joined with single spaces
 *      - Returns nullopt when the file is absent, unreadable, or empty after trimming
 */
auto read_field(const std::string& ualink_root, std::string_view subdir, std::string_view filename)
    -> std::optional<std::string> {
  const auto relative_path = join_subdirectory_file(subdir, filename);
  const auto full_path = (ualink_root + "/" + relative_path);

  errno = 0;
  auto sysfs_file_stream = std::ifstream(full_path);
  if (!sysfs_file_stream.is_open()) {
    const auto rsmi_status = ErrnoToRsmiStatus(errno);
    errno = 0;
    auto outstream = std::ostringstream();
    outstream << __func__ << " | failed to open: " << full_path << " | status: " << rsmi_status;
    LOG_DEBUG(outstream);
    return std::nullopt;
  }

  auto joined_lines = std::string();
  auto line = std::string();
  while (std::getline(sysfs_file_stream, line)) {
    if (!joined_lines.empty()) {
      joined_lines += ' ';
    }
    joined_lines += line;
  }

  const auto trimmed = trim(joined_lines);
  if (trimmed.empty()) {
    return std::nullopt;
  }

  return trimmed;
}

auto parse_u32_decimal(const std::string& text) -> std::optional<uint32_t> {
  auto value = std::uint64_t(0);
  auto instream = std::istringstream(text);
  if (!(instream >> value)) {
    return std::nullopt;
  }
  if ((value > std::numeric_limits<uint32_t>::max())) {
    return std::nullopt;
  }
  // Reject trailing non-whitespace so "12abc" or "1 2" do not parse as 12.
  auto trailing = char{};
  if (instream >> trailing) {
    return std::nullopt;
  }

  return static_cast<uint32_t>(value);
}

/**
 *  Parses "0x"-prefixed hex into bytes (inverse of format_ppod_id / format_lane_en_bitmap)
 *      - Separators between bytes ('-', ':', whitespace) are skipped so a UUID-style readback
 *        parses like the packed form the writer emits
 *      - Each byte is exactly two hex digits; an odd digit count or overflow past @p max fails
 *      - Returns the number of bytes written, or nullopt on any malformed input
 */
auto parse_hex_bytes(const std::string& text, uint8_t* out, std::size_t max)
    -> std::optional<std::size_t> {
  auto body = std::string_view(text);
  if (((body.size() >= 2) && (body[0] == '0') && ((body[1] == 'x') || (body[1] == 'X')))) {
    body.remove_prefix(2);
  }

  auto packed = std::string();
  packed.reserve(body.size());
  for (const auto ch : body) {
    if (((ch == '-') || (ch == ':') || (std::isspace(static_cast<unsigned char>(ch)) != 0))) {
      continue;
    }
    packed.push_back(ch);
  }

  if (((packed.size() % 2) != 0)) {
    return std::nullopt;
  }

  const auto byte_count = (packed.size() / 2);
  if ((byte_count > max)) {
    return std::nullopt;
  }

  /**
   *    Reject the whole string before writing a byte: bailing out mid-loop would leave the
   *    caller's buffer part sentinel, part payload, behind a mask bit reported as unread
   */
  for (const auto ch : packed) {
    if ((std::isxdigit(static_cast<unsigned char>(ch)) == 0)) {
      return std::nullopt;
    }
  }

  const auto hex_value = [](char ch) -> int {
    if (((ch >= '0') && (ch <= '9'))) {
      return (ch - '0');
    }
    if (((ch >= 'a') && (ch <= 'f'))) {
      return ((ch - 'a') + 10);
    }
    return ((ch - 'A') + 10);
  };

  for (auto idx = std::size_t(0); idx < byte_count; ++idx) {
    out[idx] = static_cast<uint8_t>(
        ((hex_value(packed[(idx * 2)]) << 4) | hex_value(packed[((idx * 2) + 1)])));
  }

  return byte_count;
}

/**
 *  Parses a space-separated decimal list (inverse of format_accel_list_space_separated)
 *      - Returns std::nullopt and leaves @p out untouched on a malformed token, a value
 *        exceeding uint32_t, or more than @p max values
 *      - A list too long to hold is reported absent rather than truncated: with no count
 *        field, a truncated list is indistinguishable from a complete one
 */
auto parse_u32_list(const std::string& text, uint32_t* out, std::size_t max)
    -> std::optional<std::size_t> {
  auto instream = std::istringstream(text);
  auto token = std::uint64_t(0);
  auto values = std::vector<uint32_t>();
  while ((instream >> token)) {
    if (((token > std::numeric_limits<uint32_t>::max()) || (values.size() >= max))) {
      return std::nullopt;
    }
    values.push_back(static_cast<uint32_t>(token));
  }
  if (!instream.eof()) {
    return std::nullopt;
  }

  std::copy(values.begin(), values.end(), out);
  return values.size();
}

auto parse_addr_mode(const std::string& text) -> std::optional<amdsmi_fabric_npa_address_mode_t> {
  auto lowered = text;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

  if ((lowered == "aliasing")) {
    return AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_ALIASING;
  }
  if ((lowered == "identification")) {
    return AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_IDENTIFICATION;
  }

  return std::nullopt;
}

auto validate_read_request(uint32_t version, uint32_t expected_version, uint32_t mask,
                           uint32_t valid_mask) -> amdsmi_status_t {
  if (version != expected_version) {
    return amdsmi_status_t::AMDSMI_STATUS_INVAL;
  }
  if (mask == 0) {
    return amdsmi_status_t::AMDSMI_STATUS_INVAL;
  }
  if ((mask & ~valid_mask) != 0) {
    return amdsmi_status_t::AMDSMI_STATUS_INVAL;
  }

  return amdsmi_status_t::AMDSMI_STATUS_SUCCESS;
}

/**
 *  Reads masked sysfs fields from one subtree, recording which fields landed
 *      - Binds the root, the requested mask, the @p subdir, and the result
 *        accumulator once so per-field call sites stay (bit, file, assign)
 *      - @p assign parses the raw value into @p config and returns whether it
 *        succeeded; only then is the field's mask bit recorded in m_fields_read
 */
struct FieldReader_t {
  const std::string& m_ualink_root;
  uint32_t m_requested_mask;
  std::string_view m_subdir;
  uint32_t& m_fields_read;

  template <typename AssignTp>
  auto read(uint32_t bit, std::string_view file, AssignTp&& assign) -> void {
    if (((m_requested_mask & bit) == 0)) {
      return;
    }
    if (const auto raw = read_field(m_ualink_root, m_subdir, file);
        raw.has_value() && assign(raw.value())) {
      m_fields_read |= bit;
    }
  }
};

enum class SurfacePhase_t { PreStage, PostCommit };

}  // namespace

/**
 *  Surface diagnostics: Re-read fields from both the write subtree and the flat surface and
 *  LOG_DEBUG any that disagree (value or presence).
 *
 *  PreStage runs before a committing request writes anything, over the full valid mask, so a
 *  disagreement means an earlier partial write left fields staged that this commit is about to
 *  apply. PostCommit runs afterwards over the committed fields only.
 *
 *  Purely observational: It never changes the apply result.
 *  The comparison runs at every log level so the sysfs read path does not vary with logging
 *  state; only the report itself is gated, inside LOG_DEBUG.
 */
auto log_surface_divergence(const AMDSmiGPUDevice& device, const amdsmi_fabric_ppod_config_t& probe,
                            SurfacePhase_t phase) -> void;
auto log_surface_divergence(const AMDSmiGPUDevice& device, const amdsmi_fabric_vpod_config_t& probe,
                            SurfacePhase_t phase) -> void;
auto log_surface_divergence(const AMDSmiGPUDevice& device,
                            const amdsmi_fabric_station_config_t& probe, SurfacePhase_t phase)
    -> void;

auto apply_ppod_config_at(const std::string& ualink_root, bool device_supports_ualink,
                          const amdsmi_fabric_ppod_config_t& config) -> amdsmi_status_t {
  if (auto status = validate_request_common(config.version, AMDSMI_FABRIC_PPOD_CONFIG_V1,
                                            config.mask, kPpodValidMask);
      status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
    return status;
  }

  /**
   *    Pure input validation runs before the hardware-capability gate so a malformed
   *    request is reported as AMDSMI_STATUS_INVAL regardless of whether this system
   *    exposes UALink (mirrors validate_request_common above)
   */
  if ((config.mask & AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID) != 0) {
    if (config.data.accelerator_id > kMaxAcceleratorId) {
      return amdsmi_status_t::AMDSMI_STATUS_INVAL;
    }
  }

  if ((config.mask & AMDSMI_FABRIC_PPOD_FIELD_PPOD_ID) != 0) {
    // All-zero is not a valid pod ID; all-guard-byte is the unset sentinel echoed back.
    if (ppod_id_all_bytes(config.data.ppod_id, AMDSMI_MAX_UUID_ELEMENTS, 0) ||
        ppod_id_all_bytes(config.data.ppod_id, AMDSMI_MAX_UUID_ELEMENTS, kPpodIdUnsetByte)) {
      return amdsmi_status_t::AMDSMI_STATUS_INVAL;
    }
  }

  if ((config.mask & AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS) != 0) {
    if (((config.data.local_accelerator_count == 0)) ||
        ((config.data.local_accelerator_count > AMDSMI_FABRIC_MAX_LOCAL_GPUS))) {
      return amdsmi_status_t::AMDSMI_STATUS_INVAL;
    }
    if (!accel_ids_in_range(config.data.local_accelerators, config.data.local_accelerator_count)) {
      return amdsmi_status_t::AMDSMI_STATUS_INVAL;
    }
  }

  if (!device_supports_ualink) {
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  if (!subdirectory_exists(ualink_root, kUALOE_UALINK_SETUP_SUBDIR)) {
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  auto request_list = WriteRequestList_t{};
  auto writer = WriteRequestBuilder_t{request_list, config.mask, kUALOE_UALINK_SETUP_SUBDIR};

  writer.add_if(AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID, kUALOE_ACCEL_ID,
                format_u32_decimal(config.data.accelerator_id));
  writer.add_if(AMDSMI_FABRIC_PPOD_FIELD_PPOD_ID, kUALOE_PPOD_ID,
                format_ppod_id(config.data.ppod_id, AMDSMI_MAX_UUID_ELEMENTS));
  writer.add_if(AMDSMI_FABRIC_PPOD_FIELD_PPOD_SIZE, kUALOE_PPOD_SIZE,
                format_u32_decimal(config.data.ppod_size));
  // add_if evaluates its payload argument eagerly, so the count must be gated
  // here: on a masked-out LOCAL_ACCELS write the count is uninitialized and
  // would drive an out-of-bounds read in the formatter.
  const auto local_accel_count = ((config.mask & AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS) != 0)
                                     ? config.data.local_accelerator_count
                                     : 0U;
  writer.add_if(
      AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS, kUALOE_LOCAL_ACCELS,
      format_accel_list_space_separated(config.data.local_accelerators, local_accel_count));
  writer.add_if(AMDSMI_FABRIC_PPOD_FIELD_BANDWIDTH, kUALOE_BANDWIDTH,
                format_u32_decimal(config.data.bandwidth));
  writer.add_if(AMDSMI_FABRIC_PPOD_FIELD_LATENCY, kUALOE_LATENCY,
                format_u32_decimal(config.data.latency));

  return flush_writes(ualink_root, request_list, kUALOE_UALINK_SETUP_SUBDIR, config.commit);
}

auto apply_ppod_config(const AMDSmiGPUDevice& device, const amdsmi_fabric_ppod_config_t& config)
    -> amdsmi_status_t {
  /**
   *    A commit applies everything staged, not only this request's fields, so residue from an
   *    earlier partial write is about to be applied too. Probe before staging, while that residue
   *    is still distinguishable from this request's own writes
   */
  if (config.commit) {
    auto probe = amdsmi_fabric_ppod_config_t{};
    probe.version = AMDSMI_FABRIC_PPOD_CONFIG_V1;
    probe.mask = kPpodValidMask;
    log_surface_divergence(device, probe, SurfacePhase_t::PreStage);
  }

  const auto status =
      apply_ppod_config_at(get_ualink_root(device), device.device_has_ualink(), config);
  if (((status == amdsmi_status_t::AMDSMI_STATUS_SUCCESS) && config.commit)) {
    log_surface_divergence(device, config, SurfacePhase_t::PostCommit);
  }
  return status;
}

auto apply_vpod_config_at(const std::string& ualink_root, bool device_supports_ualink,
                          const amdsmi_fabric_vpod_config_t& config) -> amdsmi_status_t {
  if (auto status = validate_request_common(config.version, AMDSMI_FABRIC_VPOD_CONFIG_V1,
                                            config.mask, kVpodValidMask);
      status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
    return status;
  }

  /**
   *    Pure input validation runs before the hardware-capability gate so a malformed
   *    request is reported as AMDSMI_STATUS_INVAL regardless of whether this system
   *    exposes UALink
   *
   *    The formatted value is reused for the write below
   */
  auto addr_mode_str = std::optional<std::string>{};
  if ((config.mask & AMDSMI_FABRIC_VPOD_FIELD_ADDR_MODE) != 0) {
    addr_mode_str = format_addr_mode(config.data.addr_mode);
    if (!addr_mode_str.has_value()) {
      return amdsmi_status_t::AMDSMI_STATUS_INVAL;
    }
  }

  if ((config.mask & AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID) != 0) {
    if (config.data.vpod_id == 0) {  // 0 is not a valid pod ID
      return amdsmi_status_t::AMDSMI_STATUS_INVAL;
    }
  }

  if ((config.mask & AMDSMI_FABRIC_VPOD_FIELD_VPOD_ACTIVE_ACCELS) != 0) {
    const auto run = vpod_active_run_length(config.data.vpod_active_accelerators,
                                            AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE);
    if (!accel_ids_in_range(config.data.vpod_active_accelerators, run)) {
      return amdsmi_status_t::AMDSMI_STATUS_INVAL;
    }
    // A value past the first sentinel would be silently dropped by the serializer;
    // reject the request rather than truncate the caller's list.
    for (auto idx = run;
         idx < static_cast<std::size_t>(AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE); ++idx) {
      if (config.data.vpod_active_accelerators[idx] != std::numeric_limits<uint32_t>::max()) {
        return amdsmi_status_t::AMDSMI_STATUS_INVAL;
      }
    }
  }

  if (!device_supports_ualink) {
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  if (!subdirectory_exists(ualink_root, kUALOE_UALINK_CONFIG_SUBDIR)) {
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  auto request_list = WriteRequestList_t{};
  auto writer = WriteRequestBuilder_t{request_list, config.mask, kUALOE_UALINK_CONFIG_SUBDIR};

  writer.add_if(AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID, kUALOE_VPOD_ID,
                format_u32_decimal(config.data.vpod_id));
  writer.add_if(AMDSMI_FABRIC_VPOD_FIELD_VPOD_SIZE, kUALOE_VPOD_SIZE,
                format_u32_decimal(config.data.vpod_size));
  writer.add_if(AMDSMI_FABRIC_VPOD_FIELD_VPOD_ACTIVE_ACCELS, kUALOE_VPOD_ACTIVE_ACCELS,
                format_vpod_active_accelerators(config.data.vpod_active_accelerators,
                                                AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE));
  if (addr_mode_str.has_value()) {
    writer.add_if(AMDSMI_FABRIC_VPOD_FIELD_ADDR_MODE, kUALOE_ADDR_MODE,
                  std::move(addr_mode_str.value()));
  }

  return flush_writes(ualink_root, request_list, kUALOE_UALINK_CONFIG_SUBDIR, config.commit);
}

auto apply_vpod_config(const AMDSmiGPUDevice& device, const amdsmi_fabric_vpod_config_t& config)
    -> amdsmi_status_t {
  if (config.commit) {
    auto probe = amdsmi_fabric_vpod_config_t{};
    probe.version = AMDSMI_FABRIC_VPOD_CONFIG_V1;
    probe.mask = kVpodValidMask;
    log_surface_divergence(device, probe, SurfacePhase_t::PreStage);
  }

  const auto status =
      apply_vpod_config_at(get_ualink_root(device), device.device_has_ualink(), config);
  if (((status == amdsmi_status_t::AMDSMI_STATUS_SUCCESS) && config.commit)) {
    log_surface_divergence(device, config, SurfacePhase_t::PostCommit);
  }
  return status;
}

auto apply_station_config_at(const std::string& ualink_root, bool device_supports_ualink,
                             const amdsmi_fabric_station_config_t& config) -> amdsmi_status_t {
  if (auto status = validate_request_common(config.version, AMDSMI_FABRIC_STATION_CONFIG_V1,
                                            config.mask, kStationValidMask);
      status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
    return status;
  }

  if (!device_supports_ualink) {
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  if (!subdirectory_exists(ualink_root, kUALOE_UALINK_STATIONS_SUBDIR)) {
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  /**
   *    Build phase mirrors the ppod/vpod paths, but no station field has a value constraint
   *    checkable here, so the driver is what rejects a bad value at write time
   */
  auto request_list = WriteRequestList_t{};
  auto writer = WriteRequestBuilder_t{request_list, config.mask, kUALOE_UALINK_STATIONS_SUBDIR};

  writer.add_if(AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS, kUALOE_STATION_FLAGS,
                format_u32_decimal(config.data.station_flags));
  writer.add_if(AMDSMI_FABRIC_DF_FIELD_NUM_STATIONS, kUALOE_NUM_STATIONS,
                format_u32_decimal(static_cast<uint32_t>(config.data.num_stations)));
  writer.add_if(AMDSMI_FABRIC_DF_FIELD_LANE_EN_BITMAP, kUALOE_LANE_EN_BITMAP,
                format_lane_en_bitmap(config.data.lane_en_bitmap, AMDSMI_FABRIC_MAX_BITMAP_SIZE));

  return flush_writes(ualink_root, request_list, kUALOE_UALINK_STATIONS_SUBDIR, config.commit);
}

auto apply_station_config(const AMDSmiGPUDevice& device,
                          const amdsmi_fabric_station_config_t& config) -> amdsmi_status_t {
  if (config.commit) {
    auto probe = amdsmi_fabric_station_config_t{};
    probe.version = AMDSMI_FABRIC_STATION_CONFIG_V1;
    probe.mask = kStationValidMask;
    log_surface_divergence(device, probe, SurfacePhase_t::PreStage);
  }

  const auto status =
      apply_station_config_at(get_ualink_root(device), device.device_has_ualink(), config);
  if (((status == amdsmi_status_t::AMDSMI_STATUS_SUCCESS) && config.commit)) {
    log_surface_divergence(device, config, SurfacePhase_t::PostCommit);
  }
  return status;
}

auto query_ppod_config_at(const std::string& ualink_root, bool device_supports_ualink,
                          amdsmi_fabric_ppod_config_t& config, std::string_view subdir)
    -> amdsmi_status_t {
  const auto requested_mask = config.mask;
  if (auto status = validate_read_request(config.version, AMDSMI_FABRIC_PPOD_CONFIG_V1,
                                          requested_mask, kPpodValidMask);
      status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
    return status;
  }

  if (!device_supports_ualink) {
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  if (!subdirectory_exists(ualink_root, subdir)) {
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  /**
   *    Guard values match the flat reader so both surfaces agree on "field absent"
   */
  config.data.accelerator_id = std::numeric_limits<uint32_t>::max();
  std::fill(std::begin(config.data.ppod_id), std::end(config.data.ppod_id), kPpodIdUnsetByte);
  config.data.ppod_size = std::numeric_limits<uint32_t>::max();
  std::fill(std::begin(config.data.local_accelerators), std::end(config.data.local_accelerators),
            std::numeric_limits<uint32_t>::max());
  config.data.local_accelerator_count = 0;
  config.data.bandwidth = std::numeric_limits<uint32_t>::max();
  config.data.latency = std::numeric_limits<uint32_t>::max();

  auto fields_read = uint32_t(0);
  auto reader = FieldReader_t{ualink_root, requested_mask, subdir, fields_read};

  reader.read(AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID, kUALOE_ACCEL_ID,
              [&](const std::string& str_value) {
                const auto parsed_value = parse_u32_decimal(str_value);
                if (parsed_value.has_value()) {
                  config.data.accelerator_id = parsed_value.value();
                }
                return parsed_value.has_value();
              });
  reader.read(AMDSMI_FABRIC_PPOD_FIELD_PPOD_ID, kUALOE_PPOD_ID, [&](const std::string& str_value) {
    return parse_hex_bytes(str_value, config.data.ppod_id, AMDSMI_MAX_UUID_ELEMENTS).has_value();
  });
  reader.read(AMDSMI_FABRIC_PPOD_FIELD_PPOD_SIZE, kUALOE_PPOD_SIZE,
              [&](const std::string& str_value) {
                const auto parsed_value = parse_u32_decimal(str_value);
                if (parsed_value.has_value()) {
                  config.data.ppod_size = parsed_value.value();
                }
                return parsed_value.has_value();
              });
  reader.read(AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS, kUALOE_LOCAL_ACCELS,
              [&](const std::string& str_value) {
                const auto count = parse_u32_list(str_value, config.data.local_accelerators,
                                                  AMDSMI_FABRIC_MAX_LOCAL_GPUS);
                if (!count.has_value()) {
                  return false;
                }
                config.data.local_accelerator_count = static_cast<uint32_t>(count.value());
                return (count.value() > 0);
              });
  reader.read(AMDSMI_FABRIC_PPOD_FIELD_BANDWIDTH, kUALOE_BANDWIDTH,
              [&](const std::string& str_value) {
                const auto parsed_value = parse_u32_decimal(str_value);
                if (parsed_value.has_value()) {
                  config.data.bandwidth = parsed_value.value();
                }
                return parsed_value.has_value();
              });
  reader.read(AMDSMI_FABRIC_PPOD_FIELD_LATENCY, kUALOE_LATENCY, [&](const std::string& str_value) {
    const auto parsed_value = parse_u32_decimal(str_value);
    if (parsed_value.has_value()) {
      config.data.latency = parsed_value.value();
    }
    return parsed_value.has_value();
  });

  config.mask = fields_read;
  return ((fields_read != 0) ? amdsmi_status_t::AMDSMI_STATUS_SUCCESS
                             : amdsmi_status_t::AMDSMI_STATUS_NO_DATA);
}

auto query_ppod_config(const AMDSmiGPUDevice& device, amdsmi_fabric_ppod_config_t& config,
                       std::string_view subdir = kUALOE_UALINK_SETUP_SUBDIR) -> amdsmi_status_t {
  return query_ppod_config_at(get_ualink_root(device), device.device_has_ualink(), config, subdir);
}

auto query_vpod_config_at(const std::string& ualink_root, bool device_supports_ualink,
                          amdsmi_fabric_vpod_config_t& config, std::string_view subdir)
    -> amdsmi_status_t {
  const auto requested_mask = config.mask;
  if (auto status = validate_read_request(config.version, AMDSMI_FABRIC_VPOD_CONFIG_V1,
                                          requested_mask, kVpodValidMask);
      status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
    return status;
  }

  if (!device_supports_ualink) {
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  if (!subdirectory_exists(ualink_root, subdir)) {
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  config.data.vpod_id = std::numeric_limits<uint32_t>::max();
  config.data.vpod_size = std::numeric_limits<uint32_t>::max();
  // Accelerator-ID list, as with local_accelerators: unused slots stay at the UNSET sentinel, so a
  // partial or absent read is not mistaken for accelerator ID 0 being active.
  std::fill(std::begin(config.data.vpod_active_accelerators),
            std::end(config.data.vpod_active_accelerators), std::numeric_limits<uint32_t>::max());
  config.data.addr_mode = AMDSMI_FABRIC_NPA_ADDRESS_MODE_UNKNOWN;

  auto fields_read = uint32_t(0);
  auto reader = FieldReader_t{ualink_root, requested_mask, subdir, fields_read};

  reader.read(AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID, kUALOE_VPOD_ID, [&](const std::string& str_value) {
    const auto parsed_value = parse_u32_decimal(str_value);
    if (parsed_value.has_value()) {
      config.data.vpod_id = parsed_value.value();
    }
    return parsed_value.has_value();
  });
  reader.read(AMDSMI_FABRIC_VPOD_FIELD_VPOD_SIZE, kUALOE_VPOD_SIZE,
              [&](const std::string& str_value) {
                const auto parsed_value = parse_u32_decimal(str_value);
                if (parsed_value.has_value()) {
                  config.data.vpod_size = parsed_value.value();
                }
                return parsed_value.has_value();
              });
  reader.read(AMDSMI_FABRIC_VPOD_FIELD_VPOD_ACTIVE_ACCELS, kUALOE_VPOD_ACTIVE_ACCELS,
              [&](const std::string& str_value) {
                const auto count = parse_u32_list(str_value, config.data.vpod_active_accelerators,
                                                  AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE);
                return (count.has_value() && (count.value() > 0));
              });
  reader.read(AMDSMI_FABRIC_VPOD_FIELD_ADDR_MODE, kUALOE_ADDR_MODE,
              [&](const std::string& str_value) {
                const auto parsed_value = parse_addr_mode(str_value);
                if (parsed_value.has_value()) {
                  config.data.addr_mode = parsed_value.value();
                }
                return parsed_value.has_value();
              });

  config.mask = fields_read;
  return ((fields_read != 0) ? amdsmi_status_t::AMDSMI_STATUS_SUCCESS
                             : amdsmi_status_t::AMDSMI_STATUS_NO_DATA);
}

auto query_vpod_config(const AMDSmiGPUDevice& device, amdsmi_fabric_vpod_config_t& config,
                       std::string_view subdir = kUALOE_UALINK_CONFIG_SUBDIR) -> amdsmi_status_t {
  return query_vpod_config_at(get_ualink_root(device), device.device_has_ualink(), config, subdir);
}

auto query_station_config_at(const std::string& ualink_root, bool device_supports_ualink,
                             amdsmi_fabric_station_config_t& config, std::string_view subdir)
    -> amdsmi_status_t {
  const auto requested_mask = config.mask;
  if (auto status = validate_read_request(config.version, AMDSMI_FABRIC_STATION_CONFIG_V1,
                                          requested_mask, kStationValidMask);
      status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
    return status;
  }

  if (!device_supports_ualink) {
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  if (!subdirectory_exists(ualink_root, subdir)) {
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  config.data.station_flags = std::numeric_limits<uint32_t>::max();
  config.data.num_stations = std::numeric_limits<uint8_t>::max();
  // Bitmap: an unread byte must read as "lane disabled" (0), not all-ones, so a partial or
  // absent read never reports spurious enabled lanes.
  std::fill(std::begin(config.data.lane_en_bitmap), std::end(config.data.lane_en_bitmap),
            static_cast<uint8_t>(0));

  auto fields_read = uint32_t(0);
  auto reader = FieldReader_t{ualink_root, requested_mask, subdir, fields_read};

  reader.read(AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS, kUALOE_STATION_FLAGS,
              [&](const std::string& str_value) {
                const auto parsed_value = parse_u32_decimal(str_value);
                if (parsed_value.has_value()) {
                  config.data.station_flags = parsed_value.value();
                }
                return parsed_value.has_value();
              });
  reader.read(AMDSMI_FABRIC_DF_FIELD_NUM_STATIONS, kUALOE_NUM_STATIONS,
              [&](const std::string& str_value) {
                const auto parsed_value = parse_u32_decimal(str_value);
                if (parsed_value.has_value() &&
                    (parsed_value.value() <= std::numeric_limits<uint8_t>::max())) {
                  config.data.num_stations = static_cast<uint8_t>(parsed_value.value());
                  return true;
                }
                return false;
              });
  reader.read(AMDSMI_FABRIC_DF_FIELD_LANE_EN_BITMAP, kUALOE_LANE_EN_BITMAP,
              [&](const std::string& str_value) {
                return parse_hex_bytes(str_value, config.data.lane_en_bitmap,
                                       AMDSMI_FABRIC_MAX_BITMAP_SIZE)
                    .has_value();
              });

  config.mask = fields_read;
  return ((fields_read != 0) ? amdsmi_status_t::AMDSMI_STATUS_SUCCESS
                             : amdsmi_status_t::AMDSMI_STATUS_NO_DATA);
}

auto query_station_config(const AMDSmiGPUDevice& device, amdsmi_fabric_station_config_t& config,
                          std::string_view subdir = kUALOE_UALINK_STATIONS_SUBDIR)
    -> amdsmi_status_t {
  return query_station_config_at(get_ualink_root(device), device.device_has_ualink(), config,
                                 subdir);
}

namespace {

/**
 *  Accumulates flat-vs-subtree disagreements over one config's compared mask.
 *  Both surfaces reporting a field lets us compare values.
 *
 *  Presence handling differs by phase. PostCommit expects both surfaces to hold the committed
 *  fields, so either direction is a divergence. PreStage treats only subtree-present/flat-absent
 *  as one: the reverse is a field nobody has staged since boot, which says nothing about residue
 *  and would otherwise fire on every field the caller has never written.
 *
 *  PostCommit assumes the driver leaves applied values in the write subtree. If it starts clearing
 *  those files instead, every committed field reports a spurious PRESENCE-DIFF and that phase needs
 *  reworking; PreStage degrades to silence on its own.
 */
struct SurfaceComparison_t {
  std::ostringstream& m_report;
  SurfacePhase_t m_phase;
  uint32_t m_compared_mask;
  uint32_t m_subtree_mask;
  uint32_t m_flat_mask;
  int m_mismatches = 0;

  auto comparable(uint32_t bit, const char* name) -> bool {
    if (((m_compared_mask & bit) == 0)) {
      return false;
    }

    const auto subtree_has = ((m_subtree_mask & bit) != 0);
    const auto flat_has = ((m_flat_mask & bit) != 0);
    if ((subtree_has && flat_has)) {
      return true;
    }

    const auto is_presence_diff =
        ((m_phase == SurfacePhase_t::PostCommit) ? (subtree_has != flat_has)
                                                 : (subtree_has && (!flat_has)));
    if (is_presence_diff) {
      m_report << " | PRESENCE-DIFF " << name << " subtree=" << (subtree_has ? "present" : "absent")
               << " flat=" << (flat_has ? "present" : "absent");
      ++m_mismatches;
    }

    return false;
  }

  auto scalar(uint32_t bit, const char* name, uint64_t subtree_val, uint64_t flat_val) -> void {
    if ((comparable(bit, name) && (subtree_val != flat_val))) {
      m_report << " | MISMATCH " << name << " subtree=" << subtree_val << " flat=" << flat_val;
      ++m_mismatches;
    }
  }

  template <typename ElemTp>
  auto array(uint32_t bit, const char* name, const ElemTp* subtree_arr, const ElemTp* flat_arr,
             std::size_t count) -> void {
    if ((comparable(bit, name) && (!std::equal(subtree_arr, (subtree_arr + count), flat_arr)))) {
      m_report << " | MISMATCH " << name << " (array differs)";
      ++m_mismatches;
    }
  }
};

auto phase_label(SurfacePhase_t phase) -> const char* {
  return ((phase == SurfacePhase_t::PreStage) ? "pre-stage" : "post-commit");
}

/**
 *  An empty or unreadable write subtree is the clean pre-stage state, not a divergence. It is also
 *  how a driver that clears the subtree on commit would read, so staying quiet keeps the log usable
 *  if the retention assumption above turns out to be wrong.
 *
 *  The status has to be checked, not just the mask: the query functions leave @p config.mask at the
 *  caller's request on their early-return paths, so a missing subtree would otherwise look like
 *  every field is staged
 */
auto is_quiet_phase(SurfacePhase_t phase, amdsmi_status_t subtree_status, uint32_t subtree_mask)
    -> bool {
  return ((phase == SurfacePhase_t::PreStage) &&
          ((subtree_status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) || (subtree_mask == 0)));
}

}  // namespace

auto log_surface_divergence(const AMDSmiGPUDevice& device, const amdsmi_fabric_ppod_config_t& probe,
                            SurfacePhase_t phase) -> void {
  auto subtree = amdsmi_fabric_ppod_config_t{};
  subtree.version = probe.version;
  subtree.mask = probe.mask;
  auto flat = subtree;

  const auto subtree_status = query_ppod_config(device, subtree);
  const auto flat_status = query_ppod_config(device, flat, kUALOE_UALINK_FLAT_SUBDIR);
  if (is_quiet_phase(phase, subtree_status, subtree.mask)) {
    return;
  }

  auto report = std::ostringstream();
  report << __func__ << " | ppod | " << phase_label(phase)
         << " | subtree_status: " << subtree_status << " | flat_status: " << flat_status;

  auto diff = SurfaceComparison_t{report, phase, probe.mask, subtree.mask, flat.mask};
  diff.scalar(AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID, "accelerator_id", subtree.data.accelerator_id,
              flat.data.accelerator_id);
  diff.array(AMDSMI_FABRIC_PPOD_FIELD_PPOD_ID, "ppod_id", subtree.data.ppod_id, flat.data.ppod_id,
             AMDSMI_MAX_UUID_ELEMENTS);
  diff.scalar(AMDSMI_FABRIC_PPOD_FIELD_PPOD_SIZE, "ppod_size", subtree.data.ppod_size,
              flat.data.ppod_size);
  diff.array(AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS, "local_accelerators",
             subtree.data.local_accelerators, flat.data.local_accelerators,
             AMDSMI_FABRIC_MAX_LOCAL_GPUS);
  diff.scalar(AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS, "local_accelerator_count",
              subtree.data.local_accelerator_count, flat.data.local_accelerator_count);
  diff.scalar(AMDSMI_FABRIC_PPOD_FIELD_BANDWIDTH, "bandwidth", subtree.data.bandwidth,
              flat.data.bandwidth);
  diff.scalar(AMDSMI_FABRIC_PPOD_FIELD_LATENCY, "latency", subtree.data.latency, flat.data.latency);

  if (diff.m_mismatches == 0) {
    report << " | all committed fields match across surfaces";
  }
  LOG_DEBUG(report);
}

auto log_surface_divergence(const AMDSmiGPUDevice& device, const amdsmi_fabric_vpod_config_t& probe,
                            SurfacePhase_t phase) -> void {
  auto subtree = amdsmi_fabric_vpod_config_t{};
  subtree.version = probe.version;
  subtree.mask = probe.mask;
  auto flat = subtree;

  const auto subtree_status = query_vpod_config(device, subtree);
  const auto flat_status = query_vpod_config(device, flat, kUALOE_UALINK_FLAT_SUBDIR);
  if (is_quiet_phase(phase, subtree_status, subtree.mask)) {
    return;
  }

  auto report = std::ostringstream();
  report << __func__ << " | vpod | " << phase_label(phase)
         << " | subtree_status: " << subtree_status << " | flat_status: " << flat_status;

  auto surface_compare = SurfaceComparison_t{report, phase, probe.mask, subtree.mask, flat.mask};
  surface_compare.scalar(AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID, "vpod_id", subtree.data.vpod_id,
                         flat.data.vpod_id);
  surface_compare.scalar(AMDSMI_FABRIC_VPOD_FIELD_VPOD_SIZE, "vpod_size", subtree.data.vpod_size,
                         flat.data.vpod_size);
  surface_compare.array(AMDSMI_FABRIC_VPOD_FIELD_VPOD_ACTIVE_ACCELS, "vpod_active_accelerators",
                        subtree.data.vpod_active_accelerators, flat.data.vpod_active_accelerators,
                        AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE);
  surface_compare.scalar(AMDSMI_FABRIC_VPOD_FIELD_ADDR_MODE, "addr_mode",
                         static_cast<uint64_t>(subtree.data.addr_mode),
                         static_cast<uint64_t>(flat.data.addr_mode));

  if (surface_compare.m_mismatches == 0) {
    report << " | all committed fields match across surfaces";
  }
  LOG_DEBUG(report);
}

auto log_surface_divergence(const AMDSmiGPUDevice& device,
                            const amdsmi_fabric_station_config_t& probe, SurfacePhase_t phase)
    -> void {
  auto subtree = amdsmi_fabric_station_config_t{};
  subtree.version = probe.version;
  subtree.mask = probe.mask;
  auto flat = subtree;

  const auto subtree_status = query_station_config(device, subtree);
  const auto flat_status = query_station_config(device, flat, kUALOE_UALINK_FLAT_SUBDIR);
  if (is_quiet_phase(phase, subtree_status, subtree.mask)) {
    return;
  }

  auto report = std::ostringstream();
  report << __func__ << " | station | " << phase_label(phase)
         << " | subtree_status: " << subtree_status << " | flat_status: " << flat_status;

  auto surface_compare = SurfaceComparison_t{report, phase, probe.mask, subtree.mask, flat.mask};
  surface_compare.scalar(AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS, "station_flags",
                         subtree.data.station_flags, flat.data.station_flags);
  surface_compare.scalar(AMDSMI_FABRIC_DF_FIELD_NUM_STATIONS, "num_stations",
                         subtree.data.num_stations, flat.data.num_stations);
  surface_compare.array(AMDSMI_FABRIC_DF_FIELD_LANE_EN_BITMAP, "lane_en_bitmap",
                        subtree.data.lane_en_bitmap, flat.data.lane_en_bitmap,
                        AMDSMI_FABRIC_MAX_BITMAP_SIZE);

  if (surface_compare.m_mismatches == 0) {
    report << " | all committed fields match across surfaces";
  }
  LOG_DEBUG(report);
}

}  // namespace fabric_ualink

auto AMDSmiGPUDevice::apply_fabric_ppod_config(const amdsmi_fabric_ppod_config_t& config) const
    -> amdsmi_status_t {
  return fabric_ualink::apply_ppod_config(*this, config);
}

auto AMDSmiGPUDevice::apply_fabric_vpod_config(const amdsmi_fabric_vpod_config_t& config) const
    -> amdsmi_status_t {
  return fabric_ualink::apply_vpod_config(*this, config);
}

auto AMDSmiGPUDevice::apply_fabric_station_config(
    const amdsmi_fabric_station_config_t& config) const -> amdsmi_status_t {
  return fabric_ualink::apply_station_config(*this, config);
}

auto AMDSmiGPUDevice::query_fabric_ppod_config(amdsmi_fabric_ppod_config_t& config) const
    -> amdsmi_status_t {
  return fabric_ualink::query_ppod_config(*this, config);
}

auto AMDSmiGPUDevice::query_fabric_vpod_config(amdsmi_fabric_vpod_config_t& config) const
    -> amdsmi_status_t {
  return fabric_ualink::query_vpod_config(*this, config);
}

auto AMDSmiGPUDevice::query_fabric_station_config(amdsmi_fabric_station_config_t& config) const
    -> amdsmi_status_t {
  return fabric_ualink::query_station_config(*this, config);
}

auto AMDSmiGPUDevice::query_fabric_ppod_config_flat(amdsmi_fabric_ppod_config_t& config) const
    -> amdsmi_status_t {
  return fabric_ualink::query_ppod_config(*this, config, kUALOE_UALINK_FLAT_SUBDIR);
}

auto AMDSmiGPUDevice::query_fabric_vpod_config_flat(amdsmi_fabric_vpod_config_t& config) const
    -> amdsmi_status_t {
  return fabric_ualink::query_vpod_config(*this, config, kUALOE_UALINK_FLAT_SUBDIR);
}

auto AMDSmiGPUDevice::query_fabric_station_config_flat(amdsmi_fabric_station_config_t& config) const
    -> amdsmi_status_t {
  return fabric_ualink::query_station_config(*this, config, kUALOE_UALINK_FLAT_SUBDIR);
}

}  // namespace amd::smi
