// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <amd_smi/amdsmi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>

namespace {
// State is external so an unmodified consumer can observe changes between calls.
struct State {
  uint64_t temperature, power, cap, utilization, used, total, corrected, uncorrected;
  std::string name, uuid, bdf;
};
std::mutex mutex;
uint32_t references = 0;
uint32_t gpu_count = 0;
std::string directory;
// Handles are opaque addresses. A client-provided handle is compared, never dereferenced.
char socket_token;
char processor_tokens[AMDSMI_MAX_DEVICES];

bool ReadUnsigned(std::istream& input, uint64_t* value) {
  std::string token;
  if (!(input >> token) || token.empty() ||
      token.find_first_not_of("0123456789") != std::string::npos)
    return false;
  std::istringstream number(token);
  return static_cast<bool>(number >> *value);
}

amdsmi_status_t Load(uint32_t index, State* state) {
  std::ifstream input(directory + "/gpu" + std::to_string(index));
  if (!input) return AMDSMI_STATUS_NOT_FOUND;
  std::string version, extra;
  if (!std::getline(input, version) || version != "AMDSMI_MOCK_V1") return AMDSMI_STATUS_INVAL;
  uint64_t* fields[] = {&state->temperature, &state->power, &state->cap,       &state->utilization,
                        &state->used,        &state->total, &state->corrected, &state->uncorrected};
  for (auto field : fields)
    if (!ReadUnsigned(input, field)) return AMDSMI_STATUS_INVAL;
  if (!std::getline(input, extra) || !extra.empty() || !std::getline(input, state->name) ||
      !std::getline(input, state->uuid) || !std::getline(input, state->bdf) || input >> extra)
    return AMDSMI_STATUS_INVAL;
  if (state->temperature > UINT16_MAX || state->power >= UINT32_MAX || state->cap >= UINT32_MAX ||
      state->utilization > 100 || state->total == 0 || state->total > UINT32_MAX ||
      state->used > state->total || state->name.empty() ||
      state->name.size() >= AMDSMI_MAX_STRING_LENGTH ||
      (state->uuid.empty() || state->uuid.size() >= AMDSMI_GPU_UUID_SIZE))
    return AMDSMI_STATUS_INVAL;
  unsigned domain, bus, device, function;
  int consumed = 0;
  if (std::sscanf(state->bdf.c_str(), "%x:%x:%x.%x%n", &domain, &bus, &device, &function,
                  &consumed) != 4 ||
      consumed != static_cast<int>(state->bdf.size()) || domain > UINT16_MAX || bus > UINT8_MAX ||
      device > 31 || function > 7)
    return AMDSMI_STATUS_INVAL;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t Validate(amdsmi_processor_handle handle, const void* output, State* state) {
  if (!references) return AMDSMI_STATUS_NOT_INIT;
  if (!output) return AMDSMI_STATUS_INVAL;
  for (uint32_t i = 0; i < gpu_count; ++i)
    if (handle == &processor_tokens[i]) return Load(i, state);
  return AMDSMI_STATUS_INVAL;
}

// All public functions serialize lifecycle and reads. Each read owns its State.
#define READ_STATE(handle, output)                \
  std::lock_guard<std::mutex> lock(mutex);        \
  State state;                                    \
  auto status = Validate(handle, output, &state); \
  if (status != AMDSMI_STATUS_SUCCESS) return status
}  // namespace

amdsmi_status_t amdsmi_init(uint64_t flags) {
  std::lock_guard<std::mutex> lock(mutex);
  if (flags != AMDSMI_INIT_AMD_GPUS) return AMDSMI_STATUS_NOT_SUPPORTED;
  if (references) {
    if (references == UINT32_MAX) return AMDSMI_STATUS_OUT_OF_RESOURCES;
    ++references;
    return AMDSMI_STATUS_SUCCESS;
  }
  const char* path = std::getenv("AMDSMI_MOCK_STATE_DIR");
  if (!path || !*path) return AMDSMI_STATUS_INVAL;
  directory = path;
  std::ifstream count_file(directory + "/count");
  if (!count_file) return AMDSMI_STATUS_NOT_FOUND;
  uint64_t count = 0;
  std::string extra;
  if (!ReadUnsigned(count_file, &count) || count > AMDSMI_MAX_DEVICES || count_file >> extra)
    return AMDSMI_STATUS_INVAL;
  for (uint32_t i = 0; i < count; ++i) {
    State state;
    auto status = Load(i, &state);
    if (status != AMDSMI_STATUS_SUCCESS) return status;
  }
  gpu_count = static_cast<uint32_t>(count);
  references = 1;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_shut_down() {
  std::lock_guard<std::mutex> lock(mutex);
  if (!references) return AMDSMI_STATUS_NOT_INIT;
  if (--references == 0) {
    gpu_count = 0;
    directory.clear();
  }
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_socket_handles(uint32_t* count, amdsmi_socket_handle* handles) {
  std::lock_guard<std::mutex> lock(mutex);
  if (!references) return AMDSMI_STATUS_NOT_INIT;
  if (!count) return AMDSMI_STATUS_INVAL;
  uint32_t required = gpu_count ? 1 : 0;
  uint32_t capacity = *count;
  *count = required;
  if (handles && capacity < required) return AMDSMI_STATUS_OUT_OF_RESOURCES;
  if (handles && required) handles[0] = &socket_token;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle socket, uint32_t* count,
                                             amdsmi_processor_handle* handles) {
  std::lock_guard<std::mutex> lock(mutex);
  if (!references) return AMDSMI_STATUS_NOT_INIT;
  if (!gpu_count || socket != &socket_token || !count) return AMDSMI_STATUS_INVAL;
  uint32_t capacity = *count;
  *count = gpu_count;
  if (handles && capacity < gpu_count) return AMDSMI_STATUS_OUT_OF_RESOURCES;
  if (handles)
    for (uint32_t i = 0; i < gpu_count; ++i) handles[i] = &processor_tokens[i];
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle handle, processor_type_t* type) {
  READ_STATE(handle, type);
  *type = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_socket_info(amdsmi_socket_handle socket, size_t size, char* name) {
  std::lock_guard<std::mutex> lock(mutex);
  if (!references) return AMDSMI_STATUS_NOT_INIT;
  if (!gpu_count || socket != &socket_token || !name || !size) return AMDSMI_STATUS_INVAL;
  constexpr char label[] = "AMD SMI mock socket";
  if (size < sizeof(label)) return AMDSMI_STATUS_OUT_OF_RESOURCES;
  std::memcpy(name, label, sizeof(label));
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_lib_version(amdsmi_version_t* version) {
  if (!version) return AMDSMI_STATUS_INVAL;
  *version = {AMDSMI_LIB_VERSION_MAJOR, AMDSMI_LIB_VERSION_MINOR, AMDSMI_LIB_VERSION_RELEASE,
              "cpu-only-mock"};
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_device_uuid(amdsmi_processor_handle handle, unsigned int* size,
                                           char* uuid) {
  READ_STATE(handle, size);
  // Match the public API's minimum buffer size; this is not a size-only query.
  if (!uuid || *size < AMDSMI_GPU_UUID_SIZE) return AMDSMI_STATUS_INVAL;
  std::memcpy(uuid, state.uuid.c_str(), state.uuid.size() + 1);
  *size = static_cast<unsigned int>(state.uuid.size() + 1);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_device_bdf(amdsmi_processor_handle handle, amdsmi_bdf_t* bdf) {
  READ_STATE(handle, bdf);
  unsigned domain, bus, device, function;
  std::sscanf(state.bdf.c_str(), "%x:%x:%x.%x", &domain, &bus, &device, &function);
  *bdf = {};
  bdf->domain_number = domain;
  bdf->bus_number = bus;
  bdf->device_number = device;
  bdf->function_number = function;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_asic_info(amdsmi_processor_handle handle, amdsmi_asic_info_t* info) {
  READ_STATE(handle, info);
  // Unmodeled numeric fields use the public unavailable sentinel.
  std::memset(info, 0xff, sizeof(*info));
  std::memset(info->market_name, 0, sizeof(info->market_name));
  std::memset(info->vendor_name, 0, sizeof(info->vendor_name));
  std::memset(info->asic_serial, 0, sizeof(info->asic_serial));
  std::snprintf(info->market_name, sizeof(info->market_name), "%s", state.name.c_str());
  std::snprintf(info->vendor_name, sizeof(info->vendor_name), "AMD");
  info->vendor_id = 0x1002;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_temp_metric(amdsmi_processor_handle handle,
                                       amdsmi_temperature_type_t sensor,
                                       amdsmi_temperature_metric_t metric, int64_t* temperature) {
  READ_STATE(handle, temperature);
  if (sensor != AMDSMI_TEMPERATURE_TYPE_EDGE || metric != AMDSMI_TEMP_CURRENT)
    return AMDSMI_STATUS_NOT_SUPPORTED;
  *temperature = static_cast<int64_t>(state.temperature);  // Public AMD SMI API uses Celsius.
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_activity(amdsmi_processor_handle handle,
                                        amdsmi_engine_usage_t* usage) {
  READ_STATE(handle, usage);
  std::memset(usage, 0xff, sizeof(*usage));
  usage->gfx_activity = static_cast<uint32_t>(state.utilization);
  usage->umc_activity = UINT16_MAX;
  usage->mm_activity = UINT16_MAX;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_power_info(amdsmi_processor_handle handle, amdsmi_power_info_t* info) {
  READ_STATE(handle, info);
  std::memset(info, 0xff, sizeof(*info));
  info->socket_power = info->current_socket_power = info->average_socket_power = state.power;
  info->power_limit = state.cap;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_memory_total(amdsmi_processor_handle handle,
                                            amdsmi_memory_type_t type, uint64_t* bytes) {
  READ_STATE(handle, bytes);
  if (type != AMDSMI_MEM_TYPE_VRAM) return AMDSMI_STATUS_NOT_SUPPORTED;
  *bytes = state.total * 1024 * 1024;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_memory_usage(amdsmi_processor_handle handle,
                                            amdsmi_memory_type_t type, uint64_t* bytes) {
  READ_STATE(handle, bytes);
  if (type != AMDSMI_MEM_TYPE_VRAM) return AMDSMI_STATUS_NOT_SUPPORTED;
  *bytes = state.used * 1024 * 1024;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_vram_usage(amdsmi_processor_handle handle,
                                          amdsmi_vram_usage_t* usage) {
  READ_STATE(handle, usage);
  *usage = {};
  usage->vram_total = static_cast<uint32_t>(state.total);
  usage->vram_used = static_cast<uint32_t>(state.used);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_total_ecc_count(amdsmi_processor_handle handle,
                                               amdsmi_error_count_t* count) {
  READ_STATE(handle, count);
  *count = {};
  count->correctable_count = state.corrected;
  count->uncorrectable_count = state.uncorrected;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_ecc_count(amdsmi_processor_handle handle, amdsmi_gpu_block_t block,
                                         amdsmi_error_count_t* count) {
  READ_STATE(handle, count);
  if (block != AMDSMI_GPU_BLOCK_UMC) return AMDSMI_STATUS_NOT_SUPPORTED;
  *count = {};
  count->correctable_count = state.corrected;
  count->uncorrectable_count = state.uncorrected;
  return AMDSMI_STATUS_SUCCESS;
}
