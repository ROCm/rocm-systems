// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <amd_smi/amdsmi.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void Check(bool condition, const char* expression, int line) {
  if (!condition)
    throw std::runtime_error(std::string(expression) + " at line " + std::to_string(line));
}
#define CHECK(expression) Check((expression), #expression, __LINE__)

struct Fixture {
  std::filesystem::path directory;
  Fixture() {
    char pattern[] = "/tmp/amdsmi-mock-test-XXXXXX";
    char* result = mkdtemp(pattern);
    if (!result) throw std::runtime_error("mkdtemp failed");
    directory = result;
    CHECK(setenv("AMDSMI_MOCK_STATE_DIR", result, 1) == 0);
  }
  ~Fixture() { std::filesystem::remove_all(directory); }
  void Write(const std::string& file, const std::string& text) {
    auto temporary = directory / (file + ".tmp");
    {
      std::ofstream output(temporary);
      output << text;
      CHECK(static_cast<bool>(output));
    }
    std::filesystem::rename(temporary, directory / file);
  }
  void GPU(unsigned index, const std::string& telemetry = "45 100 750 0 0 196608 0 0") {
    Write("gpu" + std::to_string(index),
          "AMDSMI_MOCK_V1\n" + telemetry +
              "\nAMD Instinct MI300X (mock)\n00000000-0000-0000-0000-00000000000" +
              std::to_string(index) + "\n0000:0" + std::to_string(index + 1) + ":00.0\n");
  }
};

void Contract() {
  Fixture fixture;
  uint32_t count = 0;
  CHECK(amdsmi_get_socket_handles(&count, nullptr) == AMDSMI_STATUS_NOT_INIT);
  CHECK(amdsmi_shut_down() == AMDSMI_STATUS_NOT_INIT);
  CHECK(amdsmi_init(AMDSMI_INIT_AMD_CPUS) == AMDSMI_STATUS_NOT_SUPPORTED);
  CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS) == AMDSMI_STATUS_NOT_FOUND);
  for (const auto& invalid : {"-1", "33", "18446744073709551616", "1 garbage"}) {
    fixture.Write("count", invalid);
    CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS) == AMDSMI_STATUS_INVAL);
  }
  fixture.Write("count", "2\n");
  fixture.GPU(0);
  CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS) == AMDSMI_STATUS_NOT_FOUND);
  fixture.GPU(1);
  CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS) == AMDSMI_STATUS_SUCCESS);
  CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS) == AMDSMI_STATUS_SUCCESS);
  CHECK(amdsmi_shut_down() == AMDSMI_STATUS_SUCCESS);  // One reference remains.
  CHECK(amdsmi_get_socket_handles(nullptr, nullptr) == AMDSMI_STATUS_INVAL);
  CHECK(amdsmi_get_socket_handles(&count, nullptr) == AMDSMI_STATUS_SUCCESS && count == 1);
  amdsmi_socket_handle socket = nullptr;
  count = 0;
  CHECK(amdsmi_get_socket_handles(&count, &socket) == AMDSMI_STATUS_OUT_OF_RESOURCES);
  CHECK(count == 1 && socket == nullptr);
  CHECK(amdsmi_get_socket_handles(&count, &socket) == AMDSMI_STATUS_SUCCESS);
  CHECK(amdsmi_get_processor_handles(nullptr, &count, nullptr) == AMDSMI_STATUS_INVAL);
  CHECK(amdsmi_get_processor_handles(socket, &count, nullptr) == AMDSMI_STATUS_SUCCESS &&
        count == 2);
  amdsmi_processor_handle devices[2] = {};
  count = 1;
  CHECK(amdsmi_get_processor_handles(socket, &count, devices) == AMDSMI_STATUS_OUT_OF_RESOURCES);
  CHECK(count == 2 && !devices[0]);
  CHECK(amdsmi_get_processor_handles(socket, &count, devices) == AMDSMI_STATUS_SUCCESS);
  CHECK(devices[0] != devices[1]);
  int64_t temperature = -1;
  CHECK(amdsmi_get_temp_metric(nullptr, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT,
                               &temperature) == AMDSMI_STATUS_INVAL);
  CHECK(amdsmi_get_temp_metric(devices[0], AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT,
                               nullptr) == AMDSMI_STATUS_INVAL);
  CHECK(amdsmi_get_temp_metric(devices[0], AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT,
                               &temperature) == AMDSMI_STATUS_SUCCESS &&
        temperature == 45);
  CHECK(amdsmi_get_temp_metric(devices[0], AMDSMI_TEMPERATURE_TYPE_HOTSPOT, AMDSMI_TEMP_CURRENT,
                               &temperature) == AMDSMI_STATUS_NOT_SUPPORTED);
  processor_type_t type;
  CHECK(amdsmi_get_processor_type(devices[0], &type) == AMDSMI_STATUS_SUCCESS);
  CHECK(type == AMDSMI_PROCESSOR_TYPE_AMD_GPU);
  char uuid[AMDSMI_GPU_UUID_SIZE];
  unsigned length = 1;
  CHECK(amdsmi_get_gpu_device_uuid(devices[0], &length, uuid) == AMDSMI_STATUS_INVAL);
  length = sizeof(uuid);
  CHECK(amdsmi_get_gpu_device_uuid(devices[0], &length, nullptr) == AMDSMI_STATUS_INVAL);
  CHECK(amdsmi_get_gpu_device_uuid(devices[0], &length, uuid) == AMDSMI_STATUS_SUCCESS);
  CHECK(std::string(uuid) == "00000000-0000-0000-0000-000000000000");
  amdsmi_bdf_t bdf;
  CHECK(amdsmi_get_gpu_device_bdf(devices[1], &bdf) == AMDSMI_STATUS_SUCCESS);
  CHECK(bdf.bus_number == 2 && bdf.domain_number == 0);
  amdsmi_asic_info_t asic;
  CHECK(amdsmi_get_gpu_asic_info(devices[0], &asic) == AMDSMI_STATUS_SUCCESS);
  CHECK(std::string(asic.market_name) == "AMD Instinct MI300X (mock)");
  CHECK(asic.vendor_id == 0x1002 && asic.device_id == UINT64_MAX);
  amdsmi_version_t version;
  CHECK(amdsmi_get_lib_version(&version) == AMDSMI_STATUS_SUCCESS);
  CHECK(version.major == AMDSMI_LIB_VERSION_MAJOR);
  CHECK(amdsmi_get_lib_version(nullptr) == AMDSMI_STATUS_INVAL);
  uint64_t bytes;
  CHECK(amdsmi_get_gpu_memory_total(devices[0], AMDSMI_MEM_TYPE_VRAM, &bytes) ==
        AMDSMI_STATUS_SUCCESS);
  CHECK(bytes == 196608ULL * 1024 * 1024);
  CHECK(amdsmi_get_gpu_memory_total(devices[0], AMDSMI_MEM_TYPE_GTT, &bytes) ==
        AMDSMI_STATUS_NOT_SUPPORTED);
  // Live updates must affect only the selected GPU, without reinitialization.
  fixture.GPU(0, "95 600 750 100 1024 196608 3 7");
  amdsmi_error_count_t ecc;
  CHECK(amdsmi_get_gpu_total_ecc_count(devices[0], &ecc) == AMDSMI_STATUS_SUCCESS);
  CHECK(ecc.correctable_count == 3 && ecc.uncorrectable_count == 7);
  CHECK(amdsmi_get_gpu_ecc_count(devices[0], AMDSMI_GPU_BLOCK_UMC, &ecc) == AMDSMI_STATUS_SUCCESS);
  CHECK(ecc.correctable_count == 3 && ecc.uncorrectable_count == 7);
  CHECK(amdsmi_get_gpu_total_ecc_count(devices[1], &ecc) == AMDSMI_STATUS_SUCCESS);
  CHECK(ecc.correctable_count == 0 && ecc.uncorrectable_count == 0);
  amdsmi_engine_usage_t usage;
  CHECK(amdsmi_get_gpu_activity(devices[0], &usage) == AMDSMI_STATUS_SUCCESS);
  CHECK(usage.gfx_activity == 100 && usage.mm_activity == UINT16_MAX);
  amdsmi_power_info_t power;
  CHECK(amdsmi_get_power_info(devices[0], &power) == AMDSMI_STATUS_SUCCESS);
  CHECK(power.socket_power == 600 && power.power_limit == 750 && power.gfx_voltage == UINT64_MAX);
  amdsmi_vram_usage_t memory;
  CHECK(amdsmi_get_gpu_vram_usage(devices[0], &memory) == AMDSMI_STATUS_SUCCESS);
  CHECK(memory.vram_used == 1024 && memory.vram_total == 196608);
  CHECK(amdsmi_get_gpu_memory_usage(devices[0], AMDSMI_MEM_TYPE_VRAM, &bytes) ==
        AMDSMI_STATUS_SUCCESS);
  CHECK(bytes == 1024ULL * 1024 * 1024);
  // Unimplemented controls must not claim to have reset hardware.
  CHECK(amdsmi_reset_gpu(devices[0]) == AMDSMI_STATUS_NOT_SUPPORTED);
  std::atomic<bool> valid{true};
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < 4; ++i) {
    threads.emplace_back([&]() {
      for (unsigned j = 0; j < 100; ++j) {
        amdsmi_error_count_t observed;
        if (amdsmi_get_gpu_total_ecc_count(devices[0], &observed) != AMDSMI_STATUS_SUCCESS ||
            observed.uncorrectable_count != 7)
          valid = false;
      }
    });
  }
  for (auto& thread : threads) thread.join();
  CHECK(valid);
  for (const auto& invalid : {"-1 100 750 0 0 196608 0 0", "45 100 750 101 0 196608 0 0",
                              "45 100 750 0 196609 196608 0 0", "45 100 750 0 0 0 0 0",
                              "45 100 750 0 0 196608 0 18446744073709551616"}) {
    fixture.GPU(0, invalid);
    CHECK(amdsmi_get_gpu_total_ecc_count(devices[0], &ecc) == AMDSMI_STATUS_INVAL);
  }
  fixture.Write("gpu0", "AMDSMI_MOCK_V1\n45 100\n");
  CHECK(amdsmi_get_gpu_total_ecc_count(devices[0], &ecc) == AMDSMI_STATUS_INVAL);
  std::filesystem::remove(fixture.directory / "gpu0");
  CHECK(amdsmi_get_gpu_total_ecc_count(devices[0], &ecc) == AMDSMI_STATUS_NOT_FOUND);
  CHECK(amdsmi_shut_down() == AMDSMI_STATUS_SUCCESS);
  CHECK(amdsmi_get_gpu_total_ecc_count(devices[1], &ecc) == AMDSMI_STATUS_NOT_INIT);
  CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS) == AMDSMI_STATUS_NOT_FOUND);
  fixture.Write("count", "0\n");
  CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS) == AMDSMI_STATUS_SUCCESS);
  CHECK(amdsmi_get_socket_handles(&count, nullptr) == AMDSMI_STATUS_SUCCESS && count == 0);
  CHECK(amdsmi_shut_down() == AMDSMI_STATUS_SUCCESS);
  unsetenv("AMDSMI_MOCK_STATE_DIR");
  CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS) == AMDSMI_STATUS_INVAL);
}
}  // namespace

int main() {
  try {
    Contract();
    std::cout << "PASS: discovery, API units, live ECC/telemetry, isolation, concurrency, "
                 "lifecycle, invalid state\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
