/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <gtest/gtest.h>

#include <vector>
#include <mutex>
#include <memory>

#include "api/rocprofiler_singleton.h"
#include "src/core/hsa/hsa_support.h"
#include "src/utils/filesystem.hpp"

namespace fs = rocprofiler::common::filesystem;
using namespace std::string_literals;

#define MAX_THREADS 10000
TEST(WhenTestingDeviceInfo, TestFailFatal) {
  fs::directory_entry dirp("/sys/class/kfd/kfd/topology/nodes");
  fs::path sysfs_nodes_path = "/sys/class/kfd/kfd/topology/nodes";
  if (!fs::exists(sysfs_nodes_path)) rocprofiler::fatal("Could not opendir `%s'", sysfs_nodes_path.c_str());
  for (auto const& dirp_entry : fs::directory_iterator{dirp}) {
    fs::path node_path = dirp_entry.path();
    long long node_id = std::stoll(std::string(dirp_entry.path().stem().string()));
    fs::path gpu_path = node_path / "gpu_id";
    std::ifstream gpu_id_file(gpu_path);
    std::string gpu_id_str;
    long long gpu_id = 0;
    if (gpu_id_file.is_open()) {
      gpu_id_file >> gpu_id_str;
      if (!gpu_id_str.empty()) {
        gpu_id = std::stoll(gpu_id_str);
        if (gpu_id == 0) {
          EXPECT_DEATH(Agent::DeviceInfo deviceInfo(node_id, gpu_id), "");
          break;
        }
      }
    }
  }
}

TEST(WhenTestingDeviceInfo, DeviceInfoReadSuccessfully) {
  fs::directory_entry dirp("/sys/class/kfd/kfd/topology/nodes");
  fs::path sysfs_nodes_path = "/sys/class/kfd/kfd/topology/nodes";
  if (!fs::exists(sysfs_nodes_path)) rocprofiler::fatal("Could not opendir `%s'", sysfs_nodes_path.c_str());
  uint64_t gpu_id = 0;
  uint32_t wave_front_size = 0;
  uint32_t location_id = 0;
  uint32_t domain = 0;
  uint32_t shader_arrays_per_se = 0;
  [[maybe_unused]] uint32_t max_waves_per_simd = 0;
  uint32_t array_count = 0;
  uint32_t se_num = 0;
  uint32_t cu_num = 0;
  uint32_t simd_count = 0;
  uint32_t simd_per_cu = 0;
  uint64_t unique_gpu_id = 0;
  uint32_t xcc_num = 1;
  uint32_t compute_units_per_sh = 0;
  long long topology_id = 0;
  uint32_t waves_per_cu = 0;
  uint32_t wave_slots_per_simd = 0;

  rocprofiler::ROCProfiler_Singleton& rocprofiler_instance =
      rocprofiler::ROCProfiler_Singleton::GetInstance();
  for (auto const& dirp_entry : fs::directory_iterator{dirp}) {
    fs::path node_path = dirp_entry.path();
    topology_id = std::stoll(dirp_entry.path().stem().string());
    fs::path gpu_path = node_path / "gpu_id";
    std::ifstream gpu_id_file(gpu_path.c_str());
    std::string gpu_id_str;
    if (gpu_id_file.is_open()) {
      gpu_id_file >> gpu_id_str;
      if (!gpu_id_str.empty()) {
        gpu_id = std::stoll(gpu_id_str);
        if (gpu_id > 0) {
          const Agent::DeviceInfo& device_info = rocprofiler_instance.GetDeviceInfo(gpu_id);
          fs::path properties_path = node_path / "properties";
          std::ifstream props_ifs(properties_path);
          if (!props_ifs.is_open())
            rocprofiler::fatal("Could not open %s/properties", properties_path.c_str());
          std::string prop_name;
          uint64_t prop_value;
          EXPECT_TRUE(gpu_id == device_info.getGPUId());
          while (props_ifs >> prop_name >> prop_value) {
            if (prop_name == "wave_front_size")
              wave_front_size = static_cast<uint32_t>(prop_value);
            else if (prop_name == "array_count")
              array_count = static_cast<uint32_t>(prop_value);
            else if (prop_name == "simd_per_cu")
              simd_per_cu = static_cast<uint32_t>(prop_value);
            else if (prop_name == "location_id")
              location_id = static_cast<uint32_t>(prop_value);
            else if (prop_name == "domain")
              domain = static_cast<uint32_t>(prop_value);
            else if (prop_name == "simd_arrays_per_engine")
              shader_arrays_per_se = static_cast<uint32_t>(prop_value);
            else if (prop_name == "max_waves_per_simd")
              max_waves_per_simd = static_cast<uint32_t>(prop_value);
            else if (prop_name == "simd_count")
              simd_count = static_cast<uint32_t>(prop_value);
            else if (prop_name == "unique_id")
              unique_gpu_id = static_cast<uint64_t>(prop_value);
            else if (prop_name == "num_xcc")
              xcc_num = static_cast<uint32_t>(prop_value);
          }
          se_num = array_count / shader_arrays_per_se;
          cu_num = simd_count / simd_per_cu;
          waves_per_cu = max_waves_per_simd * simd_per_cu;
          compute_units_per_sh = cu_num / (se_num * shader_arrays_per_se);
          wave_slots_per_simd = waves_per_cu / simd_per_cu;


          EXPECT_TRUE(wave_front_size == device_info.getMaxWaveSize())
              << "Device Info has incorrect wave_front_size ";
          EXPECT_TRUE(simd_per_cu == device_info.getSimdCountPerCU())
              << "Device Info has incorrect simd_per_cu";
          EXPECT_TRUE(location_id == device_info.getPCILocationID())
              << "Device Info has incorrect location_id";
          EXPECT_TRUE(se_num == device_info.getShaderEngineCount())
              << "Device Info has incorrect se_num";
          EXPECT_TRUE(waves_per_cu == device_info.getMaxWavesPerCU())
              << "Device Info has incorrect waves_per_cu";

          EXPECT_TRUE(domain == device_info.getPCIDomain()) << "Device Info has incorrect domain";
          EXPECT_TRUE(shader_arrays_per_se == device_info.getShaderArraysPerSE())
              << "Device Info has incorrect shader_arrays_per_se";
          EXPECT_TRUE(cu_num == device_info.getCUCount()) << "Device Info has incorrect cu_num";
          EXPECT_TRUE(compute_units_per_sh == device_info.getCUCountPerSH())
              << "Device Info has incorrect compute_units_per_sh ";
          EXPECT_TRUE(wave_slots_per_simd == device_info.getWaveSlotsPerSimd())
              << "Device Info has incorrect wave_slots_per_simd ";
          EXPECT_TRUE(unique_gpu_id == device_info.getUniqueGPUId())
              << "Device Info has incorrect unique_gpu_id ";
          EXPECT_TRUE(xcc_num == device_info.getXccCount()) << "Device Info has incorrect xcc_num ";
          ;
          EXPECT_TRUE(static_cast<uint32_t>(topology_id) == device_info.getNumaNode())
              << "Device Info has incorrect topology_id";
        }
      }
    }
  }
}

TEST(WhenTestingDeviceInfo, GetDeviceInfoFail) {
  fs::directory_entry dirp("/sys/class/kfd/kfd/topology/nodes");
  fs::path sysfs_nodes_path = "/sys/class/kfd/kfd/topology/nodes";
  if (!fs::exists(sysfs_nodes_path)) rocprofiler::fatal("Could not opendir `%s'", sysfs_nodes_path.c_str());
  uint64_t node_id = 0;
  for ([[maybe_unused]] auto const& dirp_entry : fs::directory_iterator{dirp}) {
    node_id++;
  }
  node_id++;
  fs::path node_path = sysfs_nodes_path / std::to_string(node_id);
  std::stringstream error;
  error << "Could not opendir `" << node_path.c_str() << "'";

  EXPECT_DEATH(Agent::DeviceInfo deviceInfo(node_id, 1), error.str().c_str());
}


class TestRocprofilerSingleton {
 public:
  uintptr_t ref_address;
  TestRocprofilerSingleton() {
    rocprofiler::ROCProfiler_Singleton& rocprofiler =
        rocprofiler::ROCProfiler_Singleton::GetInstance();
    ref_address = reinterpret_cast<long int>(&rocprofiler);
  }
};

void instantiateRocprofiler(uint64_t *target) {
  TestRocprofilerSingleton test;
  *target = test.ref_address;

}


//Add more threads here
TEST(WhenInvokingRocprofilerSingleton, RocprofilerSingletonInstanciation) {
  std::vector<std::thread> threads;
  uint64_t *refaddress = (uint64_t*)malloc(sizeof(uint64_t)*(MAX_THREADS));
  for(int i = 0; i < MAX_THREADS; i++) {
    threads.emplace_back(instantiateRocprofiler, &refaddress[i]);
  }
  for (auto&& thread : threads) thread.join();
  uint64_t ref_addr = refaddress[0];
  for(int i = 1; i < MAX_THREADS; i++)
   EXPECT_EQ(ref_addr, refaddress[i]) << "RocprofilerSingleton Instanciation failed";
  free(refaddress);
}
