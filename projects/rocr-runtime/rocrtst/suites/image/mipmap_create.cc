#include "suites/image/mipmap_create.h"
#include <iostream>
#include <cmath>

static void PrintStatus(const char* label, hsa_status_t s) {
  const char* msg = nullptr;
  hsa_status_string(s, &msg);
  std::cout << label << ": " << (s == HSA_STATUS_SUCCESS ? "SUCCESS" : "FAIL")
            << (msg? std::string(" (" ) + msg + ")" : std::string()) << std::endl;
}

hsa_status_t MipmapCreateTest::RunBasic() {
  // Discover default GPU agent & pools (BaseRocR variant of helper functions executed externally in test prolog)
  // Build descriptor
  hsa_ext_image_descriptor_t desc; // avoid enum narrowing by zero-init struct explicitly
  std::memset(&desc, 0, sizeof(desc));
  desc.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
  desc.width = 1024;
  desc.height = 512;
  desc.depth = 1;
  desc.array_size = 0; // 2D, not arrayed
  desc.format.channel_order = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
  desc.format.channel_type  = HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8;

  // Fallback: if gpu_device1 handle is zero, enumerate agents and pick first GPU
  if (gpu_device1()->handle == 0) {
    hsa_iterate_agents([](hsa_agent_t ag, void* data) -> hsa_status_t {
      hsa_device_type_t t; hsa_agent_get_info(ag, HSA_AGENT_INFO_DEVICE, &t);
      if (t == HSA_DEVICE_TYPE_GPU) { *reinterpret_cast<hsa_agent_t*>(data) = ag; return HSA_STATUS_INFO_BREAK; }
      return HSA_STATUS_SUCCESS; }, gpu_device1());
  }

  // Compute desired full chain levels
  uint32_t max_dim = desc.width > desc.height ? desc.width : desc.height;
  uint32_t max_levels = 1 + (uint32_t)std::floor(std::log2((double)max_dim));
  uint32_t requested_levels = max_levels; // full chain

  hsa_amd_mipmap_array_info_t info{};
  hsa_status_t s = hsa_amd_mipmap_array_get_info(*gpu_device1(), &desc, requested_levels,
        HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &info);
  PrintStatus("info_query", s);
  if (s != HSA_STATUS_SUCCESS) return s;

  std::cout << "Reported size=" << info.size << " alignment=" << info.alignment
            << " max_levels=" << info.max_levels << " levels_used=" << info.levels_used << std::endl;

  if (info.max_levels < requested_levels) {
    std::cout << "Runtime-reported max_levels < requested_levels (" << info.max_levels << " < " << requested_levels << ")\n";
    return HSA_STATUS_ERROR_INVALID_ARGUMENT; // treat as failure condition for this test
  }

  // Create without initial data (linear default)
  hsa_ext_image_t mip_array{};
  s = hsa_amd_mipmap_array_create(*gpu_device1(), &desc, nullptr, 0,
       HSA_ACCESS_PERMISSION_RO, requested_levels,
       HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &mip_array);
  PrintStatus("create", s);
  if (s != HSA_STATUS_SUCCESS) return s;

  // Optionally enumerate first level info just as a sanity check
  hsa_ext_image_descriptor_t lvl0{};
  s = hsa_amd_mipmap_get_level_info(*gpu_device1(), &mip_array, 0, &lvl0);
  PrintStatus("level0_info", s);
  if (s == HSA_STATUS_SUCCESS) {
    std::cout << "Level0 dims: " << lvl0.width << "x" << lvl0.height << std::endl;
  }

  // Destroy
  s = hsa_amd_mipmap_array_destroy(&mip_array);
  PrintStatus("destroy", s);
  return s;
}

hsa_status_t MipmapCreateTest::RunTooManyLevels() {
  hsa_ext_image_descriptor_t desc; std::memset(&desc, 0, sizeof(desc));
  desc.geometry = HSA_EXT_IMAGE_GEOMETRY_2D;
  desc.width = 64;
  desc.height = 64;
  desc.depth = 1;
  desc.array_size = 0;
  desc.format.channel_order = HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA;
  desc.format.channel_type  = HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8;

  if (gpu_device1()->handle == 0) {
    hsa_iterate_agents([](hsa_agent_t ag, void* data) -> hsa_status_t {
      hsa_device_type_t t; hsa_agent_get_info(ag, HSA_AGENT_INFO_DEVICE, &t);
      if (t == HSA_DEVICE_TYPE_GPU) { *reinterpret_cast<hsa_agent_t*>(data) = ag; return HSA_STATUS_INFO_BREAK; }
      return HSA_STATUS_SUCCESS; }, gpu_device1());
  }

  uint32_t requested_levels = 20; // intentionally excessive for 64x64 (should be 7)
  hsa_ext_image_t mip_array{};
  hsa_status_t s = hsa_amd_mipmap_array_create(*gpu_device1(), &desc, nullptr, 0,
        HSA_ACCESS_PERMISSION_RO, requested_levels,
        HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 0, 0, &mip_array);
  PrintStatus("create_excess_levels", s);
  return s; // Expect failure (INVALID_ARGUMENT)
}
