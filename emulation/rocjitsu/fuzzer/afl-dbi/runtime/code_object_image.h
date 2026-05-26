#pragma once

#include <stdint.h>

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::fuzzer::afl_dbi {

struct DeviceImage {
  std::string id;
  std::string target_id;
  std::vector<uint8_t> bytes;
  uint32_t index = 0;
};

struct CodeObjectImageSummary {
  bool top_level_raw_elf = false;
  bool top_level_bundle = false;
  bool top_level_ccob = false;
  uint32_t device_image_count = 0;
  uint32_t top_level_ccob_device_image_count = 0;
  bool raw_elf_bypass_drops_sibling_payloads = false;
};

bool is_raw_elf_image(std::span<const uint8_t> image);
bool is_clang_offload_bundle(std::span<const uint8_t> image);
bool is_ccob_image(std::span<const uint8_t> image);
bool is_ccob_memory(const uint8_t *base);

std::vector<uint8_t> read_file_bytes(const char *path);
std::vector<uint8_t> read_fd_bytes(int fd, uint64_t offset, uint64_t size);
std::vector<uint8_t> copy_module_data_image(const void *image);

std::vector<DeviceImage> extract_device_images(std::span<const uint8_t> image);
CodeObjectImageSummary summarize_code_object_image(std::span<const uint8_t> image);
std::optional<std::vector<uint8_t>>
rebuild_code_object_image_with_replaced_device_image(std::span<const uint8_t> image,
                                                     std::string_view device_image_id,
                                                     std::span<const uint8_t> replacement);
std::optional<std::vector<uint8_t>>
rebuild_code_object_image_with_replaced_device_image(std::span<const uint8_t> image,
                                                     const DeviceImage &device_image,
                                                     std::span<const uint8_t> replacement);
std::optional<std::vector<uint8_t>>
rebuild_ccob_with_replaced_device_image(std::span<const uint8_t> image,
                                        std::string_view device_image_id,
                                        std::span<const uint8_t> replacement);
std::string normalize_amdgpu_target_id(std::string_view target_id);
std::vector<DeviceImage> order_device_images_for_target(std::span<const DeviceImage> images,
                                                        std::string_view agent_target_id);

} // namespace rocjitsu::fuzzer::afl_dbi
