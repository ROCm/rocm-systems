#include "code_object_image.h"
#include "instrumentation_planner.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/code_object_patcher.h"

#ifdef ROCFUZZ_ENABLE_HSA_READER_CHECK
#include <hsa/hsa.h>
#endif
#include <hip/hip_runtime_api.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr uint32_t kInputBytes = 256;
constexpr uint32_t kWorkItems = 256;
constexpr uint32_t kThreadsPerBlock = 64;
constexpr uint32_t kForcedPrivateSegmentBytes = 256;

enum class LoaderMode {
  RawHip,
  CcobHip,
  CcobHsaReader,
};

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err__ = (expr);                                                 \
    if (err__ != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(err__));                                       \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

void fail(const char *message) {
  fprintf(stderr, "metadata_resize_loader_smoke: %s\n", message);
  exit(1);
}

LoaderMode parse_loader_mode(const char *arg) {
  if (strcmp(arg, "--raw-hip") == 0)
    return LoaderMode::RawHip;
  if (strcmp(arg, "--ccob-hip") == 0)
    return LoaderMode::CcobHip;
  if (strcmp(arg, "--ccob-hsa-reader") == 0)
    return LoaderMode::CcobHsaReader;
  fail("unknown loader mode");
  return LoaderMode::RawHip;
}

const char *loader_mode_name(LoaderMode mode) {
  switch (mode) {
  case LoaderMode::RawHip:
    return "raw-hip";
  case LoaderMode::CcobHip:
    return "ccob-hip";
  case LoaderMode::CcobHsaReader:
    return "ccob-hsa-reader";
  }
  return "unknown";
}

const rocjitsu::fuzzer::afl_dbi::DeviceImage *
select_device_image(std::span<const rocjitsu::fuzzer::afl_dbi::DeviceImage> images,
                    std::string_view gfxip) {
  for (const auto &image : images) {
    if (image.target_id == gfxip)
      return &image;
  }
  return images.empty() ? nullptr : &images.front();
}

std::vector<uint8_t> patched_loader_image(const char *path, std::string_view gfxip,
                                          LoaderMode mode) {
  namespace afl = rocjitsu::fuzzer::afl_dbi;

  std::vector<uint8_t> container = afl::read_file_bytes(path);
  if (container.empty())
    fail("input code object is empty or unreadable");
  std::vector<afl::DeviceImage> images = afl::extract_device_images(container);
  const afl::DeviceImage *selected = select_device_image(images, gfxip);
  if (selected == nullptr)
    fail("input does not contain an AMDGPU device image");

  rocjitsu::AmdGpuCodeObject co(selected->bytes.data(), selected->bytes.size());
  if (!co.is_valid())
    fail("selected AMDGPU device image does not parse");
  rocjitsu::CodeObjectPatcher patcher(co);

  const std::vector<afl::KernelSite> sites = afl::find_kernel_sites(patcher.image_bytes());
  std::optional<afl::KernelSite> target_site;
  for (const afl::KernelSite &site : sites) {
    if (site.name == "branchy_kernel_a") {
      target_site = site;
      break;
    }
  }
  if (!target_site)
    fail("branchy_kernel_a descriptor was not found");

  afl::ProbeRegisterRequirements requirements;
  requirements.sgprs = target_site->allocated_sgpr_count;
  requirements.vgprs = target_site->allocated_vgpr_count;
  requirements.private_segment_bytes = kForcedPrivateSegmentBytes;
  std::optional<afl::KernelDescriptorResourceSummary> summary =
      afl::plan_kernel_descriptor_resources(patcher.image_bytes(), *target_site,
                                            requirements);
  if (!summary)
    fail("descriptor private-segment resource planning failed");
  if (summary->patched_private_segment_fixed_size < kForcedPrivateSegmentBytes)
    fail("descriptor plan did not request the forced private segment size");

  const char *metadata_failure = nullptr;
  std::optional<afl::AmdgpuMetadataPrivateSegmentPatch> metadata_patch =
      afl::plan_amdgpu_metadata_private_segment_patch(
          patcher.image_bytes(), target_site->name,
          summary->patched_private_segment_fixed_size, &metadata_failure);
  if (!metadata_patch)
    fail(metadata_failure != nullptr ? metadata_failure
                                     : "metadata private-segment planning failed");
  if (metadata_patch->kind !=
      afl::AmdgpuMetadataPrivateSegmentPatch::Kind::RebuiltNoteSection) {
    fail("forced private segment size did not require metadata note rebuild");
  }

  const char *patch_failure = nullptr;
  if (!afl::patch_kernel_descriptor_resources(patcher, *summary, &patch_failure))
    fail(patch_failure != nullptr ? patch_failure : "descriptor resource patch failed");

  std::vector<uint8_t> patched = patcher.emit();
  if (patched.size() <= selected->bytes.size())
    fail("rebuilt metadata note did not grow the raw device image");

  if (mode == LoaderMode::RawHip)
    return patched;

  std::optional<std::vector<uint8_t>> rebuilt_container =
      afl::rebuild_code_object_image_with_replaced_device_image(container, *selected,
                                                               patched);
  if (!rebuilt_container)
    fail("rebuilt metadata note could not be inserted back into the container");
  if (rebuilt_container->size() <= container.size())
    fail("rebuilt metadata note did not grow the code-object container");
  if (!afl::is_ccob_image(*rebuilt_container))
    fail("container modes require a rebuilt CCOB image");
  return *rebuilt_container;
}

void check_hsa_reader(std::span<const uint8_t> image) {
#ifdef ROCFUZZ_ENABLE_HSA_READER_CHECK
  auto check_hsa = [](hsa_status_t status, const char *what) {
    if (status == HSA_STATUS_SUCCESS)
      return;
    const char *text = nullptr;
    (void)hsa_status_string(status, &text);
    fprintf(stderr, "metadata_resize_loader_smoke: %s failed: %u %s\n", what,
            static_cast<unsigned>(status), text != nullptr ? text : "<unknown>");
    exit(2);
  };

  check_hsa(hsa_init(), "hsa_init");
  hsa_code_object_reader_t reader{};
  hsa_status_t status =
      hsa_code_object_reader_create_from_memory(image.data(), image.size(), &reader);
  if (status != HSA_STATUS_SUCCESS) {
    (void)hsa_shut_down();
    check_hsa(status, "hsa_code_object_reader_create_from_memory");
  }
  check_hsa(hsa_code_object_reader_destroy(reader), "hsa_code_object_reader_destroy");
  check_hsa(hsa_shut_down(), "hsa_shut_down");
  printf("metadata_resize_loader_smoke mode=ccob-hsa-reader reader=%llu image_bytes=%zu\n",
         static_cast<unsigned long long>(reader.handle), image.size());
#else
  (void)image;
  fail("HSA-reader check was not enabled in this build");
#endif
}

std::string write_temp_module(std::span<const uint8_t> image) {
  std::string path = "/tmp/rocfuzz_metadata_resize_XXXXXX.co";
  std::vector<char> writable_path(path.begin(), path.end());
  writable_path.push_back('\0');
  const int fd = mkstemps(writable_path.data(), 3);
  if (fd < 0)
    fail("failed to create temporary CCOB module");
  size_t written = 0;
  while (written < image.size()) {
    const ssize_t chunk = write(fd, image.data() + written, image.size() - written);
    if (chunk <= 0) {
      (void)close(fd);
      fail("failed to write temporary CCOB module");
    }
    written += static_cast<size_t>(chunk);
  }
  if (close(fd) != 0)
    fail("failed to close temporary CCOB module");
  return std::string(writable_path.data());
}

void load_module_image(hipModule_t *module, std::span<const uint8_t> image,
                       LoaderMode mode) {
  if (mode == LoaderMode::RawHip) {
    HIP_CHECK(hipModuleLoadData(module, image.data()));
    return;
  }

  // CCOB loading is normally file-path based. Use a temporary file so this
  // smoke validates the same loader entry shape as library CCOB modules.
  std::string path = write_temp_module(image);
  hipError_t err = hipModuleLoad(module, path.c_str());
  (void)remove(path.c_str());
  if (err != hipSuccess) {
    fprintf(stderr, "HIP error loading temporary CCOB at %s: %s\n", path.c_str(),
            hipGetErrorString(err));
    exit(2);
  }
}

void run_branchy_module(std::span<const uint8_t> image, LoaderMode mode) {
  hipModule_t module = nullptr;
  hipFunction_t kernel_a = nullptr;
  hipFunction_t kernel_b = nullptr;
  load_module_image(&module, image, mode);
  HIP_CHECK(hipModuleGetFunction(&kernel_a, module, "branchy_kernel_a"));
  HIP_CHECK(hipModuleGetFunction(&kernel_b, module, "branchy_kernel_b"));

  std::vector<uint8_t> input(kInputBytes, 0);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<uint8_t>(i * 131u + 17u);

  uint8_t *device_input = nullptr;
  uint32_t *device_scratch = nullptr;
  uint32_t *device_result = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_input), input.size()));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_scratch),
                      sizeof(uint32_t) * kWorkItems));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_result),
                      sizeof(uint32_t) * kWorkItems));

  HIP_CHECK(hipMemcpy(device_input, input.data(), input.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(device_scratch, 0, sizeof(uint32_t) * kWorkItems));
  HIP_CHECK(hipMemset(device_result, 0, sizeof(uint32_t) * kWorkItems));

  uint32_t n = kWorkItems;
  uint32_t selector = 0x55;
  void *args_a[] = {&device_input, &device_scratch, &n};
  HIP_CHECK(hipModuleLaunchKernel(kernel_a, kWorkItems / kThreadsPerBlock, 1, 1,
                                  kThreadsPerBlock, 1, 1, 0, nullptr, args_a,
                                  nullptr));

  void *args_b[] = {&device_scratch, &device_result, &n, &selector};
  HIP_CHECK(hipModuleLaunchKernel(kernel_b, kWorkItems / kThreadsPerBlock, 1, 1,
                                  kThreadsPerBlock, 1, 1, 0, nullptr, args_b,
                                  nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> output(kWorkItems, 0);
  HIP_CHECK(hipMemcpy(output.data(), device_result,
                      sizeof(uint32_t) * output.size(), hipMemcpyDeviceToHost));
  uint32_t result = 0;
  for (uint32_t value : output)
    result += value;
  printf("metadata_resize_loader_smoke mode=%s result=%u\n",
         loader_mode_name(mode), result);

  HIP_CHECK(hipFree(device_result));
  HIP_CHECK(hipFree(device_scratch));
  HIP_CHECK(hipFree(device_input));
  HIP_CHECK(hipModuleUnload(module));
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <code-object> <gfxip> "
                    "<--raw-hip|--ccob-hip|--ccob-hsa-reader>\n",
            argv[0]);
    return 1;
  }

  const LoaderMode mode = parse_loader_mode(argv[3]);
  std::vector<uint8_t> patched = patched_loader_image(argv[1], argv[2], mode);
  if (mode == LoaderMode::CcobHsaReader)
    check_hsa_reader(patched);
  else
    run_branchy_module(patched, mode);
  return 0;
}
