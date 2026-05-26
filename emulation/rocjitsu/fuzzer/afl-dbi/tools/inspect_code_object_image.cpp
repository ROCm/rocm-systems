#include "code_object_image.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace afl_dbi = rocjitsu::fuzzer::afl_dbi;

namespace {

bool write_file(const char *path, std::span<const uint8_t> data) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file)
    return false;
  file.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(file);
}

void print_bool(const char *name, bool value) {
  printf(" %s=%s", name, value ? "true" : "false");
}

void print_summary(const char *path, std::span<const uint8_t> image,
                   const std::vector<afl_dbi::DeviceImage> &device_images) {
  const afl_dbi::CodeObjectImageSummary summary =
      afl_dbi::summarize_code_object_image(image);

  printf("path=%s bytes=%zu", path, image.size());
  print_bool("top_level_raw_elf", summary.top_level_raw_elf);
  print_bool("top_level_bundle", summary.top_level_bundle);
  print_bool("top_level_ccob", summary.top_level_ccob);
  printf(" device_images=%u", summary.device_image_count);
  printf(" top_level_ccob_device_images=%u", summary.top_level_ccob_device_image_count);
  print_bool("raw_elf_bypass_drops_sibling_payloads",
             summary.raw_elf_bypass_drops_sibling_payloads);
  printf("\n");

  for (size_t i = 0; i < device_images.size(); ++i) {
    const afl_dbi::DeviceImage &device_image = device_images[i];
    printf("  device[%zu] index=%u id=%s target_id=%s bytes=%zu\n", i, device_image.index,
           device_image.id.c_str(),
           device_image.target_id.empty() ? "-" : device_image.target_id.c_str(),
           device_image.bytes.size());
  }
}

void usage(const char *argv0) {
  fprintf(stderr,
          "usage: %s <code-object> [<code-object> ...]\n"
          "       %s --dump-device <index> <code-object> <output-hsaco>\n",
          argv0, argv0);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 2;
  }

  if (std::string(argv[1]) == "--dump-device") {
    if (argc != 5) {
      usage(argv[0]);
      return 2;
    }

    char *end = nullptr;
    const unsigned long requested = strtoul(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0') {
      fprintf(stderr, "inspect_code_object_image: invalid device index: %s\n", argv[2]);
      return 2;
    }

    std::vector<uint8_t> image = afl_dbi::read_file_bytes(argv[3]);
    if (image.empty()) {
      fprintf(stderr, "inspect_code_object_image: failed to read %s\n", argv[3]);
      return 1;
    }

    std::vector<afl_dbi::DeviceImage> device_images = afl_dbi::extract_device_images(image);
    print_summary(argv[3], image, device_images);
    if (requested >= device_images.size()) {
      fprintf(stderr, "inspect_code_object_image: device index %lu out of range\n", requested);
      return 1;
    }

    const afl_dbi::DeviceImage &device_image = device_images[requested];
    if (!write_file(argv[4], device_image.bytes)) {
      fprintf(stderr, "inspect_code_object_image: failed to write %s\n", argv[4]);
      return 1;
    }
    return 0;
  }

  int status = 0;
  for (int arg = 1; arg < argc; ++arg) {
    std::vector<uint8_t> image = afl_dbi::read_file_bytes(argv[arg]);
    if (image.empty()) {
      fprintf(stderr, "inspect_code_object_image: failed to read %s\n", argv[arg]);
      status = 1;
      continue;
    }
    std::vector<afl_dbi::DeviceImage> device_images = afl_dbi::extract_device_images(image);
    print_summary(argv[arg], image, device_images);
  }
  return status;
}
