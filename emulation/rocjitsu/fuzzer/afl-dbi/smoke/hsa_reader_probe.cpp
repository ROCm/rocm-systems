#include <hsa/hsa.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <vector>

namespace {

std::vector<uint8_t> read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    perror(path);
    exit(1);
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    fprintf(stderr, "hsa_reader_probe: empty input: %s\n", path);
    exit(1);
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char *>(bytes.data()), size);
  if (!file) {
    fprintf(stderr, "hsa_reader_probe: failed to read input: %s\n", path);
    exit(1);
  }
  return bytes;
}

void check_hsa(hsa_status_t status, const char *what) {
  if (status == HSA_STATUS_SUCCESS)
    return;
  const char *text = nullptr;
  (void)hsa_status_string(status, &text);
  fprintf(stderr, "hsa_reader_probe: %s failed: %u %s\n", what, static_cast<unsigned>(status),
          text != nullptr ? text : "<unknown>");
  exit(2);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <code-object>\n", argv[0]);
    return 2;
  }

  std::vector<uint8_t> image = read_file(argv[1]);
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
  printf("reader=%llu image_bytes=%zu\n", static_cast<unsigned long long>(reader.handle),
         image.size());
  return 0;
}
