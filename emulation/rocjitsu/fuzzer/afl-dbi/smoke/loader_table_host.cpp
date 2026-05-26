#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

bool points_to_preload(
    hsa_status_t (*fn)(hsa_file_t, size_t, size_t, hsa_code_object_reader_t *)) {
  if (fn == nullptr) {
    fprintf(stderr, "loader table file-offset reader is null\n");
    return false;
  }

  Dl_info info{};
  if (dladdr(reinterpret_cast<void *>(fn), &info) == 0 || info.dli_fname == nullptr) {
    fprintf(stderr, "dladdr failed for loader table file-offset reader\n");
    return false;
  }

  if (strstr(info.dli_fname, "rocjitsu_afl_preload") == nullptr) {
    fprintf(stderr, "loader table reader points to %s, not rocjitsu AFL preload\n",
            info.dli_fname);
    return false;
  }
  return true;
}

bool create_reader_from_file_offset(
    hsa_status_t (*fn)(hsa_file_t, size_t, size_t, hsa_code_object_reader_t *),
    const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    perror("open");
    return false;
  }

  struct stat st {};
  if (fstat(fd, &st) != 0) {
    perror("fstat");
    close(fd);
    return false;
  }
  if (st.st_size <= 0) {
    fprintf(stderr, "empty code object: %s\n", path);
    close(fd);
    return false;
  }

  hsa_code_object_reader_t reader {};
  hsa_status_t status = fn(fd, 0, static_cast<size_t>(st.st_size), &reader);
  close(fd);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "file-offset reader creation failed: %d\n", static_cast<int>(status));
    return false;
  }

  printf("file_offset_reader=%llu image_bytes=%lld\n",
         static_cast<unsigned long long>(reader.handle), static_cast<long long>(st.st_size));
  status = hsa_code_object_reader_destroy(reader);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "reader destroy failed: %d\n", static_cast<int>(status));
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 2) {
    fprintf(stderr, "usage: %s [hsaco]\n", argv[0]);
    return 2;
  }

  hsa_status_t status = hsa_init();
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_init failed: %d\n", static_cast<int>(status));
    return 1;
  }

  hsa_ven_amd_loader_1_03_pfn_t table{};
  status = hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(table),
                                                &table);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "hsa_system_get_major_extension_table failed: %d\n",
            static_cast<int>(status));
    hsa_shut_down();
    return 1;
  }

  auto reader = table.hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size;
  bool ok = points_to_preload(reader);
  if (ok && argc == 2)
    ok = create_reader_from_file_offset(reader, argv[1]);
  hsa_shut_down();
  return ok ? 0 : 1;
}
