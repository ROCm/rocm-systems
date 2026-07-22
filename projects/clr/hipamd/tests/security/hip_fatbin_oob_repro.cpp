/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// ============================================================================
// ROCM-26752 / SEC-00548 - Out-of-Bounds Read in the HIP fat binary parser.
//
// This standalone program is both:
//   1. A reproducer for the out-of-bounds read described in the ticket, and
//   2. A negative regression test for the fix in
//      projects/clr/hipamd/src/hip_fatbin.cpp.
//
// Background
// ----------
// hip::FatBinaryInfo::ExtractFatBinaryUsingCOMGR() parses an uncompressed
// "__CLANG_OFFLOAD_BUNDLE__" image. For each matching ISA, comgr returns an
// (offset, size) pair taken straight from the attacker-controlled bundle
// header. Before the fix, PopulateCodeObjectMap() computed
//     loc = image + item.offset
// and forwarded (loc, item.size) to the program loader with no bounds check.
// A crafted bundle with offset = 0xFFFFFFF0 makes `loc` point ~4 GiB past the
// image; the first read happens in amd::Program::addDeviceProgram() ->
// amd::Elf::isElfMagic(loc), producing an out-of-bounds read / access
// violation / SIGSEGV.
//
// What this program does
// ----------------------
// It builds a malicious uncompressed offload bundle whose code-object entries
// all claim offset = 0xFFFFFFF0 (with a non-zero size, so comgr reports
// size > 0), then feeds that image to HIP through delivery paths for which the
// fix knows the true image size and can therefore reject the crafted offset:
//
//   Case 1 (all platforms): hipModuleLoad(path) - the crafted image is written
//       to a temp file; HIP maps it, so the file size bounds the parse.
//   Case 2 (file-backed ptr): hipModuleLoadData() on a view of the same file
//       mapped into this process (mmap / MapViewOfFile). Mirrors the embedded
//       / dlopen threat model where the image lives in a file-backed mapping.
//
// Expected results
// ----------------
//   * Unpatched runtime: the process crashes (access violation / SIGSEGV, or an
//     ASan heap-buffer-overflow) inside the fatbin parser.
//   * Patched runtime: every case returns a HIP error (hipErrorInvalidImage)
//     and the program prints "ALL CASES PASSED" and exits 0.
//
// Build (Windows, against a locally built runtime)
// ------------------------------------------------
//   set PATH=<install>\bin;%PATH%
//   hipcc -O0 -g hip_fatbin_oob_repro.cpp -o hip_fatbin_oob_repro.exe
//   hip_fatbin_oob_repro.exe
//
// Build (Linux)
// -------------
//   hipcc -O0 -g hip_fatbin_oob_repro.cpp -o hip_fatbin_oob_repro
//   # optionally with AddressSanitizer for a precise report:
//   hipcc -O0 -g -fsanitize=address hip_fatbin_oob_repro.cpp -o hip_fatbin_oob_repro
//
// NOTE: requires a ROCm-capable device so the HIP runtime initializes and
// populates the device ISA query set.
// ============================================================================

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

// The attacker-controlled offset from the ticket. On a 64-bit host this points
// ~4 GiB past a small image.
constexpr uint64_t kMaliciousOffset = 0xFFFFFFF0ULL;
constexpr uint64_t kMaliciousSize = 0x1000ULL;

constexpr char kBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";  // 24 chars, no NUL stored

void AppendU64(std::vector<uint8_t>& buf, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
  }
}

void AppendBundleEntry(std::vector<uint8_t>& buf, const std::string& entry_id, uint64_t offset,
                       uint64_t size) {
  AppendU64(buf, offset);
  AppendU64(buf, size);
  AppendU64(buf, entry_id.size());
  buf.insert(buf.end(), entry_id.begin(), entry_id.end());
}

// Build a malicious uncompressed offload bundle. Every entry claims the same
// out-of-bounds offset. Multiple bundle-entry-id spellings are included so the
// crafted image matches whatever form comgr's lookup expects for the current
// device / runtime (the exact match format is comgr-version specific).
std::vector<uint8_t> BuildMaliciousBundle(const std::string& gcn_arch) {
  std::vector<std::string> entry_ids;

  // arch with and without the trailing feature list (":sramecc+:xnack-").
  const std::string arch_no_feat = gcn_arch.substr(0, gcn_arch.find(':'));
  const std::string triple = "amdgcn-amd-amdhsa--";
  for (const char* kind : {"hip", "hipv4"}) {
    entry_ids.push_back(std::string(kind) + "-" + triple + gcn_arch);
    if (arch_no_feat != gcn_arch) {
      entry_ids.push_back(std::string(kind) + "-" + triple + arch_no_feat);
    }
    // SPIR-V ISA names are always inserted into the query set by the runtime.
    entry_ids.push_back(std::string(kind) + "-spirv64-amd-amdhsa--amdgcnspirv");
  }

  std::vector<uint8_t> buf;
  buf.insert(buf.end(), kBundleMagic, kBundleMagic + (sizeof(kBundleMagic) - 1));
  AppendU64(buf, entry_ids.size());
  for (const auto& id : entry_ids) {
    AppendBundleEntry(buf, id, kMaliciousOffset, kMaliciousSize);
  }
  return buf;
}

std::string TempFilePath(const char* name) {
#ifdef _WIN32
  char dir[MAX_PATH] = {0};
  DWORD n = GetTempPathA(MAX_PATH, dir);
  std::string base = (n > 0 && n < MAX_PATH) ? std::string(dir) : std::string(".\\");
  return base + name;
#else
  return std::string("/tmp/") + name;
#endif
}

bool WriteFile(const std::string& path, const std::vector<uint8_t>& data) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    return false;
  }
  const bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
  std::fclose(f);
  return ok;
}

const char* ErrName(hipError_t e) {
  const char* n = hipGetErrorName(e);
  return n != nullptr ? n : "<null>";
}

// Case 1: crafted image on disk, loaded via hipModuleLoad().
bool TestModuleLoadFromFile(const std::vector<uint8_t>& bundle) {
  const std::string path = TempFilePath("rocm26752_malicious.hipfb");
  if (!WriteFile(path, bundle)) {
    std::printf("[case1] FAILED to write %s\n", path.c_str());
    return false;
  }

  hipModule_t module = nullptr;
  const hipError_t err = hipModuleLoad(&module, path.c_str());
  std::remove(path.c_str());

  if (err == hipSuccess) {
    std::printf("[case1] UNEXPECTED hipSuccess - crafted image was accepted\n");
    if (module != nullptr) hipModuleUnload(module);
    return false;
  }
  std::printf("[case1] hipModuleLoad rejected crafted image: %s (expected)\n", ErrName(err));
  return true;
}

// Case 2: crafted image mapped into this process as a file-backed view, loaded
// via hipModuleLoadData(). Mirrors the embedded / dlopen threat model.
bool TestModuleLoadDataFileBacked(const std::vector<uint8_t>& bundle) {
  const std::string path = TempFilePath("rocm26752_malicious_map.hipfb");
  if (!WriteFile(path, bundle)) {
    std::printf("[case2] FAILED to write %s\n", path.c_str());
    return false;
  }

  bool ok = false;
  hipModule_t module = nullptr;

#ifdef _WIN32
  HANDLE hfile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
  if (hfile == INVALID_HANDLE_VALUE) {
    std::printf("[case2] FAILED to open %s\n", path.c_str());
    std::remove(path.c_str());
    return false;
  }
  HANDLE hmap = CreateFileMappingA(hfile, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (hmap == nullptr) {
    std::printf("[case2] FAILED to create file mapping\n");
    CloseHandle(hfile);
    std::remove(path.c_str());
    return false;
  }
  void* view = MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    std::printf("[case2] FAILED to map view\n");
    CloseHandle(hmap);
    CloseHandle(hfile);
    std::remove(path.c_str());
    return false;
  }

  const hipError_t err = hipModuleLoadData(&module, view);

  UnmapViewOfFile(view);
  CloseHandle(hmap);
  CloseHandle(hfile);
#else
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    std::printf("[case2] FAILED to open %s\n", path.c_str());
    std::remove(path.c_str());
    return false;
  }
  void* view = ::mmap(nullptr, bundle.size(), PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (view == MAP_FAILED) {
    std::printf("[case2] FAILED to mmap crafted image\n");
    std::remove(path.c_str());
    return false;
  }

  const hipError_t err = hipModuleLoadData(&module, view);

  ::munmap(view, bundle.size());
#endif

  std::remove(path.c_str());

  if (err == hipSuccess) {
    std::printf("[case2] UNEXPECTED hipSuccess - crafted image was accepted\n");
    if (module != nullptr) hipModuleUnload(module);
    return false;
  }
  std::printf("[case2] hipModuleLoadData rejected crafted image: %s (expected)\n", ErrName(err));
  ok = true;
  return ok;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered so output survives a hard crash
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess || device_count == 0) {
    std::printf("SKIP: no HIP device available (%s)\n", ErrName(err));
    return 0;
  }

  hipDeviceProp_t props{};
  err = hipGetDeviceProperties(&props, 0);
  if (err != hipSuccess) {
    std::printf("FAILED: hipGetDeviceProperties: %s\n", ErrName(err));
    return 1;
  }
  const std::string gcn_arch = props.gcnArchName;
  std::printf("Device 0 gcnArchName: %s\n", gcn_arch.c_str());

  const std::vector<uint8_t> bundle = BuildMaliciousBundle(gcn_arch);
  std::printf("Crafted malicious bundle: %zu bytes, offset=0x%llx size=0x%llx\n", bundle.size(),
              static_cast<unsigned long long>(kMaliciousOffset),
              static_cast<unsigned long long>(kMaliciousSize));

  bool all_ok = true;
  all_ok = TestModuleLoadFromFile(bundle) && all_ok;
  all_ok = TestModuleLoadDataFileBacked(bundle) && all_ok;

  if (all_ok) {
    std::printf("ALL CASES PASSED - crafted fat binary rejected without OOB access\n");
    return 0;
  }
  std::printf("FAILURES DETECTED\n");
  return 1;
}

// ============================================================================
// Optional: full malicious-.so / .dll reproducer (matches the dlopen story).
//
// Embed the crafted bundle bytes into a shared library's fat-binary section and
// register them via __hipRegisterFatBinary, then load the library. Extraction
// runs through the same file-backed path as Case 2. Because the image lives in
// a file-backed mapping, the fix derives the image bound from the backing file
// and rejects the crafted offset, so no OOB read occurs.
// ============================================================================
