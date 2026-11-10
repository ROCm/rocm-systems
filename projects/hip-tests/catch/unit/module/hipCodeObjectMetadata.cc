/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_module_common.hh"
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

/**
 * @addtogroup hipLibraryLoadData hipLibraryLoadData
 * @{
 * @ingroup ModuleTest
 */

/**
 * Test Description
 * ------------------------
 * - The runtime's reading of code object metadata, checked against values that follow from the
 *   kernel source (codeObjectMetadataKernels.cc), not against another runtime: kernel names,
 *   explicit argument offsets and sizes (hipKernelGetParamInfo), static LDS, launch bounds and
 *   scratch (hipFuncGetAttribute), the arguments as the kernel receives them, and buffered
 *   printf output (amdhsa.printf).
 * - Every container the runtime reads: uncompressed and compressed (CCOB v2 and v3) fat
 *   binaries, generic targets alone and next to the native ones (native wins), a raw ELF, and
 *   a code object with two metadata notes; each from a file and from memory.
 * Test source
 * ------------------------
 * - unit/module/hipCodeObjectMetadata.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.0
 */

#if HT_AMD

namespace {

struct Pair {
  int a;
  char b;
};

struct Big {
  double x[3];
};

//! Loads @p file with hipLibraryLoadFromFile, or from a buffer with hipLibraryLoadData
class Library {
 public:
  Library(const std::string& file, bool fromMemory) {
    if (fromMemory) {
      image_ = LoadModuleIntoBuffer(file.c_str());
      status_ = hipLibraryLoadData(&library_, image_.data(), nullptr, nullptr, 0, nullptr,
                                   nullptr, 0);
    } else {
      status_ = hipLibraryLoadFromFile(&library_, file.c_str(), nullptr, nullptr, 0, nullptr,
                                       nullptr, 0);
    }
  }
  ~Library() {
    if (status_ == hipSuccess) static_cast<void>(hipLibraryUnload(library_));
  }
  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;

  hipError_t status() const { return status_; }

  hipKernel_t kernel(const char* name) const {
    hipKernel_t kernel = nullptr;
    HIP_CHECK(hipLibraryGetKernel(&kernel, library_, name));
    return kernel;
  }

  hipFunction_t function(const char* name) const {
    hipFunction_t function = nullptr;
    HIP_CHECK(hipKernelGetFunction(&function, kernel(name)));
    return function;
  }

  unsigned int kernelCount() const {
    unsigned int count = 0;
    HIP_CHECK(hipLibraryGetKernelCount(&count, library_));
    return count;
  }

 private:
  std::vector<char> image_;  //!< The image, when loaded from memory
  hipLibrary_t library_ = nullptr;
  hipError_t status_ = hipErrorUnknown;
};

//! Explicit parameters of @p kernel as offset:size pairs
std::vector<std::pair<size_t, size_t>> Params(hipKernel_t kernel) {
  std::vector<std::pair<size_t, size_t>> params;
  for (size_t i = 0; i < 1024; ++i) {
    size_t offset = 0;
    size_t size = 0;
    if (hipKernelGetParamInfo(kernel, i, &offset, &size) != hipSuccess) break;
    params.emplace_back(offset, size);
  }
  return params;
}

int Attribute(hipFunction_t function, hipFunction_attribute attribute) {
  int value = -1;
  HIP_CHECK(hipFuncGetAttribute(&value, attribute, function));
  return value;
}

//! Device target id with ':' replaced by '_', as the raw ELF files are named
std::string DeviceTag(bool withFeatures) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string tag = prop.gcnArchName;
  if (!withFeatures) tag = tag.substr(0, tag.find(':'));
  for (char& c : tag) {
    if (c == ':') c = '_';
  }
  return tag;
}

//! The per-target file @p prefix-<target>.co for this device, or "" if it wasn't built
std::string PerTargetFile(const std::string& prefix) {
  for (bool features : {true, false}) {
    const std::string file = prefix + "-" + DeviceTag(features) + ".co";
    if (std::ifstream(file).good()) return file;
  }
  return std::string();
}

#if !defined(_WIN32)
//! Runs @p fn and returns what it wrote to stdout
std::string CaptureStdout(const std::function<void()>& fn) {
  fflush(stdout);
  const int saved = dup(STDOUT_FILENO);
  FILE* capture = tmpfile();
  REQUIRE(capture != nullptr);
  dup2(fileno(capture), STDOUT_FILENO);
  fn();
  fflush(stdout);
  dup2(saved, STDOUT_FILENO);
  close(saved);
  std::string text;
  rewind(capture);
  char buffer[256];
  size_t n = 0;
  while ((n = fread(buffer, 1, sizeof(buffer), capture)) > 0) text.append(buffer, n);
  fclose(capture);
  return text;
}
#endif

//! Runs target_id: the target the loaded code object was built for, e.g. "gfx11-generic"
std::string LoadedTarget(const Library& library) {
  constexpr size_t kSize = 64;
  char* out = nullptr;
  HIP_CHECK(hipMalloc(&out, kSize));
  HIP_CHECK(hipMemset(out, 0, kSize));
  void* args[] = {&out};
  HIP_CHECK(hipModuleLaunchKernel(library.function("target_id"), 1, 1, 1, 1, 1, 1, 0, nullptr,
                                  args, nullptr));
  char host[kSize] = {};
  HIP_CHECK(hipMemcpy(host, out, kSize, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(out));
  host[kSize - 1] = '\0';
  return host;
}

//! Checks everything codeObjectMetadataKernels.cc defines
void CheckMetadataKernels(const Library& library) {
  REQUIRE(library.kernelCount() == 6);
  CHECK(Params(library.kernel("target_id")) == std::vector<std::pair<size_t, size_t>>{{0, 8}});

  // args_layout(char, short, int, long long, float, double, long long*, Pair, Big)
  const std::vector<std::pair<size_t, size_t>> expected = {
      {0, 1}, {2, 2}, {4, 4}, {8, 8}, {16, 4}, {24, 8}, {32, 8}, {40, 8}, {48, 24}};
  CHECK(Params(library.kernel("args_layout")) == expected);
  CHECK(Params(library.kernel("no_args")).empty());
  CHECK(Params(library.kernel("shared_static")) ==
        std::vector<std::pair<size_t, size_t>>{{0, 8}});
  CHECK(Params(library.kernel("scratch")) ==
        std::vector<std::pair<size_t, size_t>>{{0, 8}, {8, 4}});
  CHECK(Params(library.kernel("print_values")) ==
        std::vector<std::pair<size_t, size_t>>{{0, 4}, {8, 8}});

  const hipFunction_t shared = library.function("shared_static");
  CHECK(Attribute(shared, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES) == 1024);
  CHECK(Attribute(shared, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) == 128);
  CHECK(Attribute(library.function("scratch"), HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES) >= 256);
  CHECK(Attribute(library.function("no_args"), HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES) == 0);
  CHECK(Attribute(library.function("no_args"), HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES) == 0);

  // The arguments as the kernel receives them
  long long* out = nullptr;
  HIP_CHECK(hipMalloc(&out, 9 * sizeof(long long)));
  HIP_CHECK(hipMemset(out, 0, 9 * sizeof(long long)));
  char c = -7;
  short s = 1234;
  int i = -123456;
  long long l = 0x123456789abcll;
  float f = 1.5f;
  double d = -2.25;
  Pair p = {42, 'x'};
  Big b = {{1.0, 2.0, 3.0}};
  void* args[] = {&c, &s, &i, &l, &f, &d, &out, &p, &b};
  HIP_CHECK(hipModuleLaunchKernel(library.function("args_layout"), 1, 1, 1, 64, 1, 1, 0, nullptr,
                                  args, nullptr));
  long long host[9] = {};
  HIP_CHECK(hipMemcpy(host, out, sizeof(host), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(out));
  const long long want[9] = {-7, 1234, -123456, 0x123456789abcll, 1500, -2250, 42, 'x', 321};
  for (int k = 0; k < 9; ++k) {
    INFO("argument " << k);
    CHECK(host[k] == want[k]);
  }

#if !defined(_WIN32)
  // Buffered printf: the formats come from amdhsa.printf
  int v = 7;
  double dv = 2.5;
  void* printArgs[] = {&v, &dv};
  const std::string text = CaptureStdout([&] {
    HIP_CHECK(hipModuleLaunchKernel(library.function("print_values"), 1, 1, 1, 32, 1, 1, 0,
                                    nullptr, printArgs, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  });
  CHECK(text.find("metadata printf 7 2.50 ok\n") != std::string::npos);
  CHECK(text.find("metadata second 21\n") != std::string::npos);
#endif
}

}  // namespace

/**
 * Test Description
 * ------------------------
 * - Fat binaries: uncompressed, compressed (CCOB v3, the bundler's default) and CCOB v2, each
 *   loaded from a file and from memory.
 */
HIP_TEST_CASE(Unit_CodeObjectMetadata_FatBinary) {
  const char* file = nullptr;
  SECTION("uncompressed") { file = "codeObjectMetadata.code"; }
  SECTION("compressed v3") { file = "codeObjectMetadataCompressed.code"; }
  SECTION("compressed v2") { file = "codeObjectMetadataCompressedV2.code"; }
  for (bool fromMemory : {false, true}) {
    INFO(file << (fromMemory ? " from memory" : " from file"));
    Library library(file, fromMemory);
    HIP_CHECK(library.status());
    CheckMetadataKernels(library);
  }
}

/**
 * Test Description
 * ------------------------
 * - A fat binary with generic targets only (code object v6): the device's generic target is
 *   selected. Skipped on devices that have none.
 */
HIP_TEST_CASE(Unit_CodeObjectMetadata_GenericTarget) {
  Library library("codeObjectMetadataGeneric.code", false);
  if (library.status() == hipErrorNoBinaryForGpu) {
    HIP_SKIP_TEST("The device has no generic target");
    return;
  }
  HIP_CHECK(library.status());
  CheckMetadataKernels(library);
  CHECK(LoadedTarget(library).find("-generic") != std::string::npos);
}

/**
 * Test Description
 * ------------------------
 * - A fat binary with the native targets and all generic targets: the device's native entry
 *   is loaded, not the generic one.
 */
HIP_TEST_CASE(Unit_CodeObjectMetadata_NativeBeforeGeneric) {
  for (bool fromMemory : {false, true}) {
    INFO((fromMemory ? "from memory" : "from file"));
    Library library("codeObjectMetadataNativeAndGeneric.code", fromMemory);
    HIP_CHECK(library.status());
    CheckMetadataKernels(library);
    const std::string target = LoadedTarget(library);
    INFO("loaded " << target);
    CHECK(target.find("-generic") == std::string::npos);
    CHECK(target.substr(0, target.find(':')) == DeviceTag(false));
  }
}

#if !defined(_WIN32)
/**
 * Test Description
 * ------------------------
 * - A raw ELF code object for the device, from a file and from memory.
 */
HIP_TEST_CASE(Unit_CodeObjectMetadata_RawElf) {
  const std::string file = PerTargetFile("codeObjectMetadata");
  if (file.empty()) {
    HIP_SKIP_TEST("No raw code object was built for this device");
    return;
  }
  for (bool fromMemory : {false, true}) {
    INFO(file << (fromMemory ? " from memory" : " from file"));
    Library library(file, fromMemory);
    HIP_CHECK(library.status());
    CheckMetadataKernels(library);
  }
}

/**
 * Test Description
 * ------------------------
 * - A code object linked from two separately assembled objects carries two metadata notes;
 *   the kernels of both are found, with their own arguments, and run.
 */
HIP_TEST_CASE(Unit_CodeObjectMetadata_MultipleNotes) {
  const std::string file = PerTargetFile("codeObjectMultiNote");
  if (file.empty()) {
    HIP_SKIP_TEST("No multi-note code object was built for this device");
    return;
  }
  Library library(file, false);
  HIP_CHECK(library.status());
  REQUIRE(library.kernelCount() == 2);
  CHECK(Params(library.kernel("note_a")) == std::vector<std::pair<size_t, size_t>>{{0, 8}});
  CHECK(Params(library.kernel("note_b")) ==
        std::vector<std::pair<size_t, size_t>>{{0, 8}, {8, 4}});

  int* out = nullptr;
  HIP_CHECK(hipMalloc(&out, 2 * sizeof(int)));
  int* outB = out + 1;
  int v = 21;
  void* argsA[] = {&out};
  void* argsB[] = {&outB, &v};
  HIP_CHECK(hipModuleLaunchKernel(library.function("note_a"), 1, 1, 1, 1, 1, 1, 0, nullptr, argsA,
                                  nullptr));
  HIP_CHECK(hipModuleLaunchKernel(library.function("note_b"), 1, 1, 1, 1, 1, 1, 0, nullptr, argsB,
                                  nullptr));
  int host[2] = {};
  HIP_CHECK(hipMemcpy(host, out, sizeof(host), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(out));
  CHECK(host[0] == 11);
  CHECK(host[1] == 42);
}
#endif

/**
 * Test Description
 * ------------------------
 * - A compressed bundle whose header version (4) no runtime or comgr reads yet fails to load
 *   with hipErrorInvalidImage, from a file and from memory, without a crash.
 */
HIP_TEST_CASE(Unit_CodeObjectMetadata_UnknownCompressedVersion) {
  std::vector<char> image = LoadModuleIntoBuffer("codeObjectMetadataCompressed.code");
  REQUIRE(image.size() > 32);
  REQUIRE(std::memcmp(image.data(), "CCOB", 4) == 0);
  image[4] = 4;  // little-endian 16-bit version
  image[5] = 0;

  hipModule_t module = nullptr;
  HIP_CHECK_ERROR(hipModuleLoadData(&module, image.data()), hipErrorInvalidImage);

  const char* file = "codeObjectMetadataVersion4.code";
  {
    std::ofstream out(file, std::ios::binary);
    out.write(image.data(), static_cast<std::streamsize>(image.size()));
  }
  HIP_CHECK_ERROR(hipModuleLoad(&module, file), hipErrorInvalidImage);
  std::remove(file);
}

/**
 * Test Description
 * ------------------------
 * - The module API reads the same metadata: function count, attributes and a launch through
 *   hipModuleLoad of the compressed fat binary.
 */
HIP_TEST_CASE(Unit_CodeObjectMetadata_ModuleApi) {
  ModuleGuard module = ModuleGuard::LoadModule("codeObjectMetadataCompressed.code");
  unsigned int count = 0;
  HIP_CHECK(hipModuleGetFunctionCount(&count, module.module()));
  CHECK(count == 6);
  const hipFunction_t shared = GetKernel(module.module(), "shared_static");
  CHECK(Attribute(shared, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES) == 1024);
  CHECK(Attribute(shared, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) == 128);

  int* out = nullptr;
  HIP_CHECK(hipMalloc(&out, 128 * sizeof(int)));
  void* args[] = {&out};
  HIP_CHECK(hipModuleLaunchKernel(shared, 1, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr));
  std::vector<int> host(128);
  HIP_CHECK(hipMemcpy(host.data(), out, 128 * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(out));
  for (int t = 0; t < 128; ++t) {
    INFO("thread " << t);
    CHECK(host[t] == 127 - t + 1);
  }
}

#endif  // HT_AMD

/**
 * End doxygen group ModuleTest.
 * @}
 */
