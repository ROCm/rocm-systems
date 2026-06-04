#include "rocjitsu/code/amdgpu_elf_reader.h"
#include "rocjitsu_fuzzer/afl_runtime.h"

#include "minimal_amdgpu_elf.h"

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <fcntl.h>
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#include <sys/shm.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef ROCJITSU_AFL_PRELOAD_PATH
#error "ROCJITSU_AFL_PRELOAD_PATH must point to librocjitsu_afl_preload.so"
#endif

#ifndef ROCJITSU_AFL_FAKE_HIP_RUNTIME_PATH
#error "ROCJITSU_AFL_FAKE_HIP_RUNTIME_PATH must point to the fake HIP runtime"
#endif

#ifndef ROCJITSU_AFL_FAKE_HSA_RUNTIME_PATH
#error "ROCJITSU_AFL_FAKE_HSA_RUNTIME_PATH must point to the fake HSA runtime"
#endif

namespace {

using PersistentFn = int (*)();
using HipModuleLoadDataFn = int (*)(void **, const void *);
using HipModuleUnloadFn = int (*)(void *);
using CreateFromMemoryFn = hsa_status_t (*)(const void *, size_t, hsa_code_object_reader_t *);
using CreateFromFileFn = hsa_status_t (*)(hsa_file_t, hsa_code_object_reader_t *);
using CreateFromFileWithOffsetFn = hsa_status_t (*)(hsa_file_t, size_t, size_t,
                                                    hsa_code_object_reader_t *);
using DestroyFn = hsa_status_t (*)(hsa_code_object_reader_t);
using ResetFakeHipFn = void (*)();
using ResetFakeHsaFn = void (*)();
using LastCreateKindFn = int (*)();
using DestroyCallsFn = int (*)();
using ReaderDataFn = const uint8_t *(*)(uint64_t);
using ReaderSizeFn = size_t (*)(uint64_t);
using ModuleCallsFn = int (*)();
using ModuleImageFn = const uint8_t *(*)();
using ModuleImageSizeFn = size_t (*)();

constexpr int kCreateKindMemory = 1;
constexpr int kCreateKindFile = 2;

class DlHandle {
public:
  DlHandle() = default;
  explicit DlHandle(const char *path, int flags) : handle_(dlopen(path, flags)) {}
  DlHandle(const DlHandle &) = delete;
  DlHandle &operator=(const DlHandle &) = delete;
  ~DlHandle() {
    if (handle_ != nullptr)
      dlclose(handle_);
  }

  void open(const char *path, int flags) { handle_ = dlopen(path, flags); }
  void *symbol(const char *name) const { return dlsym(handle_, name); }
  void *get() const { return handle_; }

private:
  void *handle_ = nullptr;
};

class SharedMemory {
public:
  SharedMemory() : id_(shmget(IPC_PRIVATE, rocjitsu::fuzzer::afl::kMapSize, IPC_CREAT | 0600)) {
    EXPECT_GE(id_, 0);
    if (id_ >= 0) {
      data_ = static_cast<uint8_t *>(shmat(id_, nullptr, 0));
      EXPECT_NE(data_, reinterpret_cast<uint8_t *>(-1));
      if (data_ != reinterpret_cast<uint8_t *>(-1))
        std::memset(data_, 0, rocjitsu::fuzzer::afl::kMapSize);
    }
  }

  SharedMemory(const SharedMemory &) = delete;
  SharedMemory &operator=(const SharedMemory &) = delete;

  ~SharedMemory() {
    if (data_ != nullptr && data_ != reinterpret_cast<uint8_t *>(-1))
      shmdt(data_);
    if (id_ >= 0)
      shmctl(id_, IPC_RMID, nullptr);
  }

  int id() const { return id_; }

private:
  int id_ = -1;
  uint8_t *data_ = nullptr;
};

class TempFile {
public:
  explicit TempFile(const std::vector<uint8_t> &bytes) {
    char path[] = "/tmp/rocjitsu-afl-hsa-reader-XXXXXX";
    fd_ = mkstemp(path);
    EXPECT_GE(fd_, 0);
    if (fd_ >= 0) {
      path_ = path;
      EXPECT_EQ(write(fd_, bytes.data(), bytes.size()), static_cast<ssize_t>(bytes.size()));
      EXPECT_EQ(lseek(fd_, 0, SEEK_SET), 0);
    }
  }

  TempFile(const TempFile &) = delete;
  TempFile &operator=(const TempFile &) = delete;

  ~TempFile() {
    if (fd_ >= 0)
      close(fd_);
    if (!path_.empty())
      unlink(path_.c_str());
  }

  int fd() const { return fd_; }

private:
  int fd_ = -1;
  std::string path_;
};

struct PreloadHarness {
  SharedMemory shm;
  DlHandle preload;
  DlHandle fake_hip;
  DlHandle fake_hsa;

  PersistentFn persistent_begin = nullptr;
  HipModuleLoadDataFn hip_module_load_data = nullptr;
  HipModuleUnloadFn hip_module_unload = nullptr;
  CreateFromMemoryFn create_from_memory = nullptr;
  CreateFromFileFn create_from_file = nullptr;
  CreateFromFileWithOffsetFn create_from_file_with_offset = nullptr;
  DestroyFn destroy = nullptr;
  ResetFakeHipFn reset_fake_hip = nullptr;
  ResetFakeHsaFn reset_fake_hsa = nullptr;
  ModuleCallsFn module_load_data_calls = nullptr;
  ModuleCallsFn module_unload_calls = nullptr;
  ModuleImageFn last_module_image = nullptr;
  ModuleImageSizeFn last_module_image_size = nullptr;
  LastCreateKindFn last_create_kind = nullptr;
  DestroyCallsFn destroy_calls = nullptr;
  ReaderDataFn reader_data = nullptr;
  ReaderSizeFn reader_size = nullptr;

  PreloadHarness() {
    setenv("__AFL_SHM_ID", std::to_string(shm.id()).c_str(), 1);
    setenv("ROCJITSU_AFL_HIP_RUNTIME_PATH", ROCJITSU_AFL_FAKE_HIP_RUNTIME_PATH, 1);
    setenv("ROCJITSU_AFL_HSA_RUNTIME_PATH", ROCJITSU_AFL_FAKE_HSA_RUNTIME_PATH, 1);

    preload.open(ROCJITSU_AFL_PRELOAD_PATH, RTLD_LAZY | RTLD_LOCAL);
    fake_hip.open(ROCJITSU_AFL_FAKE_HIP_RUNTIME_PATH, RTLD_LAZY | RTLD_LOCAL);
    fake_hsa.open(ROCJITSU_AFL_FAKE_HSA_RUNTIME_PATH, RTLD_LAZY | RTLD_LOCAL);
    EXPECT_NE(preload.get(), nullptr) << dlerror();
    EXPECT_NE(fake_hip.get(), nullptr) << dlerror();
    EXPECT_NE(fake_hsa.get(), nullptr) << dlerror();

    persistent_begin =
        reinterpret_cast<PersistentFn>(preload.symbol("rocjitsu_afl_persistent_begin"));
    hip_module_load_data =
        reinterpret_cast<HipModuleLoadDataFn>(preload.symbol("hipModuleLoadData"));
    hip_module_unload = reinterpret_cast<HipModuleUnloadFn>(preload.symbol("hipModuleUnload"));
    create_from_memory = reinterpret_cast<CreateFromMemoryFn>(
        preload.symbol("hsa_code_object_reader_create_from_memory"));
    create_from_file = reinterpret_cast<CreateFromFileFn>(
        preload.symbol("hsa_code_object_reader_create_from_file"));
    create_from_file_with_offset = reinterpret_cast<CreateFromFileWithOffsetFn>(
        preload.symbol("hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size"));
    destroy = reinterpret_cast<DestroyFn>(preload.symbol("hsa_code_object_reader_destroy"));

    reset_fake_hip = reinterpret_cast<ResetFakeHipFn>(fake_hip.symbol("rocfuzz_fake_hip_reset"));
    module_load_data_calls =
        reinterpret_cast<ModuleCallsFn>(fake_hip.symbol("rocfuzz_fake_hip_module_load_data_calls"));
    module_unload_calls =
        reinterpret_cast<ModuleCallsFn>(fake_hip.symbol("rocfuzz_fake_hip_module_unload_calls"));
    last_module_image =
        reinterpret_cast<ModuleImageFn>(fake_hip.symbol("rocfuzz_fake_hip_last_module_image"));
    last_module_image_size = reinterpret_cast<ModuleImageSizeFn>(
        fake_hip.symbol("rocfuzz_fake_hip_last_module_image_size"));
    reset_fake_hsa = reinterpret_cast<ResetFakeHsaFn>(fake_hsa.symbol("rocfuzz_fake_hsa_reset"));
    last_create_kind =
        reinterpret_cast<LastCreateKindFn>(fake_hsa.symbol("rocfuzz_fake_hsa_last_create_kind"));
    destroy_calls =
        reinterpret_cast<DestroyCallsFn>(fake_hsa.symbol("rocfuzz_fake_hsa_destroy_calls"));
    reader_data = reinterpret_cast<ReaderDataFn>(fake_hsa.symbol("rocfuzz_fake_hsa_reader_data"));
    reader_size = reinterpret_cast<ReaderSizeFn>(fake_hsa.symbol("rocfuzz_fake_hsa_reader_size"));
  }

  ~PreloadHarness() {
    unsetenv("ROCJITSU_AFL_HSA_RUNTIME_PATH");
    unsetenv("ROCJITSU_AFL_HIP_RUNTIME_PATH");
    unsetenv("__AFL_SHM_ID");
  }

  void assert_ready() const {
    ASSERT_NE(persistent_begin, nullptr);
    ASSERT_NE(hip_module_load_data, nullptr);
    ASSERT_NE(hip_module_unload, nullptr);
    ASSERT_NE(create_from_memory, nullptr);
    ASSERT_NE(create_from_file, nullptr);
    ASSERT_NE(create_from_file_with_offset, nullptr);
    ASSERT_NE(destroy, nullptr);
    ASSERT_NE(reset_fake_hip, nullptr);
    ASSERT_NE(reset_fake_hsa, nullptr);
    ASSERT_NE(module_load_data_calls, nullptr);
    ASSERT_NE(module_unload_calls, nullptr);
    ASSERT_NE(last_module_image, nullptr);
    ASSERT_NE(last_module_image_size, nullptr);
    ASSERT_NE(last_create_kind, nullptr);
    ASSERT_NE(destroy_calls, nullptr);
    ASSERT_NE(reader_data, nullptr);
    ASSERT_NE(reader_size, nullptr);
  }
};

std::vector<uint8_t> reader_bytes(const PreloadHarness &harness, hsa_code_object_reader_t reader) {
  const uint8_t *data = harness.reader_data(reader.handle);
  const size_t size = harness.reader_size(reader.handle);
  EXPECT_NE(data, nullptr);
  EXPECT_GT(size, 0u);
  if (data == nullptr || size == 0)
    return {};
  return std::vector<uint8_t>(data, data + size);
}

} // namespace

TEST(RocjitsuAflPreloadHsaReaderTest, RewritesRawMemoryReaderAndOwnsBytesUntilDestroy) {
  PreloadHarness harness;
  harness.assert_ready();
  harness.reset_fake_hsa();
  ASSERT_EQ(harness.persistent_begin(), 0);

  const auto image = rocjitsu::fuzzer::afl::test::make_minimal_amdgpu_elf();
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(harness.create_from_memory(image.data(), image.size(), &reader), HSA_STATUS_SUCCESS);

  EXPECT_EQ(harness.last_create_kind(), kCreateKindMemory);
  EXPECT_NE(harness.reader_data(reader.handle), image.data());
  const auto rewritten = reader_bytes(harness, reader);
  EXPECT_TRUE(rocjitsu::is_supported_amdgpu_elf(rewritten));
  EXPECT_GT(rewritten.size(), image.size());
  EXPECT_NE(rewritten, image);

  EXPECT_EQ(harness.destroy(reader), HSA_STATUS_SUCCESS);
  EXPECT_EQ(harness.destroy_calls(), 1);
  EXPECT_EQ(harness.reader_data(reader.handle), nullptr);
}

TEST(RocjitsuAflPreloadHsaReaderTest, RewritesRawHipModuleLoadData) {
  PreloadHarness harness;
  harness.assert_ready();
  harness.reset_fake_hip();

  const auto image = rocjitsu::fuzzer::afl::test::make_minimal_amdgpu_elf();
  void *module = nullptr;
  ASSERT_EQ(harness.hip_module_load_data(&module, image.data()), 0);

  EXPECT_EQ(harness.module_load_data_calls(), 1);
  ASSERT_NE(module, nullptr);
  const uint8_t *loaded_image = harness.last_module_image();
  ASSERT_NE(loaded_image, nullptr);
  EXPECT_NE(loaded_image, image.data());

  const size_t loaded_image_size = harness.last_module_image_size();
  ASSERT_GT(loaded_image_size, image.size());
  const std::vector<uint8_t> loaded(loaded_image, loaded_image + loaded_image_size);
  EXPECT_TRUE(rocjitsu::is_supported_amdgpu_elf(loaded));
  EXPECT_NE(loaded, image);

  ASSERT_EQ(harness.hip_module_unload(module), 0);
  EXPECT_EQ(harness.module_unload_calls(), 1);
}

TEST(RocjitsuAflPreloadHsaReaderTest, ForwardsNonRawMemoryReader) {
  PreloadHarness harness;
  harness.assert_ready();
  harness.reset_fake_hsa();

  const std::vector<uint8_t> not_elf = {'n', 'o', 't', '-', 'e', 'l', 'f'};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(harness.create_from_memory(not_elf.data(), not_elf.size(), &reader),
            HSA_STATUS_SUCCESS);

  EXPECT_EQ(harness.last_create_kind(), kCreateKindMemory);
  EXPECT_EQ(harness.reader_data(reader.handle), not_elf.data());
  EXPECT_EQ(harness.reader_size(reader.handle), not_elf.size());
  EXPECT_EQ(harness.destroy(reader), HSA_STATUS_SUCCESS);
}

TEST(RocjitsuAflPreloadHsaReaderTest, RewritesRawFileWithOffsetReader) {
  PreloadHarness harness;
  harness.assert_ready();
  harness.reset_fake_hsa();
  ASSERT_EQ(harness.persistent_begin(), 0);

  const std::vector<uint8_t> prefix = {'p', 'a', 'd'};
  const auto image = rocjitsu::fuzzer::afl::test::make_minimal_amdgpu_elf();
  std::vector<uint8_t> file_bytes(prefix);
  file_bytes.insert(file_bytes.end(), image.begin(), image.end());
  TempFile file(file_bytes);
  ASSERT_GE(file.fd(), 0);

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(harness.create_from_file_with_offset(file.fd(), prefix.size(), image.size(), &reader),
            HSA_STATUS_SUCCESS);

  EXPECT_EQ(harness.last_create_kind(), kCreateKindMemory);
  const auto rewritten = reader_bytes(harness, reader);
  EXPECT_TRUE(rocjitsu::is_supported_amdgpu_elf(rewritten));
  EXPECT_GT(rewritten.size(), image.size());
  EXPECT_NE(rewritten, image);
  EXPECT_EQ(harness.destroy(reader), HSA_STATUS_SUCCESS);
}

TEST(RocjitsuAflPreloadHsaReaderTest, ForwardsNonRawFileReader) {
  PreloadHarness harness;
  harness.assert_ready();
  harness.reset_fake_hsa();

  const std::vector<uint8_t> not_elf = {'n', 'o', 't', '-', 'e', 'l', 'f'};
  TempFile file(not_elf);
  ASSERT_GE(file.fd(), 0);

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(harness.create_from_file(file.fd(), &reader), HSA_STATUS_SUCCESS);

  EXPECT_EQ(harness.last_create_kind(), kCreateKindFile);
  EXPECT_EQ(harness.destroy(reader), HSA_STATUS_SUCCESS);
}
