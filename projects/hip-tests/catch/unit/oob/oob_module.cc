#include <hip_test_common.hh>
#include <hip/hiprtc.h>
#include "hip_test_filesystem.hh"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#if HT_AMD
// POSIX-only helpers for the fat-binary readable-size bounds tests below. This
// directory (unit/oob) is gated to UNIX, so these headers are always available.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <system_error>
#endif

// Only the valid code object is shipped alongside the test binary (cwd-relative
// basename). The malformed variants that exercise getElfSize's rejection paths
// are synthesized here at runtime instead of being generated and installed as
// fixtures: installing corrupt ELFs broke DEB packaging, because dh_makeshlibs
// runs `objdump -p` on every installed ELF and aborts on a bad header. The
// mutation recipes below are ported from the former gen_evil_elfs.py.
constexpr std::string_view kValidModule = "oob_kernel.code";

namespace {
using Bytes = std::vector<char>;

// Elf64_Ehdr field offsets we mutate.
constexpr size_t kEShoff = 40;      // e_shoff (u64)
constexpr size_t kEShentsize = 58;  // e_shentsize (u16)
constexpr size_t kEShnum = 60;      // e_shnum (u16)

constexpr char kElfMagic[4] = {0x7f, 'E', 'L', 'F'};

Bytes ReadFile(std::string_view path) {
  std::ifstream f(std::string(path), std::ios::binary | std::ios::ate);
  REQUIRE(f.good());
  std::streamsize size = f.tellg();
  REQUIRE(size >= 0);
  f.seekg(0);
  Bytes buf(static_cast<size_t>(size));
  f.read(buf.data(), size);
  REQUIRE(f.good());
  return buf;
}

// AMDGPU is little-endian and so are all supported hosts, so a plain copy matches
// the '<' packing the original Python used.
uint16_t Rd16(const Bytes& b, size_t o) {
  uint16_t v;
  std::memcpy(&v, b.data() + o, sizeof(v));
  return v;
}
uint64_t Rd64(const Bytes& b, size_t o) {
  uint64_t v;
  std::memcpy(&v, b.data() + o, sizeof(v));
  return v;
}
void Wr16(Bytes& b, size_t o, uint16_t v) { std::memcpy(b.data() + o, &v, sizeof(v)); }
void Wr32(Bytes& b, size_t o, uint32_t v) { std::memcpy(b.data() + o, &v, sizeof(v)); }
void Wr64(Bytes& b, size_t o, uint64_t v) { std::memcpy(b.data() + o, &v, sizeof(v)); }

// Extract the first AMDGPU ELF from a clang offload bundle (pass a raw ELF through).
// Keeps arch aligned with whatever oob_kernel.code was just compiled for.
Bytes ExtractElf(const Bytes& data) {
  if (data.size() >= 4 && std::memcmp(data.data(), kElfMagic, 4) == 0) {
    return data;
  }
  static constexpr char kBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  // Magic (24) + entry count (8) must be present before any header read.
  REQUIRE(data.size() >= 32);
  REQUIRE(std::memcmp(data.data(), kBundleMagic, 24) == 0);
  const uint64_t total = data.size();
  size_t off = 24;
  uint64_t num = Rd64(data, off);
  off += 8;
  for (uint64_t i = 0; i < num; ++i) {
    // Each entry header is 24 bytes (offset, size, id_len) + a variable id.
    REQUIRE(off + 24 <= total);
    uint64_t entry_off = Rd64(data, off);
    uint64_t entry_size = Rd64(data, off + 8);
    uint64_t id_len = Rd64(data, off + 16);
    REQUIRE(id_len <= total - (off + 24));
    off += 24 + id_len;
    // Reject an entry whose payload range overflows or spills past the buffer.
    if (entry_size >= 4 && entry_off <= total && entry_size <= total - entry_off &&
        std::memcmp(data.data() + entry_off, kElfMagic, 4) == 0) {
      return Bytes(data.begin() + entry_off, data.begin() + entry_off + entry_size);
    }
  }
  FAIL("no ELF entry in bundle");
  return {};
}

// e_shnum past the buffer.
Bytes MakeHugeShnum(const Bytes& elf) {
  Bytes b = elf;
  Wr16(b, kEShnum, 0xFFFF);
  return b;
}

// e_shoff: section table start past end of file.
Bytes MakeBadShoff(const Bytes& elf) {
  Bytes b = elf;
  Wr64(b, kEShoff, uint64_t(1) << 40);
  return b;
}

// e_shoff + e_shnum: table starts in bounds but its end spills past the file.
Bytes MakeTableSpill(const Bytes& elf) {
  Bytes b = elf;
  Wr64(b, kEShoff, elf.size() - 10);
  Wr16(b, kEShnum, 1);
  return b;
}

// sh_offset + sh_size on one section overflows a uint64.
Bytes MakeShOverflow(const Bytes& elf) {
  Bytes b = elf;
  uint64_t shoff = Rd64(elf, kEShoff);
  uint16_t shentsize = Rd16(elf, kEShentsize);
  size_t s5 = shoff + 5 * shentsize;
  REQUIRE(s5 + 40 <= b.size());
  Wr32(b, s5 + 4, 1);              // sh_type = SHT_PROGBITS
  Wr64(b, s5 + 24, ~uint64_t(0));  // sh_offset -> overflow on + sh_size
  Wr64(b, s5 + 32, 0x10);          // sh_size
  return b;
}

// Writes a byte buffer to a uniquely-named temp .code file and removes it on scope
// exit, so the file-backed loader path (hipModuleLoad) sees an exact file size.
class TempCodeObject {
 public:
  TempCodeObject(std::string_view name, const Bytes& data) {
    // Address of the source buffer gives a unique-enough name per run without an
    // OS-specific call (getpid), matching the pattern in hrr_workload_test.cc.
    path_ = fs::temp_directory_path() /
            (std::string("oob_") + std::string(name) + "_" +
             std::to_string(reinterpret_cast<uintptr_t>(data.data())) + ".code");
    std::ofstream f(path_, std::ios::binary);
    f.write(data.data(), static_cast<std::streamsize>(data.size()));
    f.close();
    REQUIRE(f.good());
  }
  ~TempCodeObject() {
    std::error_code ec;
    fs::remove(path_, ec);
  }
  TempCodeObject(const TempCodeObject&) = delete;
  TempCodeObject& operator=(const TempCodeObject&) = delete;

  std::string path() const { return path_.string(); }

 private:
  fs::path path_;
};
}  // namespace

// File-backed path: hipModuleLoad(fname) - image_size_ is the exact file size.
HIP_TEST_CASE(OOB_hip_module_load_over) {
  Bytes elf = ExtractElf(ReadFile(kValidModule));

  SECTION("valid - sanity") {
    hipModule_t module{};
    HIP_CHECK(hipModuleLoad(&module, std::string(kValidModule).c_str()));
    HIP_CHECK(hipModuleUnload(module));
  }

  SECTION("huge shnum") {
    TempCodeObject fixture("huge_shnum", MakeHugeShnum(elf));
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoad(&module, fixture.path().c_str()), hipErrorInvalidImage);
  }

  SECTION("bad shoff") {
    TempCodeObject fixture("bad_shoff", MakeBadShoff(elf));
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoad(&module, fixture.path().c_str()), hipErrorInvalidImage);
  }

  SECTION("table spill") {
    TempCodeObject fixture("table_spill", MakeTableSpill(elf));
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoad(&module, fixture.path().c_str()), hipErrorInvalidImage);
  }

  SECTION("sh overflow") {
    TempCodeObject fixture("sh_overflow", MakeShOverflow(elf));
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoad(&module, fixture.path().c_str()), hipErrorInvalidImage);
  }
}

// In-memory path: hipModuleLoadData carries no length, so only malformations
// detectable without one can be asserted. Length-dependent cases (e.g. a
// corrupt e_shoff) are covered against the file path in OOB_hip_module_load_over.
HIP_TEST_CASE(OOB_hip_module_load_data_over) {
  Bytes elf = ExtractElf(ReadFile(kValidModule));

  SECTION("sh overflow in-memory") {
    Bytes buf = MakeShOverflow(elf);
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoadData(&module, buf.data()), hipErrorInvalidImage);
  }
}

// Runtime-compiled code objects live in an anonymous heap buffer and must still
// load through hipModuleLoadData after the bounds hardening.
HIP_TEST_CASE(OOB_hiprtc_roundtrip_loads) {
  static constexpr char kSource[] = "extern \"C\" __global__ void nop() {}\n";

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  std::string arch = std::string("--offload-arch=") + props.gcnArchName;
  const char* options[] = {arch.c_str()};

  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, kSource, "nop.cu", 0, nullptr, nullptr));
  HIPRTC_CHECK(hiprtcCompileProgram(prog, 1, options));

  size_t code_size = 0;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));
  std::vector<char> code(code_size);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  hipModule_t module{};
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  HIP_CHECK(hipModuleUnload(module));
}

// ---------------------------------------------------------------------------
// Size-bounds tests for fat-binary parsing. Only file-backed loads have an
// exact size, so bounds are asserted there; pointer inputs carry no length.
// ---------------------------------------------------------------------------
#if HT_AMD

namespace {

// Magic string for ROCm compressed clang offload bundles.
constexpr char kCompressedBundleMagic[] = "CCOB";

// Builds a `size`-byte compressed offload bundle: the "CCOB" magic with
// `total_size` written to the totalSize field (offset 8); remaining bytes zero.
Bytes MakeCompressedBundle(size_t size, uint32_t total_size) {
  REQUIRE(size >= 12);  // magic (4) through totalSize (offset 8, 4 bytes)
  Bytes image(size, 0);
  std::memcpy(image.data(), kCompressedBundleMagic, 4);
  std::memcpy(image.data() + 8, &total_size, sizeof(total_size));
  return image;
}

// Read-only file mapping used by the valid-image false-positive guard.
class FileBackedMapping {
 public:
  explicit FileBackedMapping(const char* path) {
    fd_ = open(path, O_RDONLY);
    REQUIRE(fd_ >= 0);

    struct stat st {};
    REQUIRE(fstat(fd_, &st) == 0);
    REQUIRE(st.st_size > 0);
    size_ = static_cast<size_t>(st.st_size);

    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    REQUIRE(data_ != MAP_FAILED);
  }

  ~FileBackedMapping() {
    if (data_ != nullptr && data_ != MAP_FAILED) {
      munmap(data_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  FileBackedMapping(const FileBackedMapping&) = delete;
  FileBackedMapping& operator=(const FileBackedMapping&) = delete;

  const void* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  int fd_ = -1;
  void* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace

/**
 * Loads a valid file-backed compressed module and runs its kernel, guarding
 * against the bounds checks regressing into false positives.
 */
HIP_TEST_CASE(OOB_hipModuleLoadData_Positive_ValidFileBackedImage) {
  FileBackedMapping mapping("oob_copyKernelCompressed.code");

  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, mapping.data()));
  REQUIRE(module != nullptr);

  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "copy_ker"));
  REQUIRE(kernel != nullptr);

  constexpr unsigned int kLen = 64;
  constexpr size_t kBytes = kLen * sizeof(int);
  std::vector<int> host_in(kLen), host_out(kLen, 0);
  for (unsigned int i = 0; i < kLen; ++i) {
    host_in[i] = static_cast<int>(i);
  }

  int* device_in = nullptr;
  int* device_out = nullptr;
  HIP_CHECK(hipMalloc(&device_in, kBytes));
  HIP_CHECK(hipMalloc(&device_out, kBytes));
  HIP_CHECK(hipMemcpy(device_in, host_in.data(), kBytes, hipMemcpyHostToDevice));

  struct {
    void* Ad;
    void* Bd;
    size_t size;
  } args{device_in, device_out, kLen};
  size_t args_size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &args_size, HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, kLen, 1, 1, 0, nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(host_out.data(), device_out, kBytes, hipMemcpyDeviceToHost));
  for (unsigned int i = 0; i < kLen; ++i) {
    REQUIRE(host_out[i] == host_in[i]);
  }

  HIP_CHECK(hipFree(device_in));
  HIP_CHECK(hipFree(device_out));
  HIP_CHECK(hipModuleUnload(module));
}

/**
 * Loads a file whose compressed header declares a totalSize larger than the
 * file. hipModuleLoad must reject it with hipErrorInvalidImage.
 */
HIP_TEST_CASE(OOB_hipModuleLoad_Negative_OutOfBoundsTotalSize) {
  // Full header, but totalSize exceeds the file size.
  const Bytes payload = MakeCompressedBundle(32, 0xFFFFFFFFu);

  TempCodeObject image("fatbin_bounds", payload);
  hipModule_t module = nullptr;
  HIP_CHECK_ERROR(hipModuleLoad(&module, image.path().c_str()), hipErrorInvalidImage);
}

#endif  // HT_AMD
