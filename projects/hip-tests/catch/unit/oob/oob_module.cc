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

// In-memory path: hipModuleLoadData(image) - no length, bound is derived from the
// mapping that contains the (anonymous heap) buffer. These cases reject in
// getElfSize before any device/arch check, so they are arch-independent. The valid
// in-memory load is covered by OOB_hiprtc_roundtrip_loads, which is arch-correct.
HIP_TEST_CASE(OOB_hip_module_load_data_over) {
  Bytes elf = ExtractElf(ReadFile(kValidModule));

  SECTION("bad shoff in-memory") {
    Bytes buf = MakeBadShoff(elf);
    hipModule_t module{};
    HIP_CHECK_ERROR(hipModuleLoadData(&module, buf.data()), hipErrorInvalidImage);
  }

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
// Readable-size bounds tests for fat-binary parsing.
//
// Pointer-based load APIs carry no length, so the runtime can only bound
// parsing when the image comes from a file-backed mapping.
//
// Malformed pointer inputs are tail-mapped before a no-access guard page so
// over-reads fault instead of silently succeeding, while correct parsing
// returns hipErrorInvalidImage. The bundle-header/magic and ELF-header-size
// bounds exercised here complement the getElfSize ELF-internal cases above.
// ---------------------------------------------------------------------------
#if HT_AMD

namespace {

// Magic strings for ROCm clang offload bundles.
constexpr char kCompressedBundleMagic[] = "CCOB";
constexpr char kUncompressedBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";

// AMDGPU ELF identity values the runtime checks. Defined locally so this test
// stays independent of runtime headers.
constexpr unsigned char kElfOsabiAmdgpuHsa = 64;  // ELFOSABI_AMDGPU_HSA
constexpr unsigned char kEmAmdgpuLowByte = 224;   // EM_AMDGPU (0x00E0), low byte

// Builds a `size`-byte compressed offload bundle: the "CCOB" magic with
// `total_size` written to the totalSize field (offset 8); remaining bytes zero.
Bytes MakeCompressedBundle(size_t size, uint32_t total_size) {
  REQUIRE(size >= 12);  // magic (4) through totalSize (offset 8, 4 bytes)
  Bytes image(size, 0);
  std::memcpy(image.data(), kCompressedBundleMagic, 4);
  std::memcpy(image.data() + 8, &total_size, sizeof(total_size));
  return image;
}

// Builds a `size`-byte little-endian AMDGPU ELFCLASS64 image whose identity
// fields (EM_AMDGPU + ELFOSABI_AMDGPU_HSA) pass the runtime's ELF check. Only
// the e_ident bytes and e_machine are set; all other fields are zero.
Bytes MakeAmdgpuElf64Image(size_t size) {
  REQUIRE(size >= 20);  // the last field written, spans offsets 18-19
  Bytes image(size, 0);
  image[0] = 0x7f;
  image[1] = 'E';
  image[2] = 'L';
  image[3] = 'F';
  image[4] = 2;  // EI_CLASS   = ELFCLASS64
  image[5] = 1;  // EI_DATA    = ELFDATA2LSB (little-endian)
  image[6] = 1;  // EI_VERSION = EV_CURRENT
  image[7] = static_cast<char>(kElfOsabiAmdgpuHsa);  // EI_OSABI
  image[18] = static_cast<char>(kEmAmdgpuLowByte);   // e_machine low byte (little-endian)
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

// Places `payload` at the end of a file-backed page immediately followed by a
// no-access guard page. The runtime then sees exactly payload.size() readable
// bytes, and any read past the payload faults instead of succeeding silently.
class TailMappedImage {
 public:
  explicit TailMappedImage(const Bytes& payload) {
    page_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    REQUIRE(!payload.empty());
    REQUIRE(payload.size() <= page_size_);

    char tmpl[] = "/tmp/hip_fatbin_bounds_XXXXXX";
    fd_ = mkstemp(tmpl);
    REQUIRE(fd_ >= 0);
    path_ = tmpl;

    // Back the image with a one-page file holding the payload at its very end
    // (leading bytes zero-padded). Mapped at a page boundary below, this puts
    // the payload's last byte flush against the page boundary.
    Bytes file_bytes(page_size_, 0);
    std::memcpy(file_bytes.data() + (page_size_ - payload.size()), payload.data(),
                payload.size());
    REQUIRE(write(fd_, file_bytes.data(), file_bytes.size()) ==
            static_cast<ssize_t>(file_bytes.size()));

    // Reserve two adjacent pages with no access. The second one stays
    // unmapped/no-access as a guard page, so any read past the first page faults.
    region_len_ = 2 * page_size_;
    region_ = mmap(nullptr, region_len_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    REQUIRE(region_ != MAP_FAILED);

    // Map the file read-only over the first page only, leaving the guard page
    // intact immediately after it.
    void* file_page = mmap(region_, page_size_, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd_, 0);
    REQUIRE(file_page == region_);

    // Point at the payload's start; image_ + payload.size() lands exactly on
    // the guard page, so reads past the payload fault.
    image_ = static_cast<char*>(region_) + (page_size_ - payload.size());
  }

  ~TailMappedImage() {
    if (region_ != nullptr && region_ != MAP_FAILED) {
      munmap(region_, region_len_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
    if (!path_.empty()) {
      unlink(path_.c_str());
    }
  }

  TailMappedImage(const TailMappedImage&) = delete;
  TailMappedImage& operator=(const TailMappedImage&) = delete;

  const void* image() const { return image_; }

 private:
  size_t page_size_ = 0;
  int fd_ = -1;
  std::string path_;
  void* region_ = nullptr;
  size_t region_len_ = 0;
  char* image_ = nullptr;
};

}  // namespace

/**
 * Feeds guard-page-backed malformed images to hipModuleLoadData and
 * expects hipErrorInvalidImage instead of an out-of-bounds read.
 */
HIP_TEST_CASE(OOB_hipModuleLoadData_Negative_TruncatedImages) {
  hipModule_t module = nullptr;

  SECTION("compressed magic shorter than the magic itself") {
    // Only 3 readable bytes, fewer than the 4-byte compressed magic. Checks that
    // the magic comparison doesn't read past the image.
    const Bytes payload(kCompressedBundleMagic, kCompressedBundleMagic + 3);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("compressed magic with too little readable image") {
    // Only 4 readable bytes (magic, no header). Checks that probing for a header
    // doesn't read past the image.
    const Bytes payload(kCompressedBundleMagic, kCompressedBundleMagic + 4);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("compressed header with out-of-bounds totalSize") {
    // Valid 32-byte header, but its declared totalSize (offset 8) claims a far
    // larger image than exists. Checks that the runtime doesn't trust that size
    // and read past the image.
    const Bytes payload = MakeCompressedBundle(32, 0xFFFFFFFFu);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("AMDGPU ELF header smaller than Elf64_Ehdr") {
    // 40-byte AMDGPU ELF header: big enough that the magic checks stay in
    // bounds, but smaller than a full Elf64_Ehdr (64 bytes). Checks that reading
    // ELF header fields doesn't read past the image.
    const Bytes payload = MakeAmdgpuElf64Image(40);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("uncompressed bundle magic truncated below magic length") {
    // Only 10 readable bytes of the 24-byte uncompressed magic. Checks that
    // comparing the full magic doesn't read past the image.
    const Bytes payload(kUncompressedBundleMagic, kUncompressedBundleMagic + 10);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("AMDGPU ELF whose declared size exceeds the readable image") {
    // Full 64-byte Elf64_Ehdr so the header checks pass, but e_shoff points a
    // section table far past the image. e_shnum stays 0 so computing the ELF
    // size does not itself walk out of bounds; the computed size is what is out
    // of bounds. Checks that the oversized size isn't used to read past the image
    // when the code object is handed off for loading.
    Bytes payload = MakeAmdgpuElf64Image(64);
    Wr16(payload, kEShentsize, 64);
    Wr64(payload, kEShoff, uint64_t(1) << 24);  // e_shoff: table far past 64 bytes
    Wr16(payload, kEShnum, 1);                  // one entry
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }

  SECTION("uncompressed bundle magic with no readable body") {
    // Exactly the 24-byte magic and nothing else. Checks that reading the bundle
    // header is clamped to the readable size rather than a fixed 4096-byte slice
    // that reads past the image.
    const Bytes payload(kUncompressedBundleMagic, kUncompressedBundleMagic + 24);
    TailMappedImage img(payload);
    HIP_CHECK_ERROR(hipModuleLoadData(&module, img.image()), hipErrorInvalidImage);
  }
}

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
