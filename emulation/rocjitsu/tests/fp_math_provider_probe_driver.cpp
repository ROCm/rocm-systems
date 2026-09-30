// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "fp_math_provider_probe.h"

#if defined(ROCJITSU_FP_MATH_PROBE_INVALID_PROVIDER)
#include "rocjitsu/isa/arch/amdgpu/shared/x86_v4_provider.h"

// A baseline-only fixture with the correct ABI and shape but a different build
// digest. Its callbacks must never run. Production artifacts remain untouched.
namespace rocjitsu {
extern "C" RJ_API_EXPORT const X86V4ProviderDescriptor *rj_x86_v4_provider_v1() noexcept {
  static constexpr Instruction::ExecuteFn wmma = [](Instruction &, void *) {};
  static constexpr amdgpu::fp_math::UnaryF32Kernel unary{
      [](const amdgpu::fp_math::F32UnaryWave &) {}, amdgpu::fp_math::MathKernelKind::Avx512x8, 8,
      1};
  static constexpr X86V4ProviderDescriptor incompatible_descriptor{
      kX86V4ProviderAbi,
      "intentionally-incompatible-fp-test-build",
      {wmma, wmma, wmma, wmma, wmma},
      [](const IsaExecutionBackend *) noexcept {},
      unary,
      unary};
  return &incompatible_descriptor;
}
} // namespace rocjitsu
#else

#include "cdna5_wmma_provider_probe.h"

#include <array>
#include <cerrno>
#include <charconv>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <link.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr std::string_view kProvider = "librocjitsu_x86_v4.so";
constexpr int kSkip = 77;
using WmmaResults = std::array<rj_test_cdna5_wmma_result, 5>;

const char *module(uintptr_t address, const char **image) {
  if (!address) {
    *image = "<none>";
    return "none";
  }
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void *>(address), &info) || !info.dli_fname) {
    *image = "<unresolved>";
    return "unknown";
  }
  *image = info.dli_fname;
  if (std::filesystem::path(info.dli_fname).filename() == kProvider)
    return "provider";
  if (std::strstr(info.dli_fname, "librocjitsu.so"))
    return "main";
  return "unknown";
}

bool provider_loaded() {
  bool loaded = false;
  dl_iterate_phdr(
      [](dl_phdr_info *info, size_t, void *state) {
        if (info->dlpi_name && std::filesystem::path(info->dlpi_name).filename() == kProvider)
          *static_cast<bool *>(state) = true;
        return 0;
      },
      &loaded);
  return loaded;
}

uint64_t fingerprint(const rj_test_fp_math_result &result) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (unsigned lane = 0; lane < result.output_count; ++lane)
    for (unsigned byte = 0; byte < 4; ++byte) {
      hash ^= (result.output_words[lane] >> (byte * 8)) & 0xffu;
      hash *= UINT64_C(1099511628211);
    }
  return hash;
}

constexpr bool unsupported(std::string_view message) {
  return message == "x86-v4 provider requires CPU and OS AVX-512 support" ||
         message.find("requires AVX") != std::string_view::npos ||
         message.find("unavailable x86 backend") != std::string_view::npos;
}

static_assert(unsupported("x86-v4 provider requires CPU and OS AVX-512 support"));
static_assert(!unsupported("cannot load x86-v4 provider: missing shared object"));
static_assert(!unsupported("x86-v4 provider is incompatible with librocjitsu.so"));

int run(std::string_view mode, unsigned operation, unsigned scenario, unsigned wave_size,
        unsigned iterations) {
  rj_test_fp_math_result result{};
  if (rj_test_fp_math_probe(operation, scenario, wave_size, iterations, &result) != 0) {
    std::fprintf(stderr, "FP probe mode=%.*s error=%s\n", static_cast<int>(mode.size()),
                 mode.data(), result.error);
    return (mode == "v3" || mode == "v4") && unsupported(result.error) ? kSkip : 1;
  }
  const char *instruction_image = nullptr, *wave_image = nullptr;
  const std::string_view instruction_module =
      module(result.instruction_callback_addr, &instruction_image);
  const std::string_view wave_module = module(result.wave_callback_addr, &wave_image);
  const std::string_view expected_module = mode == "v3"   ? "main"
                                           : mode == "v4" ? "provider"
                                                          : "none";
  const unsigned expected_kind = mode == "v3" ? 1 : mode == "v4" ? 2 : 0;
  const bool no_provider =
      mode == "auto" || mode == "v3" || mode == "scalar" || mode == "force-scalar";
  if (instruction_module != "main" || wave_module != expected_module ||
      result.kind != expected_kind || result.mismatches != 0 ||
      (mode != "auto" && (!result.qualified_model || !result.fenv_preserved)) ||
      (mode == "auto" && result.qualified_model) || (no_provider && provider_loaded())) {
    std::fprintf(stderr,
                 "FP probe failed op=%u scenario=%u wave=%u instruction=%s wave_kernel=%s "
                 "kind=%u qualified=%u mismatches=%u fenv=%u provider_loaded=%d\n",
                 operation, scenario, wave_size, instruction_image, wave_image, result.kind,
                 result.qualified_model, result.mismatches, result.fenv_preserved,
                 provider_loaded());
    return 1;
  }
  std::printf("operation=%s scenario=%u wave=%u tier=%u kind=%u module=%.*s image=%s "
              "first_init_ns=%" PRIu64 " elapsed_ns=%" PRIu64
              " iterations=%u ns_per_op=%.3f hash=%016" PRIx64 "\n",
              operation == 0 ? "exp" : "log", scenario, wave_size, result.tier, result.kind,
              static_cast<int>(wave_module.size()), wave_module.data(), wave_image,
              result.first_init_ns, result.elapsed_ns, result.executed,
              result.executed ? static_cast<double>(result.elapsed_ns) / result.executed : 0.0,
              fingerprint(result));
  return 0;
}

int check(std::string_view mode, unsigned iterations) {
  for (unsigned operation : {0u, 1u})
    for (unsigned wave_size : {32u, 64u})
      for (unsigned scenario = 0; scenario < (iterations ? 1u : 8u); ++scenario)
        if (const int status = run(mode, operation, scenario, wave_size, iterations); status != 0)
          return status;
  return 0;
}

std::filesystem::path main_image() {
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void *>(&rj_test_fp_math_probe), &info) || !info.dli_fname)
    throw std::runtime_error("cannot locate actual shared FP probe image");
  return std::filesystem::canonical(info.dli_fname);
}

struct TemporaryDirectory {
  std::filesystem::path path;
  TemporaryDirectory() {
    std::array<char, 64> pattern{};
    std::snprintf(pattern.data(), pattern.size(), "/tmp/rocjitsu-fp-probe-XXXXXX");
    const char *created = mkdtemp(pattern.data());
    if (!created)
      throw std::runtime_error("cannot create private FP loader fixture directory");
    path = created;
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

int wmma_results(WmmaResults &results) {
  for (unsigned form = 0; form < results.size(); ++form) {
    if (rj_test_cdna5_wmma_probe(form, 0, 0, &results[form]) != 0) {
      std::fprintf(stderr, "WMMA form=%u error=%s\n", form, results[form].error);
      return 1;
    }
    const char *image = nullptr;
    if (std::string_view(module(results[form].callback_addr, &image)) != "main") {
      std::fprintf(stderr, "optional/scalar WMMA did not execute the main image: %s\n", image);
      return 1;
    }
  }
  return 0;
}

bool transfer(int descriptor, void *buffer, size_t size, bool writing) {
  auto *cursor = static_cast<char *>(buffer);
  while (size != 0) {
    const ssize_t completed =
        writing ? write(descriptor, cursor, size) : read(descriptor, cursor, size);
    if (completed < 0 && errno == EINTR)
      continue;
    if (completed <= 0)
      return false;
    cursor += completed;
    size -= static_cast<size_t>(completed);
  }
  return true;
}

int wmma_scalar_oracle(const char *self, WmmaResults &results) {
  int channel[2];
  if (pipe(channel) != 0)
    throw std::runtime_error("cannot create WMMA scalar oracle pipe");
  const auto executable = std::filesystem::canonical(self).string();
  const auto descriptor = std::to_string(channel[1]);
  std::fflush(nullptr);
  const pid_t child = fork();
  if (child == -1) {
    close(channel[0]);
    close(channel[1]);
    throw std::runtime_error("cannot fork WMMA scalar oracle");
  }
  if (child == 0) {
    close(channel[0]);
    if (setenv("RJ_FORCE_SCALAR", "1", 1) != 0 || setenv("RJ_CDNA5_WMMA_BACKEND", "v3", 1) != 0)
      _exit(1);
    execl(executable.c_str(), executable.c_str(), "--wmma-scalar-oracle", descriptor.c_str(),
          static_cast<char *>(nullptr));
    _exit(1);
  }
  close(channel[1]);
  const bool received = transfer(channel[0], results.data(), sizeof(results), false);
  close(channel[0]);
  int status = 0;
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    return 1;
  return received ? 0 : 1;
}

int fixture_child(const char *self, std::string_view mode, const char *directory) {
  if (main_image().parent_path() != std::filesystem::canonical(directory)) {
    std::fprintf(stderr, "loader fixture did not use its private copied main image\n");
    return 1;
  }
  if (mode == "auto")
    return check(mode, 0);
  if (mode == "wmma-auto") {
    WmmaResults automatic{}, scalar{};
    if (wmma_results(automatic) != 0 || wmma_scalar_oracle(self, scalar) != 0)
      return 1;
    for (unsigned form = 0; form < automatic.size(); ++form) {
      if (automatic[form].output_count != scalar[form].output_count)
        return 1;
      for (unsigned word = 0; word < automatic[form].output_count; ++word)
        if (automatic[form].output_words[word] != scalar[form].output_words[word]) {
          std::fprintf(stderr, "optional WMMA differs from fresh scalar oracle form=%u word=%u\n",
                       form, word);
          return 1;
        }
    }
    if (provider_loaded()) {
      std::fprintf(stderr, "rejected optional WMMA provider remained loaded\n");
      return 1;
    }
    std::printf("optional_wmma_scalar_oracle=bit_exact forms=5\n");
    return 0;
  }
  rj_test_fp_math_result result{};
  rj_test_cdna5_wmma_result wmma{};
  const int status = mode == "wmma-v4" ? rj_test_cdna5_wmma_probe(0, 0, 0, &wmma)
                                       : rj_test_fp_math_probe(0, 0, 64, 0, &result);
  const char *error = mode == "wmma-v4" ? wmma.error : result.error;
  if (status != 0 && unsupported(error)) {
    std::printf("SKIP: cannot reach loader failure without supported v4: %s\n", error);
    return kSkip;
  }
  const std::string_view message(error);
  if (status == 0 || (message.find("provider") == std::string_view::npos &&
                      message.find("ABI") == std::string_view::npos &&
                      message.find("digest") == std::string_view::npos)) {
    std::fprintf(stderr, "explicit v4 did not report a provider load/validation error: %s\n",
                 error);
    return 1;
  }
  std::printf("expected_explicit_v4_error=%s\n", error);
  return 0;
}

int fixture(const char *self, std::string_view kind, std::string_view mode,
            const char *invalid_provider) {
  const auto original = main_image();
  TemporaryDirectory temporary;
  std::filesystem::copy_file(original, temporary.path / original.filename());
  if (kind == "mismatch")
    std::filesystem::copy_file(invalid_provider, temporary.path / kProvider);
  const char *previous = std::getenv("LD_LIBRARY_PATH");
  const std::string search = temporary.path.string() + ":" + original.parent_path().string() +
                             (previous ? ":" + std::string(previous) : "");
  const std::string requested(mode);
  const bool wmma = mode.starts_with("wmma-");
  const std::string math_backend = wmma ? "auto" : requested;
  const std::string wmma_backend = wmma ? requested.substr(5) : "v3";
  const auto executable = std::filesystem::canonical(self).string();
  const auto directory = temporary.path.string();
  std::fflush(nullptr);
  const pid_t child = fork();
  if (child == -1)
    throw std::runtime_error("cannot fork private FP loader fixture");
  if (child == 0) {
    if (setenv("LD_LIBRARY_PATH", search.c_str(), 1) != 0 ||
        setenv("RJ_MATH_BACKEND", math_backend.c_str(), 1) != 0 ||
        setenv("RJ_FORCE_SCALAR", "0", 1) != 0 ||
        setenv("RJ_CDNA5_WMMA_BACKEND", wmma_backend.c_str(), 1) != 0)
      _exit(1);
    execl(executable.c_str(), executable.c_str(), "--fixture-child", requested.c_str(),
          directory.c_str(), static_cast<char *>(nullptr));
    _exit(1);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status))
    return 1;
  return WEXITSTATUS(status);
}

bool valid_mode(std::string_view mode) {
  return mode == "auto" || mode == "v3" || mode == "scalar" || mode == "force-scalar" ||
         mode == "v4";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string_view(argv[1]) == "--wmma-scalar-oracle") {
      int descriptor = -1;
      const std::string_view argument(argv[2]);
      const auto [end, error] =
          std::from_chars(argument.data(), argument.data() + argument.size(), descriptor);
      if (error != std::errc{} || end != argument.data() + argument.size() || descriptor < 0)
        return 2;
      WmmaResults scalar{};
      if (wmma_results(scalar) != 0)
        return 1;
      return transfer(descriptor, scalar.data(), sizeof(scalar), true) ? 0 : 1;
    }
    if (argc == 4 && std::string_view(argv[1]) == "--fixture-child")
      return fixture_child(argv[0], argv[2], argv[3]);
    if (argc == 5 && std::string_view(argv[1]) == "--fixture" &&
        (std::string_view(argv[2]) == "missing" || std::string_view(argv[2]) == "mismatch") &&
        (std::string_view(argv[3]) == "auto" || std::string_view(argv[3]) == "v4" ||
         std::string_view(argv[3]) == "wmma-auto" || std::string_view(argv[3]) == "wmma-v4"))
      return fixture(argv[0], argv[2], argv[3], argv[4]);
    if ((argc == 2 || argc == 3) && valid_mode(argv[1])) {
      unsigned iterations = 0;
      if (argc == 3) {
        const std::string_view argument(argv[2]);
        const auto [end, error] =
            std::from_chars(argument.data(), argument.data() + argument.size(), iterations);
        if (error != std::errc{} || end != argument.data() + argument.size())
          return 2;
      }
      return check(argv[1], iterations);
    }
    std::fprintf(stderr,
                 "usage: %s <auto|v3|scalar|force-scalar|v4> [iterations] | "
                 "--fixture <missing|mismatch> <auto|v4> <invalid-provider-path>\n",
                 argv[0]);
    return 2;
  } catch (const std::exception &exception) {
    std::fprintf(stderr, "FP driver: %s\n", exception.what());
    return 1;
  }
}

#endif // ROCJITSU_FP_MATH_PROBE_INVALID_PROVIDER
