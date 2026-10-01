// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_wmma_provider_probe.h"

#include <charconv>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <string>
#include <string_view>

namespace {

constexpr int kSkip = 77;
constexpr std::string_view kForms[] = {"f32_f16", "f16_f16", "f32_bf16", "bf16_bf16",
                                       "bf16f32_bf16"};

constexpr bool unsupported_v4(std::string_view message) {
  return message == "x86-v4 provider requires CPU and OS AVX-512 support";
}

static_assert(unsupported_v4("x86-v4 provider requires CPU and OS AVX-512 support"));
static_assert(!unsupported_v4("cannot load x86-v4 provider: missing shared object"));
static_assert(!unsupported_v4("x86-v4 provider is incompatible with librocjitsu.so"));

bool parse_uint(const char *arg, uint32_t &value) {
  const std::string_view text(arg);
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  return ec == std::errc{} && end == text.data() + text.size();
}

const char *module_name(uintptr_t address, const char **image, uintptr_t *offset) {
  Dl_info info{};
  if (!address || !dladdr(reinterpret_cast<void *>(address), &info) || !info.dli_fname) {
    *image = "<unresolved>";
    *offset = 0;
    return "unknown";
  }
  *image = info.dli_fname;
  *offset = address - reinterpret_cast<uintptr_t>(info.dli_fbase);
  if (std::strstr(info.dli_fname, "librocjitsu_x86_v4.so"))
    return "provider";
  if (std::strstr(info.dli_fname, "librocjitsu.so"))
    return "main";
  return "unknown";
}

uint64_t output_hash(const rj_test_cdna5_wmma_result &result) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (uint32_t i = 0; i < result.output_count; ++i) {
    const uint32_t word = result.output_words[i];
    for (unsigned byte = 0; byte < 4; ++byte) {
      hash ^= (word >> (byte * 8)) & 0xffu;
      hash *= UINT64_C(1099511628211);
    }
  }
  return hash;
}

void dump_accesses(const char *name, const uint64_t *lanes, const uint8_t *bytes) {
  std::printf("%s=", name);
  const char *separator = "";
  for (uint32_t reg = 0; reg < RJ_TEST_WMMA_MAX_VGPRS; ++reg) {
    if (!lanes[reg])
      continue;
    std::printf("%s%u:%" PRIx64 ":%x", separator, reg, lanes[reg], unsigned{bytes[reg]});
    separator = ",";
  }
  std::printf("\n");
}

int cold_load() {
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void *>(&rj_test_cdna5_wmma_probe), &info) || !info.dli_fname) {
    std::fprintf(stderr, "cannot locate linked librocjitsu.so\n");
    return 1;
  }
  const std::string path =
      (std::filesystem::absolute(info.dli_fname).parent_path() / "librocjitsu_x86_v4.so").string();
  const auto start = std::chrono::steady_clock::now();
  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  const auto end = std::chrono::steady_clock::now();
  if (!handle) {
    std::fprintf(stderr, "dlopen(%s) failed: %s\n", path.c_str(), dlerror());
    return 1;
  }
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("cold_load_ns=%" PRId64 " image=%s\n", static_cast<int64_t>(elapsed), path.c_str());
  if (dlclose(handle) != 0) {
    std::fprintf(stderr, "dlclose(%s) failed: %s\n", path.c_str(), dlerror());
    return 1;
  }
  return 0;
}

int run(uint32_t form, uint32_t scenario, uint32_t iterations, std::string_view expected, bool dump,
        bool observe = false) {
  rj_test_cdna5_wmma_result result{};
  const int status = observe ? rj_test_cdna5_wmma_observe(form, scenario, &result)
                             : rj_test_cdna5_wmma_probe(form, scenario, iterations, &result);
  if (status != 0) {
    std::fprintf(stderr, "form=%.*s scenario=%u error=%s\n", static_cast<int>(kForms[form].size()),
                 kForms[form].data(), scenario, result.error);
    return expected == "provider" && unsupported_v4(result.error) ? kSkip : 1;
  }
  const char *image = nullptr;
  uintptr_t callback_offset = 0;
  const char *module = module_name(result.callback_addr, &image, &callback_offset);
  if (std::string_view(module) == "unknown") {
    std::fprintf(stderr, "form=%.*s unresolved callback image=%s\n",
                 static_cast<int>(kForms[form].size()), kForms[form].data(), image);
    return 1;
  }
  if (expected != "any" && expected != module) {
    std::fprintf(stderr, "form=%.*s expected=%.*s actual=%s image=%s\n",
                 static_cast<int>(kForms[form].size()), kForms[form].data(),
                 static_cast<int>(expected.size()), expected.data(), module, image);
    return 1;
  }
  const double ns_per_op =
      result.executed ? static_cast<double>(result.elapsed_ns) / result.executed : 0.0;
  std::printf("form=%.*s scenario=%u module=%s image=%s callback_offset=%" PRIxPTR
              " first_decode_ns=%" PRIu64 " elapsed_ns=%" PRIu64
              " iterations=%u ns_per_op=%.3f hash=%016" PRIx64 "\n",
              static_cast<int>(kForms[form].size()), kForms[form].data(), scenario, module, image,
              callback_offset, result.first_decode_ns, result.elapsed_ns, result.executed,
              ns_per_op, output_hash(result));
  if (dump) {
    std::printf("words=");
    for (uint32_t i = 0; i < result.output_count; ++i)
      std::printf("%08x", result.output_words[i]);
    std::printf("\n");
  }
  if (observe) {
    dump_accesses("reads", result.read_lanes, result.read_bytes);
    dump_accesses("writes", result.write_lanes, result.write_bytes);
  }
  return 0;
}

int relative_chdir_probe(const char *directory, std::string_view requested_form,
                         std::string_view expected) {
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void *>(&rj_test_cdna5_wmma_probe), &info) || !info.dli_fname ||
      !std::string_view(info.dli_fname).starts_with("./")) {
    std::fprintf(stderr, "librocjitsu.so was not first loaded through a relative ./ path\n");
    return 1;
  }
  std::printf("main_origin=%s\n", info.dli_fname);
  std::error_code ec;
  std::filesystem::current_path(directory, ec);
  if (ec) {
    std::fprintf(stderr, "cannot chdir to %s: %s\n", directory, ec.message().c_str());
    return 1;
  }
  if (requested_form == "all") {
    for (uint32_t form = 0; form < 5; ++form)
      if (const int status = run(form, 0, 0, expected, false); status != 0)
        return status;
    return 0;
  }
  uint32_t form = 0;
  const std::string text(requested_form);
  if (!parse_uint(text.c_str(), form) || form >= 5) {
    std::fprintf(stderr, "form must be all or in 0..4\n");
    return 2;
  }
  return run(form, 0, 0, expected, false);
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--cold-load")
    return cold_load();
  if (argc == 5 && std::string_view(argv[1]) == "--chdir-and-probe") {
    const std::string_view expected = argv[4];
    if (expected != "main" && expected != "provider") {
      std::fprintf(stderr, "expected module must be main or provider\n");
      return 2;
    }
    return relative_chdir_probe(argv[2], argv[3], expected);
  }
  if (argc < 3 || argc > 6) {
    std::fprintf(stderr,
                 "usage: %s --cold-load | --chdir-and-probe <dir> <all|form> <main|provider> "
                 "| <all|form 0..4> <iterations> [scenario 0..14] "
                 "[expected main|provider|any] [--dump|--observe]\n",
                 argv[0]);
    return 2;
  }
  uint32_t iterations = 0;
  if (!parse_uint(argv[2], iterations)) {
    std::fprintf(stderr, "invalid iterations: %s\n", argv[2]);
    return 2;
  }
  uint32_t scenario = 0;
  if (argc >= 4 && !parse_uint(argv[3], scenario)) {
    std::fprintf(stderr, "invalid scenario: %s\n", argv[3]);
    return 2;
  }
  if (scenario > 14) {
    std::fprintf(stderr, "scenario must be in 0..14\n");
    return 2;
  }
  const std::string_view expected = argc >= 5 ? argv[4] : "any";
  if (expected != "main" && expected != "provider" && expected != "any") {
    std::fprintf(stderr, "expected module must be main, provider, or any\n");
    return 2;
  }
  const bool observe = argc == 6 && std::string_view(argv[5]) == "--observe";
  const bool dump = observe || (argc == 6 && std::string_view(argv[5]) == "--dump");
  if (argc == 6 && !dump) {
    std::fprintf(stderr, "unknown option: %s\n", argv[5]);
    return 2;
  }
  if (observe && iterations != 0) {
    std::fprintf(stderr, "observed probes cannot be timed\n");
    return 2;
  }
  if (std::string_view(argv[1]) == "all") {
    for (uint32_t form = 0; form < 5; ++form)
      if (const int status = run(form, scenario, iterations, expected, dump, observe); status != 0)
        return status;
    return 0;
  }
  uint32_t form = 0;
  if (!parse_uint(argv[1], form) || form >= 5) {
    std::fprintf(stderr, "form must be all or in 0..4\n");
    return 2;
  }
  return run(form, scenario, iterations, expected, dump, observe);
}
