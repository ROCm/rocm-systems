// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_waitcheck_smoke_test.cpp
/// @brief End-to-end smoke test for the rj_waitcheck command-line tool.

#include "waitcheck_fixture.h"

#include <gtest/gtest.h>
#include <zstd.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

std::filesystem::path g_waitcheck_tool;

struct TempDir {
  std::filesystem::path path;

  explicit TempDir(std::filesystem::path temp_path) : path(std::move(temp_path)) {
    std::filesystem::create_directories(path);
  }

  ~TempDir() { std::filesystem::remove_all(path); }
};

std::string shell_quote(std::string_view text) {
  std::string quoted = "'";
  for (const char ch : text) {
    if (ch == '\'')
      quoted += "'\\''";
    else
      quoted += ch;
  }
  quoted += "'";
  return quoted;
}

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool contains(std::string_view haystack, std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
}

bool command_exited_with(int status, int code) {
  return status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == code;
}

bool command_succeeded(int status) { return command_exited_with(status, 0); }

bool write_binary_file(const std::filesystem::path &path, const std::vector<uint8_t> &contents) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(contents.data()),
            static_cast<std::streamsize>(contents.size()));
  return out.good();
}

template <typename T> void append_value(std::vector<uint8_t> &bytes, T value) {
  const auto *raw = reinterpret_cast<const uint8_t *>(&value);
  bytes.insert(bytes.end(), raw, raw + sizeof(value));
}

std::vector<uint8_t> make_clang_offload_bundle(const std::vector<uint8_t> &code_object) {
  constexpr std::string_view magic = "__CLANG_OFFLOAD_BUNDLE__";
  constexpr std::string_view host_id = "host-x86_64-unknown-linux-gnu-";
  constexpr std::string_view gpu_id = "hipv4-amdgcn-amd-amdhsa--gfx950";
  constexpr uint64_t bundle_count = 2;
  const uint64_t header_size = magic.size() + sizeof(bundle_count) + 2 * 3 * sizeof(uint64_t) +
                               host_id.size() + gpu_id.size();
  const uint64_t payload_offset = rocjitsu::waitcheck_test::align_up(header_size, 4096);

  std::vector<uint8_t> bundle;
  bundle.insert(bundle.end(), magic.begin(), magic.end());
  append_value(bundle, bundle_count);
  append_value(bundle, payload_offset);
  append_value(bundle, uint64_t{0});
  append_value(bundle, static_cast<uint64_t>(host_id.size()));
  bundle.insert(bundle.end(), host_id.begin(), host_id.end());
  append_value(bundle, payload_offset);
  append_value(bundle, static_cast<uint64_t>(code_object.size()));
  append_value(bundle, static_cast<uint64_t>(gpu_id.size()));
  bundle.insert(bundle.end(), gpu_id.begin(), gpu_id.end());
  bundle.resize(static_cast<size_t>(payload_offset), 0);
  bundle.insert(bundle.end(), code_object.begin(), code_object.end());
  return bundle;
}

std::vector<uint8_t> make_compressed_offload_bundle(const std::vector<uint8_t> &bundle) {
  constexpr std::string_view magic = "CCOB";
  constexpr uint16_t version = 3;
  constexpr uint16_t zstd_method = 1;
  constexpr uint64_t hash = 0;
  constexpr size_t header_size = 32;

  std::vector<uint8_t> compressed(ZSTD_compressBound(bundle.size()));
  const size_t compressed_size = ZSTD_compress(compressed.data(), compressed.size(), bundle.data(),
                                               bundle.size(), ZSTD_CLEVEL_DEFAULT);
  if (ZSTD_isError(compressed_size))
    return {};
  compressed.resize(compressed_size);

  std::vector<uint8_t> result;
  result.insert(result.end(), magic.begin(), magic.end());
  append_value(result, version);
  append_value(result, zstd_method);
  append_value(result, static_cast<uint64_t>(header_size + compressed.size()));
  append_value(result, static_cast<uint64_t>(bundle.size()));
  append_value(result, hash);
  result.insert(result.end(), compressed.begin(), compressed.end());
  return result;
}

std::vector<uint8_t>
make_gnu_host_fat_binary(const std::vector<std::vector<uint8_t>> &compressed_bundles) {
  std::vector<uint8_t> fatbin;
  for (const auto &bundle : compressed_bundles) {
    fatbin.resize(static_cast<size_t>(rocjitsu::waitcheck_test::align_up(fatbin.size(), 4096)), 0);
    fatbin.insert(fatbin.end(), bundle.begin(), bundle.end());
  }

  constexpr uint64_t fatbin_offset = 0x100;
  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t fatbin_name = rocjitsu::waitcheck_test::add_elf_name(shstrtab, ".hip_fatbin");
  const uint32_t shstrtab_name = rocjitsu::waitcheck_test::add_elf_name(shstrtab, ".shstrtab");
  const uint64_t shstrtab_offset = fatbin_offset + fatbin.size();
  const uint64_t shoff = rocjitsu::waitcheck_test::align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 3;
  std::vector<uint8_t> image(shoff + section_count * sizeof(rocjitsu::Elf64_Shdr), 0);

  rocjitsu::Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE);
  ehdr.e_ident[rocjitsu::EI_CLASS] = rocjitsu::ELFCLASS64;
  ehdr.e_ident[rocjitsu::EI_DATA] = 1;
  ehdr.e_ident[rocjitsu::EI_VERSION] = 1;
  ehdr.e_ident[rocjitsu::EI_OSABI] = rocjitsu::ELFOSABI_GNU;
  ehdr.e_type = rocjitsu::ET_DYN;
  ehdr.e_machine = rocjitsu::EM_X86_64;
  ehdr.e_version = 1;
  ehdr.e_ehsize = sizeof(rocjitsu::Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_shentsize = sizeof(rocjitsu::Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 2;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));
  std::memcpy(image.data() + fatbin_offset, fatbin.data(), fatbin.size());
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<rocjitsu::Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = fatbin_name;
  shdrs[1].sh_type = rocjitsu::SHT_PROGBITS;
  shdrs[1].sh_offset = fatbin_offset;
  shdrs[1].sh_size = fatbin.size();
  shdrs[1].sh_addralign = 4096;
  shdrs[2].sh_name = shstrtab_name;
  shdrs[2].sh_type = rocjitsu::SHT_STRTAB;
  shdrs[2].sh_offset = shstrtab_offset;
  shdrs[2].sh_size = shstrtab.size();
  shdrs[2].sh_addralign = 1;
  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(rocjitsu::Elf64_Shdr));
  return image;
}

} // namespace

TEST(RjWaitcheck, ReportsMissingWait) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto input = temp_dir.path / "missing_wait_gfx1200.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(
      write_binary_file(input, rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(input.string()) + " --target gfx1200 > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_exited_with(status, 4)) << "stderr:\n"
                                              << stderr_text << "\nstdout:\n"
                                              << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 4> expected = {
      "rj_waitcheck:", "diagnostics=1", "missing s_wait_loadcnt <= 0", "consumer .text+"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, ListsAndScansConcatenatedCompressedGfx950Bundles) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto input = temp_dir.path / "pytorch_style_gfx950.so";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";
  const auto missing = rocjitsu::waitcheck_test::make_gfx_code_object(
      {0xE0501000u, 0x80000008u, 0x7E020300u}, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950);
  const auto correct = rocjitsu::waitcheck_test::make_gfx_code_object(
      {0xE0501000u, 0x80000008u, 0xBF8C0F70u, 0x7E020300u}, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950);
  const auto executable = make_gnu_host_fat_binary(
      {make_compressed_offload_bundle(make_clang_offload_bundle(missing)),
       make_compressed_offload_bundle(make_clang_offload_bundle(correct))});
  ASSERT_TRUE(write_binary_file(input, executable));

  std::string command = shell_quote(g_waitcheck_tool.string()) + " --list-code-objects " +
                        shell_quote(input.string()) + " > " + shell_quote(output.string()) +
                        " 2> " + shell_quote(error.string());
  int status = std::system(command.c_str());
  std::string stdout_text = read_text_file(output);
  std::string stderr_text = read_text_file(error);
  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text, "gfx950: 2")) << stdout_text;

  command = shell_quote(g_waitcheck_tool.string()) + " --all-code-objects --no-fail " +
            shell_quote(input.string()) + " > " + shell_quote(output.string()) + " 2> " +
            shell_quote(error.string());
  status = std::system(command.c_str());
  stdout_text = read_text_file(output);
  stderr_text = read_text_file(error);
  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text, "missing s_waitcnt vmcnt(0)")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text,
                       "rj_waitcheck: scanned inputs=1 skipped=0 code-objects=2 diagnostics=1"))
      << stdout_text;
}

TEST(RjWaitcheck, KernelEntryChecksOnlySelectedDescriptor) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto input = temp_dir.path / "multi_kernel_gfx950.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";
  const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"hazard", {0xE0501000u, 0x80000008u, 0x7E020300u}},
       {"clean", {0xE0501000u, 0x80000008u, 0xBF8C0F70u, 0x7E020300u}}},
      rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950);
  ASSERT_TRUE(write_binary_file(input, image));

  // The second descriptor starts after the first kernel's three encoded
  // words. A whole-object check reports the first kernel's hazard; selecting
  // entry 0xc must decode and analyze only the clean second kernel.
  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(input.string()) +
                              " --target gfx950 --kernel-entry 0xc > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());
  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text, "gfx950[0]:kernel=.text+0xc")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "instructions=3 memory-events=1 diagnostics=0")) << stdout_text;
}

TEST(RjWaitcheck, ExhaustiveSchedulesKernelsAndReportsSlowest) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto input = temp_dir.path / "multi_kernel_gfx950.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";
  const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"hazard", {0xE0501000u, 0x80000008u, 0x7E020300u}},
       {"clean", {0xE0501000u, 0x80000008u, 0xBF8C0F70u, 0x7E020300u}}},
      rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950);
  ASSERT_TRUE(write_binary_file(input, image));

  const std::string command =
      shell_quote(g_waitcheck_tool.string()) + " " + shell_quote(input.string()) +
      " --exhaustive --target gfx950 --summary-only --no-fail --progress -j2 "
      "--slowest-kernels 2 > " +
      shell_quote(output.string()) + " 2> " + shell_quote(error.string());
  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(contains(stderr_text, "active=")) << stderr_text;
  EXPECT_TRUE(contains(stderr_text, "100% kernels 2/2 code-objects 1/1")) << stderr_text;
  EXPECT_TRUE(contains(stdout_text, "code-objects=1/1 kernels=2/2")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "rj_waitcheck: slowest-kernels count=2")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "name=\"hazard\"")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "name=\"clean\"")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "entry=0x0")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "entry=0xc")) << stdout_text;
}

TEST(RjWaitcheck, AnalyzesGfx942CodeObject) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto input = temp_dir.path / "missing_wait_gfx942.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(
      write_binary_file(input, rocjitsu::waitcheck_test::make_gfx942_missing_wait_code_object()));

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(input.string()) + " --target gfx942 > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_exited_with(status, 4)) << "stderr:\n"
                                              << stderr_text << "\nstdout:\n"
                                              << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 3> expected = {
      "missing_wait_gfx942.co:gfx942[0]: instructions=2 memory-events=1 diagnostics=1",
      "missing s_waitcnt vmcnt(0)", "diagnostics=1"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, AnalyzesGfx1100CodeObject) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto input = temp_dir.path / "missing_wait_gfx1100.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(
      write_binary_file(input, rocjitsu::waitcheck_test::make_gfx1100_missing_wait_code_object()));

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(input.string()) + " --target gfx1100 > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_exited_with(status, 4)) << "stderr:\n"
                                              << stderr_text << "\nstdout:\n"
                                              << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 3> expected = {
      "missing_wait_gfx1100.co:gfx1100[0]: instructions=2 memory-events=1 diagnostics=1",
      "missing s_waitcnt vmcnt(0)", "diagnostics=1"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, AnalyzesGfx1250CodeObject) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto input = temp_dir.path / "vmax_u64_gfx1250.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(
      write_binary_file(input, rocjitsu::waitcheck_test::make_gfx1250_vmax_u64_code_object()));

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(input.string()) + " --target gfx1250 > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 2> expected = {
      "vmax_u64_gfx1250.co:gfx1250[0]: instructions=1 memory-events=0 diagnostics=0",
      "diagnostics=0"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, MaxDiagnosticsZeroKeepsHazardStatusWithoutDetails) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto input = temp_dir.path / "missing_wait_gfx1200.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(
      write_binary_file(input, rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(input.string()) +
                              " --target gfx1200 --max-diagnostics 0 --no-fail > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text, "diagnostics=>=1")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "omitted at least 1 diagnostic(s) after limit")) << stdout_text;
  EXPECT_FALSE(contains(stdout_text, "missing s_wait_loadcnt")) << stdout_text;
}

TEST(RjWaitcheck, BatchSkipsUnsupportedInputs) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto missing = temp_dir.path / "missing_wait_gfx1200.co";
  const auto correct = temp_dir.path / "correct_wait_gfx1200.co";
  const auto unsupported = temp_dir.path / "unsupported.txt";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(write_binary_file(missing,
                                rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));
  ASSERT_TRUE(write_binary_file(correct,
                                rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));
  {
    std::ofstream out(unsupported);
    out << "not an ELF";
  }

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(missing.string()) + " " + shell_quote(correct.string()) +
                              " " + shell_quote(unsupported.string()) +
                              " --all-code-objects --skip-unsupported --no-fail > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 5> expected = {
      "missing_wait_gfx1200.co:gfx1200[0]: instructions=2 memory-events=1 diagnostics=1",
      "correct_wait_gfx1200.co:gfx1200[0]: instructions=3 memory-events=1 diagnostics=0",
      "unsupported.txt: skipped: failed to parse input executable or code object",
      "missing s_wait_loadcnt <= 0",
      "rj_waitcheck: scanned inputs=3 skipped=1 code-objects=2 diagnostics=1"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, BatchSkipsUndecodableCodeObjects) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto invalid = temp_dir.path / "invalid_inst_gfx1200.co";
  const auto correct = temp_dir.path / "correct_wait_gfx1200.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(write_binary_file(
      invalid, rocjitsu::waitcheck_test::make_gfx1200_invalid_instruction_code_object()));
  ASSERT_TRUE(write_binary_file(correct,
                                rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(invalid.string()) + " " + shell_quote(correct.string()) +
                              " --all-code-objects --skip-unsupported --no-fail > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 4> expected = {
      "invalid_inst_gfx1200.co: skipped: waitcheck analysis failed for gfx1200[0]: "
      "decode failed while building CFG: Invalid instruction opcode: 800000",
      "correct_wait_gfx1200.co:gfx1200[0]: instructions=3 memory-events=1 diagnostics=0",
      "rj_waitcheck: scanned inputs=2 skipped=1 code-objects=1 diagnostics=0", "diagnostics=0"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, BatchSkipsRelocatableCodeObjects) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto intermediate = temp_dir.path / "intermediate_gfx1200.o";
  const auto correct = temp_dir.path / "correct_wait_gfx1200.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  ASSERT_TRUE(write_binary_file(
      intermediate, rocjitsu::waitcheck_test::make_relocatable_code_object(
                        rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object())));
  ASSERT_TRUE(write_binary_file(correct,
                                rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));

  const std::string command =
      shell_quote(g_waitcheck_tool.string()) + " " + shell_quote(intermediate.string()) + " " +
      shell_quote(correct.string()) + " --all-code-objects --skip-unsupported --no-fail > " +
      shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 3> expected = {
      "intermediate_gfx1200.o: skipped: waitcheck analysis failed for gfx1200[0]: "
      "relocatable AMDGPU objects are not final loadable code objects",
      "correct_wait_gfx1200.co:gfx1200[0]: instructions=3 memory-events=1 diagnostics=0",
      "rj_waitcheck: scanned inputs=2 skipped=1 code-objects=1 diagnostics=0"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, RecursiveDirectorySweepScansNestedCorpusFiles) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto corpus = temp_dir.path / "corpus";
  const auto nested = corpus / "nested";
  const auto missing = nested / "missing_wait_gfx1200.co";
  const auto correct = corpus / "correct_wait_gfx1200.co";
  const auto unsupported = nested / "unsupported.txt";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";
  std::filesystem::create_directories(nested);

  ASSERT_TRUE(write_binary_file(missing,
                                rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));
  ASSERT_TRUE(write_binary_file(correct,
                                rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));
  {
    std::ofstream out(unsupported);
    out << "not an ELF";
  }

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(corpus.string()) +
                              " --recursive --all-code-objects --skip-unsupported --no-fail > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 5> expected = {
      "correct_wait_gfx1200.co:gfx1200[0]: instructions=3 memory-events=1 diagnostics=0",
      "missing_wait_gfx1200.co:gfx1200[0]: instructions=2 memory-events=1 diagnostics=1",
      "unsupported.txt: skipped: failed to parse input executable or code object",
      "missing s_wait_loadcnt <= 0",
      "rj_waitcheck: scanned inputs=3 skipped=1 code-objects=2 diagnostics=1"};
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjWaitcheck, SummaryOnlyRecursiveSweepReportsLowerBoundTotals) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto corpus = temp_dir.path / "corpus";
  const auto nested = corpus / "nested";
  const auto missing = nested / "missing_wait_gfx1200.co";
  const auto correct = corpus / "correct_wait_gfx1200.co";
  const auto unsupported = nested / "unsupported.txt";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";
  std::filesystem::create_directories(nested);

  ASSERT_TRUE(write_binary_file(missing,
                                rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));
  ASSERT_TRUE(write_binary_file(correct,
                                rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));
  {
    std::ofstream out(unsupported);
    out << "not an ELF";
  }

  const std::string command =
      shell_quote(g_waitcheck_tool.string()) + " " + shell_quote(corpus.string()) +
      " --recursive --all-code-objects --skip-unsupported --no-fail --max-diagnostics 0 "
      "--stop-after-first-diagnostic --summary-only > " +
      shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text,
                       "rj_waitcheck: scanned inputs=3 skipped=1 code-objects=2 diagnostics=>=1"))
      << stdout_text;
  EXPECT_FALSE(contains(stdout_text, "missing_wait_gfx1200.co:gfx1200")) << stdout_text;
  EXPECT_FALSE(contains(stdout_text, "unsupported.txt: skipped")) << stdout_text;
  EXPECT_FALSE(contains(stdout_text, "missing s_wait_loadcnt")) << stdout_text;
}

TEST(RjWaitcheck, ExhaustiveSweepReportsCompletedKernelTotals) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto corpus = temp_dir.path / "corpus";
  const auto missing = corpus / "missing_wait_gfx1200.co";
  const auto correct = corpus / "correct_wait_gfx1200.co";
  const auto unrelated = corpus / "unrelated.txt";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";
  std::filesystem::create_directories(corpus);

  ASSERT_TRUE(write_binary_file(missing,
                                rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object()));
  ASSERT_TRUE(write_binary_file(correct,
                                rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));
  {
    std::ofstream out(unrelated);
    out << "not an ELF";
  }

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(corpus.string()) +
                              " --exhaustive --target gfx1200 --summary-only > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_exited_with(status, 4)) << "stderr:\n"
                                              << stderr_text << "\nstdout:\n"
                                              << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text,
                       "rj_waitcheck: exhaustive files=3 ignored=1 code-objects=2/2 kernels=2/2 "
                       "instructions=5 memory-events=2 diagnostics=1 analysis-errors=0"))
      << stdout_text;
}

TEST(RjWaitcheck, ExhaustiveProgressHasKernelDenominatorWhenForced) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto corpus = temp_dir.path / "corpus";
  const auto first = corpus / "first_gfx1200.co";
  const auto second = corpus / "second_gfx1200.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";
  std::filesystem::create_directories(corpus);

  ASSERT_TRUE(
      write_binary_file(first, rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));
  ASSERT_TRUE(
      write_binary_file(second, rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(corpus.string()) +
                              " --exhaustive --target gfx1200 --summary-only --progress -j16 > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(contains(stderr_text, "kernels 0/2 code-objects 0/2")) << stderr_text;
  EXPECT_TRUE(contains(stderr_text, "100% kernels 2/2 code-objects 2/2")) << stderr_text;
  EXPECT_TRUE(contains(stdout_text, "code-objects=2/2 kernels=2/2")) << stdout_text;
}

TEST(RjWaitcheck, ExhaustiveSweepFailsWhenSelectedObjectCannotBeAnalyzed) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_waitcheck_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto corpus = temp_dir.path / "corpus";
  const auto invalid = corpus / "invalid_inst_gfx1200.co";
  const auto correct = corpus / "correct_wait_gfx1200.co";
  const auto unrelated = corpus / "unrelated.txt";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";
  std::filesystem::create_directories(corpus);

  ASSERT_TRUE(write_binary_file(
      invalid, rocjitsu::waitcheck_test::make_gfx1200_invalid_instruction_code_object()));
  ASSERT_TRUE(write_binary_file(correct,
                                rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object()));
  {
    std::ofstream out(unrelated);
    out << "not an ELF";
  }

  const std::string command = shell_quote(g_waitcheck_tool.string()) + " " +
                              shell_quote(corpus.string()) +
                              " --exhaustive --target gfx1200 --summary-only > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_exited_with(status, 2)) << "stderr:\n"
                                              << stderr_text << "\nstdout:\n"
                                              << stdout_text;
  EXPECT_TRUE(contains(stderr_text,
                       "invalid_inst_gfx1200.co: analysis error: waitcheck analysis failed for "
                       "gfx1200[0]: decode failed while building CFG"))
      << stderr_text;
  EXPECT_TRUE(contains(stdout_text,
                       "rj_waitcheck: exhaustive files=3 ignored=1 code-objects=1/2 kernels=1/2 "
                       "instructions=3 memory-events=1 diagnostics=0 analysis-errors=1"))
      << stdout_text;
}

TEST(RjWaitcheck, ExhaustiveSweepRequiresTargetAndFullAnalysis) {
  const auto output = std::filesystem::temp_directory_path() /
                      ("rj_waitcheck_usage_" + std::to_string(static_cast<long long>(getpid())));
  const std::string command = shell_quote(g_waitcheck_tool.string()) +
                              " . --exhaustive > /dev/null 2> " + shell_quote(output.string());
  int status = std::system(command.c_str());
  std::string stderr_text = read_text_file(output);
  EXPECT_TRUE(command_exited_with(status, 1)) << stderr_text;
  EXPECT_TRUE(contains(stderr_text, "--exhaustive requires --target")) << stderr_text;

  const std::string stop_command =
      shell_quote(g_waitcheck_tool.string()) +
      " . --exhaustive --target gfx1200 --stop-after-first-diagnostic > /dev/null 2> " +
      shell_quote(output.string());
  status = std::system(stop_command.c_str());
  stderr_text = read_text_file(output);
  std::filesystem::remove(output);
  EXPECT_TRUE(command_exited_with(status, 1)) << stderr_text;
  EXPECT_TRUE(
      contains(stderr_text, "--stop-after-first-diagnostic cannot be used with --exhaustive"))
      << stderr_text;

  const std::string progress_command = shell_quote(g_waitcheck_tool.string()) +
                                       " . --progress > /dev/null 2> " +
                                       shell_quote(output.string());
  status = std::system(progress_command.c_str());
  stderr_text = read_text_file(output);
  std::filesystem::remove(output);
  EXPECT_TRUE(command_exited_with(status, 1)) << stderr_text;
  EXPECT_TRUE(contains(stderr_text, "--progress requires --exhaustive")) << stderr_text;

  const std::string jobs_command = shell_quote(g_waitcheck_tool.string()) +
                                   " . --all-code-objects -j17 > /dev/null 2> " +
                                   shell_quote(output.string());
  status = std::system(jobs_command.c_str());
  stderr_text = read_text_file(output);
  std::filesystem::remove(output);
  EXPECT_TRUE(command_exited_with(status, 1)) << stderr_text;
  EXPECT_TRUE(contains(stderr_text, "invalid job count: 17 (expected 1-16)")) << stderr_text;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: rj_waitcheck_smoke_test <rj_waitcheck>\n";
    return 2;
  }

  g_waitcheck_tool = argv[1];
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
