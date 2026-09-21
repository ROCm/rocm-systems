// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_fuzz_core.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/test_encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/test_encodings.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef __AFL_HAVE_MANUAL_CONTROL
__AFL_FUZZ_INIT();
#endif

namespace {

using Window = std::array<uint8_t, rocjitsu::decode_fuzz::kWindowSize>;

void usage(std::ostream &out) {
  out << "Usage:\n"
         "  rj_decode_fuzz --afl [--target TARGET]\n"
         "  rj_decode_fuzz --input FILE --json [--target TARGET]\n"
         "  rj_decode_fuzz --emit-seeds DIRECTORY [--target TARGET]\n"
         "\n"
         "TARGET is a canonical AMDGPU target ID or registered alias.\n";
}

Window words_to_window(std::span<const uint32_t> words) {
  if (words.size() > Window{}.size() / sizeof(uint32_t))
    throw std::runtime_error("seed encoding exceeds the 16-byte decoder window");
  Window result{};
  for (size_t word = 0; word < words.size(); ++word) {
    for (size_t byte = 0; byte < sizeof(uint32_t); ++byte) {
      result[word * sizeof(uint32_t) + byte] =
          static_cast<uint8_t>((words[word] >> (byte * 8)) & 0xffu);
    }
  }
  return result;
}

Window read_window(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open input: " + path.string());
  const std::streamsize size = input.tellg();
  if (size != static_cast<std::streamsize>(Window{}.size()))
    throw std::runtime_error("input must be exactly 16 bytes: " + path.string());
  input.seekg(0);
  Window window{};
  if (!input.read(reinterpret_cast<char *>(window.data()), size))
    throw std::runtime_error("cannot read input: " + path.string());
  return window;
}

std::string safe_name(std::string_view mnemonic) {
  std::string result;
  result.reserve(mnemonic.size());
  for (const unsigned char c : mnemonic)
    result += std::isalnum(c) || c == '_' ? static_cast<char>(c) : '_';
  return result;
}

size_t emit_seeds(const std::filesystem::path &directory, std::string_view target) {
  if (std::filesystem::exists(directory)) {
    if (!std::filesystem::is_directory(directory))
      throw std::runtime_error("seed destination is not a directory: " + directory.string());
    if (std::filesystem::directory_iterator(directory) != std::filesystem::directory_iterator())
      throw std::runtime_error("seed destination is not empty: " + directory.string());
  }
  std::filesystem::create_directories(directory);
  std::set<Window> seen;
  size_t index = 0;
  const auto write = [&](std::string_view name, const Window &window) {
    if (!seen.insert(window).second)
      return;
    std::string filename = std::to_string(index++);
    filename += '_';
    filename += safe_name(name);
    filename += ".bin";
    std::ofstream output(directory / filename, std::ios::binary);
    if (!output.write(reinterpret_cast<const char *>(window.data()), window.size()))
      throw std::runtime_error("cannot write seed: " + (directory / filename).string());
  };

  const auto write_encodings = [&](const auto &encodings) {
    for (const auto &encoding : encodings)
      write(encoding.mnemonic, words_to_window(encoding.words));
  };
  if (target == "cdna1")
    write_encodings(rocjitsu::cdna1::test_data::ENCODINGS);
  else if (target == "cdna2")
    write_encodings(rocjitsu::cdna2::test_data::ENCODINGS);
  else if (target == "cdna3")
    write_encodings(rocjitsu::cdna3::test_data::ENCODINGS);
  else if (target == "cdna4")
    write_encodings(rocjitsu::cdna4::test_data::ENCODINGS);
  else if (target == "rdna1")
    write_encodings(rocjitsu::rdna1::test_data::ENCODINGS);
  else if (target == "rdna2")
    write_encodings(rocjitsu::rdna2::test_data::ENCODINGS);
  else if (target == "rdna3")
    write_encodings(rocjitsu::rdna3::test_data::ENCODINGS);
  else if (target == "rdna3_5")
    write_encodings(rocjitsu::rdna3_5::test_data::ENCODINGS);
  else if (target == "rdna4")
    write_encodings(rocjitsu::rdna4::test_data::ENCODINGS);
  else if (target == "gfx1250")
    write_encodings(rocjitsu::cdna5::test_data::ENCODINGS);
  else
    throw std::runtime_error("unsupported decoder target: " + std::string(target));

  if (target == "gfx1250") {
    // Generated test encodings currently hold two words. Keep representative
    // literal and paired four-word forms in the initial corpus explicitly.
    constexpr std::array<uint32_t, 3> kFmamkF64 = {0x46040504u, 0x00000000u, 0xC1F00000u};
    constexpr std::array<uint32_t, 3> kFmaakF64 = {0x48040504u, 0x00000000u, 0xC1F00000u};
    constexpr std::array<uint32_t, 3> kTrue16Literal = {0xD7620086u, 0x02030CFFu, 0x000000FFu};
    constexpr std::array<uint32_t, 4> kWmmaScale = {0xCC350000u, 0x04020900u, 0xCC330006u,
                                                    0x02026912u};
    write("v_fmamk_f64_literal", words_to_window(kFmamkF64));
    write("v_fmaak_f64_literal", words_to_window(kFmaakF64));
    write("v_and_b16_true16_literal", words_to_window(kTrue16Literal));
    write("v_wmma_scale_f32", words_to_window(kWmmaScale));
  }
  return index;
}

int run_replay(const std::filesystem::path &path, std::string_view target) {
  const Window window = read_window(path);
  std::unique_ptr<rocjitsu::Decoder> decoder = rocjitsu::decode_fuzz::create_decoder(target);
  if (!decoder)
    throw std::runtime_error(std::string(target) + " decoder is unavailable");
  const auto record = rocjitsu::decode_fuzz::decode_window(*decoder, window);
  std::cout << rocjitsu::decode_fuzz::record_json(record, window, path.string()) << '\n';
  return 0;
}

} // namespace

int run_afl(std::string_view target) {
#ifdef __AFL_HAVE_MANUAL_CONTROL
  std::unique_ptr<rocjitsu::Decoder> decoder = rocjitsu::decode_fuzz::create_decoder(target);
  if (!decoder)
    throw std::runtime_error(std::string(target) + " decoder is unavailable");

  __AFL_INIT();
  unsigned char *buffer = __AFL_FUZZ_TESTCASE_BUF;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
#endif
  while (__AFL_LOOP(10000)) {
    const int size = __AFL_FUZZ_TESTCASE_LEN;
    if (size != static_cast<int>(rocjitsu::decode_fuzz::kWindowSize))
      continue;
    const auto bytes = std::span<const uint8_t>(buffer, static_cast<size_t>(size));
    (void)rocjitsu::decode_fuzz::decode_window(*decoder, bytes);
  }
#ifdef __clang__
#pragma clang diagnostic pop
#endif
  return 0;
#else
  (void)target;
  std::cerr << "rj_decode_fuzz was not compiled with an AFL++ compiler wrapper\n";
  return 2;
#endif
}

int main(int argc, char **argv) {
  bool afl = false;
  bool json = false;
  std::string input;
  std::string seed_directory;
  std::string target = "gfx1250";

  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (argument == "--afl") {
      afl = true;
    } else if (argument == "--json") {
      json = true;
    } else if (argument == "--input" && i + 1 < argc) {
      input = argv[++i];
    } else if (argument == "--emit-seeds" && i + 1 < argc) {
      seed_directory = argv[++i];
    } else if (argument == "--target" && i + 1 < argc) {
      target = argv[++i];
    } else if (argument == "--help") {
      usage(std::cout);
      return 0;
    } else {
      usage(std::cerr);
      return 2;
    }
  }

  const int modes = static_cast<int>(afl) + static_cast<int>(!input.empty()) +
                    static_cast<int>(!seed_directory.empty());
  const bool replay = !input.empty();
  if (modes != 1 || json != replay) {
    usage(std::cerr);
    return 2;
  }

  const auto *descriptor = rocjitsu::default_isa_target_registry().find(target);
  if (descriptor == nullptr) {
    std::cerr << "rj_decode_fuzz: unsupported decoder target: " << target << '\n';
    return 2;
  }
  const std::string_view canonical_target = descriptor->id;

  // Unexpected decoder exceptions must terminate the AFL child by signal so
  // AFL++ retains the input. Replay and seed modes keep friendly diagnostics.
  if (afl)
    return run_afl(canonical_target);

  try {
    if (!seed_directory.empty()) {
      std::cout << "wrote " << emit_seeds(seed_directory, canonical_target) << ' '
                << canonical_target << " seeds\n";
      return 0;
    }
    return run_replay(input, canonical_target);
  } catch (const std::exception &error) {
    std::cerr << "rj_decode_fuzz: " << error.what() << '\n';
    return 2;
  }
}
