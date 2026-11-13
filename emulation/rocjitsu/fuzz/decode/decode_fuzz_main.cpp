// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/cli.h"

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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifdef __AFL_HAVE_MANUAL_CONTROL
#include <unistd.h>

__AFL_FUZZ_INIT();
#endif

namespace {

using Window = std::array<uint8_t, rocjitsu::decode_fuzz::kWindowSize>;

void usage(std::ostream &out, const rocjitsu::cli::Command &command) {
  out << rocjitsu::cli::format_help(command)
      << "\nModes:\n"
         "  rj_decode_fuzz --afl [--target TARGET]\n"
         "  rj_decode_fuzz --input FILE --json [--target TARGET]\n"
         "  rj_decode_fuzz --emit-seeds DIRECTORY [--target TARGET]\n"
         "\n"
         "TARGET is a canonical AMDGPU target ID or registered alias.\n";
}

util::FailureOr<Window> words_to_window(std::span<const uint32_t> words,
                                        util::DiagnosticEmitter diagnostic) {
  if (words.size() > Window{}.size() / sizeof(uint32_t))
    return diagnostic.emit() << "seed encoding exceeds the 16-byte decoder window";
  Window result{};
  for (size_t word = 0; word < words.size(); ++word) {
    for (size_t byte = 0; byte < sizeof(uint32_t); ++byte) {
      result[word * sizeof(uint32_t) + byte] =
          static_cast<uint8_t>((words[word] >> (byte * 8)) & 0xffu);
    }
  }
  return result;
}

util::FailureOr<Window> read_window(const std::filesystem::path &path,
                                    util::DiagnosticEmitter diagnostic) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    return diagnostic.emit() << "cannot open input: " << path.string();
  const std::streamsize size = input.tellg();
  if (size < 0)
    return diagnostic.emit() << "cannot read input: " << path.string();
  if (size != static_cast<std::streamsize>(Window{}.size()))
    return diagnostic.emit() << "input must be exactly 16 bytes: " << path.string();
  input.seekg(0);
  Window window{};
  if (!input.read(reinterpret_cast<char *>(window.data()), size))
    return diagnostic.emit() << "cannot read input: " << path.string();
  return window;
}

std::string safe_name(std::string_view mnemonic) {
  std::string result;
  result.reserve(mnemonic.size());
  for (const unsigned char c : mnemonic)
    result += std::isalnum(c) || c == '_' ? static_cast<char>(c) : '_';
  return result;
}

util::FailureOr<size_t> emit_seeds(const std::filesystem::path &directory, std::string_view target,
                                   util::DiagnosticEmitter diagnostic) {
  std::error_code error;
  const bool exists = std::filesystem::exists(directory, error);
  if (error)
    return diagnostic.emit() << "cannot inspect seed destination: " << directory.string() << ": "
                             << error.message();
  if (exists) {
    const bool is_directory = std::filesystem::is_directory(directory, error);
    if (error)
      return diagnostic.emit() << "cannot inspect seed destination: " << directory.string() << ": "
                               << error.message();
    if (!is_directory)
      return diagnostic.emit() << "seed destination is not a directory: " << directory.string();
    const std::filesystem::directory_iterator entry(directory, error);
    if (error)
      return diagnostic.emit() << "cannot inspect seed destination: " << directory.string() << ": "
                               << error.message();
    if (entry != std::filesystem::directory_iterator())
      return diagnostic.emit() << "seed destination is not empty: " << directory.string();
  }
  std::filesystem::create_directories(directory, error);
  if (error)
    return diagnostic.emit() << "cannot create seed destination: " << directory.string() << ": "
                             << error.message();
  std::set<Window> seen;
  size_t index = 0;
  const auto write = [&](std::string_view name, std::span<const uint32_t> words) -> util::Result {
    const util::FailureOr<Window> converted = words_to_window(words, diagnostic);
    if (converted.failed())
      return util::Result::failure();
    const Window &window = converted.value();
    if (!seen.insert(window).second)
      return util::Result::success();
    std::string filename = std::to_string(index++);
    filename += '_';
    filename += safe_name(name);
    filename += ".bin";
    std::ofstream output(directory / filename, std::ios::binary);
    output.write(reinterpret_cast<const char *>(window.data()), window.size());
    output.close();
    if (!output)
      return diagnostic.emit() << "cannot write seed: " << (directory / filename).string();
    return util::Result::success();
  };

  const auto write_encodings = [&](const auto &encodings) -> util::Result {
    for (const auto &encoding : encodings) {
      if (write(encoding.mnemonic, encoding.words).failed())
        return util::Result::failure();
    }
    return util::Result::success();
  };
  util::Result written;
  if (target == "cdna1")
    written = write_encodings(rocjitsu::cdna1::test_data::ENCODINGS);
  else if (target == "cdna2")
    written = write_encodings(rocjitsu::cdna2::test_data::ENCODINGS);
  else if (target == "cdna3")
    written = write_encodings(rocjitsu::cdna3::test_data::ENCODINGS);
  else if (target == "cdna4")
    written = write_encodings(rocjitsu::cdna4::test_data::ENCODINGS);
  else if (target == "rdna1")
    written = write_encodings(rocjitsu::rdna1::test_data::ENCODINGS);
  else if (target == "rdna2")
    written = write_encodings(rocjitsu::rdna2::test_data::ENCODINGS);
  else if (target == "rdna3")
    written = write_encodings(rocjitsu::rdna3::test_data::ENCODINGS);
  else if (target == "rdna3_5")
    written = write_encodings(rocjitsu::rdna3_5::test_data::ENCODINGS);
  else if (target == "rdna4")
    written = write_encodings(rocjitsu::rdna4::test_data::ENCODINGS);
  else if (target == "cdna5")
    written = write_encodings(rocjitsu::cdna5::test_data::ENCODINGS);
  else
    return diagnostic.emit() << "unsupported decoder target: " << target;

  if (written.failed())
    return util::Result::failure();

  if (target == "cdna5") {
    // Generated test encodings currently hold two words. Keep representative
    // literal and paired four-word forms in the initial corpus explicitly.
    constexpr std::array<uint32_t, 3> kFmamkF64 = {0x46040504u, 0x00000000u, 0xC1F00000u};
    constexpr std::array<uint32_t, 3> kFmaakF64 = {0x48040504u, 0x00000000u, 0xC1F00000u};
    constexpr std::array<uint32_t, 3> kTrue16Literal = {0xD7620086u, 0x02030CFFu, 0x000000FFu};
    constexpr std::array<uint32_t, 4> kWmmaScale = {0xCC350000u, 0x04020900u, 0xCC330006u,
                                                    0x02026912u};
    if (write("v_fmamk_f64_literal", kFmamkF64).failed())
      return util::Result::failure();
    if (write("v_fmaak_f64_literal", kFmaakF64).failed())
      return util::Result::failure();
    if (write("v_and_b16_true16_literal", kTrue16Literal).failed())
      return util::Result::failure();
    if (write("v_wmma_scale_f32", kWmmaScale).failed())
      return util::Result::failure();
  }
  return index;
}

util::Result run_replay(const std::filesystem::path &path, std::string_view target,
                        util::DiagnosticEmitter diagnostic) {
  const util::FailureOr<Window> input = read_window(path, diagnostic);
  if (input.failed())
    return util::Result::failure();
  const util::FailureOr<std::unique_ptr<rocjitsu::Decoder>> decoder =
      rocjitsu::decode_fuzz::create_decoder(target, diagnostic);
  if (decoder.failed())
    return util::Result::failure();
  const Window &window = input.value();
  const util::FailureOr<rocjitsu::decode_fuzz::DecodeRecord> record =
      rocjitsu::decode_fuzz::decode_window(*decoder.value(), window, diagnostic);
  if (record.failed())
    return util::Result::failure();
  std::cout << rocjitsu::decode_fuzz::record_json(record.value(), window, path.string()) << '\n';
  return util::Result::success();
}

} // namespace

int run_afl(std::string_view target, util::DiagnosticEmitter diagnostic) {
#ifdef __AFL_HAVE_MANUAL_CONTROL
  const util::FailureOr<std::unique_ptr<rocjitsu::Decoder>> decoder =
      rocjitsu::decode_fuzz::create_decoder(target, diagnostic);
  if (decoder.failed())
    return 2;

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
    if (rocjitsu::decode_fuzz::decode_window(*decoder.value(), bytes, diagnostic).failed())
      std::abort();
  }
#ifdef __clang__
#pragma clang diagnostic pop
#endif
  return 0;
#else
  (void)target;
  (void)diagnostic;
  std::cerr << "rj_decode_fuzz was not compiled with an AFL++ compiler wrapper\n";
  return 2;
#endif
}

int main(int argc, char **argv) {
  using namespace rocjitsu::cli;
  Command command{"rj_decode_fuzz"};
  command.set_positional_usage("");
  auto afl_flag = std::make_shared<BoolFlag>(
      FlagInfo{"--afl", {}, "Run the AFL fuzzing loop", ValueMode::Switch});
  auto json_flag = std::make_shared<BoolFlag>(
      FlagInfo{"--json", {}, "Emit replay result as JSON", ValueMode::Switch});
  auto input_flag =
      std::make_shared<StringFlag>(FlagInfo{"--input", {}, "Replay one 16-byte input file"});
  auto seeds_flag = std::make_shared<StringFlag>(
      FlagInfo{"--emit-seeds", {}, "Write seeds to an empty directory"});
  auto target_flag = std::make_shared<StringFlag>(
      FlagInfo{"--target", {}, "Canonical AMDGPU target ID or registered alias"}, "gfx1250");
  auto help =
      std::make_shared<BoolFlag>(FlagInfo{"--help", {"-h"}, "Show help", ValueMode::Switch});
  auto report = [](std::string_view message) { std::cerr << message << '\n'; };
  util::DiagnosticEmitter diagnostic(report);
  if (command
          .add_flags("Decoder fuzzing", {afl_flag, json_flag, input_flag, seeds_flag, target_flag},
                     diagnostic)
          .failed() ||
      command.add_flag(help, diagnostic).failed())
    return 2;
  std::vector<std::string_view> arguments(argv + 1, argv + argc);
  auto parsed = parse(command, arguments, diagnostic);
  if (parsed.failed())
    return 2;
  if (help->value()) {
    usage(std::cout, command);
    return 0;
  }
  if (!parsed.value().positional.empty()) {
    usage(std::cerr, command);
    return 2;
  }
  const bool afl = afl_flag->value();
  const bool json = json_flag->value();
  const auto &input = input_flag->value();
  const auto &seed_directory = seeds_flag->value();
  const auto &target = target_flag->value();

  const int modes = static_cast<int>(afl) + static_cast<int>(!input.empty()) +
                    static_cast<int>(!seed_directory.empty());
  const bool replay = !input.empty();
  if (modes != 1 || json != replay) {
    usage(std::cerr, command);
    return 2;
  }

  const auto *descriptor = rocjitsu::default_isa_target_registry().find(target);
  if (descriptor == nullptr) {
    std::cerr << "rj_decode_fuzz: unsupported decoder target: " << target << '\n';
    return 2;
  }
  const auto *gpu_target =
      rocjitsu::default_isa_target_registry().find_gpu_target_by_code_object_id(target);
  const std::string_view decoder_target =
      gpu_target == nullptr ? descriptor->id : gpu_target->code_object_id;

  const auto report_error = [](std::string_view message) {
    std::cerr << "rj_decode_fuzz: " << message << '\n';
  };
  const util::DiagnosticEmitter emit_error(report_error);
  if (afl)
    return run_afl(decoder_target, emit_error);

  if (!seed_directory.empty()) {
    const util::FailureOr<size_t> written = emit_seeds(seed_directory, descriptor->id, emit_error);
    if (written.failed())
      return 2;
    std::cout << "wrote " << written.value() << ' ' << decoder_target << " seeds\n";
    return 0;
  }
  return run_replay(input, decoder_target, emit_error).succeeded() ? 0 : 2;
}
