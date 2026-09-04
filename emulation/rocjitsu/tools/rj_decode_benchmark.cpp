// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_decode_benchmark.cpp
/// @brief Measure decoder throughput over the text of a real code object.

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace rocjitsu;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kLookaheadWords = 4;
constexpr std::size_t kDefaultInvalidLimit = 65536;

struct TextRange {
  std::size_t begin;
  std::size_t word_count;
};

struct DecodeCorpus {
  std::vector<uint32_t> words;
  std::vector<TextRange> ranges;
  std::vector<std::size_t> valid_offsets;
  std::vector<std::array<uint32_t, kLookaheadWords>> invalid_encodings;
  std::size_t text_bytes = 0;
  std::size_t scan_rejections = 0;
};

struct Options {
  std::string input;
  std::optional<rj_code_target_id_t> target;
  std::size_t iterations = 1;
  std::size_t invalid_limit = kDefaultInvalidLimit;
};

struct Measurement {
  std::size_t attempts = 0;
  std::size_t unexpected = 0;
  uint64_t checksum = 0;
  std::chrono::nanoseconds elapsed{};
};

void print_usage(std::ostream &os) {
  os << "Usage: rj_decode_benchmark INPUT [--target TARGET] [--iterations N] "
        "[--invalid-limit N]\n"
        "Targets: gfx942, gfx950, gfx1200, gfx1201, gfx1250, gfx1251\n";
}

[[nodiscard]] std::optional<rj_code_target_id_t> parse_target(std::string_view value) {
  if (value == "gfx942")
    return ROCJITSU_CODE_TARGET_GFX942;
  if (value == "gfx950")
    return ROCJITSU_CODE_TARGET_GFX950;
  if (value == "gfx1200")
    return ROCJITSU_CODE_TARGET_GFX1200;
  if (value == "gfx1201")
    return ROCJITSU_CODE_TARGET_GFX1201;
  if (value == "gfx1250")
    return ROCJITSU_CODE_TARGET_GFX1250;
  if (value == "gfx1251")
    return ROCJITSU_CODE_TARGET_GFX1251;
  return std::nullopt;
}

[[nodiscard]] bool parse_size(std::string_view text, std::size_t &value) {
  auto [ptr, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && ptr == text.data() + text.size();
}

[[nodiscard]] std::optional<Options> parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg(argv[index]);
    if (arg == "--help") {
      print_usage(std::cout);
      return std::nullopt;
    }
    if (arg == "--target" && index + 1 < argc) {
      options.target = parse_target(argv[++index]);
      if (!options.target) {
        std::cerr << "Unsupported target: " << argv[index] << '\n';
        return std::nullopt;
      }
      continue;
    }
    if (arg == "--iterations" && index + 1 < argc) {
      if (!parse_size(argv[++index], options.iterations) || options.iterations == 0) {
        std::cerr << "Iterations must be a positive integer\n";
        return std::nullopt;
      }
      continue;
    }
    if (arg == "--invalid-limit" && index + 1 < argc) {
      if (!parse_size(argv[++index], options.invalid_limit)) {
        std::cerr << "Invalid limit must be a non-negative integer\n";
        return std::nullopt;
      }
      continue;
    }
    if (!arg.starts_with("--") && options.input.empty()) {
      options.input = arg;
      continue;
    }
    std::cerr << "Unexpected argument: " << arg << '\n';
    return std::nullopt;
  }
  if (options.input.empty()) {
    print_usage(std::cerr);
    return std::nullopt;
  }
  return options;
}

void copy_encoding(std::span<const uint32_t> words, std::size_t offset,
                   std::array<uint32_t, kLookaheadWords> &copy) {
  copy.fill(0);
  const std::size_t available = std::min(copy.size(), words.size() - offset);
  std::copy_n(words.begin() + static_cast<std::ptrdiff_t>(offset), available, copy.begin());
}

[[nodiscard]] bool collect_corpus(const AmdGpuCodeObject &object, Decoder &decoder,
                                  std::size_t invalid_limit, DecodeCorpus &corpus,
                                  std::string &error) {
  for (const Section *section : object.text_sections()) {
    const std::size_t word_count = section->size() / sizeof(uint32_t);
    const std::size_t begin = corpus.words.size();
    corpus.words.resize(begin + word_count + kLookaheadWords, 0);
    std::memcpy(corpus.words.data() + begin, section->data(), word_count * sizeof(uint32_t));
    corpus.ranges.push_back({begin, word_count});
    corpus.text_bytes += word_count * sizeof(uint32_t);
  }

  for (const TextRange range : corpus.ranges) {
    std::size_t relative_offset = 0;
    while (relative_offset < range.word_count) {
      const std::size_t offset = range.begin + relative_offset;
      DecodeResult decoded = decoder.decode(corpus.words.data() + offset);
      if (decoded.failed()) {
        ++corpus.scan_rejections;
        if (corpus.invalid_encodings.size() < invalid_limit) {
          std::array<uint32_t, kLookaheadWords> encoding;
          copy_encoding(corpus.words, offset, encoding);
          corpus.invalid_encodings.push_back(encoding);
        }
        ++relative_offset;
        continue;
      }

      const std::size_t byte_size = decoded.value()->size();
      if (byte_size == 0 || byte_size % sizeof(uint32_t) != 0) {
        error = "decoder returned an invalid instruction size";
        return false;
      }
      const std::size_t instruction_words = byte_size / sizeof(uint32_t);
      if (instruction_words > range.word_count - relative_offset) {
        error = "decoded instruction crosses a text-section boundary";
        return false;
      }
      corpus.valid_offsets.push_back(offset);
      relative_offset += instruction_words;
    }
  }

  static constexpr std::array<uint32_t, 4> kMutationMasks = {0xffffffffu, 0xff800000u, 0x007fffffu,
                                                             0x9e3779b9u};
  for (std::size_t offset : corpus.valid_offsets) {
    if (corpus.invalid_encodings.size() >= invalid_limit)
      break;
    std::array<uint32_t, kLookaheadWords> encoding;
    copy_encoding(corpus.words, offset, encoding);
    const uint32_t original = encoding[0];
    for (uint32_t mask : kMutationMasks) {
      encoding[0] = original ^ mask;
      if (decoder.decode(encoding.data()).failed()) {
        corpus.invalid_encodings.push_back(encoding);
        break;
      }
    }
  }
  return true;
}

[[nodiscard]] Measurement measure_valid(const DecodeCorpus &corpus, Decoder &decoder,
                                        std::size_t iterations) {
  Measurement measurement;
  measurement.attempts = corpus.valid_offsets.size() * iterations;
  const auto start = Clock::now();
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    for (std::size_t offset : corpus.valid_offsets) {
      DecodeResult decoded = decoder.decode(corpus.words.data() + offset);
      if (decoded.failed()) {
        ++measurement.unexpected;
        continue;
      }
      measurement.checksum += decoded.value()->size();
    }
  }
  measurement.elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
  return measurement;
}

[[nodiscard]] Measurement measure_invalid(const DecodeCorpus &corpus, Decoder &decoder,
                                          std::size_t iterations) {
  Measurement measurement;
  measurement.attempts = corpus.invalid_encodings.size() * iterations;
  const auto start = Clock::now();
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    for (const auto &encoding : corpus.invalid_encodings) {
      DecodeResult decoded = decoder.decode(encoding.data());
      if (decoded.succeeded()) {
        ++measurement.unexpected;
        measurement.checksum += decoded.value()->size();
      }
    }
  }
  measurement.elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
  return measurement;
}

void print_measurement(std::string_view name, const Measurement &measurement) {
  const double elapsed_ns = static_cast<double>(measurement.elapsed.count());
  const double ns_per_attempt = measurement.attempts == 0 ? 0.0 : elapsed_ns / measurement.attempts;
  const double mips = elapsed_ns == 0.0 ? 0.0 : measurement.attempts / (elapsed_ns / 1e9) / 1e6;
  std::cout << name << ": " << measurement.attempts << " attempts, " << std::fixed
            << std::setprecision(2) << ns_per_attempt << " ns/attempt, " << mips << " MIPS, "
            << measurement.unexpected << " unexpected, checksum " << measurement.checksum << '\n';
}

} // namespace

int main(int argc, char **argv) {
  const std::optional<Options> options = parse_options(argc, argv);
  if (!options)
    return argc > 1 && std::string_view(argv[1]) == "--help" ? 0 : 1;

  try {
    AmdGpuCodeObject object(options->input);
    if (!object.is_valid()) {
      std::cerr << "Failed to parse input code object\n";
      return 2;
    }
    const rj_code_target_id_t object_target = object.target_id();
    if (options->target && object_target != ROCJITSU_CODE_TARGET_INVALID &&
        *options->target != object_target) {
      std::cerr << "Selected target does not match code-object target metadata\n";
      return 2;
    }
    const rj_code_target_id_t target = options->target.value_or(object_target);
    if (target == ROCJITSU_CODE_TARGET_INVALID) {
      std::cerr << "Cannot infer decoder target; pass --target\n";
      return 2;
    }
    std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
    if (!decoder) {
      std::cerr << "No decoder is available for the selected target\n";
      return 2;
    }

    DecodeCorpus corpus;
    std::string error;
    if (!collect_corpus(object, *decoder, options->invalid_limit, corpus, error)) {
      std::cerr << error << '\n';
      return 2;
    }
    if (corpus.valid_offsets.empty()) {
      std::cerr << "No valid instruction encodings found in text sections\n";
      return 2;
    }

    std::cout << "text: " << corpus.text_bytes << " bytes, " << corpus.valid_offsets.size()
              << " valid instructions, " << corpus.scan_rejections << " rejected words, "
              << corpus.invalid_encodings.size() << " invalid samples\n";
    const Measurement valid = measure_valid(corpus, *decoder, options->iterations);
    print_measurement("valid", valid);
    if (!corpus.invalid_encodings.empty()) {
      const Measurement invalid = measure_invalid(corpus, *decoder, options->iterations);
      print_measurement("invalid", invalid);
    }
    return valid.unexpected == 0 ? 0 : 3;
  } catch (const std::exception &error) {
    std::cerr << "Benchmark failed: " << error.what() << '\n';
    return 2;
  }
}
