// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/analysis/analyze.h"

#include <charconv>
#include <fstream>
#include <iostream>

namespace {

constexpr int kExitNoObservedFindings = 0;
constexpr int kExitFindings = 1;
constexpr int kExitInputError = 2;
constexpr int kExitOutputFailure = 3;

} // namespace

int main(int argc, char **argv) {
  using namespace rocjitsu::consan;
  AnalysisOptions options;
  if (argc < 2) {
    std::cerr
        << "Usage: rj_consan_analyze CAPTURE [--examples 0.." << kMaxFindingExamples << "]\n"
        << "  --examples LIMIT: retain at most LIMIT finding examples per check per execution\n"
        << "  (default " << options.examples << ", maximum " << kMaxFindingExamples
        << "; 0 hides finding examples).\n"
        << "  Counts, coverage and integrity diagnostics are unchanged by this limit.\n";
    return kExitInputError;
  }
  for (int index = 2; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "--examples" && index + 1 < argc) {
      const std::string_view number(argv[++index]);
      const auto parsed =
          std::from_chars(number.data(), number.data() + number.size(), options.examples);
      if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() ||
          options.examples > kMaxFindingExamples)
        return kExitInputError;
    } else
      return kExitInputError;
  }
  std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
  if (!file || file.tellg() < 0 || uint64_t(file.tellg()) > CaptureLimits{}.file_bytes) {
    std::cerr << "capture input error: missing, unseekable or oversized input\n";
    return kExitInputError;
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(file.tellg()));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  if (!file || file.peek() != std::char_traits<char>::eof()) {
    std::cerr << "capture input error: incomplete or changing input\n";
    return kExitInputError;
  }
  ContractError error;
  auto capture = decode_capture(bytes, &error);
  if (capture.failed()) {
    std::cerr << "capture input error: " << error.message << '\n';
    return kExitInputError;
  }
  const auto result = analyze_capture(capture.value(), options, &error);
  if (result.failed()) {
    std::cerr << "analysis input error: " << error.message << '\n';
    return kExitInputError;
  }
  std::cout << render_analysis(result.value());
  std::cout.flush();
  if (!std::cout)
    return kExitOutputFailure;
  return has_findings(result.value()) ? kExitFindings : kExitNoObservedFindings;
}
