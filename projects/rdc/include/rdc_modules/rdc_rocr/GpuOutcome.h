/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RDC_MODULES_RDC_ROCR_GPUOUTCOME_H_
#define RDC_MODULES_RDC_ROCR_GPUOUTCOME_H_

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "rdc_lib/DiagText.h"

// One GPU's outcome, which outlives the test objects that check it: a release that fails runs in
// a test object's destructor, and the GPU's result is composed from this object after that test
// object is gone, so that a failed destroy still makes that GPU's failure.

namespace amd {
namespace rdc {

// A sentence that follows "GPU <id>: " reads as part of that line, so its first letter is lower
// cased when the rest of its first word is lower case ("ROCr does not list the GPU." keeps its
// capital, as a name does).
inline std::string lower_cased(const std::string& sentence) {
  std::string out = sentence;
  const size_t word_end = std::min(out.find(' '), out.size());
  for (size_t i = 1; i < word_end; i++) {
    if (std::isupper(static_cast<unsigned char>(out[i])) != 0) {
      return out;
    }
  }
  if (!out.empty()) {
    out[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(out[0])));
  }
  return out;
}

// One GPU's outcome: the sentence of each sub-test that ran, the first failure and the first skip
// among them, and the outcome they add up to.
class GpuOutcome {
 public:
  void add(rdc_diag_result_t outcome, const std::string& sentence) {
    // Each sub-test looks the GPU's agent up, so a GPU ROCr does not list gives every one of them
    // the same sentence: a text that said it already says no more.
    if (std::find(said_.begin(), said_.end(), sentence) == said_.end()) {
      said_.push_back(sentence);
    }
    if (outcome == RDC_DIAG_RESULT_FAIL) {
      failed_ = true;
      if (first_fail_.empty()) {
        first_fail_ = sentence;
      }
    } else if (outcome == RDC_DIAG_RESULT_SKIP) {
      if (first_skip_.empty()) {
        first_skip_ = sentence;
      }
    } else {
      passed_ = true;
    }
  }

  // Fail when a sub-test failed, else Pass when one passed, else Skip: nothing was checked.
  rdc_diag_result_t status() const {
    if (failed_) {
      return RDC_DIAG_RESULT_FAIL;
    }
    return passed_ ? RDC_DIAG_RESULT_PASS : RDC_DIAG_RESULT_SKIP;
  }

  // The GPU's unit of the check's info: what a reader needs first about it.
  std::string summary(uint32_t entity, const char* passed) const {
    std::string sentence = passed;
    if (!first_fail_.empty()) {
      sentence = first_fail_;
    } else if (!first_skip_.empty()) {
      sentence = first_skip_;
    }
    return "GPU " + std::to_string(entity) + ": " + lower_cased(sentence);
  }

  // The GPU's own text: one sentence per sub-test outcome, cut at a whole sentence.
  void copy_to(char (&out)[MAX_DIAG_MSG_LENGTH]) const {
    DiagText sentences;
    for (const std::string& sentence : said_) {
      sentences.add(sentence);
    }
    sentences.copy_to(out);
  }

 private:
  std::vector<std::string> said_;  // the sentences of this GPU, each of them once
  std::string first_fail_;
  std::string first_skip_;
  bool passed_ = false;
  bool failed_ = false;
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCR_GPUOUTCOME_H_
