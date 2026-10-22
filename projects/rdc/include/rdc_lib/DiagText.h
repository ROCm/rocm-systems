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

#ifndef INCLUDE_RDC_LIB_DIAGTEXT_H_
#define INCLUDE_RDC_LIB_DIAGTEXT_H_

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/rdc_common.h"

// The text of a diagnostic result, which librdc's checks and the diagnostic
// modules build the same way. Both are hidden, so that every library that
// includes this header keeps its own copy and none exports it.

namespace amd {
namespace rdc {

// A check's text, built from units (sentences, topology pairs, process IDs) joined by single
// spaces in at most MAX_DIAG_MSG_LENGTH - 1 bytes. The fixed start, if any, is always kept. Units
// are kept while they fit and the rest only counted; a cut text ends after a whole unit with
// " (N more not shown)".
class __attribute__((visibility("hidden"))) DiagText {
 public:
  explicit DiagText(std::string start = {}) : text_(std::move(start)) {}

  void add(const std::string& unit) {
    if (unit.empty()) {
      return;  // an empty unit says nothing, and must not be counted as not shown
    }
    const size_t space = text_.empty() ? 0 : 1;
    if (more_ == 0 && text_.size() + space + unit.size() <= kBudget) {
      unit_starts_.push_back(text_.size());
      text_.append(space, ' ');
      text_ += unit;
    } else {
      more_++;
    }
  }

  bool has_units() const { return !unit_starts_.empty() || more_ != 0; }

  void copy_to(char (&out)[MAX_DIAG_MSG_LENGTH]) const {
    std::string text = text_;
    size_t kept = unit_starts_.size();
    size_t more = more_;
    while (more != 0) {
      // The suffix is the one for the count it would show, so what is kept is decided against
      // the text that will be written.
      const std::string suffix =
          (text.empty() ? "(" : " (") + std::to_string(more) + " more not shown)";
      if (text.size() + suffix.size() <= kBudget) {
        text += suffix;
        break;
      }
      if (kept != 0) {
        text.resize(unit_starts_[--kept]);
        more++;
        continue;
      }
      // No unit can be kept: the fixed start is cut to leave room for the suffix, at the last
      // space before that length or at that length when it has none, so the suffix always fits.
      const size_t room = kBudget - suffix.size();
      const size_t space = text.rfind(' ', room);
      text.resize(space == std::string::npos ? room : space);
      text += suffix;
      break;
    }
    strncpy_with_null(out, text.c_str(), sizeof(out));
  }

 private:
  static constexpr size_t kBudget = MAX_DIAG_MSG_LENGTH - 1;
  std::string text_;
  std::vector<size_t> unit_starts_;  // where each kept unit, with the space before it, starts
  size_t more_ = 0;                  // units not kept
};

// A check of a group with no GPU: nothing was checked, and the result says so. The caller sets
// *result and result->test_case first, so that the result it leaves names its check.
inline __attribute__((visibility("hidden"))) rdc_status_t
skip_empty_group(rdc_diag_test_result_t* result) {
  result->status = RDC_DIAG_RESULT_SKIP;
  strncpy_with_null(result->info, "The group has no GPU.", MAX_DIAG_MSG_LENGTH);
  return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_DIAGTEXT_H_
