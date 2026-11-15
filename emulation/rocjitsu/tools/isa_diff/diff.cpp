// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "diff.h"

#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cstring>

namespace rocjitsu::isa_diff {
namespace {

std::string key_of(const InstRef &ref) {
  std::vector<uint32_t> words = ref.words;
  if (ref.inst && (ref.inst->flags() & (BRANCH | COND_BRANCH)) &&
      !(ref.inst->flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) && words.size() == 1)
    words[0] &= 0xffff0000u;
  return std::string(reinterpret_cast<const char *>(words.data()), words.size() * 4);
}

/// LCS over [ob, oe) x [nb, ne); appends matched pairs in order. Returns false if over budget.
bool lcs(const std::vector<std::string> &a, const std::vector<std::string> &b, size_t ob, size_t oe,
         size_t nb, size_t ne, size_t max_cells, std::vector<std::pair<size_t, size_t>> &out) {
  const size_t n = oe - ob, m = ne - nb;
  if (n == 0 || m == 0)
    return true;
  if ((n + 1) > max_cells / (m + 1))
    return false;
  std::vector<uint32_t> dp((n + 1) * (m + 1), 0);
  auto at = [&](size_t i, size_t j) -> uint32_t & { return dp[i * (m + 1) + j]; };
  for (size_t i = n; i-- > 0;)
    for (size_t j = m; j-- > 0;)
      at(i, j) =
          a[ob + i] == b[nb + j] ? at(i + 1, j + 1) + 1 : std::max(at(i + 1, j), at(i, j + 1));
  size_t i = 0, j = 0;
  while (i < n && j < m) {
    if (a[ob + i] == b[nb + j]) {
      out.emplace_back(ob + i, nb + j);
      ++i;
      ++j;
    } else if (at(i + 1, j) >= at(i, j + 1)) {
      ++i;
    } else {
      ++j;
    }
  }
  return true;
}

} // namespace

const char *change_kind_name(ChangeKind kind) {
  switch (kind) {
  case ChangeKind::Instruction:
    return "instruction";
  case ChangeKind::ControlFlow:
    return "control_flow";
  case ChangeKind::Layout:
    return "layout";
  case ChangeKind::Metadata:
    return "metadata";
  case ChangeKind::Unexplained:
    return "unexplained";
  }
  return "?";
}

DiffResult diff_kernels(const KernelModel &old_model, const KernelModel &new_model,
                        size_t max_lcs_cells) {
  const auto &oi = old_model.insts();
  const auto &ni = new_model.insts();
  std::vector<std::string> ok(oi.size()), nk(ni.size());
  for (size_t i = 0; i < oi.size(); ++i)
    ok[i] = key_of(oi[i]);
  for (size_t j = 0; j < ni.size(); ++j)
    nk[j] = key_of(ni[j]);

  DiffResult r;
  // Common prefix and suffix, then LCS of the middle.
  size_t pre = 0;
  while (pre < oi.size() && pre < ni.size() && ok[pre] == nk[pre])
    ++pre;
  size_t suf = 0;
  while (suf < oi.size() - pre && suf < ni.size() - pre &&
         ok[oi.size() - 1 - suf] == nk[ni.size() - 1 - suf])
    ++suf;
  std::vector<std::pair<size_t, size_t>> matched;
  for (size_t i = 0; i < pre; ++i)
    matched.emplace_back(i, i);
  if (!lcs(ok, nk, pre, oi.size() - suf, pre, ni.size() - suf, max_lcs_cells, matched)) {
    r.alignment_complete = false;
    r.notes.push_back(
        {ChangeKind::Unexplained, "alignment budget exceeded; middle treated as one hunk", true});
  }
  for (size_t k = suf; k-- > 0;)
    matched.emplace_back(oi.size() - 1 - k, ni.size() - 1 - k);

  r.old_to_new.assign(oi.size(), -1);
  r.new_to_old.assign(ni.size(), -1);
  for (auto [i, j] : matched) {
    r.old_to_new[i] = static_cast<int64_t>(j);
    r.new_to_old[j] = static_cast<int64_t>(i);
  }

  // Validate matched direct branches: their targets must correspond.
  std::vector<bool> branch_mismatch(oi.size(), false);
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < oi.size(); ++i) {
      const int64_t j = r.old_to_new[i];
      if (j < 0)
        continue;
      const int64_t ti = oi[i].branch_target_index;
      const int64_t tj = ni[static_cast<size_t>(j)].branch_target_index;
      const bool is_branch = ti >= 0 || tj >= 0;
      if (!is_branch)
        continue;
      if (ti >= 0 && tj >= 0 && r.old_to_new[static_cast<size_t>(ti)] == tj)
        continue;
      r.old_to_new[i] = -1;
      r.new_to_old[static_cast<size_t>(j)] = -1;
      branch_mismatch[i] = true;
      changed = true;
    }
  }

  // Hunks between consecutive matched pairs.
  size_t i = 0, j = 0;
  while (i < oi.size() || j < ni.size()) {
    if (i < oi.size() && j < ni.size() && r.old_to_new[i] == static_cast<int64_t>(j)) {
      ++i;
      ++j;
      continue;
    }
    Hunk h;
    h.old_begin = i;
    h.new_begin = j;
    while (i < oi.size() && r.old_to_new[i] < 0)
      ++i;
    while (j < ni.size() && r.new_to_old[j] < 0)
      ++j;
    h.old_end = i;
    h.new_end = j;
    bool control = false;
    for (size_t k = h.old_begin; k < h.old_end; ++k)
      control |= oi[k].fx.control_flow || branch_mismatch[k];
    for (size_t k = h.new_begin; k < h.new_end; ++k)
      control |= ni[k].fx.control_flow;
    h.kind = control ? ChangeKind::ControlFlow : ChangeKind::Instruction;
    if (control)
      h.note = "hunk contains a branch, terminator or a branch whose target changed";
    if (h.old_begin == h.old_end && h.new_begin == h.new_end) {
      // Crossed matches (should not happen with LCS); avoid an infinite loop.
      if (i < oi.size())
        ++i;
      if (j < ni.size())
        ++j;
      continue;
    }
    r.hunks.push_back(std::move(h));
  }

  // Layout: matched instructions at different kernel-relative offsets.
  const auto &oa = old_model.artifact();
  const auto &na = new_model.artifact();
  for (size_t k = 0; k < oi.size(); ++k) {
    const int64_t m = r.old_to_new[k];
    if (m < 0)
      continue;
    if (oi[k].offset - oa.kernel_begin != ni[static_cast<size_t>(m)].offset - na.kernel_begin) {
      r.layout_shift = true;
      break;
    }
  }
  if (oa.text_vaddr + oa.kernel_begin != na.text_vaddr + na.kernel_begin)
    r.notes.push_back({ChangeKind::Layout, "kernel entry address differs", false});
  if (r.layout_shift) {
    const bool pc_dependent = old_model.reads_pc() || new_model.reads_pc();
    r.notes.push_back({ChangeKind::Layout,
                       pc_dependent ? "instruction layout shifted in a kernel that reads the PC"
                                    : "instruction layout shifted (no PC reads in either kernel)",
                       pc_dependent});
  }

  // Descriptor and metadata.
  bool layout_only = false;
  const auto fields = descriptor_semantic_differences(oa.descriptor, na.descriptor, layout_only);
  for (const auto &f : fields)
    r.notes.push_back({ChangeKind::Metadata, "kernel descriptor field differs: " + f, true});
  if (layout_only)
    r.notes.push_back({ChangeKind::Layout,
                       "kernel descriptor differs only in kernel_code_entry_byte_offset", false});
  if (oa.metadata_sha256 != na.metadata_sha256)
    r.notes.push_back({ChangeKind::Metadata, "code-object note (AMDHSA metadata) differs", true});
  if (old_model.uncovered_bytes() != new_model.uncovered_bytes())
    r.notes.push_back({ChangeKind::Unexplained,
                       "bytes not covered by reachable instructions differ (unreachable code or "
                       "padding)",
                       true});
  if (!old_model.cfg_complete() || !new_model.cfg_complete())
    r.notes.push_back({ChangeKind::ControlFlow,
                       "CFG has unknown successors (indirect flow, calls or unknown effects): "
                       "liveness is widened to everything there, and single entry of regions "
                       "cannot be checked",
                       true});
  return r;
}

} // namespace rocjitsu::isa_diff
