// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "diff.h"
#include "effects.h"
#include "json.h"

#include "rocjitsu/code/kernel_descriptor_abi.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cstring>
#include <set>

namespace rocjitsu::isa_diff {
namespace {

std::string key_of(const InstRef &ref) {
  if (ref.inline_asm >= 0) {
    // An inline-asm block matches only the block with the same position in the other
    // function's sequence and the same text: a renamed operand makes it a change, and a
    // matched pair receives the same values. Machine-word keys have a length that is a
    // multiple of 4; this one never does.
    std::string key =
        std::string(1, '\0') + "asm#" + std::to_string(ref.inline_asm) + "\n" + ref.asm_text;
    if (key.size() % 4 == 0)
      key.push_back('\0');
    return key;
  }
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

/// Common prefix and suffix, then LCS of the middle of @p a and @p b; appends the matched
/// pairs in order. Returns false if the middle was over budget (left unmatched).
bool align(const std::vector<std::string> &a, const std::vector<std::string> &b, size_t max_cells,
           std::vector<std::pair<size_t, size_t>> &out) {
  size_t pre = 0;
  while (pre < a.size() && pre < b.size() && a[pre] == b[pre])
    ++pre;
  size_t suf = 0;
  while (suf < a.size() - pre && suf < b.size() - pre &&
         a[a.size() - 1 - suf] == b[b.size() - 1 - suf])
    ++suf;
  for (size_t i = 0; i < pre; ++i)
    out.emplace_back(i, i);
  const bool ok = lcs(a, b, pre, a.size() - suf, pre, b.size() - suf, max_cells, out);
  for (size_t k = suf; k-- > 0;)
    out.emplace_back(a.size() - 1 - k, b.size() - 1 - k);
  return ok;
}

/// A scheduling hint or dependency wait: no functional effect on an in-order wave.
bool is_hint(const InstRef &ref) { return ref.inline_asm < 0 && is_region_noop(ref.mnemonic); }

} // namespace

namespace {

/// Highest VGPR any reachable instruction names, plus one.
unsigned vgpr_usage(const KernelModel &m) {
  VSet used;
  for (const auto &ref : m.insts())
    used |= ref.fx.vreads | ref.fx.vwrites;
  return vgpr_extent(used);
}

/// VGPRs the descriptor's granulated count allocates (the granule depends on the target
/// and wave size), or 0 without a descriptor.
unsigned vgpr_total(const std::vector<uint8_t> &descriptor, const Target &target) {
  if (descriptor.size() < sizeof(amdhsa::kernel_descriptor_t))
    return 0;
  amdhsa::kernel_descriptor_t kd{};
  std::memcpy(&kd, descriptor.data(), sizeof(kd));
  using namespace amdhsa;
  return (RJ_AMDHSA_BITS_GET(kd.compute_pgm_rsrc1,
                             COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT) +
          1) *
         descriptor_vgpr_granularity_for_wavefront(target.arch, target.wave_size);
}

/// Architectural VGPRs the descriptor allocates: below ACCUM_OFFSET on a unified register
/// file (gfx90a and later CDNA, where AGPRs follow), the granulated count otherwise; 0
/// without a descriptor.
unsigned vgpr_allocation(const std::vector<uint8_t> &descriptor, const Target &target) {
  if (descriptor.size() < sizeof(amdhsa::kernel_descriptor_t))
    return 0;
  if (!target.unified_vgprs)
    return vgpr_total(descriptor, target);
  amdhsa::kernel_descriptor_t kd{};
  std::memcpy(&kd, descriptor.data(), sizeof(kd));
  using namespace amdhsa;
  return (RJ_AMDHSA_BITS_GET(kd.compute_pgm_rsrc3, COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) + 1) * 4;
}

/// Highest SGPR any reachable instruction names, plus one.
unsigned sgpr_usage(const KernelModel &m) {
  unsigned used = 0;
  for (const auto &ref : m.insts())
    for (unsigned l = 0; l < kNumSgprs; ++l)
      if (ref.fx.reads.test(l) || ref.fx.writes.test(l))
        used = std::max(used, l + 1);
  return used;
}

/// The descriptor's register allocation covers what the kernel names: arch VGPRs within
/// ACCUM_OFFSET and the granulated VGPR count; on GFX9, SGPRs plus the six extra ones (VCC,
/// FLAT_SCRATCH, XNACK) within the granule of 16 (GFX10+ always gives a wave s0-s105).
bool allocation_covers_usage(const KernelModel &m) {
  const auto &desc = m.artifact().descriptor;
  const Target &target = m.artifact().target;
  if (desc.size() < sizeof(amdhsa::kernel_descriptor_t))
    return false;
  amdhsa::kernel_descriptor_t kd{};
  std::memcpy(&kd, desc.data(), sizeof(kd));
  using namespace amdhsa;
  const unsigned vused = vgpr_usage(m);
  if (vused > vgpr_allocation(desc, target) || vused > vgpr_total(desc, target))
    return false;
  if (target.major >= 10)
    return true;
  const unsigned sgpr_total =
      (RJ_AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) +
       1) *
      16;
  return sgpr_usage(m) + 6 <= sgpr_total;
}

/// The two descriptors differ only in register-allocation bits of compute_pgm_rsrc1
/// (granulated VGPR/SGPR counts) and compute_pgm_rsrc3 (ACCUM_OFFSET).
bool allocation_only_difference(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b,
                                const std::string &field) {
  if (a.size() < sizeof(amdhsa::kernel_descriptor_t) || a.size() != b.size())
    return false;
  amdhsa::kernel_descriptor_t ka{}, kb{};
  std::memcpy(&ka, a.data(), sizeof(ka));
  std::memcpy(&kb, b.data(), sizeof(kb));
  using namespace amdhsa;
  if (field == "compute_pgm_rsrc1")
    return ((ka.compute_pgm_rsrc1 ^ kb.compute_pgm_rsrc1) &
            ~(COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT |
              COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT)) == 0;
  if (field == "compute_pgm_rsrc3")
    return ((ka.compute_pgm_rsrc3 ^ kb.compute_pgm_rsrc3) &
            ~COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) == 0;
  return false;
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
  case ChangeKind::Hint:
    return "hint";
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
  // Instructions with an effect first (common prefix and suffix, then LCS of the middle);
  // hints are left out, so a moved or re-encoded s_delay_alu cannot change which real
  // instructions pair up.
  std::vector<size_t> ox, nx;
  std::vector<std::string> okx, nkx;
  for (size_t i = 0; i < oi.size(); ++i)
    if (!is_hint(oi[i])) {
      ox.push_back(i);
      okx.push_back(ok[i]);
    }
  for (size_t j = 0; j < ni.size(); ++j)
    if (!is_hint(ni[j])) {
      nx.push_back(j);
      nkx.push_back(nk[j]);
    }
  std::vector<std::pair<size_t, size_t>> core;
  if (!align(okx, nkx, max_lcs_cells, core)) {
    r.alignment_complete = false;
    r.notes.push_back(
        {ChangeKind::Unexplained, "alignment budget exceeded; middle treated as one hunk", true});
  }
  // Then the gaps between matched pairs: only hints can still match there (any other
  // match would have lengthened the common subsequence above). A gap over budget leaves
  // its hints unmatched, which only makes hint hunks.
  std::vector<std::pair<size_t, size_t>> matched;
  size_t gap_i = 0, gap_j = 0;
  auto fill_gap = [&](size_t end_i, size_t end_j) {
    if (gap_i < end_i && gap_j < end_j)
      (void)lcs(ok, nk, gap_i, end_i, gap_j, end_j, max_lcs_cells, matched);
  };
  for (auto [x, y] : core) {
    fill_gap(ox[x], nx[y]);
    matched.emplace_back(ox[x], nx[y]);
    gap_i = ox[x] + 1;
    gap_j = nx[y] + 1;
  }
  fill_gap(oi.size(), ni.size());

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
    bool control = false, hint = true;
    for (size_t k = h.old_begin; k < h.old_end; ++k) {
      control |= oi[k].fx.control_flow || branch_mismatch[k];
      hint &= is_hint(oi[k]);
    }
    for (size_t k = h.new_begin; k < h.new_end; ++k) {
      control |= ni[k].fx.control_flow;
      hint &= is_hint(ni[k]);
    }
    h.kind = control ? ChangeKind::ControlFlow : hint ? ChangeKind::Hint : ChangeKind::Instruction;
    if (control)
      h.note = "hunk contains a branch, terminator or a branch whose target changed";
    else if (hint)
      h.note = "scheduling hints or dependency waits only: no functional effect on an in-order "
               "wave (hardware hazards are not checked)";
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
  // Register-allocation-only descriptor changes (granulated counts, ACCUM_OFFSET) are
  // informational when each kernel's allocation covers the registers it names; anything
  // else in the descriptor is semantic.
  const bool both_fit = allocation_covers_usage(old_model) && allocation_covers_usage(new_model);
  std::vector<std::string> semantic_fields;
  for (const auto &f : fields) {
    if (both_fit && allocation_only_difference(oa.descriptor, na.descriptor, f)) {
      r.notes.push_back({ChangeKind::Metadata,
                         "kernel descriptor field differs: " + f +
                             " (register allocation only; informational: each kernel's "
                             "registers fit its allocation)",
                         false});
      continue;
    }
    semantic_fields.push_back(f);
    r.notes.push_back({ChangeKind::Metadata, "kernel descriptor field differs: " + f, true});
  }
  // Register-count metadata (.vgpr_count, .sgpr_count, .agpr_count) is informational
  // when the descriptors agree on every semantic field and each kernel's VGPRs fit its
  // allocation; a kernel naming VGPRs beyond its allocation is reported as semantic.
  bool counts_informational = semantic_fields.empty();
  for (const auto *m : {&old_model, &new_model}) {
    const unsigned alloc = vgpr_allocation(m->artifact().descriptor, m->artifact().target);
    const unsigned used = vgpr_usage(*m);
    if (alloc == 0) {
      counts_informational = false;
      continue;
    }
    if (used > alloc) {
      counts_informational = false;
      r.notes.push_back({ChangeKind::Metadata,
                         std::string(m == &old_model ? "old" : "new") + " kernel names v" +
                             std::to_string(used - 1) + " beyond its descriptor allocation of " +
                             std::to_string(alloc) + " VGPRs",
                         true});
    }
  }
  if (layout_only)
    r.notes.push_back({ChangeKind::Layout,
                       "kernel descriptor differs only in kernel_code_entry_byte_offset", false});
  if (oa.metadata_parsed && na.metadata_parsed) {
    // Field-level comparison of this kernel's metadata and the global fields; a note that
    // differs only in other kernels' entries does not concern this kernel.
    // Argument records are compared as units: one inserted argument shifts every later
    // index, which is one change, not dozens.
    auto split_args = [](const std::map<std::string, std::string> &m,
                         std::map<std::string, std::string> &rest) {
      std::map<size_t, std::string> args;
      for (const auto &[k, v] : m) {
        const size_t close = k.find(']');
        if (k.rfind(".args[", 0) != 0 || close == std::string::npos) {
          rest[k] = v;
          continue;
        }
        size_t idx = 0;
        for (size_t i = 6; i < close; ++i)
          idx = idx * 10 + static_cast<size_t>(k[i] - '0');
        std::string &rec = args[idx];
        rec += (rec.empty() ? "" : " ") +
               (close + 2 <= k.size() ? k.substr(close + 2) : std::string()) + "=" + v;
      }
      return args;
    };
    auto compare_args = [&](const std::map<std::string, std::string> &a,
                            const std::map<std::string, std::string> &b,
                            std::map<std::string, std::string> &rest_a,
                            std::map<std::string, std::string> &rest_b) {
      const auto aa = split_args(a, rest_a), ab = split_args(b, rest_b);
      if (aa == ab)
        return size_t{0};
      size_t first = 0;
      while (aa.count(first) && ab.count(first) && aa.at(first) == ab.at(first))
        ++first;
      auto rec = [&](const std::map<size_t, std::string> &m) {
        return m.count(first) ? "{" + m.at(first) + "}" : std::string("<absent>");
      };
      r.notes.push_back({ChangeKind::Metadata,
                         "kernel metadata .args differs (old " + std::to_string(aa.size()) +
                             " args, new " + std::to_string(ab.size()) +
                             " args); first difference at .args[" + std::to_string(first) +
                             "]: old " + rec(aa) + ", new " + rec(ab),
                         true});
      return size_t{1};
    };
    auto compare = [&](const std::map<std::string, std::string> &a,
                       const std::map<std::string, std::string> &b, const char *what) {
      std::set<std::string> keys;
      for (const auto &[k, v] : a)
        keys.insert(k);
      for (const auto &[k, v] : b)
        keys.insert(k);
      size_t n = 0;
      for (const auto &k : keys) {
        auto ia = a.find(k), ib = b.find(k);
        const std::string va = ia == a.end() ? "<absent>" : ia->second;
        const std::string vb = ib == b.end() ? "<absent>" : ib->second;
        if (va == vb)
          continue;
        const bool count_field = k == ".vgpr_count" || k == ".sgpr_count" || k == ".agpr_count";
        if (count_field && counts_informational && std::string_view(what) == "kernel metadata") {
          r.notes.push_back({ChangeKind::Metadata,
                             std::string(what) + " field differs: " + k + " (old " + va + ", new " +
                                 vb +
                                 "); informational: descriptors agree and both kernels' VGPRs "
                                 "fit their allocation",
                             false});
          continue;
        }
        if (++n <= 32)
          r.notes.push_back(
              {ChangeKind::Metadata,
               std::string(what) + " field differs: " + k + " (old " + va + ", new " + vb + ")",
               true});
      }
      if (n > 32)
        r.notes.push_back({ChangeKind::Metadata,
                           std::string(what) + ": " + std::to_string(n - 32) +
                               " further differing field(s) not listed",
                           true});
      return n;
    };
    std::map<std::string, std::string> rest_old, rest_new;
    const size_t nk = compare_args(oa.kernel_metadata, na.kernel_metadata, rest_old, rest_new) +
                      compare(rest_old, rest_new, "kernel metadata");
    const size_t ng = compare(oa.global_metadata, na.global_metadata, "code-object metadata");
    if (nk == 0 && ng == 0 && oa.metadata_sha256 != na.metadata_sha256)
      r.notes.push_back({ChangeKind::Metadata,
                         "AMDHSA note differs only outside this kernel's metadata entry", false});
  } else if (oa.metadata_sha256 != na.metadata_sha256) {
    r.notes.push_back({ChangeKind::Metadata,
                       "code-object note (AMDHSA metadata) differs and was not parsed field by "
                       "field",
                       true});
  }
  // Relocations (handoff 4/5.1): matched instructions must carry the same relocations,
  // otherwise identical bytes can resolve to different values.
  {
    size_t listed = 0, total = 0;
    for (size_t i = 0; i < r.old_to_new.size(); ++i) {
      const int64_t j = r.old_to_new[i];
      if (j < 0)
        continue;
      const InstRef &a = old_model.insts()[i];
      const InstRef &b = new_model.insts()[static_cast<size_t>(j)];
      if (a.relocations == b.relocations)
        continue;
      ++total;
      if (++listed <= 16)
        r.notes.push_back({ChangeKind::Layout,
                           "relocation differs at matched instruction old " + hex64(a.pc) + " (" +
                               (a.relocations.empty() ? "none" : a.relocations) + ") / new " +
                               hex64(b.pc) + " (" +
                               (b.relocations.empty() ? "none" : b.relocations) + ")",
                           true});
    }
    if (total > 16)
      r.notes.push_back({ChangeKind::Layout,
                         std::to_string(total - 16) + " further relocation difference(s)", true});
  }
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
  // Inline-asm blocks are matched by their position in each function's sequence. That is
  // only meaningful when both sequences have the same length and operand shapes.
  const auto &oasm = old_model.artifact().inline_asm;
  const auto &nasm = new_model.artifact().inline_asm;
  if (oasm.size() != nasm.size()) {
    r.notes.push_back({ChangeKind::Unexplained,
                       "inline asm blocks: old has " + std::to_string(oasm.size()) + ", new has " +
                           std::to_string(nasm.size()) + "; blocks cannot be matched",
                       true});
  } else {
    for (size_t k = 0; k < oasm.size(); ++k)
      if (inline_asm_shape(oasm[k].text) != inline_asm_shape(nasm[k].text)) {
        r.notes.push_back({ChangeKind::Unexplained,
                           "inline asm block " + std::to_string(k) + " differs in shape: '" +
                               oasm[k].text + "' vs '" + nasm[k].text + "'",
                           true});
        break;
      }
  }
  if (!oasm.empty() || !nasm.empty()) {
    for (const auto &[m, side] : {std::pair{&old_model, "old"}, {&new_model, "new"}})
      for (const InstRef &ref : m->insts())
        if (ref.inline_asm >= 0 && ref.fx.unknown_effect) {
          r.notes.push_back({ChangeKind::Unexplained,
                             std::string(side) + " inline asm block " +
                                 std::to_string(ref.inline_asm) + " has unknown effects (" +
                                 ref.fx.reject_reason + ")",
                             true});
          break;
        }
  }
  return r;
}

} // namespace rocjitsu::isa_diff
