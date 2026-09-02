// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

namespace rocjitsu {

/// A branch-only body reached through zero or more ordinary SOPP relays.
struct ConSanBranchOnlyRelayEntry {
  std::vector<uint64_t> relay_offsets;
};

/// A preplaced indirect entry whose scalar jump tuple is preserved in one
/// liveness-proven vector reservoir. The optional continuation skips the probe
/// after restoring the tuple when EXEC is empty.
struct ConSanBranchOnlyBorrowedEntry {
  ConSanIndirectJumpSgprs jump;
  uint16_t backup_vgpr = 0;
  std::optional<uint64_t> empty_exec_continuation_offset;
};

/// A kernel-entry prologue that supersedes the body's original entry relays
/// and chains directly into its trampoline.
struct ConSanBranchOnlyPrologueEntry {
  uint64_t prologue_offset = 0;
};

/// Complete control-flow effect for a body whose continuation uses branch-only
/// routing. Its entry is exactly one ordinary relay path, borrowed indirect
/// entry, or composed kernel prologue; mixed partial schemas are unrepresentable.
struct ConSanBranchOnlyContinuation {
  std::variant<ConSanBranchOnlyRelayEntry, ConSanBranchOnlyBorrowedEntry,
               ConSanBranchOnlyPrologueEntry>
      entry;
  std::vector<uint64_t> return_relay_offsets;

  [[nodiscard]] const std::vector<uint64_t> &entry_relay_offsets() const {
    static const std::vector<uint64_t> empty;
    if (const auto *relays = std::get_if<ConSanBranchOnlyRelayEntry>(&entry))
      return relays->relay_offsets;
    return empty;
  }

  [[nodiscard]] const ConSanBranchOnlyBorrowedEntry *borrowed_entry() const {
    return std::get_if<ConSanBranchOnlyBorrowedEntry>(&entry);
  }

  [[nodiscard]] ConSanBranchOnlyBorrowedEntry *borrowed_entry() {
    return std::get_if<ConSanBranchOnlyBorrowedEntry>(&entry);
  }

  [[nodiscard]] const ConSanBranchOnlyPrologueEntry *prologue_entry() const {
    return std::get_if<ConSanBranchOnlyPrologueEntry>(&entry);
  }
};

} // namespace rocjitsu
