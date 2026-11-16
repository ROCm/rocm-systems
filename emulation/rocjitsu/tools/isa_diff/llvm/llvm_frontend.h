// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file llvm_frontend.h
/// @brief LLVM plugin: `.s` and `.ll` inputs through the official LLVM tools (handoff 4).
///
/// @details RocJITsu has no in-tree assembler, and the core tool deliberately does not
/// link LLVM. This plugin drives an external LLVM toolchain as an assembler adapter:
///   - `.s`:  llvm-mc -filetype=obj  ->  ld.lld -shared  ->  code object (binary mode);
///   - `.ll`: llc -filetype=asm (common args + side args)  ->  the `.s` path.
/// The executed object is always the machine code these tools emit, decoded by RocJITsu.
/// It is the re-assembled code, not necessarily the bytes an original binary emission
/// would contain; the provenance says so. Every command line, its exit status, the input
/// hash and the toolchain identity are recorded, so that a pair is reproducible and old and
/// new are known to share the input, target and features. Arguments are passed through
/// posix_spawn without a shell.
#pragma once

#include "artifact.h"
#include "json.h"

#include <optional>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff::llvm {

struct Toolchain {
  std::string bin_dir;
  std::string llc, llvm_mc, ld_lld, llvm_objcopy;
  std::string version;  ///< First line of `llc --version`.
  std::string revision; ///< `clang --version` source/revision line when available.

  /// @brief Locate the tools: @p hint, else $RJ_ISA_DIFF_LLVM_BIN, else /opt/rocm/llvm/bin.
  static std::optional<Toolchain> discover(const std::string &hint, std::string &error);
  [[nodiscard]] Json to_json() const;
};

struct ToolCommand {
  std::vector<std::string> argv;
  int exit_code = -1;
  std::string stderr_tail;
  uint64_t elapsed_ns = 0; ///< Wall time of the tool run (handoff 11 "compile" cost).
};

struct Provenance {
  std::string input_path;
  std::string input_sha256;
  std::string input_kind; ///< "ll" or "s".
  std::string triple = "amdgcn-amd-amdhsa";
  std::string mcpu = "gfx950";
  /// Wave size asked for (0: the target's default). On GFX10+ it becomes
  /// -mattr=+wavefrontsize32/64 for llc and llvm-mc; a raw .text fallback gets it too.
  unsigned wave_size = 0;
  /// Target features (-mattr) for llc and llvm-mc, e.g. "+real-true16" from a RUN line.
  std::string mattr;
  std::vector<std::string> common_args;
  std::vector<std::string> side_args;
  std::vector<ToolCommand> commands;
  std::string executed_object; ///< "linked_code_object" or "raw_text".
  std::string note;
  Json toolchain;
  /// Comment-only inline-asm blocks marked in the assembled copy of the source.
  size_t inline_asm_blocks = 0;

  [[nodiscard]] Json to_json() const;
};

/// @brief Assemble a `.s` file into a code object and load @p kernel (or the only kernel).
[[nodiscard]] bool assemble(const Toolchain &tc, const std::string &s_path,
                            const std::string &work_prefix, const std::string &kernel,
                            Artifact &out, Provenance &prov, std::string &error);

/// @brief Compile a `.ll` file with llc (@p common_args then @p side_args), then assemble.
[[nodiscard]] bool compile(const Toolchain &tc, const std::string &ll_path,
                           const std::vector<std::string> &common_args,
                           const std::vector<std::string> &side_args,
                           const std::string &work_prefix, const std::string &kernel, Artifact &out,
                           Provenance &prov, std::string &error);

/// @brief Mark comment-only inline-asm blocks (`;;#ASMSTART`, comment lines, `;;#ASMEND`)
/// of an LLVM assembly listing. Each block gets a label `__rj_isa_diff_asm.<n>` in front of
/// it, and a record "<label>\t<before_label 0|1>\t<text>" in the non-loaded section
/// `.rj_isa_diff_inline_asm` appended at the end. The machine code is unchanged.
/// @param marked Set to the number of blocks marked (0: the text is returned unchanged).
[[nodiscard]] std::string mark_inline_asm(const std::string &source, size_t &marked);

/// @brief Split a whitespace-separated argument string (no quoting).
[[nodiscard]] std::vector<std::string> split_args(const std::string &text);

/// @brief Reject side arguments that would change the target or features between sides.
[[nodiscard]] std::optional<std::string> check_side_args(const std::vector<std::string> &args);

/// @brief Run a program without a shell; stdout/stderr go to the given files.
[[nodiscard]] int run_tool(const std::vector<std::string> &argv, const std::string &stdout_path,
                           const std::string &stderr_path);

} // namespace rocjitsu::isa_diff::llvm
