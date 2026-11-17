// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file llvm_plugin_test.cpp
/// @brief LLVM plugin tests: run only when an LLVM toolchain is available at configure time.

#include "case_io.h"
#include "kernel_model.h"
#include "llvm_frontend.h"
#include "mutation.h"
#include "region_runner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#ifndef RJ_ISA_DIFF_TEST_LLVM_BIN
#error "RJ_ISA_DIFF_TEST_LLVM_BIN must name the LLVM bin directory"
#endif

namespace {

using namespace rocjitsu::isa_diff;

// Uniform (kernel-argument) i16 arithmetic lowers to SALU on gfx950. With the DAG combiner
// disabled, llc emits a different but equivalent SALU sequence between identical loads.
constexpr const char *kUniformI16 = R"(target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @k3(ptr addrspace(1) %out, i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %a16 = trunc i32 %a to i16
  %b16 = trunc i32 %b to i16
  %c16 = trunc i32 %c to i16
  %s0 = add i16 %a16, %b16
  %s1 = shl i16 %s0, 3
  %s2 = xor i16 %s1, %c16
  %s3 = lshr i16 %s2, 2
  %s4 = mul i16 %s3, 5
  %cmp = icmp ult i16 %s4, %b16
  %s5 = select i1 %cmp, i16 %s4, i16 %a16
  %e = zext i16 %s5 to i32
  %f = add i32 %e, %d
  %g = and i32 %f, 4095
  %h = sub i32 %g, %c
  %p = getelementptr i32, ptr addrspace(1) %out, i64 1
  store i32 %h, ptr addrspace(1) %out
  store i32 %e, ptr addrspace(1) %p
  ret void
}
)";

// With the machine scheduler disabled, llc reorders the kernel-argument loads (SMEM).
constexpr const char *kUniformI32 = R"(target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @k2(ptr addrspace(1) %out, i32 %a, i32 %b, i32 %c, i64 %x) {
entry:
  %t0 = add i32 %a, %b
  %t1 = shl i32 %t0, 3
  %t2 = xor i32 %t1, %c
  %t3 = lshr i32 %t2, 5
  %t4 = and i32 %t3, 65535
  %cmp = icmp ugt i32 %t4, %b
  %t7 = select i1 %cmp, i32 %t4, i32 %a
  %x1 = shl i64 %x, 7
  %xl = trunc i64 %x1 to i32
  %r0 = add i32 %t7, %xl
  store i32 %r0, ptr addrspace(1) %out
  ret void
}
)";

std::string temp_dir(const std::string &name) {
  auto dir = std::filesystem::temp_directory_path() / ("rj_isa_diff_llvm_test_" + name);
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir.string();
}

llvm::Toolchain toolchain() {
  std::string error;
  auto tc = llvm::Toolchain::discover(RJ_ISA_DIFF_TEST_LLVM_BIN, error);
  EXPECT_TRUE(tc.has_value()) << error;
  return tc.value_or(llvm::Toolchain{});
}

struct Pair {
  Artifact old_a, new_a;
  llvm::Provenance old_p, new_p;
};

Pair compile_pair(const std::string &dir, const char *ir, const std::vector<std::string> &common,
                  const std::vector<std::string> &old_args,
                  const std::vector<std::string> &new_args) {
  const auto tc = toolchain();
  const std::string ll = dir + "/k.ll";
  EXPECT_TRUE(write_text_file(ll, ir));
  Pair p;
  std::string error;
  EXPECT_TRUE(llvm::compile(tc, ll, common, old_args, dir + "/old", "", p.old_a, p.old_p, error))
      << error;
  EXPECT_TRUE(llvm::compile(tc, ll, common, new_args, dir + "/new", "", p.new_a, p.new_p, error))
      << error;
  return p;
}

TEST(IsaDiffLlvm, ToolchainIdentityAndArgumentChecks) {
  const auto tc = toolchain();
  EXPECT_NE(tc.version.find("LLVM"), std::string::npos) << tc.version;
  EXPECT_EQ(llvm::split_args("  -O3   -enable-misched=false "),
            (std::vector<std::string>{"-O3", "-enable-misched=false"}));
  EXPECT_TRUE(llvm::check_side_args({"-mcpu=gfx942"}).has_value());
  EXPECT_TRUE(llvm::check_side_args({"-mattr=+xnack"}).has_value());
  EXPECT_FALSE(llvm::check_side_args({"-O1", "-combiner-disabled"}).has_value());
  EXPECT_NE(llvm::run_tool({"/nonexistent/tool"}, "", ""), 0);
}

TEST(IsaDiffLlvm, OverlayTakesASidesOwnAssemblerAndKeepsTheLinker) {
  const auto base = toolchain();
  const std::string dir = temp_dir("overlay");
  std::string error;
  // A directory with neither llc nor llvm-mc is no overlay.
  auto empty = base;
  EXPECT_FALSE(empty.overlay(dir, error));
  EXPECT_NE(error.find("no llc or llvm-mc"), std::string::npos) << error;
  // An LLVM PR's build: llvm-mc (and llc), no lld.
  std::filesystem::create_symlink(base.llvm_mc, dir + "/llvm-mc");
  auto tc = base;
  ASSERT_TRUE(tc.overlay(dir, error)) << error;
  EXPECT_EQ(tc.llvm_mc, dir + "/llvm-mc");
  EXPECT_EQ(tc.ld_lld, base.ld_lld);
  EXPECT_EQ(tc.llc, base.llc);
  EXPECT_NE(tc.mc_version.find("version"), std::string::npos) << tc.mc_version;
  const Json j = tc.to_json();
  EXPECT_EQ(j["overlay"]["dir"].as_string(), dir);
  ASSERT_EQ(j["overlay"]["tools"].size(), 1u);
  EXPECT_EQ(j["overlay"]["tools"].items()[0].as_string(), "llvm-mc");
  // The overlaid assembler is the one that runs.
  const std::string s = dir + "/f.s";
  ASSERT_TRUE(write_text_file(s, "\t.text\n\t.globl\tf\n\t.p2align\t8\n\t.type\tf,@function\n"
                                 "f:\n\tv_add_u32_e32 v0, v0, v1\n\ts_setpc_b64 s[30:31]\n"
                                 ".Lfunc_end0:\n\t.size\tf, .Lfunc_end0-f\n"));
  Artifact a;
  llvm::Provenance p;
  ASSERT_TRUE(llvm::assemble(tc, s, dir + "/f", "f", a, p, error)) << error;
  ASSERT_FALSE(p.commands.empty());
  EXPECT_EQ(p.commands.front().argv.front(), dir + "/llvm-mc");
  EXPECT_EQ(p.to_json()["toolchain"]["overlay"]["dir"].as_string(), dir);
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffLlvm, AssemblesKernelSourceAndBareListings) {
  const auto tc = toolchain();
  const std::string dir = temp_dir("asm");
  // A kernel with a descriptor is linked into a code object.
  const std::string kernel_s = dir + "/kernel.s";
  ASSERT_TRUE(write_text_file(kernel_s, R"(.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl k
.p2align 8
.type k,@function
k:
  s_add_u32 s4, s2, s3
  s_cselect_b32 s5, s4, 0
  s_endpgm
.Lfunc_end0:
  .size k, .Lfunc_end0-k
.rodata
.p2align 6
.amdhsa_kernel k
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 16
  .amdhsa_accum_offset 4
.end_amdhsa_kernel
)"));
  Artifact a;
  llvm::Provenance p;
  std::string error;
  ASSERT_TRUE(llvm::assemble(tc, kernel_s, dir + "/kernel", "k", a, p, error)) << error;
  EXPECT_EQ(a.kind, "code_object");
  EXPECT_EQ(a.kernel_name, "k");
  EXPECT_EQ(a.kernel_end - a.kernel_begin, 12u); // three 4-byte instructions
  EXPECT_EQ(p.executed_object, "linked_code_object");
  ASSERT_EQ(p.commands.size(), 2u);
  EXPECT_EQ(p.commands[0].exit_code, 0);
  EXPECT_NE(p.commands[0].argv[0].find("llvm-mc"), std::string::npos);
  EXPECT_NE(p.commands[1].argv[0].find("ld.lld"), std::string::npos);
  EXPECT_EQ(p.input_kind, "s");
  EXPECT_FALSE(p.input_sha256.empty());

  // A bare listing has no descriptor: the raw .text is used, and the provenance says so.
  const std::string bare_s = dir + "/bare.s";
  ASSERT_TRUE(write_text_file(bare_s, "s_add_u32 s4, s2, s3\ns_endpgm\n"));
  Artifact b;
  llvm::Provenance q;
  ASSERT_TRUE(llvm::assemble(tc, bare_s, dir + "/bare", "", b, q, error)) << error;
  EXPECT_EQ(b.kind, "raw_words");
  EXPECT_EQ(q.executed_object, "raw_text");
  EXPECT_EQ(b.text.size(), 8u);

  // Invalid assembly is a reported failure, not a guess.
  const std::string bad_s = dir + "/bad.s";
  ASSERT_TRUE(write_text_file(bad_s, "s_not_an_instruction s1\n"));
  Artifact c;
  llvm::Provenance r;
  EXPECT_FALSE(llvm::assemble(tc, bad_s, dir + "/bad", "", c, r, error));
  EXPECT_NE(error.find("llvm-mc failed"), std::string::npos) << error;
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffLlvm, RealPairCombinerSwitchIsVerifiedEndToEnd) {
  const std::string dir = temp_dir("combiner");
  Pair p = compile_pair(dir, kUniformI16, {"-O3"}, {}, {"-combiner-disabled"});
  EXPECT_EQ(p.old_p.input_sha256, p.new_p.input_sha256); // same LLVM input
  EXPECT_EQ(p.old_p.commands.size(), 3u);                // llc, llvm-mc, ld.lld
  EXPECT_EQ(p.old_a.kernel_name, "k3");
  std::string error;
  auto data = analyze_artifacts(p.old_a, p.new_a, {}, error);
  ASSERT_TRUE(data) << error;
  ASSERT_FALSE(data->regions.empty());
  for (const auto &r : data->regions) {
    EXPECT_EQ(r.status, RegionStatus::Ready) << r.reason;
    EXPECT_EQ(r.contract.pending_effects, "NONE");
    EXPECT_EQ(r.contract.composition, "HOLDS");
  }
  TestOptions opt;
  opt.cases = 2000;
  std::vector<RegionResult> results;
  Json report = test_case(*data, opt, "", &results);
  EXPECT_EQ(report["overall"]["test_status"].as_string(), "NO_MISMATCH_FOUND");
  EXPECT_EQ(report["overall"]["contract_status"].as_string(), "ESTABLISHED");
  for (const auto &r : results)
    EXPECT_GT(r.valid_pairs, 2000u);
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffLlvm, RealPairReorderingScalarLoadsRunsTheLoads) {
  // Disabling the scheduler reorders the kernel-argument loads. Those loads now execute in
  // the region against a shared memory input, so the pair is tested rather than rejected.
  const std::string dir = temp_dir("misched");
  Pair p = compile_pair(dir, kUniformI32, {"-O3"}, {}, {"-enable-misched=false"});
  std::string error;
  auto data = analyze_artifacts(p.old_a, p.new_a, {}, error);
  ASSERT_TRUE(data) << error;
  ASSERT_FALSE(data->regions.empty());
  bool loads = false;
  for (const auto &r : data->regions) {
    EXPECT_EQ(r.status, RegionStatus::Ready) << r.reason;
    loads |= r.contract.memory_reads;
  }
  EXPECT_TRUE(loads);
  TestOptions opt;
  opt.cases = 300;
  opt.case_log_dir.clear();
  std::vector<RegionResult> results;
  Json report = test_case(*data, opt, "", &results);
  EXPECT_EQ(report["overall"]["test_status"].as_string(), "NO_MISMATCH_FOUND");
  EXPECT_EQ(report["overall"]["contract_status"].as_string(), "ESTABLISHED");
  for (const auto &r : results)
    EXPECT_GT(r.valid_pairs, 300u);
  std::filesystem::remove_all(dir);
}

// A kernel that reads a global: llc emits s_getpc + GOT-relative literals, which the
// linker resolves; --emit-relocs keeps the record of that relocation.
constexpr const char *kGlobalTemplate = R"(target triple = "amdgcn-amd-amdhsa"
@SYM = addrspace(1) global i32 7, align 4
define amdgpu_kernel void @kg(ptr addrspace(1) %out, i32 %x) {
  %v = load i32, ptr addrspace(1) @SYM, align 4
  %s = add i32 %v, %x
  %m = mul i32 %s, 3
  store i32 %m, ptr addrspace(1) %out, align 4
  ret void
}
)";

std::string with_symbol(const std::string &name) {
  std::string ir = kGlobalTemplate;
  for (size_t at = ir.find("SYM"); at != std::string::npos; at = ir.find("SYM", at))
    ir.replace(at, 3, name);
  return ir;
}

Pair compile_two(const std::string &dir, const std::string &old_ir, const std::string &new_ir) {
  const auto tc = toolchain();
  Pair p;
  std::string error;
  EXPECT_TRUE(write_text_file(dir + "/old.ll", old_ir));
  EXPECT_TRUE(write_text_file(dir + "/new.ll", new_ir));
  EXPECT_TRUE(
      llvm::compile(tc, dir + "/old.ll", {"-O3"}, {}, dir + "/old", "", p.old_a, p.old_p, error))
      << error;
  EXPECT_TRUE(
      llvm::compile(tc, dir + "/new.ll", {"-O3"}, {}, dir + "/new", "", p.new_a, p.new_p, error))
      << error;
  return p;
}

TEST(IsaDiffLlvm, RelocationsAreKeptComparedAndNeverExecuted) {
  const std::string dir = temp_dir("reloc");
  Pair p = compile_two(dir, with_symbol("g"), with_symbol("h"));
  // GOTPCREL32_LO/HI on the two s_add/s_addc literals after s_getpc_b64.
  ASSERT_EQ(p.old_a.text_relocations.size(), 2u);
  EXPECT_EQ(p.old_a.text_relocations[0].symbol, "g");
  EXPECT_EQ(p.new_a.text_relocations[0].symbol, "h");
  EXPECT_EQ(p.old_a.text_relocations[0].addend, 4);
  EXPECT_EQ(p.old_a.text, p.new_a.text); // identical executed bytes...
  std::string error;
  auto data = analyze_artifacts(p.old_a, p.new_a, {}, error);
  ASSERT_TRUE(data) << error;
  EXPECT_TRUE(data->changes.items().empty());
  // ...but the relocation target differs: a semantic layout change, never "no diff".
  bool reloc_note = false;
  for (const auto &u : data->unresolved)
    reloc_note |= u.kind == "layout" && u.what.find("relocation differs") != std::string::npos;
  EXPECT_TRUE(reloc_note);
  TestOptions opt;
  opt.cases = 10;
  Json report = test_case(*data, opt, "", nullptr);
  EXPECT_NE(report["overall"]["test_status"].as_string(), "NO_SEMANTIC_DIFF");
  EXPECT_NE(report["overall"]["coverage_scope"].as_string(), "COMPLETE_FOR_DECLARED_SCOPE");

  // A region holding a relocated instruction is unsupported, not executed.
  PlanOptions manual;
  manual.manual_region = "old:1-2,new:1-2"; // s_add_u32 s0, s0, <GOTPCREL32_LO>
  auto m = analyze_artifacts(p.old_a, p.new_a, manual, error);
  ASSERT_TRUE(m) << error;
  ASSERT_EQ(m->regions.size(), 1u);
  EXPECT_EQ(m->regions[0].status, RegionStatus::Unsupported);
  EXPECT_NE(m->regions[0].reason.find("relocated instruction"), std::string::npos)
      << m->regions[0].reason;
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffLlvm, KernelMetadataIsComparedFieldByField) {
  const std::string dir = temp_dir("metadata");
  const std::string old_ir = with_symbol("g");
  std::string new_ir = old_ir;
  const std::string sig = "i32 %x)";
  new_ir.replace(new_ir.find(sig), sig.size(), "i32 %x, i32 %unused)");
  Pair p = compile_two(dir, old_ir, new_ir);
  ASSERT_TRUE(p.old_a.metadata_parsed);
  EXPECT_EQ(p.old_a.kernel_metadata.at(".name"), "\"kg\"");
  EXPECT_TRUE(p.old_a.kernel_metadata.count(".sgpr_count"));
  EXPECT_TRUE(p.old_a.global_metadata.count("amdhsa.target"));
  std::string error;
  auto data = analyze_artifacts(p.old_a, p.new_a, {}, error);
  ASSERT_TRUE(data) << error;
  // One inserted argument is one grouped change, not a cascade of shifted fields. The
  // hidden arguments are realigned, so .kernarg_segment_size itself does not change.
  size_t metadata_items = 0;
  bool args = false;
  for (const auto &u : data->unresolved) {
    metadata_items += u.kind == "metadata";
    args |= u.what.find("kernel metadata .args differs (old 21 args, new 22 args); first "
                        "difference at .args[2]") != std::string::npos;
  }
  EXPECT_TRUE(args);
  EXPECT_EQ(metadata_items, 1u);
  std::filesystem::remove_all(dir);
}

// Disabling the machine scheduler reorders the kernarg loads, and register allocation then
// puts the loaded i64 in s[8:9] instead of s[4:5]: the SALU tail is the same computation
// with renamed inputs. Bound to the ROCm LLVM codegen recorded in implementation_map.md.
constexpr const char *kRenamedTail = R"(target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @k(ptr addrspace(1) %out, i32 %a, i32 %b, i32 %c, i64 %x) {
entry:
  %t0 = add i32 %a, %b
  %t1 = shl i32 %t0, 3
  %t2 = xor i32 %t1, %c
  %t3 = lshr i32 %t2, 5
  %t4 = mul i32 %t3, 7
  %t5 = and i32 %t4, 65535
  %cmp = icmp ugt i32 %t5, %b
  %t6 = select i1 %cmp, i32 %t5, i32 %a
  %x1 = shl i64 %x, 7
  %x2 = add i64 %x1, 12345
  %xl = trunc i64 %x2 to i32
  %xh64 = lshr i64 %x2, 32
  %xh = trunc i64 %xh64 to i32
  %r0 = add i32 %t6, %xl
  %r1 = sub i32 %r0, %xh
  %r2 = or i32 %r1, %c
  %r3 = mul i32 %r2, %a
  %q = udiv i32 %r3, 13
  %m = urem i32 %r3, 13
  %r4 = add i32 %q, %m
  %r5 = call i32 @llvm.umin.i32(i32 %r4, i32 %t5)
  %r6 = call i32 @llvm.smax.i32(i32 %r5, i32 %b)
  %p = getelementptr i32, ptr addrspace(1) %out, i64 1
  store i32 %r6, ptr addrspace(1) %out
  store i32 %r3, ptr addrspace(1) %p
  ret void
}
declare i32 @llvm.umin.i32(i32, i32)
declare i32 @llvm.smax.i32(i32, i32)
)";

/// Index into `side` of the instruction at the PC that starts a hunk listing line.
size_t index_of_listed_pc(const RegionSide &side, const Json &listing) {
  const uint64_t pc = std::stoull(listing.items().at(0).as_string(), nullptr, 16);
  for (size_t i = 0; i < side.insts.size(); ++i)
    if (side.insts[i].pc == pc)
      return side.begin + i;
  return SIZE_MAX;
}

TEST(IsaDiffLlvm, RealRegisterRenamingIsInternalToTheMergedRegion) {
  const std::string dir = temp_dir("renamed");
  Pair p = compile_pair(dir, kRenamedTail, {"-O3"}, {}, {"-enable-misched=false"});
  std::string error;
  auto automatic = analyze_artifacts(p.old_a, p.new_a, {}, error);
  ASSERT_TRUE(automatic) << error;
  // The reordered argument loads and the renamed tail merge into one region across the
  // scalar loads, so the renaming is internal to it; rsrc1 and .sgpr_count differ only in
  // register allocation, which both kernels fit.
  ASSERT_EQ(automatic->regions.size(), 1u);
  const RegionPair &whole = automatic->regions[0];
  ASSERT_EQ(whole.status, RegionStatus::Ready) << whole.reason;
  EXPECT_TRUE(whole.contract.memory_reads);
  EXPECT_EQ(whole.contract.composition, "HOLDS");
  EXPECT_EQ(whole.contract.status, ContractStatus::Established);
  EXPECT_TRUE(automatic->unresolved.empty());
  TestOptions opt;
  opt.cases = 500;
  opt.case_log_dir.clear();
  Json report = test_case(*automatic, opt, "", nullptr);
  EXPECT_EQ(report["overall"]["test_status"].as_string(), "NO_MISMATCH_FOUND");
  EXPECT_EQ(report["overall"]["contract_status"].as_string(), "ESTABLISHED");

  // The renamed tail alone (from the first hunk that rewrites instructions to the region
  // end): identity mapping is wrong there, an entry map reconciles it.
  const Json *rewrite = nullptr;
  for (const auto &h : automatic->changes.items())
    if (!rewrite && !h["old"].items().empty() && !h["new"].items().empty())
      rewrite = &h;
  ASSERT_NE(rewrite, nullptr);
  const size_t old_tail = index_of_listed_pc(whole.old_side, (*rewrite)["old"]);
  const size_t new_tail = index_of_listed_pc(whole.new_side, (*rewrite)["new"]);
  ASSERT_NE(old_tail, SIZE_MAX);
  ASSERT_NE(new_tail, SIZE_MAX);
  const std::string range =
      "old:" + std::to_string(old_tail) + "-" + std::to_string(whole.old_side.end) +
      ",new:" + std::to_string(new_tail) + "-" + std::to_string(whole.new_side.end);
  PlanOptions manual;
  manual.manual_region = range;
  auto identity = analyze_artifacts(p.old_a, p.new_a, manual, error);
  ASSERT_TRUE(identity) << error;
  std::vector<RegionResult> results;
  (void)test_case(*identity, opt, "", &results);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].test_status, TestStatus::LocalMismatch);

  manual.entry_map = "s4=s8,s5=s9";
  auto mapped = analyze_artifacts(p.old_a, p.new_a, manual, error);
  ASSERT_TRUE(mapped) << error;
  ASSERT_EQ(mapped->regions.size(), 1u);
  EXPECT_EQ(mapped->regions[0].contract.mapping, "user_register_map");
  Json r = test_case(*mapped, opt, "", &results);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].test_status, TestStatus::NoMismatchFound) << results[0].reason;
  // A hand-written relation is at best ASSUMED, never ESTABLISHED.
  EXPECT_NE(results[0].contract_status, ContractStatus::Established);
  EXPECT_GT(results[0].valid_pairs, 500u);
  // The other hunks (the reordered loads) are outside the region: never a whole pass.
  EXPECT_NE(r["overall"]["coverage_scope"].as_string(), "COMPLETE_FOR_DECLARED_SCOPE");

  // Mapping the exit too is wrong for this tail (the result is in s4 on both sides): the
  // translated exit liveness no longer agrees, so the region is not even tested.
  manual.entry_map.clear();
  manual.register_map = "s4=s8,s5=s9";
  auto both = analyze_artifacts(p.old_a, p.new_a, manual, error);
  ASSERT_TRUE(both) << error;
  ASSERT_EQ(both->regions.size(), 1u);
  EXPECT_EQ(both->regions[0].status, RegionStatus::Inconclusive);
  EXPECT_NE(both->regions[0].reason.find("does not reconcile"), std::string::npos)
      << both->regions[0].reason;
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffLlvm, SideArgumentsThatChangeTheTargetAreRejected) {
  const auto tc = toolchain();
  const std::string dir = temp_dir("reject");
  ASSERT_TRUE(write_text_file(dir + "/k.ll", kUniformI16));
  Artifact a;
  llvm::Provenance p;
  std::string error;
  EXPECT_FALSE(
      llvm::compile(tc, dir + "/k.ll", {"-O3"}, {"-mcpu=gfx942"}, dir + "/x", "", a, p, error));
  EXPECT_NE(error.find("-mcpu"), std::string::npos) << error;
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// Callable functions and comment-only inline asm (LLVM test listings).

TEST(IsaDiffLlvm, CommentOnlyInlineAsmIsMarkedWithItsPosition) {
  const std::string src = "f:\n"
                          "\t;;#ASMSTART\n\t; def v[0:1]\n\t;;#ASMEND\n"
                          "\tv_mov_b32_e32 v4, 0\n"
                          "\t;;#ASMSTART\n\tv_mov_b32 v1, 0\n\t;;#ASMEND\n" // real code: kept
                          "\t;;#ASMSTART\n\t; use v2\n\t;;#ASMEND\n"
                          ".LBB0_2:                                ; %Flow\n"
                          "\ts_setpc_b64 s[30:31]\n";
  size_t marked = 0;
  const std::string out = llvm::mark_inline_asm(src, marked);
  EXPECT_EQ(marked, 2u);
  EXPECT_NE(out.find("__rj_isa_diff_asm.0:\n\t;;#ASMSTART\n\t; def v[0:1]"), std::string::npos)
      << out;
  EXPECT_NE(out.find(".section\t.rj_isa_diff_inline_asm"), std::string::npos);
  EXPECT_NE(out.find("\"__rj_isa_diff_asm.0\\t0\\t; def v[0:1]\""), std::string::npos) << out;
  // The use is followed by a label: it ends the preceding block.
  EXPECT_NE(out.find("\"__rj_isa_diff_asm.1\\t1\\t; use v2\""), std::string::npos) << out;
  EXPECT_NE(out.find("v_mov_b32 v1, 0"), std::string::npos);
  size_t none = 0;
  EXPECT_EQ(llvm::mark_inline_asm("s_endpgm\n", none), "s_endpgm\n");
  EXPECT_EQ(none, 0u);
}

/// A callable function as llc prints it, with inline-asm placeholders: two defs feed a
/// packed move whose result is stored through the inreg pointer.
constexpr const char *kShuffleFn = R"(	.text
	.globl	f
	.p2align	8
	.type	f,@function
f:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	;;#ASMSTART
	; def DEF0
	;;#ASMEND
	v_mov_b32_e32 v4, 0
	;;#ASMSTART
	; def DEF1
	;;#ASMEND
	s_nop 0
	v_pk_mov_b32 v[0:1], PKMOV op_sel:[1,0]
	global_store_dwordx2 v4, v[0:1], s[0:1]
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
)";

/// A function returning a value and passing an intermediate to an inline-asm use.
constexpr const char *kUseFn = R"(	.text
	.globl	g
	.p2align	8
	.type	g,@function
g:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	;;#ASMSTART
	; def TMP_REG
	;;#ASMEND
	s_nop 0
	v_add_u32_e32 TMP_REG, v0, TMP_REG
	v_mul_lo_u32 v0, TMP_REG, v1
	;;#ASMSTART
	; use USE_REG
	;;#ASMEND
	EXTRA_INST
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	g, .Lfunc_end0-g
)";

std::string fill(std::string text, const std::vector<std::pair<std::string, std::string>> &subst) {
  for (const auto &[from, to] : subst)
    for (size_t pos; (pos = text.find(from)) != std::string::npos;)
      text.replace(pos, from.size(), to);
  return text;
}

/// Assemble two listings of one function and run the whole pipeline.
/// The ABI and comparison options of one end-to-end pair.
struct PairOptions {
  std::string mcpu = "gfx950";
  unsigned wave_size = 0; ///< 0: the target's default.
  std::string mattr;
  unsigned return_vgprs = 0;
  int entry_vgprs = -2; ///< -2: the loader's default.
  int entry_sgprs = -2;
  std::vector<uint32_t> return_masks;
  bool strict_nan = false;
  uint32_t poison_filter = 0;
};

std::string verdict_of(const std::string &dir, const std::string &fn, const std::string &old_s,
                       const std::string &new_s, const PairOptions &o);

std::string verdict_of(const std::string &dir, const std::string &fn, const std::string &old_s,
                       const std::string &new_s, unsigned return_vgprs) {
  PairOptions o;
  o.return_vgprs = return_vgprs;
  return verdict_of(dir, fn, old_s, new_s, o);
}

Json report_of(const std::string &dir, const std::string &fn, const std::string &old_s,
               const std::string &new_s, const PairOptions &o);

std::string verdict_of(const std::string &dir, const std::string &fn, const std::string &old_s,
                       const std::string &new_s, const PairOptions &o) {
  const Json report = report_of(dir, fn, old_s, new_s, o);
  if (report.is_null())
    return "error";
  return report["overall"]["test_status"].as_string() + "/" +
         report["overall"]["contract_status"].as_string();
}

std::optional<CaseData> case_of(const std::string &dir, const std::string &fn,
                                const std::string &old_s, const std::string &new_s,
                                const PairOptions &o);

Json report_of(const std::string &dir, const std::string &fn, const std::string &old_s,
               const std::string &new_s, const PairOptions &o) {
  auto data = case_of(dir, fn, old_s, new_s, o);
  if (!data)
    return Json();
  TestOptions opt;
  opt.cases = 1000;
  opt.strict_nan = o.strict_nan;
  opt.poison_filter = o.poison_filter;
  return test_case(*data, opt, "", nullptr);
}

/// Assemble two listings of one function and analyze them.
std::optional<CaseData> case_of(const std::string &dir, const std::string &fn,
                                const std::string &old_s, const std::string &new_s,
                                const PairOptions &o) {
  const auto tc = toolchain();
  Artifact a[2];
  for (int side = 0; side < 2; ++side) {
    const std::string path = dir + (side ? "/new.s" : "/old.s");
    EXPECT_TRUE(write_text_file(path, side ? new_s : old_s));
    llvm::Provenance p;
    p.mcpu = o.mcpu;
    p.wave_size = o.wave_size;
    p.mattr = o.mattr;
    std::string error;
    EXPECT_TRUE(llvm::assemble(tc, path, dir + (side ? "/new" : "/old"), fn, a[side], p, error))
        << error;
    EXPECT_TRUE(a[side].callable);
    EXPECT_EQ(a[side].target.name, o.mcpu);
    if (o.wave_size)
      if (auto w = with_wave_size(a[side].target, o.wave_size))
        a[side].target = *w;
    a[side].function_mode = default_function_mode(a[side].target);
    a[side].return_vgprs = o.return_vgprs;
    a[side].return_masks = o.return_masks;
    if (o.entry_vgprs != -2)
      a[side].entry_vgprs = o.entry_vgprs;
    if (o.entry_sgprs != -2)
      a[side].entry_sgprs = o.entry_sgprs;
  }
  std::string error;
  auto data = analyze_artifacts(a[0], a[1], {}, error);
  EXPECT_TRUE(data) << error;
  return data;
}

/// Sum of a region count over the regions and the whole function's result.
uint64_t count_of(const Json &report, const std::string &key) {
  uint64_t n = 0;
  for (const Json &g : report["regions"].items())
    n += g["counts"][key].as_uint();
  if (report["whole_function"]["result"].is_object())
    n += report["whole_function"]["result"]["counts"][key].as_uint();
  return n;
}

TEST(IsaDiffLlvm, CallableFunctionWithInlineAsmIsTestedEndToEnd) {
  const std::string dir = temp_dir("callable");
  const std::string base =
      fill(kShuffleFn, {{"DEF0", "v[0:1]"}, {"DEF1", "v[2:3]"}, {"PKMOV", "v[0:1], v[2:3]"}});
  // The function loads as a callable function; both blocks are found in order, and the
  // return makes the CFG complete.
  {
    const auto tc = toolchain();
    ASSERT_TRUE(write_text_file(dir + "/f.s", base));
    Artifact a;
    llvm::Provenance p;
    std::string error;
    ASSERT_TRUE(llvm::assemble(tc, dir + "/f.s", dir + "/f", "f", a, p, error)) << error;
    EXPECT_TRUE(a.callable);
    EXPECT_TRUE(a.descriptor.empty());
    EXPECT_EQ(p.inline_asm_blocks, 2u);
    ASSERT_EQ(a.inline_asm.size(), 2u);
    EXPECT_EQ(a.inline_asm[0].text, "; def v[0:1]");
    EXPECT_EQ(a.inline_asm[1].text, "; def v[2:3]");
    EXPECT_LT(a.inline_asm[0].offset, a.inline_asm[1].offset);
    auto model = KernelModel::build(a, error);
    ASSERT_TRUE(model) << error;
    EXPECT_TRUE(model->cfg_complete());
    EXPECT_TRUE(model->blocks().back().returns);
    size_t pseudo = 0;
    for (const auto &ref : model->insts())
      pseudo += ref.inline_asm >= 0;
    EXPECT_EQ(pseudo, 2u);
  }
  // The defs trade registers and the packed move follows: equivalent (void: no return
  // registers observed).
  const std::string swapped =
      fill(kShuffleFn, {{"DEF0", "v[2:3]"}, {"DEF1", "v[0:1]"}, {"PKMOV", "v[2:3], v[0:1]"}});
  EXPECT_EQ(verdict_of(dir, "f", base, swapped, 0), "NO_MISMATCH_FOUND/ESTABLISHED");
  // The defs trade registers but the move does not follow: wrong.
  const std::string wrong =
      fill(kShuffleFn, {{"DEF0", "v[2:3]"}, {"DEF1", "v[0:1]"}, {"PKMOV", "v[0:1], v[2:3]"}});
  EXPECT_EQ(verdict_of(dir, "f", base, wrong, 0), "LOCAL_MISMATCH/ESTABLISHED");

  const std::string g =
      fill(kUseFn, {{"TMP_REG", "v2"}, {"USE_REG", "v2"}, {"EXTRA_INST", "s_nop 0"}});
  // Renaming the temporary everywhere, its asm use included: equivalent.
  EXPECT_EQ(verdict_of(
                dir, "g", g,
                fill(kUseFn, {{"TMP_REG", "v3"}, {"USE_REG", "v3"}, {"EXTRA_INST", "s_nop 0"}}), 1),
            "NO_MISMATCH_FOUND/ESTABLISHED");
  // The asm use reads another register: what it observes differs.
  EXPECT_EQ(verdict_of(
                dir, "g", g,
                fill(kUseFn, {{"TMP_REG", "v2"}, {"USE_REG", "v0"}, {"EXTRA_INST", "s_nop 0"}}), 1),
            "LOCAL_MISMATCH/ESTABLISHED");
  // A caller-saved SGPR may be clobbered; a callee-saved one may not.
  EXPECT_EQ(
      verdict_of(
          dir, "g", g,
          fill(kUseFn, {{"TMP_REG", "v2"}, {"USE_REG", "v2"}, {"EXTRA_INST", "s_mov_b32 s20, 0"}}),
          1),
      "NO_MISMATCH_FOUND/ESTABLISHED");
  EXPECT_EQ(
      verdict_of(
          dir, "g", g,
          fill(kUseFn, {{"TMP_REG", "v2"}, {"USE_REG", "v2"}, {"EXTRA_INST", "s_mov_b32 s34, 0"}}),
          1),
      "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

/// A void function f: @p body, then the return.
std::string fn_listing(const std::vector<std::string> &body) {
  std::string t = "\t.text\n\t.globl\tf\n\t.p2align\t8\n\t.type\tf,@function\nf:\n";
  for (const auto &l : body)
    t += "\t" + l + "\n";
  return t + "\ts_setpc_b64 s[30:31]\n.Lfunc_end0:\n\t.size\tf, .Lfunc_end0-f\n";
}

/// Store v0 through the inreg pointer s[0:1] after @p ops.
std::string store_fn(const std::vector<std::string> &ops) {
  std::vector<std::string> body = {"v_mov_b32_e32 v4, 0"};
  body.insert(body.end(), ops.begin(), ops.end());
  body.push_back("global_store_dword v4, v0, s[0:1]");
  body.push_back("s_waitcnt vmcnt(0)");
  return fn_listing(body);
}

TEST(IsaDiffLlvm, NanResultsOfTheSameFormatCompareEqual) {
  const std::string dir = temp_dir("nan");
  PairOptions o;
  o.entry_vgprs = 2;
  // fadd is commutative; with two NaN inputs the hardware returns the first operand's NaN.
  const std::string ab = store_fn({"v_add_f32_e32 v0, v0, v1"});
  const std::string ba = store_fn({"v_add_f32_e32 v0, v1, v0"});
  EXPECT_EQ(verdict_of(dir, "f", ab, ba, o), "NO_MISMATCH_FOUND/ESTABLISHED");
  PairOptions strict = o;
  strict.strict_nan = true;
  EXPECT_EQ(verdict_of(dir, "f", ab, ba, strict), "LOCAL_MISMATCH/ESTABLISHED");
  // Packed f16 halves are compared half by half.
  EXPECT_EQ(verdict_of(dir, "f", store_fn({"v_pk_add_f16 v0, v0, v1"}),
                       store_fn({"v_pk_add_f16 v0, v1, v0"}), o),
            "NO_MISMATCH_FOUND/ESTABLISHED");
  // fneg must keep the NaN's bits: a bitwise sign flip (old) and a multiplication by -1,
  // which quiets a signalling NaN (new), differ. Only old's value is not a floating-point
  // result, so no equivalence applies.
  EXPECT_EQ(verdict_of(dir, "f", store_fn({"v_xor_b32_e32 v0, 0x80000000, v0"}),
                       store_fn({"v_mul_f32_e32 v0, -1.0, v0"}), o),
            "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffLlvm, RescheduledNanProducersDrawTheSameNans) {
  const std::string dir = temp_dir("nan_order");
  PairOptions o;
  o.entry_vgprs = 7;
  // Two independent multiplications whose NaN results meet in integer code (v2 - v3);
  // new runs them in the other order. The same operations on the same inputs draw the
  // same alternative NaNs, so the difference of payloads is the same on both sides.
  const std::string a = "v_mul_f32_e32 v2, v0, v1", b = "v_mul_f32_e32 v3, v6, v5";
  auto f = [](const std::string &x, const std::string &y) {
    return store_fn({x, y, "v_sub_u32_e32 v0, v2, v3"});
  };
  EXPECT_EQ(verdict_of(dir, "f", f(a, b), f(b, a), o), "NO_MISMATCH_FOUND/ESTABLISHED");
  // Directly: both sides of the whole function under the random NaN choice, with NaN
  // inputs in v0 and v6 (other payloads per lane), give each product the same NaN.
  auto data = case_of(dir, "f", f(a, b), f(b, a), o);
  ASSERT_TRUE(data && data->whole_function);
  const RegionPair &r = *data->whole_function;
  std::string error;
  const Target target = r.contract.gpu().value_or(Target{});
  auto run_side = [&](const std::vector<RegionInst> &insts) {
    auto runner = RegionRunner::create(insts, "nan_order", error, r.contract.vgpr_count, target);
    EXPECT_TRUE(runner) << error;
    MachineState s;
    s.frame = runner->baseline_frame();
    s.set(kLocExecLo, 0xffffffffu);
    s.set(kLocExecHi, 0xffffffffu);
    s.set(1, 0x10u); // s[0:1]: the store's base, far from the code
    s.resize_vgprs(r.contract.vgpr_count);
    for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
      s.vset(0, lane, 0x7f800001u + lane); // signalling NaNs
      s.vset(1, lane, 0x3f800000u);        // 1.0
      s.vset(6, lane, 0xffc00000u + lane); // quiet NaNs, sign set
      s.vset(5, lane, 0x40000000u);        // 2.0
    }
    runner->set_nan_choice(3, 0x1234);
    runner->load(s);
    EXPECT_EQ(runner->run(insts.size()).status, RunStatus::Ok);
    return runner->capture();
  };
  const MachineState so = run_side(r.old_side.insts), sn = run_side(r.new_side.insts);
  for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
    EXPECT_EQ(so.vget(2, lane), sn.vget(2, lane)) << "lane " << lane;
    EXPECT_EQ(so.vget(3, lane), sn.vget(3, lane)) << "lane " << lane;
  }
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffLlvm, UndefinedValuesAreDontCare) {
  const std::string dir = temp_dir("undef");
  // The third stored element is undefined in the IR: old stores whatever v2 holds, new
  // stores 0. v0 and v1 are the arguments.
  const std::string old_s = fn_listing(
      {"v_mov_b32_e32 v4, 0", "global_store_dwordx3 v4, v[0:2], s[0:1]", "s_waitcnt vmcnt(0)"});
  const std::string new_s =
      fn_listing({"v_mov_b32_e32 v4, 0", "v_mov_b32_e32 v2, 0",
                  "global_store_dwordx3 v4, v[0:2], s[0:1]", "s_waitcnt vmcnt(0)"});
  PairOptions o;
  o.entry_vgprs = 2;
  EXPECT_EQ(verdict_of(dir, "f", old_s, new_s, o), "NO_MISMATCH_FOUND/ESTABLISHED");
  // Without the argument count v2 counts as defined, and the difference is a mismatch.
  EXPECT_EQ(verdict_of(dir, "f", old_s, new_s, PairOptions{}), "LOCAL_MISMATCH/ESTABLISHED");
  // An undefined value read from another register (an IMPLICIT_DEF allocated elsewhere).
  PairOptions none;
  none.entry_vgprs = 0;
  EXPECT_EQ(verdict_of(dir, "f", store_fn({"v_mov_b32_e32 v0, v5"}),
                       store_fn({"v_mov_b32_e32 v0, v7"}), none),
            "NO_MISMATCH_FOUND/ESTABLISHED");
  // New stores garbage where old stores the argument: a real bug.
  PairOptions one;
  one.entry_vgprs = 1;
  EXPECT_EQ(verdict_of(dir, "f", store_fn({}), store_fn({"v_mov_b32_e32 v0, v7"}), one),
            "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffLlvm, NarrowReturnValuesCompareTheirBitsOnly) {
  const std::string dir = temp_dir("retmask");
  // A half returned in v0: the high half is undefined.
  const std::string old_s = fn_listing({"v_cvt_f16_f32_e32 v0, v0"});
  const std::string new_s =
      fn_listing({"v_cvt_f16_f32_e32 v0, v0", "v_or_b32_e32 v0, 0x12340000, v0"});
  PairOptions o;
  o.return_vgprs = 1;
  o.return_masks = {0xffff};
  EXPECT_EQ(verdict_of(dir, "f", old_s, new_s, o), "NO_MISMATCH_FOUND/ESTABLISHED");
  o.return_masks.clear();
  EXPECT_EQ(verdict_of(dir, "f", old_s, new_s, o), "LOCAL_MISMATCH/ESTABLISHED");
  // A wrong low half is still a mismatch under the mask.
  o.return_masks = {0xffff};
  EXPECT_EQ(verdict_of(dir, "f", old_s,
                       fn_listing({"v_cvt_f16_f32_e32 v0, v0", "v_xor_b32_e32 v0, 1, v0"}), o),
            "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffLlvm, PoisonFilterExcludesInputsThatArePoisonInTheIR) {
  const std::string dir = temp_dir("poison");
  PairOptions o;
  o.entry_vgprs = 2;
  // shl by >= 32 is poison: a 64-bit shift (new) and a 32-bit one (old) agree below 32.
  const std::string shl32 = store_fn({"v_lshlrev_b32_e32 v0, v1, v0"});
  const std::string shl64 = store_fn({"v_mov_b32_e32 v3, 0", "v_mov_b32_e32 v2, v0",
                                      "v_lshlrev_b64 v[2:3], v1, v[2:3]", "v_mov_b32_e32 v0, v2"});
  EXPECT_EQ(verdict_of(dir, "f", shl32, shl64, o), "LOCAL_MISMATCH/ESTABLISHED");
  PairOptions shift = o;
  shift.poison_filter = kPoisonShift;
  EXPECT_EQ(verdict_of(dir, "f", shl32, shl64, shift), "NO_MISMATCH_FOUND/ESTABLISHED");
  // add nsw: a saturating add (new) agrees with the wrapping one (old) without overflow.
  const std::string wrap = store_fn({"v_add_u32_e32 v0, v0, v1"});
  const std::string sat = store_fn({"v_add_i32 v0, v0, v1 clamp"});
  EXPECT_EQ(verdict_of(dir, "f", wrap, sat, o), "LOCAL_MISMATCH/ESTABLISHED");
  PairOptions nsw = o;
  nsw.poison_filter = kPoisonNsw;
  EXPECT_EQ(verdict_of(dir, "f", wrap, sat, nsw), "NO_MISMATCH_FOUND/ESTABLISHED");
  // The filter does not excuse a difference on ordinary inputs.
  EXPECT_EQ(verdict_of(dir, "f", wrap, store_fn({"v_sub_u32_e32 v0, v0, v1"}), nsw),
            "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffLlvm, NanBitsThroughIntegerOperationsAreDontCare) {
  const std::string dir = temp_dir("nanint");
  PairOptions o;
  o.entry_vgprs = 2;
  // fadd commutes; the NaN's low half then goes through an integer AND before the store.
  // Old re-runs with alternative NaN choices show those bits depend on the NaN.
  const std::string ab = store_fn({"v_add_f32_e32 v0, v0, v1", "v_and_b32_e32 v0, 0xffff, v0"});
  const std::string ba = store_fn({"v_add_f32_e32 v0, v1, v0", "v_and_b32_e32 v0, 0xffff, v0"});
  EXPECT_EQ(verdict_of(dir, "f", ab, ba, o), "NO_MISMATCH_FOUND/ESTABLISHED");
  PairOptions strict = o;
  strict.strict_nan = true;
  EXPECT_EQ(verdict_of(dir, "f", ab, ba, strict), "LOCAL_MISMATCH/ESTABLISHED");
  // A wrong mask is wrong for ordinary values too: still a mismatch.
  EXPECT_EQ(verdict_of(dir, "f", ab,
                       store_fn({"v_add_f32_e32 v0, v1, v0", "v_and_b32_e32 v0, 0xfffe, v0"}), o),
            "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffLlvm, UndefinedScalarsAreDontCare) {
  const std::string dir = temp_dir("sgprundef");
  PairOptions o;
  o.entry_vgprs = 0;
  // s40 and s41 are neither arguments (s0-s29 by default) nor callee-saved: undefined.
  const std::string s40 = store_fn({"v_mov_b32_e32 v0, s40"});
  const std::string s41 = store_fn({"v_mov_b32_e32 v0, s41"});
  EXPECT_EQ(verdict_of(dir, "f", s40, s41, o), "NO_MISMATCH_FOUND/ESTABLISHED");
  // With every SGPR defined, the reallocation is a difference.
  PairOptions all = o;
  all.entry_sgprs = 102;
  EXPECT_EQ(verdict_of(dir, "f", s40, s41, all), "LOCAL_MISMATCH/ESTABLISHED");
  // New reads garbage where old reads an argument: a real bug.
  PairOptions four = o;
  four.entry_sgprs = 4;
  EXPECT_EQ(verdict_of(dir, "f", store_fn({"v_mov_b32_e32 v0, s2"}), s41, four),
            "LOCAL_MISMATCH/ESTABLISHED");
  // A callee-saved SGPR holds the caller's value: defined.
  EXPECT_EQ(verdict_of(dir, "f", store_fn({"v_mov_b32_e32 v0, s34"}),
                       store_fn({"v_mov_b32_e32 v0, s35"}), o),
            "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

/// A void function f for @p cpu: @p body, then the return of that generation.
std::string target_fn(const std::string &cpu, const std::vector<std::string> &body) {
  std::string t = "\t.text\n\t.globl\tf\n\t.p2align\t8\n\t.type\tf,@function\nf:\n";
  for (const auto &l : body)
    t += "\t" + l + "\n";
  const std::string ret = cpu.rfind("gfx125", 0) == 0 ? "s_set_pc_i64" : "s_setpc_b64";
  return t + "\t" + ret + " s[30:31]\n.Lfunc_end0:\n\t.size\tf, .Lfunc_end0-f\n";
}

/// Store v0 through s[0:1] after @p ops, in @p cpu's syntax.
std::string target_store_fn(const std::string &cpu, const std::vector<std::string> &ops) {
  const bool gfx9 = cpu.rfind("gfx9", 0) == 0;
  const bool gfx10 = cpu.rfind("gfx10", 0) == 0;
  const bool gfx12 = cpu.rfind("gfx12", 0) == 0;
  std::vector<std::string> body = {"v_mov_b32_e32 v4, 0"};
  body.insert(body.end(), ops.begin(), ops.end());
  // GFX11 renamed the dword forms to b32; GFX10 counts stores with vscnt, GFX12 storecnt.
  body.push_back(gfx9 || gfx10 ? "global_store_dword v4, v0, s[0:1]"
                               : "global_store_b32 v4, v0, s[0:1]");
  body.push_back(gfx12  ? "s_wait_storecnt 0x0"
                 : gfx9 ? "s_waitcnt vmcnt(0)"
                        : "s_waitcnt_vscnt null, 0x0");
  return target_fn(cpu, body);
}

TEST(IsaDiffTargets, EveryTargetPassesEquivalentCodeAndCatchesABug) {
  const std::string dir = temp_dir("targets");
  struct Spec {
    const char *cpu;
    unsigned wave;
  };
  for (const Spec s :
       {Spec{"gfx908", 64}, Spec{"gfx90a", 64}, Spec{"gfx942", 64}, Spec{"gfx1030", 32},
        Spec{"gfx1030", 64}, Spec{"gfx1100", 32}, Spec{"gfx1100", 64}, Spec{"gfx1151", 32},
        Spec{"gfx1200", 32}, Spec{"gfx1200", 64}, Spec{"gfx1250", 32}}) {
    SCOPED_TRACE(std::string(s.cpu) + " wave" + std::to_string(s.wave));
    PairOptions o;
    o.mcpu = s.cpu;
    o.wave_size = s.wave;
    o.entry_vgprs = 2;
    const bool gfx9 = std::string(s.cpu).rfind("gfx9", 0) == 0;
    const std::string add = gfx9 ? "v_add_u32_e32" : "v_add_nc_u32_e32";
    const std::string sub = gfx9 ? "v_sub_u32_e32" : "v_sub_nc_u32_e32";
    // A compare into the wave's VCC and a select on it, then integer arithmetic.
    const std::string vcc = s.wave == 32 ? "vcc_lo" : "vcc";
    auto body = [&](const std::string &op, const std::string &a, const std::string &b) {
      return target_store_fn(s.cpu,
                             {"v_cmp_gt_u32_e32 " + vcc + ", v0, v1",
                              "v_cndmask_b32_e32 v2, v1, v0, " + vcc, op + " v0, " + a + ", " + b});
    };
    const std::string base = body(add, "v2", "v1");
    EXPECT_EQ(verdict_of(dir, "f", base, body(add, "v1", "v2"), o),
              "NO_MISMATCH_FOUND/ESTABLISHED");
    EXPECT_EQ(verdict_of(dir, "f", base, body(sub, "v2", "v1"), o), "LOCAL_MISMATCH/ESTABLISHED");
  }
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffTargets, Wave32MasksAreOneSgprAndWave64MasksAPair) {
  const std::string dir = temp_dir("wave32");
  const auto tc = toolchain();
  for (unsigned wave : {32u, 64u}) {
    SCOPED_TRACE("wave" + std::to_string(wave));
    const std::string vcc = wave == 32 ? "vcc_lo" : "vcc";
    const std::string s4 = wave == 32 ? "s4" : "s[4:5]";
    const std::string sx = wave == 32 ? "_b32" : "_b64";
    const std::string text = target_fn("gfx1100", {"v_cmp_eq_u32_e32 " + vcc + ", v0, v1",
                                                   "v_cndmask_b32_e32 v2, v1, v0, " + vcc,
                                                   "s_and_saveexec" + sx + " " + s4 + ", " + vcc});
    ASSERT_TRUE(write_text_file(dir + "/f.s", text));
    Artifact a;
    llvm::Provenance p;
    p.mcpu = "gfx1100";
    p.wave_size = wave;
    std::string error;
    ASSERT_TRUE(llvm::assemble(tc, dir + "/f.s", dir + "/f", "f", a, p, error)) << error;
    a.target = *with_wave_size(a.target, wave);
    auto model = KernelModel::build(a, error);
    ASSERT_TRUE(model) << error;
    const auto &in = model->insts();
    ASSERT_GE(in.size(), 3u);
    auto locs = [](std::initializer_list<unsigned> l) {
      LocSet s;
      for (unsigned x : l)
        s.set(x);
      return s;
    };
    if (wave == 32) {
      EXPECT_EQ(format_locs(in[0].fx.writes), "vcc_lo");
      EXPECT_EQ(in[1].fx.reads, locs({kLocVccLo, kLocExecLo}));
      EXPECT_EQ(in[2].fx.writes, locs({4, kLocExecLo, kLocScc}));
      EXPECT_EQ(in[2].fx.reads, locs({kLocVccLo, kLocExecLo}));
    } else {
      EXPECT_EQ(in[0].fx.writes, locs({kLocVccLo, kLocVccHi}));
      EXPECT_EQ(in[1].fx.reads, locs({kLocVccLo, kLocVccHi, kLocExecLo, kLocExecHi}));
      EXPECT_EQ(in[2].fx.writes, locs({4, 5, kLocExecLo, kLocExecHi, kLocScc}));
    }
    EXPECT_TRUE(in[0].fx.p0 && in[1].fx.p0 && in[2].fx.p0)
        << in[0].fx.reject_reason << in[1].fx.reject_reason << in[2].fx.reject_reason;
  }
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffTargets, NanProvenanceFollowsAWave32Select) {
  const std::string dir = temp_dir("wave32nan");
  PairOptions o;
  o.mcpu = "gfx1100";
  o.wave_size = 32;
  o.entry_vgprs = 4;
  auto body = [&](const std::string &add) {
    return target_store_fn("gfx1100", {"v_cmp_lt_u32_e32 vcc_lo, v2, v3", add,
                                       "v_cndmask_b32_e32 v0, v2, v0, vcc_lo"});
  };
  const std::string ab = body("v_add_f32_e32 v0, v0, v1");
  const std::string ba = body("v_add_f32_e32 v0, v1, v0");
  EXPECT_EQ(verdict_of(dir, "f", ab, ba, o), "NO_MISMATCH_FOUND/ESTABLISHED");
  PairOptions strict = o;
  strict.strict_nan = true;
  EXPECT_EQ(verdict_of(dir, "f", ab, ba, strict), "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffTargets, Gfx12SplitWaitsRetireTheirCounters) {
  const std::string dir = temp_dir("gfx12wait");
  const auto tc = toolchain();
  const std::string text =
      target_fn("gfx1200", {"global_load_b32 v1, v0, s[0:1]", "s_load_b32 s4, s[0:1], 0x0",
                            "s_wait_loadcnt 0x0", "v_mov_b32_e32 v2, v1", "s_wait_kmcnt 0x0",
                            "s_mov_b32 s5, s4"});
  ASSERT_TRUE(write_text_file(dir + "/f.s", text));
  Artifact a;
  llvm::Provenance p;
  p.mcpu = "gfx1200";
  std::string error;
  ASSERT_TRUE(llvm::assemble(tc, dir + "/f.s", dir + "/f", "f", a, p, error)) << error;
  EXPECT_EQ(a.target.name, "gfx1200");
  auto model = KernelModel::build(a, error);
  ASSERT_TRUE(model) << error;
  const auto &in = model->insts();
  ASSERT_GE(in.size(), 6u);
  // Before the load wait: the load (v1) and the scalar load (s4) are outstanding.
  EXPECT_TRUE(model->pending_before(2).all_vdests().test(1));
  EXPECT_TRUE(model->pending_before(2).all_dests().test(4));
  // After s_wait_loadcnt 0: v1 has arrived, s4 may not have.
  EXPECT_FALSE(model->pending_before(3).all_vdests().test(1));
  EXPECT_TRUE(model->pending_before(3).all_dests().test(4));
  // After s_wait_kmcnt 0: nothing is outstanding.
  EXPECT_FALSE(model->pending_before(5).any());
  for (size_t i : {2u, 4u})
    EXPECT_TRUE(in[i].fx.p0) << in[i].text << ": " << in[i].fx.reject_reason;
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffTargets, SaluFloatAndModeWithoutIeeeOnGfx12) {
  const std::string dir = temp_dir("salufloat");
  PairOptions o;
  o.mcpu = "gfx1200";
  o.entry_sgprs = 8;
  auto body = [&](const std::string &op) {
    return target_store_fn("gfx1200", {op, "v_mov_b32_e32 v0, s6"});
  };
  // Commuted f32 multiply of the same operands: equal (NaN bits are unspecified).
  EXPECT_EQ(verdict_of(dir, "f", body("s_mul_f32 s6, s2, s3"), body("s_mul_f32 s6, s3, s2"), o),
            "NO_MISMATCH_FOUND/ESTABLISHED");
  EXPECT_EQ(verdict_of(dir, "f", body("s_mul_f32 s6, s2, s3"), body("s_add_f32 s6, s2, s3"), o),
            "LOCAL_MISMATCH/ESTABLISHED");
  EXPECT_EQ(default_function_mode(*target_by_name("gfx1200")) & ((1u << 8) | (1u << 9)), 0u);
  std::filesystem::remove_all(dir);
}

/// Assemble two listings of function f (gfx950) and analyse them.
std::optional<CaseData> analyze_pair(const std::string &dir, const std::string &old_s,
                                     const std::string &new_s, int entry_vgprs,
                                     const PlanOptions &plan = {}) {
  const auto tc = toolchain();
  Artifact a[2];
  for (int side = 0; side < 2; ++side) {
    const std::string path = dir + (side ? "/new.s" : "/old.s");
    EXPECT_TRUE(write_text_file(path, side ? new_s : old_s));
    llvm::Provenance p;
    std::string error;
    EXPECT_TRUE(llvm::assemble(tc, path, dir + (side ? "/new" : "/old"), "f", a[side], p, error))
        << error;
    a[side].return_vgprs = 0;
    a[side].entry_vgprs = entry_vgprs;
  }
  std::string error;
  auto data = analyze_artifacts(a[0], a[1], plan, error);
  EXPECT_TRUE(data) << error;
  return data;
}

TEST(IsaDiffReview, PerBitMaskPassIsNotAPlainPass) {
  const std::string dir = temp_dir("maskpass");
  // u is undefined (no entry VGPRs): old stores (u & 1) * 3, which is 0 or 3; new stores
  // 1, which old can never produce. Masking the two bits that vary accepts it, so the
  // result is marked as relying on the mask, not a plain pass.
  auto data =
      analyze_pair(dir, store_fn({"v_and_b32_e32 v0, 1, v0", "v_mul_u32_u24_e32 v0, 3, v0"}),
                   store_fn({"v_mov_b32_e32 v0, 1"}), 0);
  ASSERT_TRUE(data);
  TestOptions opt;
  opt.cases = 300;
  const Json report = test_case(*data, opt, "", nullptr);
  EXPECT_EQ(report["overall"]["test_status"].as_string(), "NO_MISMATCH_FOUND");
  EXPECT_GT(report["overall"]["dont_care_masked_cases"].as_uint(), 0u);
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffReview, AlternativeNanCounterexamplesReplay) {
  const std::string dir = temp_dir("nanreplay");
  // inf + -inf: the hardware's NaN is canonical (bit 0 clear), so both sides agree; under
  // the all-ones NaN choice new's bit 0 leaks into its output. The bundle records the
  // choice and replay reproduces it.
  // The producer differs too (another temporary), so it is inside the region.
  const std::vector<std::string> base = {"v_mov_b32_e32 v2, 0xff800000",
                                         "v_add_f32_e32 v2, 0x7f800000, v2",
                                         "v_and_b32_e32 v0, 0x7fc00000, v2"};
  const std::vector<std::string> leak = {
      "v_mov_b32_e32 v5, 0xff800000", "v_add_f32_e32 v2, 0x7f800000, v5",
      "v_and_b32_e32 v0, 0x7fc00000, v2", "v_and_b32_e32 v3, 1, v2", "v_or_b32_e32 v0, v0, v3"};
  auto data = analyze_pair(dir, store_fn(base), store_fn(leak), 0);
  ASSERT_TRUE(data);
  TestOptions opt;
  opt.cases = 50;
  opt.counterexample_dir = dir + "/cex";
  const Json report = test_case(*data, opt, "", nullptr);
  ASSERT_EQ(report["overall"]["test_status"].as_string(), "LOCAL_MISMATCH");
  size_t replayed = 0;
  for (const auto &e : std::filesystem::directory_iterator(dir + "/cex")) {
    const auto text = read_text_file(e.path().string());
    ASSERT_TRUE(text);
    const auto bundle = Json::parse(*text);
    ASSERT_TRUE(bundle);
    EXPECT_TRUE((*bundle)["nan_choice"].is_object()) << e.path();
    std::string error;
    const ReplayOutcome out = replay_counterexample(*bundle, error);
    EXPECT_TRUE(out.reproduced) << error << " " << out.summary;
    ++replayed;
  }
  EXPECT_GT(replayed, 0u);
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffReview, HintOnlyChangesAreNoSemanticDiff) {
  const std::string dir = temp_dir("hints");
  PairOptions o;
  o.mcpu = "gfx1100";
  o.entry_vgprs = 2;
  // A moved and re-encoded s_delay_alu and an added s_nop (LLVM #230390 changed only
  // s_delay_alu): no functional effect on an in-order wave, so nothing is tested.
  const std::string a =
      target_store_fn("gfx1100", {"v_add_f32_e32 v0, v0, v1", "s_delay_alu instid0(VALU_DEP_1)",
                                  "v_mul_lo_u32 v0, v0, v0"});
  const std::string b =
      target_store_fn("gfx1100", {"s_nop 0", "v_add_f32_e32 v0, v0, v1",
                                  "s_delay_alu instid0(VALU_DEP_2)", "v_mul_lo_u32 v0, v0, v0"});
  EXPECT_EQ(verdict_of(dir, "f", a, b, o), "NO_SEMANTIC_DIFF/NOT_APPLICABLE");
  // A real change elsewhere is still tested; the hint hunk is a non-semantic note.
  auto data = analyze_pair(dir, store_fn({"s_nop 1", "v_add_u32_e32 v0, v0, v1"}),
                           store_fn({"v_add_u32_e32 v0, v1, v0"}), 2);
  ASSERT_TRUE(data);
  size_t hint_changes = 0;
  for (const Json &c : data->changes.items())
    hint_changes += c["kind"].as_string() == "hint" ? 1 : 0;
  EXPECT_EQ(hint_changes, 0u); // the s_nop sits next to the real change: one hunk
  // Separated from it by common code, the s_nop is a hunk of its own.
  data = analyze_pair(
      dir,
      store_fn(
          {"v_add_u32_e32 v0, v0, v1", "v_mov_b32_e32 v2, 0", "s_nop 1", "v_mov_b32_e32 v5, 0"}),
      store_fn({"v_add_u32_e32 v0, v1, v0", "v_mov_b32_e32 v2, 0", "v_mov_b32_e32 v5, 0"}), 2);
  ASSERT_TRUE(data);
  hint_changes = 0;
  for (const Json &c : data->changes.items())
    hint_changes += c["kind"].as_string() == "hint" ? 1 : 0;
  EXPECT_EQ(hint_changes, 1u);
  ASSERT_EQ(data->regions.size(), 1u);
  bool noted = false;
  for (const auto &n : data->notes)
    noted = noted || (n.kind == ChangeKind::Hint && !n.semantic);
  EXPECT_TRUE(noted);
  TestOptions opt;
  opt.cases = 300;
  EXPECT_EQ(test_case(*data, opt, "", nullptr)["overall"]["test_status"].as_string(),
            "NO_MISMATCH_FOUND");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffReview, NanChoicesFollowTheOperationNotItsIndex) {
  const std::string dir = temp_dir("nanorder");
  PairOptions o;
  o.entry_vgprs = 2;
  // The product's NaN bits go through an integer square, which no per-bit mask of old's
  // alternatives covers. New renames the temporary and gains an s_nop in front, which
  // shifts every instruction index; the same multiply must still draw the same NaN.
  const std::string a = store_fn({"v_mul_f32_e32 v2, v0, v1", "v_mul_lo_u32 v0, v2, v2"});
  const std::string b =
      store_fn({"s_nop 0", "v_mul_f32_e32 v3, v0, v1", "v_mul_lo_u32 v0, v3, v3"});
  EXPECT_EQ(verdict_of(dir, "f", a, b, o), "NO_MISMATCH_FOUND/ESTABLISHED");
  // New squaring a different NaN-dependent value is still caught.
  const std::string c = store_fn({"s_nop 0", "v_mul_f32_e32 v3, v0, v1", "v_add_u32_e32 v3, 1, v3",
                                  "v_mul_lo_u32 v0, v3, v3"});
  EXPECT_EQ(verdict_of(dir, "f", a, c, o), "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffReview, VgprMutantsAreComparedNotRejected) {
  // A VALU mutant that writes another VGPR must be compared, not rejected as an
  // undeclared effect: make_synthetic_mutation re-derives the VGPR writes too.
  const std::string dir = temp_dir("vmutants");
  auto data = analyze_pair(dir, store_fn({"v_add_u32_e32 v0, v0, v1"}),
                           store_fn({"v_add_u32_e32 v0, v1, v0"}), 2);
  ASSERT_TRUE(data);
  ASSERT_EQ(data->regions.size(), 1u);
  size_t valu = 0, killed = 0;
  for (const Mutant &m : enumerate_mutants(data->regions[0])) {
    if (m.unit != "valu" || m.operator_class != "register")
      continue;
    std::string error;
    auto syn = make_synthetic_mutation(data->regions[0], m.spec, error);
    ASSERT_TRUE(syn) << error;
    BaselinePolicy policy(1);
    TestOptions opt;
    opt.cases = 200;
    opt.reduce = false;
    const RegionResult r = test_region(*syn, opt, policy);
    EXPECT_EQ(r.mismatch_kinds.count("effect_model"), 0u) << m.mutated;
    killed += r.test_status == TestStatus::LocalMismatch ? 1 : 0;
    ++valu;
  }
  EXPECT_GT(valu, 0u);
  EXPECT_GT(killed, 0u);
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffWholeFunction, RenamingAcrossBlocksIsSettledByTheWholeFunction) {
  const std::string dir = temp_dir("wholefn");
  // s4 renamed to s5 in three blocks: every cut sees different continuation liveness,
  // so the regions alone are INCONCLUSIVE_MAPPING; the whole function has no cut.
  auto body = [](const std::string &t, const std::string &add) {
    return fn_listing({"s_add_u32 " + t + ", s0, 1", "s_cmp_eq_u32 s1, 0", "s_cbranch_scc1 .LBB0_2",
                       "s_add_u32 " + t + ", " + t + ", " + add,
                       ".LBB0_2:", "v_mov_b32_e32 v0, " + t, "v_mov_b32_e32 v4, 0",
                       "global_store_dword v4, v0, s[2:3]", "s_waitcnt vmcnt(0)"});
  };
  TestOptions opt;
  opt.cases = 500;
  PlanOptions regions_only;
  regions_only.whole_function = false;
  auto data = analyze_pair(dir, body("s4", "2"), body("s5", "2"), 0, regions_only);
  ASSERT_TRUE(data);
  EXPECT_FALSE(data->whole_function);
  EXPECT_EQ(test_case(*data, opt, "", nullptr)["overall"]["test_status"].as_string(),
            "INCONCLUSIVE");

  data = analyze_pair(dir, body("s4", "2"), body("s5", "2"), 0);
  ASSERT_TRUE(data);
  ASSERT_TRUE(data->whole_function);
  EXPECT_EQ(data->whole_function->status, RegionStatus::Ready) << data->whole_function->reason;
  EXPECT_EQ(data->whole_function->contract.status, ContractStatus::Established);
  Json report = test_case(*data, opt, "", nullptr);
  EXPECT_EQ(report["overall"]["test_status"].as_string(), "NO_MISMATCH_FOUND");
  EXPECT_EQ(report["overall"]["coverage_scope"].as_string(), "COMPLETE_FOR_DECLARED_SCOPE");
  EXPECT_EQ(report["overall"]["resolved_by"].as_string(), "whole_function");
  EXPECT_TRUE(report["whole_function"]["decides"].as_bool());
  // The case round-trips with its whole-function region.
  std::string error;
  auto again = CaseData::from_json(data->to_json(), error);
  ASSERT_TRUE(again) << error;
  ASSERT_TRUE(again->whole_function);
  EXPECT_EQ(again->whole_function->prepared_key, data->whole_function->prepared_key);

  // A real difference in the renamed code: the whole function makes it a verdict.
  data = analyze_pair(dir, body("s4", "2"), body("s5", "3"), 0);
  ASSERT_TRUE(data);
  report = test_case(*data, opt, "", nullptr);
  EXPECT_EQ(report["overall"]["test_status"].as_string(), "LOCAL_MISMATCH");
  EXPECT_EQ(report["overall"]["resolved_by"].as_string(), "whole_function");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffWholeFunction, MismatchesRelyingOnADifferingUpstreamAreWitnesses) {
  const std::string dir = temp_dir("wholefn_dep");
  // Old copies (a, b) into (v6, v7), new into (v7, v6); one is incremented on a branch
  // and the final subtraction is swapped too: equivalent. The last region's mismatch
  // relies on the swapped registers of the first: a witness, not a verdict (memmove
  // functions of LLVM's memmove-var-size.ll looked like this).
  auto body = [](bool swapped) {
    const std::string x = swapped ? "v7" : "v6", y = swapped ? "v6" : "v7";
    return fn_listing({"v_mov_b32_e32 " + x + ", v0", "v_mov_b32_e32 " + y + ", v1",
                       "s_cmp_eq_u32 s4, 0", "s_cbranch_scc1 .LBB0_2",
                       "v_add_u32_e32 " + x + ", 1, " + x,
                       ".LBB0_2:", "v_sub_u32_e32 v0, " + x + ", " + y, "v_mov_b32_e32 v4, 0",
                       "global_store_dword v4, v0, s[2:3]", "s_waitcnt vmcnt(0)"});
  };
  TestOptions opt;
  opt.cases = 500;
  PlanOptions regions_only;
  regions_only.whole_function = false;
  auto data = analyze_pair(dir, body(false), body(true), 2, regions_only);
  ASSERT_TRUE(data);
  Json report = test_case(*data, opt, "", nullptr);
  EXPECT_EQ(report["overall"]["test_status"].as_string(), "INCONCLUSIVE");
  bool relies = false;
  for (const Json &u : report["unresolved"].items())
    relies = relies || u["what"].as_string().find("relies on the R_out") != std::string::npos ||
             u["what"].as_string().find("may be compensated") != std::string::npos;
  EXPECT_TRUE(relies);
  data = analyze_pair(dir, body(false), body(true), 2);
  ASSERT_TRUE(data);
  report = test_case(*data, opt, "", nullptr);
  EXPECT_EQ(report["overall"]["test_status"].as_string(), "NO_MISMATCH_FOUND");
  EXPECT_EQ(report["overall"]["resolved_by"].as_string(), "whole_function");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffWholeFunction, PassingRegionsDoNotRunTheWholeFunction) {
  const std::string dir = temp_dir("wholefn_pass");
  auto data = analyze_pair(dir, store_fn({"v_add_u32_e32 v0, v0, v1"}),
                           store_fn({"v_add_u32_e32 v0, v1, v0"}), 2);
  ASSERT_TRUE(data);
  ASSERT_TRUE(data->whole_function);
  TestOptions opt;
  opt.cases = 300;
  const Json report = test_case(*data, opt, "", nullptr);
  EXPECT_EQ(report["overall"]["test_status"].as_string(), "NO_MISMATCH_FOUND");
  EXPECT_FALSE(report.has("whole_function"));
  EXPECT_FALSE(report["overall"].has("resolved_by"));
  std::filesystem::remove_all(dir);
}

/// A spill to stack slot @p store and a reload from @p load around a dead overwrite.
std::vector<std::string> spill_body(bool scratch, int store, int load) {
  const std::string st =
      scratch ? "scratch_store_dword off, v1, s32 offset:" + std::to_string(store)
              : "buffer_store_dword v1, off, s[0:3], s32 offset:" + std::to_string(store);
  const std::string ld =
      scratch ? "scratch_load_dword v3, off, s32 offset:" + std::to_string(load)
              : "buffer_load_dword v3, off, s[0:3], s32 offset:" + std::to_string(load);
  return {"v_add_u32_e32 v1, v0, v2", st, "v_mov_b32_e32 v1, 0",
          "s_waitcnt vmcnt(0)",       ld, "s_waitcnt vmcnt(0)",
          "v_add_u32_e32 v0, v3, v1"};
}

TEST(IsaDiffMemory, SpillSlotsAreDeadWhereNothingElseReadsTheStack) {
  const std::string dir = temp_dir("spills");
  for (const auto &[cpu, scratch] : {std::pair{"gfx908", false}, {"gfx942", true}}) {
    SCOPED_TRACE(cpu);
    PairOptions o;
    o.mcpu = cpu;
    o.return_vgprs = 1;
    o.entry_vgprs = 3;
    const std::string base = target_fn(cpu, spill_body(scratch, 12, 12));
    // Another stack slot, stored and reloaded inside the region: the frame is dead after
    // it, so the slots are not compared, only the reloaded value.
    EXPECT_EQ(verdict_of(dir, "f", base, target_fn(cpu, spill_body(scratch, 16, 16)), o),
              "NO_MISMATCH_FOUND/ESTABLISHED");
    // Reloading another slot than the one stored: the value differs.
    EXPECT_EQ(verdict_of(dir, "f", base, target_fn(cpu, spill_body(scratch, 16, 12)), o),
              "LOCAL_MISMATCH/ESTABLISHED");
  }
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffMemory, LdsAccessesRunInTheWindow) {
  const std::string dir = temp_dir("lds");
  PairOptions o;
  o.return_vgprs = 1;
  o.entry_vgprs = 2;
  const std::string old_s =
      target_fn("gfx950", {"ds_write_b32 v0, v1 offset:12", "s_waitcnt lgkmcnt(0)",
                           "ds_read_b32 v0, v0 offset:12", "s_waitcnt lgkmcnt(0)"});
  // The offset folded into the address: equivalent.
  const std::string folded =
      target_fn("gfx950", {"v_add_u32_e32 v2, 12, v0", "ds_write_b32 v2, v1",
                           "s_waitcnt lgkmcnt(0)", "ds_read_b32 v0, v2", "s_waitcnt lgkmcnt(0)"});
  auto data = analyze_pair(dir, old_s, folded, 2);
  ASSERT_TRUE(data);
  ASSERT_EQ(data->regions.size(), 1u);
  ASSERT_EQ(data->regions[0].status, RegionStatus::Ready) << data->regions[0].reason;
  // The address input lies in the LDS window.
  ASSERT_EQ(data->regions[0].contract.vinput_masks.size(), 1u);
  EXPECT_EQ(data->regions[0].contract.vinput_masks[0].first, 0u);
  EXPECT_EQ(verdict_of(dir, "f", old_s, folded, o), "NO_MISMATCH_FOUND/ESTABLISHED");
  // A wrong offset stores to another LDS word.
  const std::string wrong = target_fn(
      "gfx950", {"v_add_u32_e32 v2, 16, v0", "ds_write_b32 v2, v1", "s_waitcnt lgkmcnt(0)",
                 "ds_read_b32 v0, v0 offset:12", "s_waitcnt lgkmcnt(0)"});
  EXPECT_EQ(verdict_of(dir, "f", old_s, wrong, o), "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffMemory, AtomicsReadAndWriteMemory) {
  const std::string dir = temp_dir("atomics");
  PairOptions o;
  o.return_vgprs = 1;
  o.entry_vgprs = 2;
  auto fn = [](const std::string &op) {
    return target_fn("gfx950", {"v_mov_b32_e32 v4, 0", op, "s_waitcnt vmcnt(0)"});
  };
  // add x == sub -x, with and without the returned old value.
  const std::string add = fn("global_atomic_add v0, v4, v1, s[4:5] sc0");
  const std::string sub =
      target_fn("gfx950", {"v_mov_b32_e32 v4, 0", "v_sub_u32_e32 v2, 0, v1",
                           "global_atomic_sub v0, v4, v2, s[4:5] sc0", "s_waitcnt vmcnt(0)"});
  EXPECT_EQ(verdict_of(dir, "f", add, sub, o), "NO_MISMATCH_FOUND/ESTABLISHED");
  EXPECT_EQ(verdict_of(dir, "f", add, fn("global_atomic_sub v0, v4, v1, s[4:5] sc0"), o),
            "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffAgpr, AgprMovesAndCalleeSavedAgprs) {
  const std::string dir = temp_dir("agpr");
  PairOptions o;
  o.mcpu = "gfx90a";
  o.return_vgprs = 1;
  o.entry_vgprs = 1;
  const std::string mov = target_fn("gfx90a", {"v_mov_b32_e32 v1, v0", "v_add_u32_e32 v0, v1, v1"});
  // A round trip through a caller-saved AGPR (a0) is a move.
  EXPECT_EQ(
      verdict_of(dir, "f",
                 target_fn("gfx90a", {"v_accvgpr_write_b32 a0, v0", "v_accvgpr_read_b32 v1, a0",
                                      "v_add_u32_e32 v0, v1, v1"}),
                 mov, o),
      "NO_MISMATCH_FOUND/ESTABLISHED");
  // a32-a255 are callee-saved from gfx90a on (CSR_AMDGPU_AGPRs): clobbering a40 is a bug.
  const std::string clobber =
      target_fn("gfx90a", {"v_accvgpr_write_b32 a40, v0", "v_mov_b32_e32 v1, v0",
                           "v_add_u32_e32 v0, v1, v1"});
  EXPECT_EQ(verdict_of(dir, "f", mov, clobber, o), "LOCAL_MISMATCH/ESTABLISHED");
  // gfx908 saves no AGPR.
  PairOptions o908 = o;
  o908.mcpu = "gfx908";
  EXPECT_EQ(verdict_of(dir, "f",
                       target_fn("gfx908", {"v_mov_b32_e32 v1, v0", "v_add_u32_e32 v0, v1, v1"}),
                       target_fn("gfx908", {"v_accvgpr_write_b32 a40, v0", "v_mov_b32_e32 v1, v0",
                                            "v_add_u32_e32 v0, v1, v1"}),
                       o908),
            "NO_MISMATCH_FOUND/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffAgpr, MfmaRunsInRegions) {
  const std::string dir = temp_dir("mfma");
  PairOptions o;
  o.mcpu = "gfx90a";
  o.return_vgprs = 1;
  o.entry_vgprs = 2;
  auto mfma = [](const std::string &acc, const std::string &a, const std::string &b,
                 const std::string &read) {
    return target_fn("gfx90a", {"v_mfma_f32_4x4x1f32 " + acc + ", " + a + ", " + b + ", 0",
                                "s_nop 3", "v_accvgpr_read_b32 v0, " + read});
  };
  const std::string base = mfma("a[0:3]", "v0", "v1", "a0");
  // Other accumulation registers: equivalent.
  EXPECT_EQ(verdict_of(dir, "f", base, mfma("a[4:7]", "v0", "v1", "a4"), o),
            "NO_MISMATCH_FOUND/ESTABLISHED");
  // A and B swapped: the 4x4 blocks are transposed.
  EXPECT_EQ(verdict_of(dir, "f", base, mfma("a[0:3]", "v1", "v0", "a0"), o),
            "LOCAL_MISMATCH/ESTABLISHED");
  // gfx950: the result in VGPRs or in AGPRs.
  PairOptions o950 = o;
  o950.mcpu = "gfx950";
  o950.entry_vgprs = 4;
  EXPECT_EQ(verdict_of(dir, "f",
                       target_fn("gfx950", {"v_mfma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], 0",
                                            "s_nop 7", "v_mov_b32_e32 v0, v5"}),
                       target_fn("gfx950", {"v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], 0",
                                            "s_nop 7", "v_accvgpr_read_b32 v0, a1"}),
                       o950),
            "NO_MISMATCH_FOUND/ESTABLISHED");
  // An AGPR write moved across unrelated code before the MFMA that reads it. The MFMA
  // reads every lane, so inactive lanes must be entry-equal (no kill before it), and the
  // entry EXEC that reaches it is all ones (mfma_full_exec).
  o.entry_vgprs = 6;
  const std::string early = target_fn(
      "gfx90a", {"v_accvgpr_write_b32 a4, v0", "v_mov_b32_e32 v5, 0",
                 "v_mfma_f32_4x4x1f32 a[0:3], a4, v1, 0", "s_nop 3", "v_accvgpr_read_b32 v0, a0"});
  const std::string late = target_fn("gfx90a", {"v_mov_b32_e32 v5, 0", "v_accvgpr_write_b32 a4, v0",
                                                "v_mfma_f32_4x4x1f32 a[0:3], a4, v1, 0", "s_nop 3",
                                                "v_accvgpr_read_b32 v0, a0"});
  EXPECT_EQ(verdict_of(dir, "f", early, late, o), "NO_MISMATCH_FOUND/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffControlFlow, BranchConditionsAreFollowedAndExitsCompared) {
  const std::string dir = temp_dir("cflow");
  PairOptions o;
  // if (s4 == 0) skip the increment; then store v0.
  auto ifthen = [](const std::string &cmp, const std::string &br) {
    return store_fn({cmp + " s4, 0", br + " .LBB0_2", "v_add_u32_e32 v0, 1, v0", ".LBB0_2:"});
  };
  const std::string base = ifthen("s_cmp_eq_u32", "s_cbranch_scc1");
  // The same condition, written the other way round: equivalent.
  EXPECT_EQ(verdict_of(dir, "f", base, ifthen("s_cmp_lg_u32", "s_cbranch_scc0"), o),
            "NO_MISMATCH_FOUND/ESTABLISHED");
  // The inverted condition takes the other exit.
  EXPECT_EQ(verdict_of(dir, "f", base, ifthen("s_cmp_lg_u32", "s_cbranch_scc1"), o),
            "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffControlFlow, LoopsRunInsideTheRegionUpToAStepLimit) {
  const std::string dir = temp_dir("loop");
  PairOptions o;
  // s5 = 2 * (s4 & 7) by a counted loop; the body and the back edge change together.
  auto loop = [](const std::string &add, const std::string &cmp, const std::string &br) {
    return store_fn({"s_and_b32 s4, s4, 7", "s_mov_b32 s5, 0", "s_cmp_eq_u32 s4, 0",
                     "s_cbranch_scc1 .LBB0_3", ".LBB0_2:", add, "s_add_u32 s4, s4, -1",
                     cmp + " s4, 0", br + " .LBB0_2", ".LBB0_3:", "v_mov_b32_e32 v0, s5"});
  };
  const std::string base = loop("s_add_u32 s5, s5, 2", "s_cmp_lg_u32", "s_cbranch_scc1");
  EXPECT_EQ(
      verdict_of(dir, "f", base, loop("s_add_u32 s5, 2, s5", "s_cmp_eq_u32", "s_cbranch_scc0"), o),
      "NO_MISMATCH_FOUND/ESTABLISHED");
  EXPECT_EQ(
      verdict_of(dir, "f", base, loop("s_add_u32 s5, s5, 3", "s_cmp_eq_u32", "s_cbranch_scc0"), o),
      "LOCAL_MISMATCH/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffControlFlow, OneSidedStepLimitIsInconclusiveWhateverTheOtherCases) {
  const std::string dir = temp_dir("loop_hang");
  PairOptions o;
  // Count s4 & 1 down to zero; new subtracts 0, so it never exits for s4 & 1 == 1, while
  // every s4 & 1 == 0 case passes. Old exits where new runs past the step limit (also at
  // the re-run's larger limit): a possible non-termination, never offset by the passes.
  auto loop = [](const std::string &dec) {
    return store_fn({"s_and_b32 s4, s4, 1", "s_cmp_eq_u32 s4, 0", "s_cbranch_scc1 .LBB0_3",
                     ".LBB0_2:", dec, "s_cmp_lg_u32 s4, 0", "s_cbranch_scc1 .LBB0_2",
                     ".LBB0_3:", "v_mov_b32_e32 v0, s4"});
  };
  const Json report =
      report_of(dir, "f", loop("s_sub_u32 s4, s4, 1"), loop("s_sub_u32 s4, s4, 0"), o);
  ASSERT_TRUE(report.is_object());
  EXPECT_EQ(report["overall"]["test_status"].as_string(), "INCONCLUSIVE");
  EXPECT_NE(report["overall"]["coverage_scope"].as_string(), "COMPLETE_FOR_DECLARED_SCOPE");
  EXPECT_GT(count_of(report, "step_limit_divergences"), 0u);
  ASSERT_TRUE(report["whole_function"].is_object());
  EXPECT_FALSE(report["whole_function"]["decides"].as_bool());
  EXPECT_NE(report["whole_function"]["reason"].as_string().find("possible non-termination"),
            std::string::npos)
      << report["whole_function"]["reason"].as_string();
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffControlFlow, SlowerLoopThatTerminatesIsReRunAndCompared) {
  const std::string dir = temp_dir("loop_slow");
  PairOptions o;
  // s5 = 2 * (s4 & 255). Old adds 2 per decrement; new adds 1 per iteration and decrements
  // every other one (by s6, toggled), so it needs about 2.5x the steps: past the step
  // limit where old is not. The looping side re-runs with a larger limit and the case is
  // compared like any other.
  const std::string old_s = store_fn(
      {"s_and_b32 s4, s4, 0xff", "s_mov_b32 s5, 0", "s_mov_b32 s6, 0", "s_cmp_eq_u32 s4, 0",
       "s_cbranch_scc1 .LBB0_3", ".LBB0_2:", "s_add_u32 s5, s5, 2", "s_sub_u32 s4, s4, 1",
       "s_cmp_lg_u32 s4, 0", "s_cbranch_scc1 .LBB0_2", ".LBB0_3:", "v_mov_b32_e32 v0, s5"});
  const std::string new_s =
      store_fn({"s_and_b32 s4, s4, 0xff", "s_mov_b32 s5, 0", "s_mov_b32 s6, 0",
                "s_cmp_eq_u32 s4, 0", "s_cbranch_scc1 .LBB0_3", ".LBB0_2:", "s_add_u32 s5, s5, 1",
                "s_sub_u32 s4, s4, s6", "s_xor_b32 s6, s6, 1", "s_cmp_lg_u32 s4, 0",
                "s_cbranch_scc1 .LBB0_2", ".LBB0_3:", "v_mov_b32_e32 v0, s5"});
  const Json report = report_of(dir, "f", old_s, new_s, o);
  ASSERT_TRUE(report.is_object());
  EXPECT_EQ(report["overall"]["test_status"].as_string(), "NO_MISMATCH_FOUND") << report.dump();
  EXPECT_GT(count_of(report, "step_limit_reruns"), 0u);
  std::filesystem::remove_all(dir);
}

/// A kernel f listed without its descriptor, as rj_isa_diff_pr writes kernels: @p body,
/// then s_endpgm.
std::string kernel_listing(const std::vector<std::string> &body) {
  std::string t = "\t.text\n\t.globl\tf\n\t.p2align\t8\n\t.type\tf,@function\nf:\n";
  for (const auto &l : body)
    t += "\t" + l + "\n";
  return t + "\ts_endpgm\n.Lfunc_end0:\n\t.size\tf, .Lfunc_end0-f\n";
}

/// The first-mismatch details of every region and of the whole function.
std::string mismatch_details(const Json &report) {
  std::string s;
  for (const Json &g : report["regions"].items())
    s += g["first_mismatch"]["detail"].as_string() + "\n";
  return s + report["whole_function"]["result"]["first_mismatch"]["detail"].as_string();
}

TEST(IsaDiffBarrier, SharedMemoryAccessesKeepTheirBarrierEpoch) {
  const std::string dir = temp_dir("barrier");
  PairOptions o;
  o.entry_vgprs = 2; // v0 data, v1 an LDS address (kept inside the window)
  auto kernel = [](std::vector<std::string> before_barrier,
                   std::vector<std::string> after_barrier) {
    std::vector<std::string> b = std::move(before_barrier);
    b.insert(b.end(), {"s_waitcnt lgkmcnt(0)", "s_barrier"});
    b.insert(b.end(), after_barrier.begin(), after_barrier.end());
    b.insert(b.end(), {"ds_read_b32 v2, v1 offset:4", "s_waitcnt lgkmcnt(0)", "v_mov_b32_e32 v4, 0",
                       "global_store_dword v4, v2, s[0:1]", "s_waitcnt vmcnt(0)"});
    return kernel_listing(b);
  };
  // Two stores before the barrier, in either order: the same stores in the same epoch.
  // LDS address inputs are below 32768 (masked with 0x7ffc), so the two stores' addresses
  // never meet (at v1 and v1 + 4 two lanes could store to one dword, and the order would
  // decide which value stays).
  const Json same =
      report_of(dir, "f", kernel({"ds_write_b32 v1, v0 offset:32768", "ds_write_b32 v1, v0"}, {}),
                kernel({"ds_write_b32 v1, v0", "ds_write_b32 v1, v0 offset:32768"}, {}), o);
  ASSERT_TRUE(same.is_object());
  EXPECT_EQ(same["overall"]["test_status"].as_string(), "NO_MISMATCH_FOUND")
      << mismatch_details(same);
  // A store moved after the barrier: this wave reads the same value, other waves of the
  // workgroup would not.
  const Json moved =
      report_of(dir, "f", kernel({"ds_write_b32 v1, v0 offset:4"}, {}),
                kernel({}, {"ds_write_b32 v1, v0 offset:4", "s_waitcnt lgkmcnt(0)"}), o);
  ASSERT_TRUE(moved.is_object());
  EXPECT_EQ(moved["overall"]["test_status"].as_string(), "LOCAL_MISMATCH") << moved.dump();
  EXPECT_NE(mismatch_details(moved).find("barrier"), std::string::npos) << mismatch_details(moved);
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffWholeFunction, KernelEarlyExitAndItsPrivateMemory) {
  const std::string dir = temp_dir("kernel_exit");
  PairOptions o;
  o.entry_vgprs = 1;
  // A value computed into another register on each side (renamed across blocks, so the
  // regions cannot decide), an early s_endpgm (s4 == 0), then the value kept in private
  // memory at another offset on each side (a spill slot an allocation change moved): the
  // whole function decides, and private memory is dead at the kernel's end.
  auto kernel = [](const std::string &add, const std::string &reg, const std::string &offset) {
    return kernel_listing({add, "s_cmp_eq_u32 s4, 0", "s_cbranch_scc0 .LBB0_1", "s_endpgm",
                           ".LBB0_1:", "v_mov_b32_e32 v5, 0",
                           "scratch_store_dword v5, " + reg + ", off offset:" + offset,
                           "s_waitcnt vmcnt(0)", "scratch_load_dword v6, v5, off offset:" + offset,
                           "s_waitcnt vmcnt(0)", "v_mov_b32_e32 v4, 0",
                           "global_store_dword v4, v6, s[0:1]", "s_waitcnt vmcnt(0)"});
  };
  const std::string old_s = kernel("v_add_u32_e32 v2, 1, v0", "v2", "16");
  const Json report = report_of(dir, "f", old_s, kernel("v_add_u32_e32 v3, 1, v0", "v3", "32"), o);
  ASSERT_TRUE(report.is_object());
  EXPECT_EQ(report["overall"]["test_status"].as_string(), "NO_MISMATCH_FOUND")
      << mismatch_details(report);
  ASSERT_TRUE(report["whole_function"]["result"].is_object()) << report["whole_function"].dump();
  EXPECT_EQ(report["whole_function"]["result"]["test_status"].as_string(), "NO_MISMATCH_FOUND");
  EXPECT_TRUE(report["whole_function"]["decides"].as_bool());
  // A real difference after the early exit is still found.
  const Json wrong = report_of(dir, "f", old_s, kernel("v_add_u32_e32 v3, 2, v0", "v3", "32"), o);
  ASSERT_TRUE(wrong.is_object());
  EXPECT_EQ(wrong["overall"]["test_status"].as_string(), "LOCAL_MISMATCH")
      << mismatch_details(wrong);
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffMemoryState, OnlyANewFaultIsAMismatch) {
  const std::string dir = temp_dir("fault_side");
  PairOptions o;
  o.entry_vgprs = 2; // v0 data, v1 an LDS address below 32768
  // A dead LDS load at v1 + 40000 leaves the 64 KiB window for about a fifth of the
  // inputs. Removing it is fine: those inputs are undefined for old, which faults there.
  // Adding it makes new fault where old completed: a mismatch.
  const std::string with_load = kernel_listing(
      {"ds_read_b32 v2, v1 offset:40000", "s_waitcnt lgkmcnt(0)", "v_mov_b32_e32 v4, 0",
       "global_store_dword v4, v0, s[0:1]", "s_waitcnt vmcnt(0)"});
  const std::string without = kernel_listing(
      {"v_mov_b32_e32 v4, 0", "global_store_dword v4, v0, s[0:1]", "s_waitcnt vmcnt(0)"});
  const Json removed = report_of(dir, "f", with_load, without, o);
  ASSERT_TRUE(removed.is_object());
  EXPECT_EQ(removed["overall"]["test_status"].as_string(), "NO_MISMATCH_FOUND")
      << mismatch_details(removed);
  EXPECT_NE(removed.dump().find("memory fault on the old side only"), std::string::npos);
  const Json added = report_of(dir, "f", without, with_load, o);
  ASSERT_TRUE(added.is_object());
  EXPECT_EQ(added["overall"]["test_status"].as_string(), "LOCAL_MISMATCH");
  EXPECT_NE(mismatch_details(added).find("new side"), std::string::npos) << mismatch_details(added);
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffMemoryState, AStoreIntoTheCodeDoesNotOutliveItsCase) {
  const std::string dir = temp_dir("code_store");
  PairOptions o;
  o.entry_vgprs = 1;
  // A case whose store address is the store instruction itself overwrites the code: that
  // case faults (excluded on both sides), and the next case must fetch the original instruction
  // (it runs exactly as on a fresh runner).
  auto k = [](const std::string &zero) {
    return kernel_listing({zero, "global_store_dword v4, v0, s[0:1]", "s_waitcnt vmcnt(0)"});
  };
  auto data = case_of(dir, "f", k("v_mov_b32_e32 v4, 0"), k("v_and_b32_e32 v4, 0, v0"), o);
  ASSERT_TRUE(data && data->whole_function);
  const RegionPair &r = *data->whole_function;
  uint64_t store_pc = 0;
  for (const RegionInst &ri : r.old_side.insts)
    if (ri.text.rfind("global_store", 0) == 0)
      store_pc = ri.pc;
  ASSERT_NE(store_pc, 0u);
  std::string error;
  const Target target = r.contract.gpu().value_or(Target{});
  auto runner =
      RegionRunner::create(r.old_side.insts, "code_store", error, r.contract.vgpr_count, target);
  auto fresh = RegionRunner::create(r.old_side.insts, "code_store_fresh", error,
                                    r.contract.vgpr_count, target);
  ASSERT_TRUE(runner && fresh) << error;
  auto state = [&](uint64_t base) {
    MachineState s;
    s.frame = runner->baseline_frame();
    s.set(kLocExecLo, 0xffffffffu);
    s.set(kLocExecHi, 0xffffffffu);
    s.set(0, static_cast<uint32_t>(base));
    s.set(1, static_cast<uint32_t>(base >> 32));
    s.resize_vgprs(r.contract.vgpr_count);
    for (unsigned lane = 0; lane < kWaveLanes; ++lane)
      s.vset(0, lane, 0x12345678u);
    return s;
  };
  const size_t n = r.old_side.insts.size();
  runner->load(state(store_pc));
  EXPECT_EQ(runner->run(n).status, RunStatus::MemoryFault);
  runner->load(state(0x0000'0010'0000'0000ull));
  EXPECT_EQ(runner->run(n).status, RunStatus::Ok);
  fresh->load(state(0x0000'0010'0000'0000ull));
  EXPECT_EQ(fresh->run(n).status, RunStatus::Ok);
  EXPECT_EQ(runner->capture().mem_written, fresh->capture().mem_written);
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffMemoryState, ALoadFromTheCodeIsExcluded) {
  const std::string dir = temp_dir("code_load");
  PairOptions o;
  o.entry_vgprs = 1;
  // A pointer that lands on the code reads each side's own instructions (different bytes
  // on the two sides): the case is excluded, not compared.
  auto k = [](const std::string &zero) {
    return kernel_listing({zero, "global_load_dword v1, v4, s[0:1]", "s_waitcnt vmcnt(0)",
                           "global_store_dword v4, v1, s[2:3]", "s_waitcnt vmcnt(0)"});
  };
  auto data = case_of(dir, "f", k("v_mov_b32_e32 v4, 0"), k("v_and_b32_e32 v4, 0, v0"), o);
  ASSERT_TRUE(data && data->whole_function);
  const RegionPair &r = *data->whole_function;
  std::string error;
  const Target target = r.contract.gpu().value_or(Target{});
  auto old_runner =
      RegionRunner::create(r.old_side.insts, "code_load_old", error, r.contract.vgpr_count, target);
  auto new_runner =
      RegionRunner::create(r.new_side.insts, "code_load_new", error, r.contract.vgpr_count, target);
  ASSERT_TRUE(old_runner && new_runner) << error;
  share_code_guard(*old_runner, *new_runner);
  MachineState s;
  s.frame = old_runner->baseline_frame();
  s.set(kLocExecLo, 0xffffffffu);
  s.set(kLocExecHi, 0xffffffffu);
  const uint64_t code = r.old_side.insts.front().pc;
  s.set(0, static_cast<uint32_t>(code));
  s.set(1, static_cast<uint32_t>(code >> 32));
  s.set(3, 0x10u); // s[2:3]: the store, away from the code
  s.resize_vgprs(r.contract.vgpr_count);
  for (auto *runner : {old_runner.get(), new_runner.get()}) {
    runner->load(s);
    const RunOutcome out = runner->run(r.old_side.insts.size());
    EXPECT_EQ(out.status, RunStatus::MemoryFault);
    EXPECT_NE(out.detail.find("load from the code"), std::string::npos) << out.detail;
  }
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffMemoryState, StoredBytesAreSortedAboveTheLdsTag) {
  const std::string dir = temp_dir("sorted_written");
  PairOptions o;
  o.entry_vgprs = 2;
  // An LDS store and a global store at an address above the LDS tag (wild address
  // arithmetic reaches 0xffff...): the captured bytes are sorted by address, which the
  // comparison's lookups rely on.
  auto k = [](const std::string &data) {
    return kernel_listing({"ds_write_b32 v1, " + data, "s_waitcnt lgkmcnt(0)",
                           "v_mov_b32_e32 v4, 0", "global_store_dword v4, v0, s[0:1]",
                           "s_waitcnt vmcnt(0)"});
  };
  auto data = case_of(dir, "f", k("v0"), k("v1"), o);
  ASSERT_TRUE(data && data->whole_function);
  const RegionPair &r = *data->whole_function;
  std::string error;
  auto runner = RegionRunner::create(r.old_side.insts, "sorted", error, r.contract.vgpr_count,
                                     r.contract.gpu().value_or(Target{}));
  ASSERT_TRUE(runner) << error;
  MachineState s;
  s.frame = runner->baseline_frame();
  s.set(kLocExecLo, 0xffffffffu);
  s.set(kLocExecHi, 0xffffffffu);
  s.set(1, 0xffff8100u); // s[0:1] = 0xffff8100'00000000
  s.resize_vgprs(r.contract.vgpr_count);
  for (unsigned lane = 0; lane < kWaveLanes; ++lane) {
    s.vset(0, lane, 0x11u * lane);
    s.vset(1, lane, 4 * lane);
  }
  runner->load(s);
  ASSERT_EQ(runner->run(r.old_side.insts.size()).status, RunStatus::Ok);
  const MachineState f = runner->capture();
  ASSERT_FALSE(f.mem_written.empty());
  EXPECT_GE(f.mem_written.back().first, kLdsTag);
  EXPECT_TRUE(std::is_sorted(f.mem_written.begin(), f.mem_written.end()));
  EXPECT_TRUE(std::ranges::any_of(f.mem_written,
                                  [](const auto &b) { return b.first > kLdsTag + 0x10000; }));
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffWholeFunction, VectorLoadsOverwriteInOrder) {
  const std::string dir = temp_dir("vmem_waw");
  PairOptions o;
  o.entry_vgprs = 1;
  auto pending = [&](const std::string &second) {
    auto k = [&](const std::string &zero) {
      return kernel_listing({zero, "global_load_dword v1, v4, s[0:1]", second, "s_waitcnt vmcnt(0)",
                             "global_store_dword v4, v1, s[2:3]", "s_waitcnt vmcnt(0)"});
    };
    auto data = case_of(dir, "f", k("v_mov_b32_e32 v4, 0"), k("v_and_b32_e32 v4, 0, v0"), o);
    EXPECT_TRUE(data && data->whole_function);
    return data && data->whole_function ? data->whole_function->contract.pending_effects
                                        : std::string("?");
  };
  // A second load into v1 while the first is pending: loads return in issue order.
  EXPECT_NE(pending("global_load_dword v1, v4, s[0:1] offset:4"), "CONFLICT");
  // A load whose address v1 is still pending reads it too early: a conflict.
  EXPECT_EQ(pending("global_load_dword v2, v1, s[0:1]"), "CONFLICT");
  std::filesystem::remove_all(dir);
}

TEST(IsaDiffWholeFunction, LdsLoadsCompleteInOrder) {
  const std::string dir = temp_dir("lds_order");
  PairOptions o;
  o.entry_vgprs = 2;
  // lgkmcnt(1) after two LDS loads retires the first (LDS returns in issue order): reading
  // its destination is no pending conflict, so the region's contract holds.
  auto kernel = [](const std::string &add) {
    return kernel_listing({"ds_read_b32 v2, v1", "ds_read_b32 v3, v1 offset:4",
                           "s_waitcnt lgkmcnt(1)", add, "s_waitcnt lgkmcnt(0)",
                           "v_add_u32_e32 v2, v2, v3", "v_mov_b32_e32 v4, 0",
                           "global_store_dword v4, v2, s[0:1]", "s_waitcnt vmcnt(0)"});
  };
  EXPECT_EQ(verdict_of(dir, "f", kernel("v_add_u32_e32 v2, v2, v0"),
                       kernel("v_add_u32_e32 v2, v0, v2"), o),
            "NO_MISMATCH_FOUND/ESTABLISHED");
  std::filesystem::remove_all(dir);
}

} // namespace
