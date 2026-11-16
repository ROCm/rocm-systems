// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file llvm_plugin_test.cpp
/// @brief LLVM plugin tests: run only when an LLVM toolchain is available at configure time.

#include "case_io.h"
#include "kernel_model.h"
#include "llvm_frontend.h"

#include <gtest/gtest.h>

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
  unsigned return_vgprs = 0;
  int entry_vgprs = -2; ///< -2: the loader's default.
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

std::string verdict_of(const std::string &dir, const std::string &fn, const std::string &old_s,
                       const std::string &new_s, const PairOptions &o) {
  const auto tc = toolchain();
  Artifact a[2];
  for (int side = 0; side < 2; ++side) {
    const std::string path = dir + (side ? "/new.s" : "/old.s");
    EXPECT_TRUE(write_text_file(path, side ? new_s : old_s));
    llvm::Provenance p;
    std::string error;
    EXPECT_TRUE(llvm::assemble(tc, path, dir + (side ? "/new" : "/old"), fn, a[side], p, error))
        << error;
    EXPECT_TRUE(a[side].callable);
    a[side].return_vgprs = o.return_vgprs;
    a[side].return_masks = o.return_masks;
    if (o.entry_vgprs != -2)
      a[side].entry_vgprs = o.entry_vgprs;
  }
  std::string error;
  auto data = analyze_artifacts(a[0], a[1], {}, error);
  EXPECT_TRUE(data) << error;
  if (!data)
    return "error";
  TestOptions opt;
  opt.cases = 1000;
  opt.strict_nan = o.strict_nan;
  opt.poison_filter = o.poison_filter;
  Json report = test_case(*data, opt, "", nullptr);
  return report["overall"]["test_status"].as_string() + "/" +
         report["overall"]["contract_status"].as_string();
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

} // namespace
