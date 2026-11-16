// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file llvm_plugin_test.cpp
/// @brief LLVM plugin tests: run only when an LLVM toolchain is available at configure time.

#include "case_io.h"
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

} // namespace
