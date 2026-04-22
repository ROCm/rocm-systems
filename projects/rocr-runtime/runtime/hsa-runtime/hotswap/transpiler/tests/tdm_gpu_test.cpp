// Tests for cross-target TDM (Tensor Data Mover) emulation.
//
// The cross-target VIMAGE handler (`handle_vimage.cpp`) emits calls to
// `salmon_tdm_load_to_lds` / `salmon_tdm_store_from_lds`; the raiser
// link-merges the embedded HIP-authored runtime bitcode (`runtime/tdm.hip`)
// into the kernel module before `verifyModule` runs. This file covers
// three things end-to-end:
//
//   1. `TdmRuntime.LinkerWiring`  — link-only smoke test (no GPU): parse
//      the embedded bitcode into a fresh module + assert the two helper
//      bodies materialise. Catches packaging / blob-generation regressions
//      independent of whether any test kernel uses TDM.
//
//   2. `TdmGpu.CrossTargetCorpus` — codegen-smoke pass over the gfx1250
//      test data dir. Scans every `.hsaco` for the VIMAGE TENSOR
//      encoding byte signature, raises matches to gfx942, asserts the
//      resulting IR contains a call to one of the helper symbols, and
//      (when HIP is available) `hipModuleLoadData`s the lifted HSACO.
//      Intentionally does NOT dispatch — the corpus carries no
//      reference outputs. See `DescriptorCoverage` below for the
//      dispatch-level correctness coverage.
//
//   3. `TdmDescriptorCoverage.*` — parameterised dispatch test on
//      gfx942 hardware. For each (direction x dim) in
//      {load, store} x {1D, 2D, 3D, 4D, 5D}, the harness builds a
//      dense contiguous Tensor Descriptor, lifts a gfx1250 kernel
//      through `runPipeline(gfx1250 -> gfx942)`, dispatches it, and
//      byte-compares the output against the known input pattern.
//      This is the walker's functional regression fence — drives both
//      branches of `handleVIMAGE`'s load/store switch and every level
//      of `walk()`'s nested b/a/z/yy outer loops.
//
// All tests `GTEST_SKIP` cleanly when their preconditions are missing —
// no test data, no hipcc at build time, no GPU — so a developer running
// the suite without all the pieces in place gets clear feedback without
// a false failure.

#include "test_common.hpp"

#include "../code_object_utils.hpp"
#include "../pipeline.hpp"
#include "../tdm_runtime.hpp"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifdef GFX1250_TEST_DATA_DIR
static const char *kGfx1250DataDir = GFX1250_TEST_DATA_DIR;
#else
static const char *kGfx1250DataDir = nullptr;
#endif

namespace {

// True iff `haystack` contains at least one 12-byte VIMAGE TENSOR
// encoding (tensor_load_to_lds or tensor_store_from_lds, _d2 or _d4
// forms). Used as a cheap pre-filter to skip raising kernels that
// don't use TDM.
//
// We scan the raw HSACO byte stream for the 3-byte signature that
// identifies the tensor pseudo op family:
//   * byte+2 == 0x71  (Inst{21:14} low 6 bits of TENSOR op + top 2
//                      bits cleared by the encoding)
//   * byte+3 == 0xd0  (Inst{31:26} == 0x34 VIMAGE encoding class,
//                      with cpol{5}=0)
// The tensor pseudo shares `op=0x71` across _d2 and _d4 variants
// (TableGen emits the same opcode slot; the form is recovered from
// the SGPR/NULL pattern in vaddr2/3). `tensor_store_from_lds` only
// differs in bits 14-13 of dmask (byte+1 == 0x40 vs 0x00) so those
// are not part of the signature. The false-positive rate is bounded
// by the density of other legitimate VIMAGE opcodes sharing the
// same high-6-bit encoding prefix (none in gfx1250 today; adjacent
// image_* ops use different op slots), and a false positive simply
// means `runPipeline` processes a kernel that has no tensor op —
// the IR check downstream still gates the real assertion. The
// mnemonic strings are NOT present in the compiled HSACO (the
// assembler emits machine bytes, not mnemonics), so the older
// string-substring approach was a no-op.
bool hsacoContainsTensorOp(const std::vector<uint8_t> &bytes) {
  if (bytes.size() < 4)
    return false;
  for (size_t i = 0; i + 4 <= bytes.size(); ++i)
    if (bytes[i + 2] == 0x71 && bytes[i + 3] == 0xd0)
      return true;
  return false;
}

std::vector<std::string> listHsacos(const std::string &dir) {
  std::vector<std::string> out;
  DIR *d = opendir(dir.c_str());
  if (!d)
    return out;
  while (struct dirent *e = readdir(d)) {
    std::string name = e->d_name;
    if (name.size() < 6)
      continue;
    if (name.substr(name.size() - 6) != ".hsaco")
      continue;
    out.push_back(dir + "/" + name);
  }
  closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

} // namespace

// ============================================================================
// Link-wiring smoke test (no test kernel required, no GPU required).
// ============================================================================
TEST(TdmRuntime, LinkerWiring) {
  if (!transpiler::tdmRuntimeAvailable())
    GTEST_SKIP() << "TDM runtime bitcode not embedded "
                    "(transpiler built without hipcc).";

  // Build a tiny throwaway module so we can exercise `linkTDMRuntime`
  // without a real raise pass. The module needs a triple/datalayout
  // matching what `raiser.cpp` sets so the linker accepts the merge.
  llvm::LLVMContext C;
  llvm::Module M("tdm_link_probe", C);
  M.setTargetTriple(llvm::Triple("amdgcn-amd-amdhsa"));

  // Declare both helpers (no body) just like the real handler does.
  transpiler::declareTDMLoad(M);
  transpiler::declareTDMStore(M);
  ASSERT_TRUE(transpiler::moduleUsesTDMRuntime(M));

  ASSERT_TRUE(transpiler::linkTDMRuntime(M, "gfx942"))
      << "linkTDMRuntime failed; check transpiler stderr above.";

  // After the link, both helpers must have a body — the bitcode merge
  // is the only thing that can resolve them, so a presence check is
  // also a "we linked the right .bc" check.
  llvm::Function *fl = M.getFunction(transpiler::kTDMLoadSymbol);
  llvm::Function *fs = M.getFunction(transpiler::kTDMStoreSymbol);
  ASSERT_NE(fl, nullptr);
  ASSERT_NE(fs, nullptr);
  EXPECT_FALSE(fl->isDeclaration())
      << "salmon_tdm_load_to_lds has no body after link "
         "— the embedded bitcode does not define it.";
  EXPECT_FALSE(fs->isDeclaration())
      << "salmon_tdm_store_from_lds has no body after link "
         "— the embedded bitcode does not define it.";
}

// ============================================================================
// End-to-end: any gfx1250 test kernel that uses TDM must raise to gfx942
// and the emitted IR must reference the helper symbol. When HIP is
// available we additionally load+dispatch to smoke that codegen succeeds.
// ============================================================================
#ifdef __HIP_PLATFORM_AMD__
class TdmGpu : public GpuTest {};
#else
class TdmGpu : public ::testing::Test {};
#endif

TEST_F(TdmGpu, CrossTargetCorpus) {
  if (!transpiler::tdmRuntimeAvailable())
    GTEST_SKIP() << "TDM runtime bitcode not embedded "
                    "(transpiler built without hipcc).";
  if (!kGfx1250DataDir)
    GTEST_SKIP() << "No GFX1250_TEST_DATA_DIR configured.";

  auto hsacos = listHsacos(kGfx1250DataDir);
  if (hsacos.empty())
    GTEST_SKIP() << "No .hsaco files in " << kGfx1250DataDir;

  int exercised = 0;
  for (const auto &path : hsacos) {
    auto bytes = transpiler::readFile(path);
    if (bytes.empty())
      continue;

    // Cheap pre-filter: skip any kernel that obviously does not use TDM.
    // Avoids spending raise time on kernels that wouldn't contribute to
    // the test's coverage signal.
    if (!hsacoContainsTensorOp(bytes))
      continue;

    auto names = transpiler::listKernelNames(bytes);
    ASSERT_FALSE(names.empty()) << "No kernels in " << path;

    for (const auto &kname : names) {
      auto result = transpiler::runPipeline(bytes, "gfx1250", "gfx942", kname);
      ASSERT_TRUE(result.success)
          << "TDM lift failed for " << kname << " in " << path
          << " (mnemonic=" << result.failMnemonic << ")";
      EXPECT_TRUE(result.irText.find("salmon_tdm_load_to_lds") !=
                      std::string::npos ||
                  result.irText.find("salmon_tdm_store_from_lds") !=
                      std::string::npos)
          << "TDM kernel " << kname
          << " raised but the helper symbol does not appear in the IR.";

#ifdef __HIP_PLATFORM_AMD__
      // Smoke: codegen + module load survives the link-merged helper.
      // No correctness check on the output here — that requires a
      // bespoke per-kernel reference, which the corpus does not carry.
      hipModule_t mod;
      hipError_t err = hipModuleLoadData(&mod, result.hsaco.data());
      EXPECT_EQ(err, hipSuccess)
          << "hipModuleLoadData failed for " << kname << ": "
          << hipGetErrorString(err);
      if (err == hipSuccess)
        (void)hipModuleUnload(mod);
#endif
      ++exercised;
    }
  }

  if (exercised == 0)
    GTEST_SKIP() << "No TDM-using kernels in " << kGfx1250DataDir
                 << " (add a hsaco containing tensor_load_to_lds or "
                    "tensor_store_from_lds to exercise this test).";
  printf("TdmGpu.CrossTargetCorpus exercised %d kernel(s)\n", exercised);
}

// ============================================================================
// Parameterised descriptor-coverage tests.
//
// The {load, store} x {1D..5D} matrix below is the walker's functional
// regression fence. For each (direction, rank) point:
//
//   1. Host builds a dense contiguous Tensor Descriptor where the
//      `rank` innermost dims carry non-trivial (tensor_dim, tile_dim,
//      stride) triples and the higher dims are left zero. Row-major
//      stride convention: S0 = T0, S1 = T0*T1, S2 = T0*T1*T2,
//      S3 = T0*T1*T2*T3.
//   2. For LOAD: host fills `in[tid]` with a known pattern, dispatches
//      `tdm_load_kernel` (which issues `tensor_load_to_lds`, then
//      copies lds[tid] to out[tid]). Expected: out[i] == in[i] for
//      every tile element.
//      For STORE: host fills `in[tid]` with the known pattern, the
//      kernel stages it into LDS and issues `tensor_store_from_lds`,
//      host reads `out[]` back. Expected: out[i] == in[i] for every
//      tile element.
//   3. Pipeline is driven by `transpiler::runPipeline(gfx1250, gfx942)`,
//      so every case exercises the full cross-target lift +
//      `salmon_tdm_{load,store}` emission + bitcode link-merge.
//
// `HIP_ASSERT` escalates any HIP-layer failure to a test failure;
// mismatches are counted and the first few indices are printed so a
// regression points straight at the offending walker branch.
// ============================================================================
#ifdef __HIP_PLATFORM_AMD__
namespace {

enum class TdmDirection { Load, Store };

// Parameter bundle. `rank` is the number of non-trivial dims (1..5);
// `tile[0..rank-1]` are the tile_dim sizes, higher tile dims are
// implicitly 0 (walker normalises to 1). `name` is the gtest
// parameter label.
struct TdmCase {
  const char *name;
  TdmDirection dir;
  int rank;
  uint32_t tile[5];
};

// 4-byte elements (data_size_log2 = 2). Product(tile) <= 64 so a
// single 64-thread workgroup copies the whole tile.
constexpr uint32_t kElementBytes = 4;

// Raw TDM Tensor Descriptor, packed exactly as the gfx1250
// hardware / `runtime/tdm.hip` decoders expect. Host builds this
// in device memory and the kernel reads it via `s_load_b128` /
// `s_load_b256` into SGPRs.
struct TDMDescriptor {
  uint32_t g0[4];
  uint32_t g1[8];
  uint32_t g2[4];
  uint32_t g3[4];
};

// Build a dense contiguous descriptor for `rank` dims with the given
// tile extents. `global_addr` is the byte address of either the
// source (load) or destination (store) buffer — the descriptor's
// semantics are direction-agnostic at this layer.
TDMDescriptor buildDescriptor(uint64_t global_addr, int rank,
                              const uint32_t tile[5]) {
  TDMDescriptor d{};
  const uint32_t T0 = tile[0];
  const uint32_t T1 = (rank >= 2) ? tile[1] : 0u;
  const uint32_t T2 = (rank >= 3) ? tile[2] : 0u;
  const uint32_t T3 = (rank >= 4) ? tile[3] : 0u;
  const uint32_t T4 = (rank >= 5) ? tile[4] : 0u;

  // Row-major strides: S0=T0, S1=T0*T1, S2=T0*T1*T2, S3=T0*T1*T2*T3.
  // A dim that is not in the rank contributes 1 to downstream products
  // so its stride would be well-defined if the walker ever read it,
  // but the walker only reads S_k when tile_dim_{k+1} > 0 — dims
  // outside the rank collapse the outer loop to a single iteration
  // and their stride is never consumed.
  const uint64_t S0 = T0;
  const uint64_t S1 = (uint64_t)T0 * std::max(T1, 1u);
  const uint64_t S2 = S1 * std::max(T2, 1u);
  const uint64_t S3 = S2 * std::max(T3, 1u);

  // Group 0.
  d.g0[0] = 0x00000001u;                          // count=1
  d.g0[1] = 0u;                                   // lds_byte_addr
  d.g0[2] = (uint32_t)(global_addr & 0xffffffffu);
  d.g0[3] = (uint32_t)((global_addr >> 32) & 0x01ffffffu);

  // Group 1.
  d.g1[0] = 2u << 16;                             // data_size_log2 = 2
  d.g1[1] = (T0 & 0xffffu) << 16;                 // tensor_dim0 low 16
  d.g1[2] = ((T0 >> 16) & 0xffffu)                // tensor_dim0 high 16
          | ((T1 & 0xffffu) << 16);               // tensor_dim1 low 16
  d.g1[3] = ((T1 >> 16) & 0xffffu)                // tensor_dim1 high 16
          | ((T0 & 0xffffu) << 16);               // tile_dim0
  d.g1[4] = (T1 & 0xffffu)                        // tile_dim1
          | ((T2 & 0xffffu) << 16);               // tile_dim2
  d.g1[5] = (uint32_t)(S0 & 0xffffffffu);         // tensor_dim0_stride lo32
  d.g1[6] = (uint32_t)((S0 >> 32) & 0xffffu)      // tensor_dim0_stride hi16
          | ((uint32_t)(S1 & 0xffffu) << 16);     // tensor_dim1_stride lo16
  d.g1[7] = (uint32_t)((S1 >> 16) & 0xffffffffu); // tensor_dim1_stride hi48

  // Group 2 (tensor_dim2, tensor_dim3, tensor_dim2_stride,
  // tile_dim3).
  d.g2[0] = T2;                                   // tensor_dim2
  d.g2[1] = T3;                                   // tensor_dim3
  d.g2[2] = (uint32_t)(S2 & 0xffffffffu);         // tensor_dim2_stride lo32
  d.g2[3] = (uint32_t)((S2 >> 32) & 0xffffu)      // tensor_dim2_stride hi16
          | ((T3 & 0xffffu) << 16);               // tile_dim3

  // Group 3 (tensor_dim3_stride, tensor_dim4, tile_dim4).
  d.g3[0] = (uint32_t)(S3 & 0xffffffffu);         // tensor_dim3_stride lo32
  d.g3[1] = (uint32_t)((S3 >> 32) & 0xffffu)      // tensor_dim3_stride hi16
          | ((T4 & 0xffffu) << 16);               // tensor_dim4 low 16
  d.g3[2] = ((T4 >> 16) & 0xffffu)                // tensor_dim4 high 16
          | ((T4 & 0xffffu) << 16);               // tile_dim4
  d.g3[3] = 0;

  return d;
}

// Pattern fill — deliberately data-dependent so a stride miscalc in
// the walker surfaces as a mis-indexed element rather than a
// matches-by-luck all-zeros pass.
uint32_t patternValue(uint32_t i) { return 0xa5c30000u ^ (i * 0x01010101u); }

// Local HIP error check that fits a helper with a non-void return
// type — `HIP_ASSERT` uses GoogleTest's `ASSERT_*` which only works
// from a void-returning test body. Mirrors HIP_ASSERT's diagnostic
// format.
#define TDM_CHECK_HIP(call)                                                    \
  do {                                                                         \
    hipError_t _err = (call);                                                  \
    if (_err != hipSuccess) {                                                  \
      ADD_FAILURE() << "HIP error " << static_cast<int>(_err) << " ("          \
                    << hipGetErrorString(_err) << ") at " << __FILE__ << ":"   \
                    << __LINE__ << ": " << #call;                              \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Run one parameter point through the full raise -> dispatch ->
// compare cycle. Returns the number of mismatches (0 == pass),
// or -1 if any prerequisite step failed before the compare could run.
int runOneCase(const TdmCase &tc, const char *hsaco_path) {
  auto bytes = transpiler::readFile(hsaco_path);
  if (bytes.empty()) {
    ADD_FAILURE() << "cannot read " << hsaco_path;
    return -1;
  }

  const char *kernel_name = (tc.dir == TdmDirection::Load)
                                ? "tdm_load_kernel"
                                : "tdm_store_kernel";

  auto r = transpiler::runPipeline(bytes, "gfx1250", "gfx942", kernel_name);
  if (!r.success) {
    ADD_FAILURE() << "raise failed for " << kernel_name
                  << " (mnemonic=" << r.failMnemonic << ")";
    return -1;
  }
  const char *expected_helper = (tc.dir == TdmDirection::Load)
                                    ? "salmon_tdm_load_to_lds"
                                    : "salmon_tdm_store_from_lds";
  EXPECT_NE(r.irText.find(expected_helper), std::string::npos)
      << "raised IR does not call " << expected_helper;

  // Tile total = product of the non-trivial tile dims.
  uint32_t total = 1;
  for (int i = 0; i < tc.rank; ++i) total *= tc.tile[i];

  // Device buffers. The load kernel reads `in` through the D#, the
  // store kernel writes `out` through the D#; the other buffer is
  // kernarg-visible via the 5th pointer parameter.
  uint32_t *d_in = nullptr;
  uint32_t *d_out = nullptr;
  TDMDescriptor *d_desc = nullptr;
  TDM_CHECK_HIP(hipMalloc(&d_in,   256 * sizeof(uint32_t)));
  TDM_CHECK_HIP(hipMalloc(&d_out,  256 * sizeof(uint32_t)));
  TDM_CHECK_HIP(hipMalloc(&d_desc, sizeof(TDMDescriptor)));

  std::vector<uint32_t> host_in(256, 0xdeadbeefu);
  for (uint32_t i = 0; i < total; ++i) host_in[i] = patternValue(i);
  TDM_CHECK_HIP(hipMemcpy(d_in, host_in.data(), 256 * sizeof(uint32_t),
                       hipMemcpyHostToDevice));
  TDM_CHECK_HIP(hipMemset(d_out, 0xcd, 256 * sizeof(uint32_t)));

  // D# points at whichever of {in, out} the direction reads from
  // through the TDM path. Load reads input via global -> LDS, store
  // writes output via LDS -> global.
  uint64_t tdm_addr = (uint64_t)(uintptr_t)(
      (tc.dir == TdmDirection::Load) ? d_in : d_out);
  TDMDescriptor desc = buildDescriptor(tdm_addr, tc.rank, tc.tile);
  TDM_CHECK_HIP(hipMemcpy(d_desc, &desc, sizeof(desc), hipMemcpyHostToDevice));

  hipModule_t mod;
  TDM_CHECK_HIP(hipModuleLoadData(&mod, r.hsaco.data()));
  hipFunction_t fn;
  TDM_CHECK_HIP(hipModuleGetFunction(&fn, mod, kernel_name));

  struct Args {
    void *g0_ptr;
    void *g1_ptr;
    void *g2_ptr;
    void *g3_ptr;
    uint32_t *extra; // `out` for load direction, `in` for store.
  } args{
      (void *)&d_desc->g0,
      (void *)&d_desc->g1,
      (void *)&d_desc->g2,
      (void *)&d_desc->g3,
      (tc.dir == TdmDirection::Load) ? d_out : d_in,
  };
  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &argSize,
                    HIP_LAUNCH_PARAM_END};
  TDM_CHECK_HIP(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr,
                                   nullptr, config));
  TDM_CHECK_HIP(hipDeviceSynchronize());

  std::vector<uint32_t> host_out(256);
  TDM_CHECK_HIP(hipMemcpy(host_out.data(), d_out, 256 * sizeof(uint32_t),
                       hipMemcpyDeviceToHost));

  int mism = 0;
  for (uint32_t i = 0; i < total; ++i) {
    if (host_out[i] != patternValue(i)) {
      if (mism < 4)
        fprintf(stderr,
                "  [%s rank=%d] mismatch at %u: got 0x%08x expected 0x%08x\n",
                tc.name, tc.rank, i, host_out[i], patternValue(i));
      ++mism;
    }
  }

  (void)hipModuleUnload(mod);
  TDM_CHECK_HIP(hipFree(d_in));
  TDM_CHECK_HIP(hipFree(d_out));
  TDM_CHECK_HIP(hipFree(d_desc));
  return mism;
}

} // namespace

class TdmDescriptorCoverage : public GpuTest,
                              public ::testing::WithParamInterface<TdmCase> {};

TEST_P(TdmDescriptorCoverage, DispatchDenseContiguous) {
  if (!transpiler::tdmRuntimeAvailable())
    GTEST_SKIP() << "TDM runtime bitcode not embedded "
                    "(transpiler built without hipcc).";
  if (!kGfx1250DataDir)
    GTEST_SKIP() << "No GFX1250_TEST_DATA_DIR configured.";

  const TdmCase &tc = GetParam();
  std::string path = std::string(kGfx1250DataDir) + "/" +
                     ((tc.dir == TdmDirection::Load) ? "tdm_load_gfx1250.hsaco"
                                                     : "tdm_store_gfx1250.hsaco");
  struct stat st;
  if (stat(path.c_str(), &st) != 0)
    GTEST_SKIP() << "missing fixture: " << path
                 << " (regenerate via the recipe in "
                    "test_data/gfx1250/tdm_{load,store}_kernel.hip)";

  int mism = runOneCase(tc, path.c_str());
  uint32_t total = 1;
  for (int i = 0; i < tc.rank; ++i) total *= tc.tile[i];
  EXPECT_EQ(mism, 0) << tc.name << " (rank " << tc.rank
                     << ") : " << mism << " / " << total << " mismatches";
}

INSTANTIATE_TEST_SUITE_P(
    AllDirsAndDims, TdmDescriptorCoverage,
    ::testing::Values(
        // Load direction.
        TdmCase{"Load_1D", TdmDirection::Load, 1, {64, 0, 0, 0, 0}},
        TdmCase{"Load_2D", TdmDirection::Load, 2, { 8, 8, 0, 0, 0}},
        TdmCase{"Load_3D", TdmDirection::Load, 3, { 8, 4, 2, 0, 0}},
        TdmCase{"Load_4D", TdmDirection::Load, 4, { 4, 4, 2, 2, 0}},
        TdmCase{"Load_5D", TdmDirection::Load, 5, { 4, 2, 2, 2, 2}},
        // Store direction.
        TdmCase{"Store_1D", TdmDirection::Store, 1, {64, 0, 0, 0, 0}},
        TdmCase{"Store_2D", TdmDirection::Store, 2, { 8, 8, 0, 0, 0}},
        TdmCase{"Store_3D", TdmDirection::Store, 3, { 8, 4, 2, 0, 0}},
        TdmCase{"Store_4D", TdmDirection::Store, 4, { 4, 4, 2, 2, 0}},
        TdmCase{"Store_5D", TdmDirection::Store, 5, { 4, 2, 2, 2, 2}}),
    [](const ::testing::TestParamInfo<TdmCase> &info) {
      return info.param.name;
    });
#endif // __HIP_PLATFORM_AMD__
