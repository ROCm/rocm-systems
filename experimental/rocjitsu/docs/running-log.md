
## 2026-04-28 session start
- User goal: fix currently failing chained MFMA tests, then add LDS loop example, shared-memory loop matmul without MFMA, and shared-memory loop matmul with MFMA.
- Constraints: keep generated ISA directories read-only; commit after each feature-sized change; avoid hacks; keep working through failures.
- First step: inspect repository state and locate failing chained MFMA tests.
- Reproducing focused chained MFMA tests with rocjitsu_tests gtest filter.
- Focused BinaryTranslatorE2E chained MFMA passed; running HSA dispatch MFMA tests and reading relevant files.
- HSA dispatch MFMA tests both timed out on iter 0; investigating common MFMA->WMMA lowering runtime sequence.
- Static lowering code shows MFMA is converted through v_wmma plus ds_bpermute lane remaps; next checking test harness translation path and available disassembly tools.
- HSA tests dumped translated MFMA code objects under /tmp; disassembling source and translated code to inspect the runtime sequence.
- Disassembly shows translated WMMA as an 8-VGPR destination while current lane-remap code only handles four result VGPRs; checking generated operand metadata.
- Generated metadata confirms RDNA4 WMMA f32 16x16x16 has 256-bit dst/src2 while CDNA MFMA has 128-bit dst/src2; current direct mapping is not width-equivalent.
- Disassembly root cause: liveness picked s[0:1] for EXEC save, clobbering the live C pointer used after the MFMA lowering. Fix will be in SGPR liveness/temp allocation, not a one-off register hack.
- Root metadata gap: CDNA flat/global raw encodings include saddr but generated instruction classes do not expose it to def/use, so liveness misses live scalar base pointers. Next step: patch generator or a generator-derived metadata path.
- Patched codegen to emit implicit FLAT saddr SGPR-pair uses when XML operands omit saddr; regenerating CDNA4 instruction files from artifacts/amdgpu_isa_cdna4.xml.
- Rebuilding hsa_translate_test and rocjitsu_tests after regenerating CDNA4 instruction metadata.
- Initial CDNA4 regeneration missed --use-shared and caused broad generated churn; reran with --use-shared to match repository generation mode.
- Restored accidental regenerated CDNA4 file churn; keeping generator fix but using a non-generated conservative temp allocation fix for current build.
- Applied non-generated fix: MFMA lowering SGPR temps now start at s8 and still require liveness-dead registers, preventing clobber of low kernarg/user pointer SGPRs like s[0:1].
- Rebuild succeeded; rerunning BinaryTranslatorE2E chained MFMA and HSA MFMA dispatch tests.
- Result: BinaryTranslatorE2E.MfmaChainedUnrolledReusesAccumulator passed. HsaTranslateTest.TranslateAndDispatchMfma16x16 and TranslateAndDispatchMfmaChainedUnrolled both passed 10 iterations with 0 mismatches.
- Next: commit MFMA fix, then add an LDS read/write loop fixture and HSA dispatch coverage for loop translation.

## LDS loop feature
- Starting LDS read/write loop fixture. Plan: add HIP kernel under tests/kernels, register it in tests/kernels/CMakeLists.txt, and add focused HSA translated dispatch coverage using group_segment_size.
- Added tests/kernels/lds_loop.hip with a 4-iteration LDS write/barrier/read/barrier loop and deterministic shifted-lane accumulation.

## LDS loop fixture
- Added `tests/kernels/lds_loop.hip`, a one-workgroup LDS read/write loop with barriers on both sides of a loop-carried LDS read.
- Registered the kernel in `tests/kernels/CMakeLists.txt` and added `HsaTranslateTest.TranslateAndDispatchLdsLoop` to compare translated dispatch output against a CPU simulation.
- Next: build `kernel_lds_loop` and run the focused HSA translate dispatch test.
- Built with `ninja -C build kernel_lds_loop tests/hsa_translate_test`.
- Verified with `./build/tests/hsa_translate_test --gtest_filter='HsaTranslateTest.TranslateAndDispatchLdsLoop'`; passed in 71 ms.
- Committing this feature before starting the shared-memory matmul loop without MFMA.
