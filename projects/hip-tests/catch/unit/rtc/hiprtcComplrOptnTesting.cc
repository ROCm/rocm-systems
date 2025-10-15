/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip/hiprtc.h>
#include <hip/hip_fp16.h>
#include <picojson.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "headers/RtcUtility.h"
#include "headers/RtcFunctions.h"
#include "headers/RtcKernels.h"
#include "headers/printf_common.h"

/*
Unit_hiprtcSingleComplrOptnTst is a test scenario which validates each
HIPRTC supported compiler option idividually.
*/
// SINGLE COMPILER OPTION TESTING
const char** null = {};
TEST_CASE("Unit_hiprtcGpuArchComplrOptnTst") {
  INFO("Testing '--gpu-architecture=gfx906:sramecc+:xnack-' compiler opt")
  REQUIRE(check_architecture(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcGpuRdcComplrOptnTst") {
  INFO("Testing '-fgpu-rdc' compiler option")
  REQUIRE(check_rdc(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcEnabledDenormalsComplrOptnTst") {
  INFO("Testing '-fgpu-flush-denormals-to-zero' compiler option")
  REQUIRE(check_denormals_enabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDisabledDenormalsComplrOptnTst") {
  INFO("Testing '-fno-gpu-flush-denormals-to-zero' compiler option")
  REQUIRE(check_denormals_disabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcOff_ffpContractComplrOptnTst") {
  INFO("Testing '-ffp-contract=off' compiler option")
  REQUIRE(check_ffp_contract_off(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcOnffpContractComplrOptnTst") {
  INFO("Testing '-ffp-contract=on' compiler option")
  REQUIRE(check_ffp_contract_on(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcFastffpContractComplrOptnTst") {
  INFO("Testing '-ffp-contract=fast' compiler option")
  REQUIRE(check_ffp_contract_fast(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcEnabledFastMathComplrOptnTst") {
  INFO("Testing '-ffast-math' compiler option")
  REQUIRE(check_fast_math_enabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDisabledFastMathComplrOptnTst") {
  INFO("Testing '-fno-fast-math' compiler option")
  REQUIRE(check_fast_math_disabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcEnabledSlpVectorizeComplrOptnTst") {
  INFO("Testing '-fslp-vectorize' compiler option")
  REQUIRE(check_slp_vectorize_enabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDisabledSlpVectorizeComplrOptnTst") {
  INFO("Testing '-fno-slp-vectorize' compiler option")
  REQUIRE(check_slp_vectorize_disabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDefineMacroComplrOptnTst") {
  INFO("Testing '-D' compiler option")
  REQUIRE(check_macro(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcUndefMacroComplrOptnTst") {
  INFO("Testing '-U' compiler option")
  REQUIRE(check_undef_macro(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcHeaderDirectoryComplrOptnTst") {
  INFO("Testing '-I' compiler option")
  REQUIRE(check_header_dir(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcWarningComplrOptnTst") {
  INFO("Testing '-w' compiler option")
  REQUIRE(check_warning(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcRpassInlineComplrOptnTst") {
  INFO("Testing '-Rpass=inline' compiler option")
  REQUIRE(check_Rpass_inline(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcEnabledConversionErrComplrOptnTst") {
  INFO("Testing '-Werror=conversion' compiler option")
  REQUIRE(check_conversionerror_enabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDisabledConversionErrComplrOptnTst") {
  INFO("Testing '-Wno-error=conversion' compiler option")
  REQUIRE(check_conversionerror_disabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcEnabledConversionWarningComplrOptnTst") {
  INFO("Testing '-Wconversion' compiler option")
  REQUIRE(check_conversionwarning_enabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDisabledConversionWarningComplrOptnTst") {
  INFO("Testing '-Wno-conversion' compiler option")
  REQUIRE(check_conversionwarning_disabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcGpuMaxThreadPerBlockComplrOptnTst") {
  INFO("Testing '--gpu-max-threads-per-block=n' compiler option")
  REQUIRE(check_max_thread(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcEnabledUnsafeAtomicComplrOptnTst") {
  INFO("Testing '-munsafe-fp-atomics' compiler option")
  REQUIRE(check_unsafe_atomic_enabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDisabledUnsafeAtomicComplrOptnTst") {
  INFO("Testing '-mno-unsafe-fp-atomics' compiler option")
  REQUIRE(check_unsafe_atomic_disabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcEnabledInfiniteNumComplrOptnTst") {
  INFO("Testing '-fhonor-infinities' compiler option")
  REQUIRE(check_infinite_num_enabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDisabledInfiniteNumComplrOptnTst") {
  INFO("Testing '-fno-honor-infinities' compiler option")
  REQUIRE(check_infinite_num_disabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcEnabledNANComplrOptnTst") {
  INFO("Testing '-fhonor-nans' compiler option")
  REQUIRE(check_NAN_num_enabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDisabledNANComplrOptnTst") {
  INFO("Testing '-fno-honor-nans' compiler option")
  REQUIRE(check_NAN_num_disabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcEnabledFiniteMathComplrOptnTst") {
  INFO("Testing '-ffinite-math-only' compiler option")
  REQUIRE(check_finite_math_enabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDisabledFiniteMathComplrOptnTst") {
  INFO("Testing '-fno-finite-math-only' compiler option")
  REQUIRE(check_finite_math_disabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcEnabledAssociativeMathComplrOptnTst") {
  INFO("Testing '-fassociative-math' compiler option")
  REQUIRE(check_associative_math_enabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDisabledAssociativeMathComplrOptnTst") {
  INFO("Testing '-fno-associative-math' compiler option")
  REQUIRE(check_associative_math_disabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcEnabledSignedZerosComplrOptnTst") {
  INFO("Testing '-fsigned-zeros' compiler option")
  REQUIRE(check_signed_zeros_enabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDisabledSignedZerosComplrOptnTst") {
  INFO("Testing '-fno-signed-zeros' compiler option")
  REQUIRE(check_signed_zeros_disabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcEnabledTrappingMathComplrOptnTst") {
  INFO("Testing '-ftrapping-math' compiler option")
  REQUIRE(check_trapping_math_enabled(null, -1, -1, -1));
}

TEST_CASE("Unit_hiprtcDisabledTrappingMathComplrOptnTst") {
  INFO("Testing '-fno-trapping-math' compiler option")
  REQUIRE(check_trapping_math_disabled(null, -1, -1, -1));
}

/*
Unit_hiprtcCombiComplrOptnTst is a test scenario which validates
a combination of HIPRTC supported compiler options which a retrieved from
RtcConfig.jason file.
*/

TEST_CASE("Unit_hiprtcCombiComplrOptnTst") {
  // COMBINATION COMPILER OPTIONS
  std::vector<std::string> CombiCompOptions = get_combi_string_vec();
  int TotalCombos = CombiCompOptions.size();
  REQUIRE(TotalCombos != -1);
  /*
  use '-Werror=conversion' and '-Wconversion' compiler option individually as
  the generate ERROR and WARNING message which might effect when used in
  combination.

  These can be used only if the ERROR and WARNING messages are required.
  '-fgpu-rdc' has to be tested in ISOLATION, cannot be validated with
  combi compiler options.
  */
  int TotalErrors = 0;
  for (int i = 0; i < TotalCombos; i++) {
    std::string one_combi = CombiCompOptions[i];
    TotalErrors += split_comb_string(one_combi);
  }
  if (TotalErrors) {
    WARN("TOTAL FAILED CASES : " << TotalErrors);
  }
  REQUIRE(!TotalErrors);
}
