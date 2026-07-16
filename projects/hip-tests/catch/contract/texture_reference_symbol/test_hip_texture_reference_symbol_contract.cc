/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

// Deprecated texture-reference (hipTexRef*/bind-texture) contracts that require a
// texture reference backed by a real device texture symbol rather than a stack
// `textureReference`. The bind/array/address entry points gate on the reference
// being registered with the runtime: the runtime rejects a bare stack reference
// with hipErrorInvalidSymbol, but accepts a reference that is either (a) a
// file-scope `texture<>` global auto-registered by the compiler (static route),
// or (b) obtained from a loaded module via hipModuleGetTexRef (module route).
//
// Every test is image-gated via CHECK_IMAGE_SUPPORT: on devices/runtime paths
// without image/array support each test skips (as on the local WSL2 iGPU) and
// runs on image-capable datacenter GPUs. A hipErrorNotSupported at a key call is
// treated as a graceful capability skip.
//
// The deprecated API family sets -Wno-deprecated-declarations globally.

// File-scope device texture globals with EXTERNAL linkage. The compiler emits
// __hipRegisterTexture for these, so the runtime can register and resolve the
// bind/query entry points against them (unlike a bare stack textureReference).
// They must NOT live in an anonymous namespace: internal linkage produces a
// mangled local symbol the runtime cannot register (it aborts with "Cannot
// create GlobalVar Obj for symbol").
texture<float, 1, hipReadModeElementType> g_tex_ref_symbol_1d;
texture<float, 2, hipReadModeElementType> g_tex_ref_symbol_2d;

namespace {
bool IsUnsupported(hipError_t status) { return status == hipErrorNotSupported; }

hipChannelFormatDesc FloatChannel() {
  return hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
}

// HIPRTC-compiled module that declares a `texture<>` global named "tex" so
// hipModuleGetTexRef can resolve a module-backed reference. Returns false when
// HIPRTC is unavailable so the caller can skip.
constexpr char const kModuleSource[] =
    "texture<float, 1, hipReadModeElementType> tex;\n"
    "extern \"C\" __global__ void touch() {}\n";

bool CompileModuleSource(std::vector<char>& code) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, kModuleSource,
                                   "texture_reference_symbol_contract.cu", 0, nullptr, nullptr));
#ifdef __HIP_PLATFORM_AMD__
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  const std::string offload_arch = std::string("--offload-arch=") + properties.gcnArchName;
  const char* options[] = {offload_arch.c_str()};
  const int num_options = 1;
#else
  const std::string fmad = "--fmad=false";
  const char* options[] = {fmad.c_str()};
  const int num_options = 1;
#endif
  const hiprtcResult compile_result = hiprtcCompileProgram(program, num_options, options);
  if (compile_result != HIPRTC_SUCCESS) {
    size_t log_size = 0;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
    std::string log(log_size, '\0');
    if (log_size > 0) {
      HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
    }
    INFO("HIPRTC compile log:\n" << log);
    HIPRTC_CHECK(hiprtcDestroyProgram(&program));
    HIPRTC_CHECK(compile_result);
    return false;
  }
  size_t code_size = 0;
  HIPRTC_CHECK(hiprtcGetCodeSize(program, &code_size));
  code.assign(code_size, 0);
  HIPRTC_CHECK(hiprtcGetCode(program, code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  return true;
}
}  // namespace

// ---------------------------------------------------------------------------
// Static-symbol route: a file-scope texture<> global is accepted by the bind
// and query entry points that reject a stack reference.
// ---------------------------------------------------------------------------

// hipGetTextureReference resolves a registered texture symbol to a usable
// reference handle.
HIP_TEST_CASE(Contract_TextureReferenceSymbol_GetTextureReference_ResolvesSymbol) {
  CHECK_IMAGE_SUPPORT;

  const textureReference* reference = nullptr;
  const hipError_t status = hipGetTextureReference(&reference, &g_tex_ref_symbol_1d);
  if (IsUnsupported(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipGetTextureReference is not supported by this runtime path.");
  }
  HIP_CHECK(status);
  REQUIRE(reference != nullptr);
}

// hipBindTexture binds linear device memory to a registered 1D texture symbol and
// hipUnbindTexture releases it; hipGetTextureAlignmentOffset reports the binding
// alignment offset. All three reject a stack reference but accept the symbol.
HIP_TEST_CASE(Contract_TextureReferenceSymbol_BindUnbindLinearMemory_Succeeds) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  constexpr size_t kBytes = 4096;
  void* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, kBytes));
  cleanup.Add([&] { (void)hipFree(device_ptr); });

  const hipChannelFormatDesc channel = FloatChannel();
  size_t offset = 0;
  const hipError_t status = hipBindTexture(&offset, &g_tex_ref_symbol_1d, device_ptr, &channel, kBytes);
  if (IsUnsupported(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipBindTexture is not supported by this runtime path.");
  }
  HIP_CHECK(status);
  cleanup.Add([&] { (void)hipUnbindTexture(&g_tex_ref_symbol_1d); });

  size_t alignment = 1;
  HIP_CHECK(hipGetTextureAlignmentOffset(&alignment, &g_tex_ref_symbol_1d));
  // A freshly bound base allocation has a zero alignment offset.
  REQUIRE(alignment == 0);
}

// hipBindTexture2D binds pitched device memory to a registered 2D texture symbol.
HIP_TEST_CASE(Contract_TextureReferenceSymbol_BindTexture2D_Succeeds) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  constexpr size_t kWidth = 64;
  constexpr size_t kHeight = 64;
  void* device_ptr = nullptr;
  size_t pitch = 0;
  HIP_CHECK(hipMallocPitch(&device_ptr, &pitch, kWidth * sizeof(float), kHeight));
  cleanup.Add([&] { (void)hipFree(device_ptr); });

  const hipChannelFormatDesc channel = FloatChannel();
  size_t offset = 0;
  const hipError_t status =
      hipBindTexture2D(&offset, &g_tex_ref_symbol_2d, device_ptr, &channel, kWidth, kHeight, pitch);
  if (IsUnsupported(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipBindTexture2D is not supported by this runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipUnbindTexture(&g_tex_ref_symbol_2d));
}

// hipBindTextureToArray binds a HIP array to a registered 2D texture symbol.
HIP_TEST_CASE(Contract_TextureReferenceSymbol_BindTextureToArray_Succeeds) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  const hipChannelFormatDesc channel = FloatChannel();
  hipArray_t array = nullptr;
  const hipError_t alloc_status = hipMallocArray(&array, &channel, 64, 64, hipArrayDefault);
  if (IsUnsupported(alloc_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipMallocArray is not supported by this runtime path.");
  }
  HIP_CHECK(alloc_status);
  cleanup.Add([&] { (void)hipFreeArray(array); });

  const hipError_t status = hipBindTextureToArray(&g_tex_ref_symbol_2d, array, &channel);
  if (IsUnsupported(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipBindTextureToArray is not supported by this runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipUnbindTexture(&g_tex_ref_symbol_2d));
}

// ---------------------------------------------------------------------------
// Module route: a module-derived reference (hipModuleGetTexRef) is accepted by
// the driver-style set/get address and array entry points, which require a
// module-registered reference (they reject both a stack and a static-symbol
// reference).
// ---------------------------------------------------------------------------

// hipModuleGetTexRef resolves a texture declared in a loaded module, and the
// resulting reference round-trips a device address through hipTexRefSetAddress /
// hipTexRefGetAddress and a HIP array through hipTexRefSetArray / hipTexRefGetArray.
HIP_TEST_CASE(Contract_TextureReferenceSymbol_ModuleTexRef_AddressAndArrayRoundTrip) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  std::vector<char> code;
  if (!CompileModuleSource(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }

  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  REQUIRE(module != nullptr);
  cleanup.Add([&] { (void)hipModuleUnload(module); });

  textureReference* reference = nullptr;
  const hipError_t ref_status = hipModuleGetTexRef(&reference, module, "tex");
  if (IsUnsupported(ref_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipModuleGetTexRef is not supported by this runtime path.");
  }
  HIP_CHECK(ref_status);
  REQUIRE(reference != nullptr);

  // Address round-trip: bind a linear device allocation and read the bound
  // pointer back.
  constexpr size_t kBytes = 4096;
  void* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, kBytes));
  cleanup.Add([&] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipTexRefSetAddress(nullptr, reference, reinterpret_cast<hipDeviceptr_t>(device_ptr),
                                kBytes));
  hipDeviceptr_t bound = 0;
  HIP_CHECK(hipTexRefGetAddress(&bound, reference));
  REQUIRE(bound == reinterpret_cast<hipDeviceptr_t>(device_ptr));

  // Array round-trip: set the format the array uses, bind a HIP array, and read
  // the bound array handle back.
  HIP_CHECK(hipTexRefSetFormat(reference, HIP_AD_FORMAT_FLOAT, 1));
  const hipChannelFormatDesc channel = FloatChannel();
  hipArray_t array = nullptr;
  HIP_CHECK(hipMallocArray(&array, &channel, 64, 64, hipArrayDefault));
  cleanup.Add([&] { (void)hipFreeArray(array); });

  HIP_CHECK(hipTexRefSetArray(reference, array, HIP_TRSA_OVERRIDE_FORMAT));
  hipArray_t bound_array = nullptr;
  HIP_CHECK(hipTexRefGetArray(&bound_array, reference));
  REQUIRE(bound_array == array);
}

// ---------------------------------------------------------------------------
// Deprecated-stub contracts: the border-color and mipmap-parameter getters do
// not read back their set value on the AMD backend by design. The contract is
// the documented stub behavior, not a CUDA-style round-trip.
// ---------------------------------------------------------------------------

// hipTexRefSetBorderColor and hipTexRefGetBorderColor both succeed, but the value
// does not round-trip on this backend: the reference has no border-color storage
// (the getter is a documented stub), so the set value is not read back. The
// contract asserts that both calls succeed and that the set value is NOT returned.
HIP_TEST_CASE(Contract_TextureReferenceSymbol_BorderColor_SucceedsWithoutRoundTrip) {
  CHECK_IMAGE_SUPPORT;

  constexpr float kSetValue = 0.25f;
  float border[4] = {kSetValue, 0.5f, 0.75f, 1.0f};
  const hipError_t set_status = hipTexRefSetBorderColor(&g_tex_ref_symbol_1d, border);
  if (IsUnsupported(set_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefSetBorderColor is not supported by this runtime path.");
  }
  HIP_CHECK(set_status);

  float returned[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
  const hipError_t get_status = hipTexRefGetBorderColor(returned, &g_tex_ref_symbol_1d);
  if (IsUnsupported(get_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefGetBorderColor is not supported by this runtime path.");
  }
  HIP_CHECK(get_status);
  // Both calls succeed, but the getter does not round-trip the set value on this
  // backend (the border-color state is not stored on the reference).
  REQUIRE(returned[0] != kSetValue);
}

// The mipmap-parameter getters are stubbed to report hipErrorInvalidValue on this
// backend regardless of any set value. The contract is this documented rejection.
HIP_TEST_CASE(Contract_TextureReferenceSymbol_MipmapParameterGetters_ReturnInvalidValue) {
  CHECK_IMAGE_SUPPORT;

  hipTextureFilterMode filter_mode = hipFilterModePoint;
  const hipError_t filter_status = hipTexRefGetMipmapFilterMode(&filter_mode, &g_tex_ref_symbol_1d);
  if (IsUnsupported(filter_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefGetMipmapFilterMode is not supported by this runtime path.");
  }
  REQUIRE(filter_status == hipErrorInvalidValue);
  (void)hipGetLastError();

  float bias = -1.0f;
  const hipError_t bias_status = hipTexRefGetMipmapLevelBias(&bias, &g_tex_ref_symbol_1d);
  REQUIRE(bias_status == hipErrorInvalidValue);
  (void)hipGetLastError();

  float min_clamp = -1.0f;
  float max_clamp = -1.0f;
  const hipError_t clamp_status = hipTexRefGetMipmapLevelClamp(&min_clamp, &max_clamp, &g_tex_ref_symbol_1d);
  REQUIRE(clamp_status == hipErrorInvalidValue);
  (void)hipGetLastError();
}
