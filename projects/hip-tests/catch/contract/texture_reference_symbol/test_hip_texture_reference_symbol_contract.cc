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
//
// The legacy `texture<>` template and the tex-ref bind/query entry points were
// removed in CUDA 12, so this whole domain is gated to the AMD backend (or a
// CUDA runtime older than 12), matching the guard used by the unit texture-
// reference tests. On CUDA 12+/NVIDIA the translation unit compiles empty. The
// backend portion uses HT_AMD/HT_NVIDIA for consistency with the rest of the
// suite; the CUDA-version dependency is guarded behind defined(CUDA_VERSION).
#if HT_AMD || (HT_NVIDIA && defined(CUDA_VERSION) && CUDA_VERSION < CUDA_12000)

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

int CurrentDevice() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

hipChannelFormatDesc FloatChannel() {
  return hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
}

// HIPRTC-compiled module that declares a `texture<>` global named "tex" so
// hipModuleGetTexRef can resolve a module-backed reference. Returns false when
// HIPRTC is unavailable so the caller can skip. No runtime header is included in
// the source: HIPRTC's compiler (comgr) implicitly provides the `texture<>`
// template and hipReadModeElementType, and cannot resolve a
// `#include <hip/hip_runtime.h>` from its in-memory input (it fails with "file
// not found"). This whole domain is AMD-only (see the file-scope guard), so the
// AMD HIPRTC behavior is the only one that applies.
constexpr char const kModuleSource[] =
    "texture<float, 1, hipReadModeElementType> tex;\n"
    "extern \"C\" __global__ void touch() {}\n";

// A 2D module texture named "tex" for the mipmapped-array round-trip: mipmapped
// arrays are 2D fixtures, and the deprecated set/get pair requires a 2D-typed
// reference (a 1D reference is rejected with hipErrorInvalidTexture).
constexpr char const kModuleSource2D[] =
    "texture<float, 2, hipReadModeElementType> tex;\n"
    "extern \"C\" __global__ void touch() {}\n";

bool CompileModuleSourceText(const char* source, std::vector<char>& code) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, source, "texture_reference_symbol_contract.cu", 0,
                                   nullptr, nullptr));
#if HT_AMD
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, CurrentDevice()));
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

// Compiles the default 1D-texture module used by the address/array round-trip.
bool CompileModuleSource(std::vector<char>& code) {
  return CompileModuleSourceText(kModuleSource, code);
}

// Compiles the 2D-texture module used by the mipmapped-array round-trip.
bool CompileModuleSource2D(std::vector<char>& code) {
  return CompileModuleSourceText(kModuleSource2D, code);
}
}  // namespace

// ---------------------------------------------------------------------------
// Static-symbol route: a file-scope texture<> global is accepted by the bind
// and query entry points that reject a stack reference.
// ---------------------------------------------------------------------------

// hipGetTextureReference resolves a registered texture symbol to a usable
// reference handle.
// @asserts: hipGetTextureReference - resolves a registered file-scope texture symbol to a non-null reference handle
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
// @asserts: hipBindTexture - binds/unbinds linear memory to a registered 1D texture symbol with a zero alignment offset
HIP_TEST_CASE(Contract_TextureReferenceSymbol_BindUnbindLinearMemory_Succeeds) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  constexpr size_t kBytes = 4096;
  void* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, kBytes));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

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
// @asserts: hipBindTexture2D - binds pitched device memory to a registered 2D texture symbol
HIP_TEST_CASE(Contract_TextureReferenceSymbol_BindTexture2D_Succeeds) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  constexpr size_t kWidth = 64;
  constexpr size_t kHeight = 64;
  void* device_ptr = nullptr;
  size_t pitch = 0;
  HIP_CHECK(hipMallocPitch(&device_ptr, &pitch, kWidth * sizeof(float), kHeight));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

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
// @asserts: hipBindTextureToArray - binds a HIP array to a registered 2D texture symbol
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
  cleanup.Add([array] { (void)hipFreeArray(array); });

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
// @asserts: hipTexRefSetAddress - module-backed texref round-trips a bound device address and HIP array through set/get
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
  cleanup.Add([module] { (void)hipModuleUnload(module); });

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
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

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
  cleanup.Add([array] { (void)hipFreeArray(array); });

  HIP_CHECK(hipTexRefSetArray(reference, array, HIP_TRSA_OVERRIDE_FORMAT));
  hipArray_t bound_array = nullptr;
  HIP_CHECK(hipTexRefGetArray(&bound_array, reference));
  REQUIRE(bound_array == array);
}

// ---------------------------------------------------------------------------
// Deprecated-stub contracts: the mipmap-parameter getters do not read back their
// set value on the AMD backend by design. The contract is the documented stub
// behavior, not a CUDA-style round-trip.
//
// hipTexRefSetBorderColor/hipTexRefGetBorderColor are intentionally NOT tested:
// the AMD runtime implements both with an unconditional assert(false) after the
// image-support check (clr/hipamd/src/hip_texture.cpp), so calling them aborts
// the whole test binary in assert-enabled builds. There is no defined contract
// to assert until the runtime stores border-color state on the reference.
// ---------------------------------------------------------------------------

// The mipmap-parameter getters are stubbed to report hipErrorInvalidValue on this
// backend regardless of any set value. The contract is this documented rejection.
// @asserts: hipTexRefGetMipmapFilterMode - mipmap-parameter getters return hipErrorInvalidValue on this backend regardless of set value
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

// The mipmap-parameter setters write their value into the textureReference and
// report success on an image-capable device. Unlike the bind/array entry points,
// these operate on a bare stack reference (no registered device symbol needed):
// the runtime writes the field directly (clr/hipamd/src/hip_texture.cpp) after
// the image-support gate. The value is observable through the matching getter's
// out-parameter, which the runtime populates before returning its documented
// hipErrorInvalidValue sentinel, so the set value round-trips even though the
// getter's return code is the stub error. A null reference is rejected with
// hipErrorInvalidValue before the image gate, so that check is backend-neutral.
// @asserts: hipTexRefSetMipmapFilterMode - mipmap-parameter setters write values observable via getters and reject a null reference
HIP_TEST_CASE(Contract_TextureReferenceSymbol_MipmapParameterSetters_WriteAndReject) {
  // Null-reference rejection is checked first because it does not depend on image
  // support (the runtime null-checks before the device image-capability gate).
  REQUIRE(hipTexRefSetMipmapFilterMode(nullptr, hipFilterModeLinear) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipTexRefSetMipmapLevelBias(nullptr, 1.0f) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipTexRefSetMipmapLevelClamp(nullptr, 0.0f, 1.0f) == hipErrorInvalidValue);
  (void)hipGetLastError();

  CHECK_IMAGE_SUPPORT;

  textureReference reference{};

  const hipError_t filter_set = hipTexRefSetMipmapFilterMode(&reference, hipFilterModeLinear);
  if (IsUnsupported(filter_set)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefSetMipmapFilterMode is not supported by this runtime path.");
  }
  REQUIRE(filter_set == hipSuccess);
  // The getter writes the stored value into the out-parameter before returning
  // its documented hipErrorInvalidValue, so the set value is observable here.
  hipTextureFilterMode read_filter = hipFilterModePoint;
  (void)hipTexRefGetMipmapFilterMode(&read_filter, &reference);
  (void)hipGetLastError();
  REQUIRE(read_filter == hipFilterModeLinear);

  constexpr float kBias = 2.5f;
  REQUIRE(hipTexRefSetMipmapLevelBias(&reference, kBias) == hipSuccess);
  float read_bias = -1.0f;
  (void)hipTexRefGetMipmapLevelBias(&read_bias, &reference);
  (void)hipGetLastError();
  REQUIRE(read_bias == kBias);

  constexpr float kMinClamp = 1.0f;
  constexpr float kMaxClamp = 7.0f;
  REQUIRE(hipTexRefSetMipmapLevelClamp(&reference, kMinClamp, kMaxClamp) == hipSuccess);
  float read_min = -1.0f;
  float read_max = -1.0f;
  (void)hipTexRefGetMipmapLevelClamp(&read_min, &read_max, &reference);
  (void)hipGetLastError();
  REQUIRE(read_min == kMinClamp);
  REQUIRE(read_max == kMaxClamp);
}

// ---------------------------------------------------------------------------
// Mipmapped-array texref contracts: a module-backed reference round-trips a
// mipmapped array handle through the deprecated set/get entry points, and the
// runtime bind entry point accepts a mipmapped array bound to a static symbol.
// ---------------------------------------------------------------------------

// The deprecated set/get mipmapped-array pair must round-trip on a module-backed
// reference: after hipTexRefSetMipmappedArray binds the array,
// hipTexRefGetMipMappedArray must return the same handle. A stack reference is
// rejected with hipErrorInvalidSymbol, so the module route (which yields a
// registered reference) is used, matching the address/array round-trip above.
// @asserts: hipTexRefSetMipmappedArray - module-backed texref round-trips a bound mipmapped array handle through set/get
HIP_TEST_CASE(Contract_TextureReferenceSymbol_ModuleTexRef_MipmappedArrayRoundTrip) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  std::vector<char> code;
  if (!CompileModuleSource2D(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }

  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  REQUIRE(module != nullptr);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  textureReference* reference = nullptr;
  const hipError_t ref_status = hipModuleGetTexRef(&reference, module, "tex");
  if (IsUnsupported(ref_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipModuleGetTexRef is not supported by this runtime path.");
  }
  HIP_CHECK(ref_status);
  REQUIRE(reference != nullptr);

  // Create a small float mipmapped array; where the device/runtime path does not
  // implement mipmapped arrays this reports unsupported and skips. numLevels
  // follows the in-tree unit-test fixture (2*mipmap_level with mipmap_level=2).
  const hipChannelFormatDesc channel = FloatChannel();
  const hipExtent extent = make_hipExtent(256, 256, 0);
  hipMipmappedArray_t mipmap = nullptr;
  const hipError_t alloc_status =
      hipMallocMipmappedArray(&mipmap, &channel, extent, 4, hipArrayDefault);
  if (IsUnsupported(alloc_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("Mipmapped arrays are not supported by this device/runtime path.");
  }
  HIP_CHECK(alloc_status);
  cleanup.Add([mipmap] { (void)hipFreeMipmappedArray(mipmap); });

  // Normalized coordinates are required before a mipmapped array is bound to the
  // reference (the runtime rejects the bind as an invalid texture otherwise).
  HIP_CHECK(hipTexRefSetFlags(reference, HIP_TRSF_NORMALIZED_COORDINATES));

  // Bind the mipmapped array to the reference, then read the bound handle back.
  const hipError_t set_status =
      hipTexRefSetMipmappedArray(reference, mipmap, HIP_TRSA_OVERRIDE_FORMAT);
  if (IsUnsupported(set_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefSetMipmappedArray is not supported by this runtime path.");
  }
  HIP_CHECK(set_status);

  hipMipmappedArray_t bound = nullptr;
  const hipError_t get_status = hipTexRefGetMipMappedArray(&bound, reference);
  if (IsUnsupported(get_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefGetMipMappedArray is not supported by this runtime path.");
  }
  HIP_CHECK(get_status);
  REQUIRE(bound == mipmap);
}

// hipBindTextureToMipmappedArray is intentionally NOT covered here: on the AMD
// Linux runtime it rejects a mipmapped array bound to a texture reference with
// hipErrorInvalidTexture (the runtime's positive bind path is Windows-only, as
// the in-tree unit test hipBindTextureToMipmappedArray.cc guards it behind
// #if defined(_WIN32)). There is no device-side positive bind contract to assert
// on this platform, so only the deprecated set/get round-trip above is exercised.

// hipTexRefSetAddress2D binds a pitched 2D device allocation to a module-backed
// reference through a HIP_ARRAY_DESCRIPTOR. There is no hipTexRefGetAddress2D
// getter, so the contract is that the bind is accepted; the base device pointer
// it records is cross-checked through hipTexRefGetAddress (which returns the
// bound linear address). A 2D-typed reference is required (a 1D reference is
// rejected), matching the mipmapped-array round-trip above.
// @asserts: hipTexRefSetAddress2D - binds a pitched 2D allocation to a module-backed texref and records the base device address
HIP_TEST_CASE(Contract_TextureReferenceSymbol_ModuleTexRef_SetAddress2D_IsAccepted) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  std::vector<char> code;
  if (!CompileModuleSource2D(code)) {
    HIP_SKIP_TEST("HIPRTC compilation is not supported by this device/runtime path.");
  }

  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  REQUIRE(module != nullptr);
  cleanup.Add([module] { (void)hipModuleUnload(module); });

  textureReference* reference = nullptr;
  const hipError_t ref_status = hipModuleGetTexRef(&reference, module, "tex");
  if (IsUnsupported(ref_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipModuleGetTexRef is not supported by this runtime path.");
  }
  HIP_CHECK(ref_status);
  REQUIRE(reference != nullptr);

  // A pitched 2D float allocation is the operand for the 2D address bind.
  constexpr size_t kWidth = 256;
  constexpr size_t kHeight = 256;
  void* device_ptr = nullptr;
  size_t pitch = 0;
  HIP_CHECK(hipMallocPitch(&device_ptr, &pitch, kWidth * sizeof(float), kHeight));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipTexRefSetFormat(reference, HIP_AD_FORMAT_FLOAT, 1));

  HIP_ARRAY_DESCRIPTOR descriptor{};
  descriptor.Width = kWidth;
  descriptor.Height = kHeight;
  descriptor.Format = HIP_AD_FORMAT_FLOAT;
  descriptor.NumChannels = 1;

  const hipError_t set_status = hipTexRefSetAddress2D(
      reference, &descriptor, reinterpret_cast<hipDeviceptr_t>(device_ptr), pitch);
  if (IsUnsupported(set_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefSetAddress2D is not supported by this runtime path.");
  }
  HIP_CHECK(set_status);

  // No 2D getter exists; cross-check the recorded base address through the linear
  // getter. Where that getter is not implemented for a 2D binding it reports
  // unsupported, which is a capability skip rather than a contract failure.
  hipDeviceptr_t bound = 0;
  const hipError_t get_status = hipTexRefGetAddress(&bound, reference);
  if (IsUnsupported(get_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefGetAddress is not supported for a 2D binding on this runtime path.");
  }
  HIP_CHECK(get_status);
  REQUIRE(bound == reinterpret_cast<hipDeviceptr_t>(device_ptr));
}

#endif  // HT_AMD || (HT_NVIDIA && CUDA_VERSION < CUDA_12000)
