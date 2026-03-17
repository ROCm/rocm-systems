/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @file hipGLContextSwitch.cc
 * @brief Comprehensive tests for HIP-GL interop when switching between multiple GL contexts.
 *
 * This test file verifies that the HIP GL interop implementation correctly handles
 * scenarios where the application switches between different OpenGL contexts.
 * The implementation uses a shared_mutex to allow parallel operations when the
 * GL context is stable (fast path with shared lock), while serializing context
 * switches (slow path with exclusive lock).
 *
 * Test categories:
 * 1. Basic context switching - single switch between two contexts
 * 2. Multiple context switches - back and forth, rapid switching
 * 3. Implicit context switch detection - APIs detect switches without explicit hipGLGetDevices
 * 4. Sequential operations - multiple operations on stable/switching contexts
 * 5. Edge cases - same context repeated, return to previous context, resource association
 * 6. Data integrity - verify data correctness across switches
 * 7. Resource lifecycle - registration, mapping across switches
 * 8. Stress tests - rapid switching, high load
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <hip/hip_gl_interop.h>

#include "gl_interop_common.hh"

#ifdef _WIN32
#include <windows.h>
#else
#include <GL/glx.h>
#endif

#include <chrono>

namespace {

//==============================================================================
// Test Constants
//==============================================================================

constexpr int kDefaultIterations = 3;
constexpr int kStressIterations = 10;
constexpr int kRapidSwitchIterations = 50;
constexpr int kOperationsPerBatch = 4;
constexpr size_t kLargeBufferSize = 4096 * sizeof(float);  // For tests needing different buffer sizes

//==============================================================================
// Helper Classes
//==============================================================================

// Use a separate once_flag for context switch tests to avoid conflicts
// with the common header's glut_init_flag when multiple windows are needed
static std::once_flag context_switch_glut_init_flag;

static void initGlutForContextSwitch() {
  static char proc_name[] = "";
  static std::array<char*, 2> glut_argv = {proc_name, nullptr};
  static int glut_argc = 1;
  glutInitErrorFunc(&GlutError);  // Use GlutError from gl_interop_common.hh
  glutInit(&glut_argc, glut_argv.data());
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  // Use smaller window size than default (512x512) for better performance
  // in rapid context switching tests that create many windows
  glutInitWindowSize(64, 64);
}

/**
 * @brief RAII wrapper for a GLUT window with its own GL context.
 *
 * Extends the common GLUT initialization pattern with support for multiple
 * windows and context switching. Each window has an associated GL context.
 */
class GLUTWindow {
 public:
  GLUTWindow() {
    std::call_once(context_switch_glut_init_flag, initGlutForContextSwitch);
    window_id_ = glutCreateWindow("");
    REQUIRE(window_id_ > 0);

#ifdef USE_GLEW
    makeCurrent();
    GLenum err = glewInit();
    if (err != GLEW_OK) {
      fprintf(stderr, "GLEW init failed: %s\n", glewGetErrorString(err));
      HipTest::HIP_SKIP_TEST("GLEW Init Failed");
      exit(1);
    }
#endif
  }

  ~GLUTWindow() {
    if (window_id_ > 0) {
      glutDestroyWindow(window_id_);
    }
  }

  void makeCurrent() { glutSetWindow(window_id_); }
  int id() const { return window_id_; }

  GLUTWindow(const GLUTWindow&) = delete;
  GLUTWindow& operator=(const GLUTWindow&) = delete;
  GLUTWindow(GLUTWindow&&) = delete;
  GLUTWindow& operator=(GLUTWindow&&) = delete;

 private:
  int window_id_ = 0;
};

/**
 * @brief Extended GL buffer with configurable size and data operations.
 *
 * Extends GLBufferObject from gl_interop_common.hh with custom size support
 * and data fill/read methods needed for data integrity tests.
 */
class GLBufferWithData {
 public:
  explicit GLBufferWithData(size_t size) : size_(size) {
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, size_, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    REQUIRE(glGetError() == GL_NO_ERROR);
  }

  ~GLBufferWithData() {
    if (vbo_ != 0) {
      glDeleteBuffers(1, &vbo_);
    }
  }

  operator GLuint() const { return vbo_; }
  size_t size() const { return size_; }

  void fill(const void* data, size_t dataSize) {
    REQUIRE(dataSize <= size_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, dataSize, data);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    REQUIRE(glGetError() == GL_NO_ERROR);
  }

  void read(void* data, size_t dataSize) {
    REQUIRE(dataSize <= size_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    void* mapped = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    REQUIRE(mapped != nullptr);
    memcpy(data, mapped, dataSize);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    REQUIRE(glGetError() == GL_NO_ERROR);
  }

  GLBufferWithData(const GLBufferWithData&) = delete;
  GLBufferWithData& operator=(const GLBufferWithData&) = delete;

 private:
  GLuint vbo_ = 0;
  size_t size_;
};
 
// Use GLImageObject from gl_interop_common.hh for texture interop tests
 
/**
 * @brief Result of a buffer interop cycle operation.
 */
struct BufferInteropResult {
  bool success;
  void* devicePtr;
  size_t size;
};

/**
 * @brief Perform a complete buffer interop cycle using GLBufferObject from common.
 *
 * Creates a buffer, registers it, maps it, verifies device pointer,
 * unmaps, and unregisters.
 */
BufferInteropResult performBufferInteropCycle(unsigned int flags = hipGraphicsRegisterFlagsNone) {
  BufferInteropResult result = {false, nullptr, 0};

  GLBufferObject buffer;  // Use common header's buffer class
  hipGraphicsResource* resource = nullptr;

  hipError_t err = hipGraphicsGLRegisterBuffer(&resource, buffer, flags);
  if (err != hipSuccess) return result;

  err = hipGraphicsMapResources(1, &resource, 0);
  if (err != hipSuccess) {
    (void)hipGraphicsUnregisterResource(resource);
    return result;
  }

  void* devPtr = nullptr;
  size_t size = 0;
  err = hipGraphicsResourceGetMappedPointer(&devPtr, &size, resource);
  if (err != hipSuccess || devPtr == nullptr) {
    (void)hipGraphicsUnmapResources(1, &resource, 0);
    (void)hipGraphicsUnregisterResource(resource);
    return result;
  }

  result.devicePtr = devPtr;
  result.size = size;

  err = hipGraphicsUnmapResources(1, &resource, 0);
  if (err != hipSuccess) {
    (void)hipGraphicsUnregisterResource(resource);
    return result;
  }

  err = hipGraphicsUnregisterResource(resource);
  if (err != hipSuccess) return result;

  result.success = true;
  return result;
}
 
 /**
  * @brief Helper to verify HIP GL device enumeration works.
  */
 bool verifyGLDeviceEnumeration() {
   const int device_count = HipTest::getDeviceCount();
   unsigned int gl_device_count = 0;
   std::vector<int> gl_devices(device_count, -1);
 
   hipError_t err = hipGLGetDevices(&gl_device_count, gl_devices.data(),
                                    device_count, hipGLDeviceListAll);
   return (err == hipSuccess && gl_device_count >= 1);
 }
 
 /**
  * @brief Simple kernel for data manipulation on device.
  */
 __global__ void simpleAddKernel(float* data, size_t count, float value) {
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < count) {
     data[idx] += value;
   }
 }
 
 }  // anonymous namespace
 
 //==============================================================================
 // SECTION 1: Basic Context Switching Tests
 //==============================================================================
 
 /**
  * @brief Test basic HIP GL interop with two GL contexts, switching once.
  *
  * Verifies that after switching from context 1 to context 2, GL interop
  * operations correctly use the new context.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_Basic") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   // Create first GL context
   GLUTWindow window1;
   window1.makeCurrent();
 
   INFO("Created first GL context (window " << window1.id() << ")");
   REQUIRE(verifyGLDeviceEnumeration());
 
   // Perform interop operations with first context
   auto result1 = performBufferInteropCycle();
   REQUIRE(result1.success);
   REQUIRE(result1.devicePtr != nullptr);
   INFO("First context: Mapped buffer at " << result1.devicePtr << ", size " << result1.size);
 
   // Create second GL context and switch
   GLUTWindow window2;
   window2.makeCurrent();
 
   INFO("Created second GL context (window " << window2.id() << ")");
   REQUIRE(window1.id() != window2.id());
 
   // Verify interop re-initialization for second context
   REQUIRE(verifyGLDeviceEnumeration());
 
   // Perform interop operations with second context
   auto result2 = performBufferInteropCycle();
   REQUIRE(result2.success);
   REQUIRE(result2.devicePtr != nullptr);
   INFO("Second context: Mapped buffer at " << result2.devicePtr << ", size " << result2.size);
 
   INFO("Successfully used HIP-GL interop with two different GL contexts");
 }
 
 /**
  * @brief Test that different GL contexts get separate interop setups.
  *
  * Verifies that resources registered in one context are truly associated
  * with that context and not shared incorrectly.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_SeparateContextSetups") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   // Create two contexts
   GLUTWindow window1;
   GLUTWindow window2;
 
  // Use context 1: create and use buffer (standard size from common header)
  window1.makeCurrent();
  REQUIRE(verifyGLDeviceEnumeration());

  GLBufferObject buffer1;
  hipGraphicsResource* resource1 = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource1, buffer1, hipGraphicsRegisterFlagsNone));
  REQUIRE(resource1 != nullptr);

  // Switch to context 2: create another buffer (larger size)
  window2.makeCurrent();
  REQUIRE(verifyGLDeviceEnumeration());

  GLBufferWithData buffer2(kLargeBufferSize);
  hipGraphicsResource* resource2 = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource2, buffer2, hipGraphicsRegisterFlagsNone));
  REQUIRE(resource2 != nullptr);

  // Verify sizes match expectations
  HIP_CHECK(hipGraphicsMapResources(1, &resource2, 0));
  void* devPtr2 = nullptr;
  size_t size2 = 0;
  HIP_CHECK(hipGraphicsResourceGetMappedPointer(&devPtr2, &size2, resource2));
  REQUIRE(size2 == kLargeBufferSize);
  HIP_CHECK(hipGraphicsUnmapResources(1, &resource2, 0));

  // Switch back to context 1: verify original resource
  window1.makeCurrent();
  HIP_CHECK(hipGraphicsMapResources(1, &resource1, 0));
  void* devPtr1 = nullptr;
  size_t size1 = 0;
  HIP_CHECK(hipGraphicsResourceGetMappedPointer(&devPtr1, &size1, resource1));
  REQUIRE(size1 == GLBufferObject::kSize);
  HIP_CHECK(hipGraphicsUnmapResources(1, &resource1, 0));
 
   // Cleanup
   window1.makeCurrent();
   HIP_CHECK(hipGraphicsUnregisterResource(resource1));
   window2.makeCurrent();
   HIP_CHECK(hipGraphicsUnregisterResource(resource2));
 }
 
 //==============================================================================
 // SECTION 2: Multiple Context Switch Tests
 //==============================================================================
 
 /**
  * @brief Test switching back and forth between GL contexts multiple times.
  *
  * Verifies that the interop continues to work correctly after multiple
  * context switches in an alternating pattern.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_BackAndForth") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
   REQUIRE(window1.id() != window2.id());
 
   for (int iter = 0; iter < kDefaultIterations; ++iter) {
     INFO("Iteration " << iter + 1 << " of " << kDefaultIterations);
 
     // Use first context
     window1.makeCurrent();
     REQUIRE(verifyGLDeviceEnumeration());
     auto result1 = performBufferInteropCycle();
     REQUIRE(result1.success);
 
     // Use second context
     window2.makeCurrent();
     REQUIRE(verifyGLDeviceEnumeration());
     auto result2 = performBufferInteropCycle();
     REQUIRE(result2.success);
   }
 
   INFO("Successfully switched between GL contexts " << kDefaultIterations << " times");
 }
 
 /**
  * @brief Test rapid context switching under stress.
  *
  * Performs many rapid context switches to verify the locking mechanism
  * handles high-frequency switches correctly.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_RapidSwitching") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
   GLUTWindow window3;
 
   // Rapid switching between 3 contexts
   for (int iter = 0; iter < kRapidSwitchIterations; ++iter) {
     int contextChoice = iter % 3;
 
     switch (contextChoice) {
       case 0: window1.makeCurrent(); break;
       case 1: window2.makeCurrent(); break;
       case 2: window3.makeCurrent(); break;
     }
 
     REQUIRE(verifyGLDeviceEnumeration());
   }
 
   INFO("Successfully completed " << kRapidSwitchIterations << " rapid context switches");
 }
 
 /**
  * @brief Test context switching with different switch patterns.
  *
  * Tests various patterns: sequential, round-robin, random-like.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_Patterns") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   std::vector<std::unique_ptr<GLUTWindow>> windows;
   constexpr int kNumContexts = 4;
 
   for (int i = 0; i < kNumContexts; ++i) {
     windows.push_back(std::make_unique<GLUTWindow>());
   }
 
   SECTION("Sequential pattern: 0-1-2-3-0-1-2-3...") {
     for (int iter = 0; iter < kStressIterations; ++iter) {
       for (int ctx = 0; ctx < kNumContexts; ++ctx) {
         windows[ctx]->makeCurrent();
         REQUIRE(verifyGLDeviceEnumeration());
       }
     }
   }
 
   SECTION("Reverse pattern: 3-2-1-0-3-2-1-0...") {
     for (int iter = 0; iter < kStressIterations; ++iter) {
       for (int ctx = kNumContexts - 1; ctx >= 0; --ctx) {
         windows[ctx]->makeCurrent();
         REQUIRE(verifyGLDeviceEnumeration());
       }
     }
   }
 
   SECTION("Ping-pong pattern: 0-3-1-2-0-3-1-2...") {
     constexpr std::array<int, 4> pattern = {0, 3, 1, 2};
     for (int iter = 0; iter < kStressIterations; ++iter) {
       for (int idx : pattern) {
         windows[idx]->makeCurrent();
         REQUIRE(verifyGLDeviceEnumeration());
       }
     }
   }
 
   SECTION("Same context repeated then switch") {
     for (int ctx = 0; ctx < kNumContexts; ++ctx) {
       // Use same context multiple times
       for (int repeat = 0; repeat < 5; ++repeat) {
         windows[ctx]->makeCurrent();
         REQUIRE(verifyGLDeviceEnumeration());
         auto result = performBufferInteropCycle();
         REQUIRE(result.success);
       }
     }
   }
 }
 
 //==============================================================================
 // SECTION 3: Context Switch Without Explicit hipGLGetDevices
 //==============================================================================
 
 /**
  * @brief Test that GL interop functions detect context switch implicitly.
  *
  * Some applications might not call hipGLGetDevices before other GL interop
  * functions. This test verifies that registration functions also properly
  * detect and handle context switches.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_RegisterWithoutGetDevices") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   // First context - explicit initialization
   GLUTWindow window1;
   window1.makeCurrent();
   REQUIRE(verifyGLDeviceEnumeration());
 
  GLBufferObject buffer1;
  hipGraphicsResource* resource1 = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource1, buffer1, hipGraphicsRegisterFlagsNone));
  HIP_CHECK(hipGraphicsMapResources(1, &resource1, 0));
  HIP_CHECK(hipGraphicsUnmapResources(1, &resource1, 0));
  HIP_CHECK(hipGraphicsUnregisterResource(resource1));

  // Second context - NO explicit hipGLGetDevices call
  GLUTWindow window2;
  window2.makeCurrent();

  INFO("Attempting to register buffer with second context without calling hipGLGetDevices");

  // This should work because hipGraphicsGLRegisterBuffer should detect
  // the context change and re-setup the interop
  GLBufferObject buffer2;
   hipGraphicsResource* resource2 = nullptr;
   HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource2, buffer2, hipGraphicsRegisterFlagsNone));
   REQUIRE(resource2 != nullptr);
 
   HIP_CHECK(hipGraphicsMapResources(1, &resource2, 0));
   void* devPtr = nullptr;
   size_t size = 0;
   HIP_CHECK(hipGraphicsResourceGetMappedPointer(&devPtr, &size, resource2));
   REQUIRE(devPtr != nullptr);
   HIP_CHECK(hipGraphicsUnmapResources(1, &resource2, 0));
   HIP_CHECK(hipGraphicsUnregisterResource(resource2));
 
   INFO("Successfully used GL interop after context switch without hipGLGetDevices");
 }
 
 /**
  * @brief Test that all GL interop APIs detect context switches.
  *
  * Verifies that each API (hipGLGetDevices, hipGraphicsGLRegisterBuffer,
  * hipGraphicsGLRegisterImage, hipGraphicsMapResources) properly detects
  * context switches.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_AllAPIsDetectSwitch") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
 
   // Initialize with context 1
   window1.makeCurrent();
   REQUIRE(verifyGLDeviceEnumeration());
 
  SECTION("hipGraphicsGLRegisterBuffer detects switch") {
    window2.makeCurrent();
    // Don't call hipGLGetDevices, directly try to register
    GLBufferObject buffer;
    hipGraphicsResource* resource = nullptr;
    HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource, buffer, hipGraphicsRegisterFlagsNone));
    REQUIRE(resource != nullptr);
    HIP_CHECK(hipGraphicsUnregisterResource(resource));
  }

  SECTION("hipGraphicsGLRegisterImage detects switch") {
    window2.makeCurrent();
    GLImageObject texture;
    hipGraphicsResource* resource = nullptr;
    HIP_CHECK(hipGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D,
                                         hipGraphicsRegisterFlagsNone));
    REQUIRE(resource != nullptr);
    HIP_CHECK(hipGraphicsUnregisterResource(resource));
  }

  SECTION("hipGraphicsMapResources detects switch") {
    // Register in context 1
    GLBufferObject buffer1;
    hipGraphicsResource* resource1 = nullptr;
    HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource1, buffer1, hipGraphicsRegisterFlagsNone));

    // Switch to context 2 and register new buffer
    window2.makeCurrent();
    GLBufferObject buffer2;
    hipGraphicsResource* resource2 = nullptr;
    HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource2, buffer2, hipGraphicsRegisterFlagsNone));
 
     // Map should work with context 2's buffer
     HIP_CHECK(hipGraphicsMapResources(1, &resource2, 0));
     HIP_CHECK(hipGraphicsUnmapResources(1, &resource2, 0));
 
     // Cleanup
     HIP_CHECK(hipGraphicsUnregisterResource(resource2));
     window1.makeCurrent();
     HIP_CHECK(hipGraphicsUnregisterResource(resource1));
   }
 }
 
//==============================================================================
// SECTION 4: Sequential Operations Tests
//==============================================================================

/**
 * @brief Test sequential GL interop operations on a stable context.
 *
 * Multiple sequential operations are performed when the GL context is stable.
 * This validates the fast path (shared lock) works correctly under load.
 * Note: True multi-threaded GL interop is not possible because OpenGL
 * contexts are thread-local - a context can only be current on one thread.
 */
 TEST_CASE("Unit_hipGL_ContextSwitch_SequentialStableContext") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window;
   window.makeCurrent();
   REQUIRE(verifyGLDeviceEnumeration());
 
   // NOTE: OpenGL contexts are thread-local. A GL context can only be current
   // on ONE thread at a time. All GL operations and HIP GL interop operations
   // (which internally use GL functions) must be called from the thread that
   // has the GL context current.
   //
   // This test validates multiple sequential interop operations on a stable
   // context (no context switching). True multi-threaded GL interop would
   // require each thread to have its own GL context.
 
   int successCount = 0;
   int failureCount = 0;
   constexpr int kTotalOps = kOperationsPerBatch * kStressIterations;
 
  for (int i = 0; i < kTotalOps; ++i) {
    GLBufferObject buffer;
    hipGraphicsResource* resource = nullptr;
 
     hipError_t err = hipGraphicsGLRegisterBuffer(&resource, buffer,
                                                  hipGraphicsRegisterFlagsNone);
     if (err != hipSuccess) {
       ++failureCount;
       continue;
     }
 
     err = hipGraphicsMapResources(1, &resource, 0);
     if (err != hipSuccess) {
       (void)hipGraphicsUnregisterResource(resource);
       ++failureCount;
       continue;
     }
 
     void* devPtr = nullptr;
     size_t size = 0;
     err = hipGraphicsResourceGetMappedPointer(&devPtr, &size, resource);
     if (err != hipSuccess || devPtr == nullptr) {
       (void)hipGraphicsUnmapResources(1, &resource, 0);
       (void)hipGraphicsUnregisterResource(resource);
       ++failureCount;
       continue;
     }
 
     err = hipGraphicsUnmapResources(1, &resource, 0);
     if (err != hipSuccess) {
       (void)hipGraphicsUnregisterResource(resource);
       ++failureCount;
       continue;
     }
 
     err = hipGraphicsUnregisterResource(resource);
     if (err != hipSuccess) {
       ++failureCount;
       continue;
     }
 
     ++successCount;
   }
 
  INFO("Successes: " << successCount << ", Failures: " << failureCount);
  REQUIRE(successCount == kTotalOps);
}
 
/**
 * @brief Test interop operations interleaved with context switches.
 *
 * This test performs interop operations interleaved with context switches
 * on the same thread. OpenGL contexts are thread-local and cannot be made
 * current on multiple threads simultaneously (GLX returns BadAccess if you try).
 *
 * This validates that the HIP GL interop implementation correctly handles
 * context switches and re-establishes interop state when the context changes.
 */
 TEST_CASE("Unit_hipGL_ContextSwitch_InterleavedWithSwitch") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
   window1.makeCurrent();
   REQUIRE(verifyGLDeviceEnumeration());
 
   int switchCount = 0;
   int operationCount = 0;
 
   // Interleave context switches with interop operations on the same thread
   // (OpenGL contexts cannot be shared across threads simultaneously)
   for (int i = 0; i < kStressIterations; ++i) {
     // Switch context
     if (i % 2 == 0) {
       window1.makeCurrent();
     } else {
       window2.makeCurrent();
     }
     ++switchCount;
 
     // Perform interop operations on the now-current context
     for (int j = 0; j < kOperationsPerBatch; ++j) {
       if (verifyGLDeviceEnumeration()) {
         ++operationCount;
       }
     }
   }
 
   INFO("Switches: " << switchCount << ", Operations: " << operationCount);
   REQUIRE(switchCount > 0);
   REQUIRE(operationCount > 0);
 }
 
/**
 * @brief Test rapid switching between multiple contexts with device enumeration.
 *
 * OpenGL contexts are thread-local - a context can only be current on one
 * thread at a time. This test validates rapid context switching on a single
 * thread across 4 different contexts, ensuring HIP GL interop correctly
 * re-establishes state for each context on every switch.
 */
 TEST_CASE("Unit_hipGL_ContextSwitch_RapidMultiContextSwitching") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   std::vector<std::unique_ptr<GLUTWindow>> windows;
   constexpr int kNumContexts = 4;
 
   for (int i = 0; i < kNumContexts; ++i) {
     windows.push_back(std::make_unique<GLUTWindow>());
   }
 
   // Initialize first context
   windows[0]->makeCurrent();
   REQUIRE(verifyGLDeviceEnumeration());
 
   int successCount = 0;
 
   // Rapidly switch between all contexts on the same thread
   // This tests the HIP GL interop's ability to handle context changes
   for (int round = 0; round < kStressIterations; ++round) {
     for (int contextIdx = 0; contextIdx < kNumContexts; ++contextIdx) {
       windows[contextIdx]->makeCurrent();
 
       // Try to enumerate devices (requires GL interop setup for this context)
       if (verifyGLDeviceEnumeration()) {
         ++successCount;
       }
     }
   }
 
   INFO("Successful rapid context switches: " << successCount);
   // All should succeed since we're on a single thread with proper context switching
   REQUIRE(successCount == kStressIterations * kNumContexts);
 }
 
 //==============================================================================
 // SECTION 5: Edge Cases and Error Handling
 //==============================================================================
 
 /**
  * @brief Test behavior when switching to the same context repeatedly.
  *
  * Verifies that switching to the same context (no actual change) works
  * correctly and doesn't cause issues with the locking mechanism.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_SameContextRepeated") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window;
   window.makeCurrent();
   REQUIRE(verifyGLDeviceEnumeration());
 
   // Repeatedly "switch" to the same context
   for (int i = 0; i < kStressIterations; ++i) {
     window.makeCurrent();  // Same context, should use fast path
     auto result = performBufferInteropCycle();
     REQUIRE(result.success);
   }
 
   INFO("Successfully performed " << kStressIterations <<
        " operations on same context without switching");
 }
 
 /**
  * @brief Test interop when switching back to a previously used context.
  *
  * Verifies that returning to a previously used context (A->B->A) works.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_ReturnToPreviousContext") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
 
   // Context A -> B -> A -> B -> A
   for (int cycle = 0; cycle < kDefaultIterations; ++cycle) {
     // Use context A
     window1.makeCurrent();
     REQUIRE(verifyGLDeviceEnumeration());
     auto result1 = performBufferInteropCycle();
     REQUIRE(result1.success);
 
     // Use context B
     window2.makeCurrent();
     REQUIRE(verifyGLDeviceEnumeration());
     auto result2 = performBufferInteropCycle();
     REQUIRE(result2.success);
 
     // Return to context A
     window1.makeCurrent();
     REQUIRE(verifyGLDeviceEnumeration());
     auto result3 = performBufferInteropCycle();
     REQUIRE(result3.success);
   }
 }
 
 /**
  * @brief Test interop with resources created in different contexts.
  *
  * Verifies proper handling when resources are associated with specific
  * GL contexts.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_ResourceContextAssociation") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
 
  // Create buffer in context 1
  window1.makeCurrent();
  REQUIRE(verifyGLDeviceEnumeration());
  GLBufferObject buffer1;

  // Register buffer from context 1
  hipGraphicsResource* resource1 = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource1, buffer1, hipGraphicsRegisterFlagsNone));

  // Create buffer in context 2
  window2.makeCurrent();
  REQUIRE(verifyGLDeviceEnumeration());
  GLBufferObject buffer2;

  // Register buffer from context 2
  hipGraphicsResource* resource2 = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource2, buffer2, hipGraphicsRegisterFlagsNone));
 
   // Use context 2's resource
   HIP_CHECK(hipGraphicsMapResources(1, &resource2, 0));
   void* devPtr2 = nullptr;
   size_t size2 = 0;
   HIP_CHECK(hipGraphicsResourceGetMappedPointer(&devPtr2, &size2, resource2));
   REQUIRE(devPtr2 != nullptr);
   HIP_CHECK(hipGraphicsUnmapResources(1, &resource2, 0));
   HIP_CHECK(hipGraphicsUnregisterResource(resource2));
 
   // Switch back and use context 1's resource
   window1.makeCurrent();
   HIP_CHECK(hipGraphicsMapResources(1, &resource1, 0));
   void* devPtr1 = nullptr;
   size_t size1 = 0;
   HIP_CHECK(hipGraphicsResourceGetMappedPointer(&devPtr1, &size1, resource1));
   REQUIRE(devPtr1 != nullptr);
   HIP_CHECK(hipGraphicsUnmapResources(1, &resource1, 0));
   HIP_CHECK(hipGraphicsUnregisterResource(resource1));
 }
 
 //==============================================================================
 // SECTION 6: Data Integrity Tests
 //==============================================================================
 
 /**
  * @brief Test data integrity across context switches.
  *
  * Verifies that data written to a buffer in one context switch cycle
  * is preserved and can be read correctly.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_DataIntegrity") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
 
   // Test data
   constexpr size_t kFloatCount = 256;
   std::vector<float> inputData(kFloatCount);
   for (size_t i = 0; i < kFloatCount; ++i) {
     inputData[i] = static_cast<float>(i) * 1.5f;
   }
 
  window1.makeCurrent();
  REQUIRE(verifyGLDeviceEnumeration());

  // Create buffer and fill with data
  GLBufferWithData buffer(kFloatCount * sizeof(float));
  buffer.fill(inputData.data(), inputData.size() * sizeof(float));
 
   // Register and map buffer
   hipGraphicsResource* resource = nullptr;
   HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource, buffer, hipGraphicsRegisterFlagsNone));
   HIP_CHECK(hipGraphicsMapResources(1, &resource, 0));
 
   void* devPtr = nullptr;
   size_t size = 0;
   HIP_CHECK(hipGraphicsResourceGetMappedPointer(&devPtr, &size, resource));
   REQUIRE(devPtr != nullptr);
 
   // Modify data on device
   constexpr float kAddValue = 100.0f;
   int blockSize = 256;
   int numBlocks = (kFloatCount + blockSize - 1) / blockSize;
   simpleAddKernel<<<numBlocks, blockSize>>>(static_cast<float*>(devPtr), kFloatCount, kAddValue);
   HIP_CHECK(hipDeviceSynchronize());
 
   HIP_CHECK(hipGraphicsUnmapResources(1, &resource, 0));
 
   // Switch to another context and back
   window2.makeCurrent();
   REQUIRE(verifyGLDeviceEnumeration());
   auto result = performBufferInteropCycle();
   REQUIRE(result.success);
 
   // Switch back to original context
   window1.makeCurrent();
 
   // Read back and verify data
   std::vector<float> outputData(kFloatCount);
   buffer.read(outputData.data(), outputData.size() * sizeof(float));
 
   for (size_t i = 0; i < kFloatCount; ++i) {
     float expected = inputData[i] + kAddValue;
     REQUIRE(outputData[i] == Catch::Approx(expected).epsilon(0.001f));
   }
 
   HIP_CHECK(hipGraphicsUnregisterResource(resource));
   INFO("Data integrity verified after context switch");
 }
 
 /**
  * @brief Test data operations in alternating contexts.
  *
  * Performs data operations alternating between two contexts, verifying
  * each context's data remains independent and correct.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_AlternatingDataOperations") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
 
   constexpr size_t kFloatCount = 128;
   constexpr float kValue1 = 1.0f;
   constexpr float kValue2 = 2.0f;
 
  // Initialize context 1
  window1.makeCurrent();
  REQUIRE(verifyGLDeviceEnumeration());
  std::vector<float> data1(kFloatCount, kValue1);
  GLBufferWithData buffer1(kFloatCount * sizeof(float));
  buffer1.fill(data1.data(), data1.size() * sizeof(float));

  // Initialize context 2
  window2.makeCurrent();
  REQUIRE(verifyGLDeviceEnumeration());
  std::vector<float> data2(kFloatCount, kValue2);
  GLBufferWithData buffer2(kFloatCount * sizeof(float));
  buffer2.fill(data2.data(), data2.size() * sizeof(float));
 
   // Alternating operations
   for (int iter = 0; iter < kDefaultIterations; ++iter) {
     // Modify buffer1
     window1.makeCurrent();
     hipGraphicsResource* res1 = nullptr;
     HIP_CHECK(hipGraphicsGLRegisterBuffer(&res1, buffer1, hipGraphicsRegisterFlagsNone));
     HIP_CHECK(hipGraphicsMapResources(1, &res1, 0));
     void* devPtr1 = nullptr;
     size_t size1 = 0;
     HIP_CHECK(hipGraphicsResourceGetMappedPointer(&devPtr1, &size1, res1));
     simpleAddKernel<<<1, kFloatCount>>>(static_cast<float*>(devPtr1), kFloatCount, 0.1f);
     HIP_CHECK(hipDeviceSynchronize());
     HIP_CHECK(hipGraphicsUnmapResources(1, &res1, 0));
     HIP_CHECK(hipGraphicsUnregisterResource(res1));
 
     // Modify buffer2
     window2.makeCurrent();
     hipGraphicsResource* res2 = nullptr;
     HIP_CHECK(hipGraphicsGLRegisterBuffer(&res2, buffer2, hipGraphicsRegisterFlagsNone));
     HIP_CHECK(hipGraphicsMapResources(1, &res2, 0));
     void* devPtr2 = nullptr;
     size_t size2 = 0;
     HIP_CHECK(hipGraphicsResourceGetMappedPointer(&devPtr2, &size2, res2));
     simpleAddKernel<<<1, kFloatCount>>>(static_cast<float*>(devPtr2), kFloatCount, 0.2f);
     HIP_CHECK(hipDeviceSynchronize());
     HIP_CHECK(hipGraphicsUnmapResources(1, &res2, 0));
     HIP_CHECK(hipGraphicsUnregisterResource(res2));
   }
 
   // Verify data
   window1.makeCurrent();
   std::vector<float> result1(kFloatCount);
   buffer1.read(result1.data(), result1.size() * sizeof(float));
   float expected1 = kValue1 + (kDefaultIterations * 0.1f);
   REQUIRE(result1[0] == Catch::Approx(expected1).epsilon(0.001f));
 
   window2.makeCurrent();
   std::vector<float> result2(kFloatCount);
   buffer2.read(result2.data(), result2.size() * sizeof(float));
   float expected2 = kValue2 + (kDefaultIterations * 0.2f);
   REQUIRE(result2[0] == Catch::Approx(expected2).epsilon(0.001f));
 
   INFO("Alternating data operations verified across context switches");
 }
 
 //==============================================================================
 // SECTION 7: Image/Texture Tests
 //==============================================================================
 
 /**
  * @brief Test GL texture interop across context switches.
  *
  * Verifies that texture resources work correctly when switching contexts.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_TextureInterop") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
 
  // Context 1: register and use a texture
  window1.makeCurrent();
  REQUIRE(verifyGLDeviceEnumeration());

  GLImageObject texture1;
  hipGraphicsResource* texResource1 = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterImage(&texResource1, texture1, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsNone));
  REQUIRE(texResource1 != nullptr);

  HIP_CHECK(hipGraphicsMapResources(1, &texResource1, 0));
  hipArray_t array1 = nullptr;
  HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&array1, texResource1, 0, 0));
  REQUIRE(array1 != nullptr);
  HIP_CHECK(hipGraphicsUnmapResources(1, &texResource1, 0));

  // Context 2: register another texture
  window2.makeCurrent();
  REQUIRE(verifyGLDeviceEnumeration());

  GLImageObject texture2;
  hipGraphicsResource* texResource2 = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterImage(&texResource2, texture2, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsNone));
  REQUIRE(texResource2 != nullptr);
 
   HIP_CHECK(hipGraphicsMapResources(1, &texResource2, 0));
   hipArray_t array2 = nullptr;
   HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&array2, texResource2, 0, 0));
   REQUIRE(array2 != nullptr);
   HIP_CHECK(hipGraphicsUnmapResources(1, &texResource2, 0));
 
   // Cleanup
   HIP_CHECK(hipGraphicsUnregisterResource(texResource2));
   window1.makeCurrent();
   HIP_CHECK(hipGraphicsUnregisterResource(texResource1));
 
   INFO("Successfully tested texture interop across context switches");
 }
 
 /**
  * @brief Test mixed buffer and texture resources across context switches.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_MixedResources") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
 
  // Context 1: buffer
  window1.makeCurrent();
  REQUIRE(verifyGLDeviceEnumeration());

  GLBufferObject buffer1;
  hipGraphicsResource* bufResource = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterBuffer(&bufResource, buffer1, hipGraphicsRegisterFlagsNone));

  // Context 2: texture
  window2.makeCurrent();
  REQUIRE(verifyGLDeviceEnumeration());

  GLImageObject texture2;
  hipGraphicsResource* texResource = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterImage(&texResource, texture2, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsNone));
 
   // Use texture in context 2
   HIP_CHECK(hipGraphicsMapResources(1, &texResource, 0));
   hipArray_t array = nullptr;
   HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&array, texResource, 0, 0));
   REQUIRE(array != nullptr);
   HIP_CHECK(hipGraphicsUnmapResources(1, &texResource, 0));
 
   // Switch back to context 1 and use buffer
   window1.makeCurrent();
   HIP_CHECK(hipGraphicsMapResources(1, &bufResource, 0));
   void* devPtr = nullptr;
   size_t size = 0;
   HIP_CHECK(hipGraphicsResourceGetMappedPointer(&devPtr, &size, bufResource));
   REQUIRE(devPtr != nullptr);
   HIP_CHECK(hipGraphicsUnmapResources(1, &bufResource, 0));
 
   // Cleanup
   HIP_CHECK(hipGraphicsUnregisterResource(bufResource));
   window2.makeCurrent();
   HIP_CHECK(hipGraphicsUnregisterResource(texResource));
 }
 
 //==============================================================================
 // SECTION 8: Stream-Based Operations
 //==============================================================================
 
 /**
  * @brief Test context switching with stream-based operations.
  *
  * Verifies that stream-based mapping/unmapping works correctly
  * across context switches.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_StreamOperations") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
 
   window1.makeCurrent();
   REQUIRE(verifyGLDeviceEnumeration());
 
  // Create stream
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  GLBufferObject buffer1;
  hipGraphicsResource* resource1 = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource1, buffer1, hipGraphicsRegisterFlagsNone));
 
   // Map with stream
   HIP_CHECK(hipGraphicsMapResources(1, &resource1, stream));
 
   void* devPtr = nullptr;
   size_t size = 0;
   HIP_CHECK(hipGraphicsResourceGetMappedPointer(&devPtr, &size, resource1));
   REQUIRE(devPtr != nullptr);
 
   // Unmap with stream
   HIP_CHECK(hipGraphicsUnmapResources(1, &resource1, stream));
   HIP_CHECK(hipStreamSynchronize(stream));
 
  // Switch context and do same with new buffer
  window2.makeCurrent();
  REQUIRE(verifyGLDeviceEnumeration());

  GLBufferObject buffer2;
  hipGraphicsResource* resource2 = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource2, buffer2, hipGraphicsRegisterFlagsNone));

  HIP_CHECK(hipGraphicsMapResources(1, &resource2, stream));
  HIP_CHECK(hipGraphicsResourceGetMappedPointer(&devPtr, &size, resource2));
  REQUIRE(devPtr != nullptr);
  HIP_CHECK(hipGraphicsUnmapResources(1, &resource2, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Cleanup
  HIP_CHECK(hipGraphicsUnregisterResource(resource2));
  window1.makeCurrent();
  HIP_CHECK(hipGraphicsUnregisterResource(resource1));
  HIP_CHECK(hipStreamDestroy(stream));
}
 
 /**
  * @brief Test multiple streams across context switches.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_MultipleStreams") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
 
   constexpr int kNumStreams = 4;
   std::array<hipStream_t, kNumStreams> streams;
 
   for (int i = 0; i < kNumStreams; ++i) {
     HIP_CHECK(hipStreamCreate(&streams[i]));
   }
 
   for (int iter = 0; iter < kDefaultIterations; ++iter) {
    // Context 1
    window1.makeCurrent();
    REQUIRE(verifyGLDeviceEnumeration());

    for (int s = 0; s < kNumStreams; ++s) {
      GLBufferObject buffer;
      hipGraphicsResource* resource = nullptr;
      HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource, buffer, hipGraphicsRegisterFlagsNone));
      HIP_CHECK(hipGraphicsMapResources(1, &resource, streams[s]));
      HIP_CHECK(hipGraphicsUnmapResources(1, &resource, streams[s]));
      HIP_CHECK(hipGraphicsUnregisterResource(resource));
    }

    // Context 2
    window2.makeCurrent();
    REQUIRE(verifyGLDeviceEnumeration());

    for (int s = 0; s < kNumStreams; ++s) {
      GLBufferObject buffer;
      hipGraphicsResource* resource = nullptr;
      HIP_CHECK(hipGraphicsGLRegisterBuffer(&resource, buffer, hipGraphicsRegisterFlagsNone));
      HIP_CHECK(hipGraphicsMapResources(1, &resource, streams[s]));
      HIP_CHECK(hipGraphicsUnmapResources(1, &resource, streams[s]));
      HIP_CHECK(hipGraphicsUnregisterResource(resource));
    }
  }
 
   // Synchronize and cleanup streams
   for (int i = 0; i < kNumStreams; ++i) {
     HIP_CHECK(hipStreamSynchronize(streams[i]));
     HIP_CHECK(hipStreamDestroy(streams[i]));
   }
 }
 
 //==============================================================================
 // SECTION 9: Stress Tests
 //==============================================================================
 
/**
 * @brief High-frequency context switch stress test with buffer operations.
 *
 * Unlike Unit_hipGL_ContextSwitch_RapidSwitching which only tests device
 * enumeration, this test performs full buffer interop cycles on each switch,
 * stressing both context switching and resource management together.
 */
TEST_CASE("Unit_hipGL_ContextSwitch_HighFrequencyStress") {
  if (!HipTest::isImageSupported()) {
    HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
    return;
  }

  constexpr int kNumContexts = 3;
  std::vector<std::unique_ptr<GLUTWindow>> windows;

  for (int i = 0; i < kNumContexts; ++i) {
    windows.push_back(std::make_unique<GLUTWindow>());
  }

  // High-frequency switching with full buffer interop operations
  int successCount = 0;
  for (int iter = 0; iter < kRapidSwitchIterations; ++iter) {
    int contextIdx = iter % kNumContexts;
    windows[contextIdx]->makeCurrent();

    // Perform full buffer interop cycle (not just device enumeration)
    auto result = performBufferInteropCycle();
    if (result.success) {
      ++successCount;
    }
  }

  INFO("Successful buffer operations: " << successCount << " / " << kRapidSwitchIterations);
  REQUIRE(successCount == kRapidSwitchIterations);
}
 
 /**
  * @brief Test many buffers registered across multiple context switches.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_ManyBuffersStress") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
 
   constexpr int kBuffersPerContext = 10;
 
   for (int round = 0; round < kDefaultIterations; ++round) {
    // Context 1: create and use many buffers
    window1.makeCurrent();
    REQUIRE(verifyGLDeviceEnumeration());

    std::vector<std::unique_ptr<GLBufferObject>> buffers1;
    std::vector<hipGraphicsResource*> resources1;

    for (int i = 0; i < kBuffersPerContext; ++i) {
      buffers1.push_back(std::make_unique<GLBufferObject>());
      hipGraphicsResource* res = nullptr;
      HIP_CHECK(hipGraphicsGLRegisterBuffer(&res, *buffers1.back(), hipGraphicsRegisterFlagsNone));
      resources1.push_back(res);
    }

    // Map all
    for (auto res : resources1) {
      HIP_CHECK(hipGraphicsMapResources(1, &res, 0));
    }

    // Unmap all
    for (auto res : resources1) {
      HIP_CHECK(hipGraphicsUnmapResources(1, &res, 0));
    }

    // Unregister all
    for (auto res : resources1) {
      HIP_CHECK(hipGraphicsUnregisterResource(res));
    }

    // Context 2: repeat
    window2.makeCurrent();
    REQUIRE(verifyGLDeviceEnumeration());

    std::vector<std::unique_ptr<GLBufferObject>> buffers2;
    std::vector<hipGraphicsResource*> resources2;

    for (int i = 0; i < kBuffersPerContext; ++i) {
      buffers2.push_back(std::make_unique<GLBufferObject>());
      hipGraphicsResource* res = nullptr;
      HIP_CHECK(hipGraphicsGLRegisterBuffer(&res, *buffers2.back(), hipGraphicsRegisterFlagsNone));
      resources2.push_back(res);
    }
 
     for (auto res : resources2) {
       HIP_CHECK(hipGraphicsMapResources(1, &res, 0));
     }
     for (auto res : resources2) {
       HIP_CHECK(hipGraphicsUnmapResources(1, &res, 0));
     }
     for (auto res : resources2) {
       HIP_CHECK(hipGraphicsUnregisterResource(res));
     }
   }
 
   INFO("Successfully handled many buffers across context switches");
 }
 
 /**
  * @brief Long-running context switch stress test.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_LongRunningStress") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
   GLUTWindow window3;
 
   constexpr int kTotalOperations = 100;
   int successCount = 0;
 
   auto startTime = std::chrono::steady_clock::now();
 
   for (int iter = 0; iter < kTotalOperations; ++iter) {
     GLUTWindow* currentWindow = nullptr;
     switch (iter % 3) {
       case 0: currentWindow = &window1; break;
       case 1: currentWindow = &window2; break;
       case 2: currentWindow = &window3; break;
     }
 
     currentWindow->makeCurrent();
 
     if (verifyGLDeviceEnumeration()) {
       auto result = performBufferInteropCycle();
       if (result.success) {
         ++successCount;
       }
     }
   }
 
   auto endTime = std::chrono::steady_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
 
   INFO("Completed " << successCount << " / " << kTotalOperations <<
        " operations in " << duration.count() << " ms");
   REQUIRE(successCount == kTotalOperations);
 }
 
 //==============================================================================
 // SECTION 10: Resource Flags Tests
 //==============================================================================
 
 /**
  * @brief Test different registration flags across context switches.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_RegistrationFlags") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window1;
   GLUTWindow window2;
 
   std::array<unsigned int, 3> flags = {
     hipGraphicsRegisterFlagsNone,
     hipGraphicsRegisterFlagsReadOnly,
     hipGraphicsRegisterFlagsWriteDiscard
   };
 
  for (auto flag : flags) {
    INFO("Testing flag: " << flag);

    // Context 1
    window1.makeCurrent();
    REQUIRE(verifyGLDeviceEnumeration());

    GLBufferObject buffer1;
    hipGraphicsResource* res1 = nullptr;
    HIP_CHECK(hipGraphicsGLRegisterBuffer(&res1, buffer1, flag));
    HIP_CHECK(hipGraphicsMapResources(1, &res1, 0));
    HIP_CHECK(hipGraphicsUnmapResources(1, &res1, 0));
    HIP_CHECK(hipGraphicsUnregisterResource(res1));

    // Context 2
    window2.makeCurrent();
    REQUIRE(verifyGLDeviceEnumeration());

    GLBufferObject buffer2;
    hipGraphicsResource* res2 = nullptr;
    HIP_CHECK(hipGraphicsGLRegisterBuffer(&res2, buffer2, flag));
    HIP_CHECK(hipGraphicsMapResources(1, &res2, 0));
    HIP_CHECK(hipGraphicsUnmapResources(1, &res2, 0));
    HIP_CHECK(hipGraphicsUnregisterResource(res2));
  }
}
 
 //==============================================================================
 // SECTION 11: Context Destruction Tests
 //==============================================================================
 
 /**
  * @brief Test behavior when a GL context is destroyed and a new one is created.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_ContextDestruction") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   // Create and use first context
   {
     GLUTWindow window1;
     window1.makeCurrent();
     REQUIRE(verifyGLDeviceEnumeration());
     auto result = performBufferInteropCycle();
     REQUIRE(result.success);
   }  // window1 destroyed here
 
   // Create new context after first one is gone
   {
     GLUTWindow window2;
     window2.makeCurrent();
     // Should detect context change and re-setup
     REQUIRE(verifyGLDeviceEnumeration());
     auto result = performBufferInteropCycle();
     REQUIRE(result.success);
   }
 
   INFO("Successfully handled context destruction and recreation");
 }
 
 /**
  * @brief Test multiple context creation/destruction cycles.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_MultipleDestructionCycles") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   for (int cycle = 0; cycle < kDefaultIterations; ++cycle) {
     INFO("Cycle " << cycle + 1 << " of " << kDefaultIterations);
 
     GLUTWindow window;
     window.makeCurrent();
     REQUIRE(verifyGLDeviceEnumeration());
     auto result = performBufferInteropCycle();
     REQUIRE(result.success);
   }  // Window destroyed each iteration
 
   INFO("Successfully completed " << kDefaultIterations << " context destruction cycles");
 }
 
 //==============================================================================
 // SECTION 12: Special Cases
 //==============================================================================
 
 /**
  * @brief Test hipGLGetDevices with different device list types across context switches.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_DeviceListTypes") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   const int device_count = HipTest::getDeviceCount();
   GLUTWindow window1;
   GLUTWindow window2;
 
   auto testDeviceList = [&](hipGLDeviceList listType, const char* name) {
     INFO("Testing device list type: " << name);
 
     window1.makeCurrent();
     unsigned int count1 = 0;
     std::vector<int> devices1(device_count, -1);
     // Note: hipGLDeviceListNextFrame is not supported
     if (listType != hipGLDeviceListNextFrame) {
       HIP_CHECK(hipGLGetDevices(&count1, devices1.data(), device_count, listType));
       REQUIRE(count1 >= 1);
     }
 
     window2.makeCurrent();
     unsigned int count2 = 0;
     std::vector<int> devices2(device_count, -1);
     if (listType != hipGLDeviceListNextFrame) {
       HIP_CHECK(hipGLGetDevices(&count2, devices2.data(), device_count, listType));
       REQUIRE(count2 >= 1);
     }
   };
 
   testDeviceList(hipGLDeviceListAll, "hipGLDeviceListAll");
   testDeviceList(hipGLDeviceListCurrentFrame, "hipGLDeviceListCurrentFrame");
 }
 
 /**
  * @brief Test that the first GL interop operation properly initializes context.
  */
 TEST_CASE("Unit_hipGL_ContextSwitch_FirstOperationInitialization") {
   if (!HipTest::isImageSupported()) {
     HipTest::HIP_SKIP_TEST("Image is not supported on the device. Skipped.");
     return;
   }
 
   GLUTWindow window;
   window.makeCurrent();
 
   // First operation - should trigger initialization
   const int device_count = HipTest::getDeviceCount();
   unsigned int gl_device_count = 0;
   std::vector<int> gl_devices(device_count, -1);
   HIP_CHECK(hipGLGetDevices(&gl_device_count, gl_devices.data(), device_count, hipGLDeviceListAll));
   REQUIRE(gl_device_count >= 1);
 
   // Subsequent operations should work
   auto result = performBufferInteropCycle();
   REQUIRE(result.success);
 
  INFO("First operation successfully initialized GL interop");
}
