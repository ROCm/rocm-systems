/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#ifdef USE_GLEW
#include <GL/glew.h>
#elif defined(__linux__)
#define GL_GLEXT_PROTOTYPES
#else
#error "GLEW is required on non-Linux platforms. Define USE_GLEW and link" \
       " against the GLEW library, or build on Linux."
#endif

#include <GL/freeglut.h>

#ifdef USE_EGL
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

#include <hip_test_common.hh>

class GLBufferObject {
 public:
  static constexpr size_t kSize = 512 * 512 * 4 * sizeof(float);

  GLBufferObject() {
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, kSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    REQUIRE(glGetError() == GL_NO_ERROR);
  }

  ~GLBufferObject() { glDeleteBuffers(1, &vbo_); }

  operator GLuint() const { return vbo_; }

 private:
  GLuint vbo_;
};

class GLImageObject {
 public:
  static constexpr size_t kWidth = 512, kHeight = 512;

  GLImageObject() {
    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, kWidth, kHeight, 0, GL_RGBA_INTEGER_EXT,
                 GL_UNSIGNED_BYTE, NULL);
    REQUIRE(glGetError() == GL_NO_ERROR);
  }

  ~GLImageObject() { glDeleteTextures(1, &tex_); }

  operator GLuint() const { return tex_; }

 private:
  GLuint tex_;
};

class IContextScopeGuard {
public:
  virtual ~IContextScopeGuard() = default;
};

// inline (one instance across all translation units that include this header) so GLUT is
// initialized exactly once per process. A per-TU flag caused glutInit() to run once per TU, which
// on Windows raises "illegal glutInit() reinitialization attempt" and can hang the test.
inline std::once_flag glut_init_flag;
inline bool glut_init_failed = false;
// Single window/context shared by every test. 0 means "not created". Created once in init() and
// kept alive for the whole process: freeglut accumulates native window handles across repeated
// create/destroy, so opening/closing one window per test hangs the window manager (Windows+Linux).
inline int glut_shared_window = 0;
inline void GlutError(const char *fmt, va_list ap)
{
    // Print what error occurred
    fprintf(stderr, "GlutError:");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");

    glut_init_failed = true;
}

// Initialize freeglut exactly once per process. glutInit() must be called only once; calling it
// again raises "illegal glutInit() reinitialization attempt" and can hang the test. ALL GL interop
// test files must go through EnsureGlutInitialized() (including the context-switch tests in
// hipGLContextSwitch.cc) so there is a single glutInit() for the whole process.
inline void GlutInitOnce() {
  static char proc_name[] = "";
  static std::array<char*, 2> glut_argv = {proc_name, nullptr};
  static int glut_argc = 1;
  glutInitErrorFunc(&GlutError);
  glutInit(&glut_argc, glut_argv.data());
  if (glut_init_failed) return;
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
}

inline void EnsureGlutInitialized() { std::call_once(glut_init_flag, GlutInitOnce); }

class GLUTContextScopeGuard : public IContextScopeGuard {
 public:
  GLUTContextScopeGuard() {
    EnsureGlutInitialized();
    if (glut_init_failed) {
      HIP_SKIP_TEST("GLUT Init Failed");
    }
    // Reuse one persistent window/context for every test that uses this guard instead of creating
    // and destroying a window per test. freeglut accumulates native window handles across repeated
    // create/destroy, and opening hundreds of windows over a full run hangs the window manager
    // (observed on Windows and Linux). Tests run serially (Catch2 single-threaded), so this lazy
    // create needs no synchronization.
    if (glut_shared_window == 0) {
      glutInitWindowSize(512, 512);
      glut_shared_window = glutCreateWindow("");
    }
    glutSetWindow(glut_shared_window);
  }

  ~GLUTContextScopeGuard() override {
    // Intentionally do NOT destroy the shared window; it is reused by subsequent tests and torn
    // down when the process exits. Per-test destroy/recreate is what caused the window-count hang.
  }

  GLUTContextScopeGuard(const GLUTContextScopeGuard&) = delete;
  GLUTContextScopeGuard& operator=(const GLUTContextScopeGuard&) = delete;

  GLUTContextScopeGuard(GLUTContextScopeGuard&&) = delete;
  GLUTContextScopeGuard& operator=(GLUTContextScopeGuard&&) = delete;
};

#ifdef USE_EGL
class EGLContextScopeGuard : public IContextScopeGuard {
 public:
  EGLContextScopeGuard() {

    // 1. Initialize EGL
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");

    eglQueryDevicesEXT(egl_devices_.max_size(), egl_devices_.data(), &num_devices_);

    INFO("Detected " << num_devices_ << " devices");

    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

    egl_display_ = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, egl_devices_.at(0), 0);

    REQUIRE(eglInitialize(egl_display_, &major_, &minor_));

    // 2. Select an appropriate configuration
    REQUIRE(eglChooseConfig(egl_display_, kConfigAttribs, &egl_config_, 1, &num_configs_));

    // 3. Create a surface
    egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config_, kPbufferAttribs);

    // 4. Bind the API
    REQUIRE(eglBindAPI(EGL_OPENGL_API));

    // 5. Create a context and make it current
    egl_context_ = eglCreateContext(egl_display_, egl_config_, EGL_NO_CONTEXT, NULL);

    REQUIRE(eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_));
  }

  ~EGLContextScopeGuard() override {
    // 6. Terminate EGL when finished
    eglTerminate(egl_display_);
  }

  EGLContextScopeGuard(const EGLContextScopeGuard&) = delete;
  EGLContextScopeGuard& operator=(const EGLContextScopeGuard&) = delete;

  EGLContextScopeGuard(EGLContextScopeGuard&&) = delete;
  EGLContextScopeGuard& operator=(EGLContextScopeGuard&&) = delete;

 private:
  // clang-format off
  static constexpr EGLint kConfigAttribs[] = {
      EGL_SURFACE_TYPE,
      EGL_PBUFFER_BIT,
      EGL_BLUE_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_RED_SIZE, 8,
      EGL_DEPTH_SIZE, 8,
      EGL_RENDERABLE_TYPE,
      EGL_OPENGL_BIT,
      EGL_NONE
  };
  // clang-format on

  static constexpr int kPbufferWidth = 9;
  static constexpr int kPbufferHeight = 9;

  static constexpr EGLint kPbufferAttribs[] = {
      EGL_WIDTH, kPbufferWidth, EGL_HEIGHT, kPbufferHeight, EGL_NONE,
  };

  std::array<EGLDeviceEXT, 8> egl_devices_;
  EGLint num_devices_;
  EGLDisplay egl_display_;
  EGLint major_, minor_;
  EGLint num_configs_;
  EGLConfig egl_config_;
  EGLSurface egl_surface_;
  EGLContext egl_context_;
};
#endif

class GLContextScopeGuard {
 public:
  using GLContextScopeGuardPtr = std::unique_ptr<IContextScopeGuard>;

  static constexpr char kEnvarName[] = "GL_CONTEXT_TYPE";

  GLContextScopeGuard() {

    char* val = std::getenv(kEnvarName);
    std::string val_str = val == NULL ? "" : val;

    if (val_str.empty() || val_str == "GLUT") {
      gl_context_ = std::make_unique<GLUTContextScopeGuard>();
#ifdef USE_EGL
    } else if (val_str == "EGL") {
      gl_context_ = std::make_unique<EGLContextScopeGuard>();
#endif
    } else {
      INFO("Unsupported " << kEnvarName << " value '" << val_str << "'");
      INFO("Supported values are ['GLUT'"
#ifdef USE_EGL
        << ", 'EGL'"
#endif
        << "]");
      REQUIRE(false);
    }

#ifdef USE_GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
      fprintf(stderr, "GLEW initialization failed: %s\n", glewGetErrorString(err));
      HIP_SKIP_TEST(HipTest::SkipReason::kGlewInitFailed);
    }
#endif
  }

  GLContextScopeGuard(const GLContextScopeGuard&) = delete;
  GLContextScopeGuard& operator=(const GLContextScopeGuard&) = delete;

  GLContextScopeGuard(GLContextScopeGuard&&) = delete;
  GLContextScopeGuard& operator=(GLContextScopeGuard&&) = delete;

 private:
  GLContextScopeGuardPtr gl_context_;
};
