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
#include <cstdint>

#ifdef __linux__
#include <GL/glx.h>
#endif

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
inline const char* glut_init_skip_reason = nullptr;
// Single window/context shared by every test. 0 means "not created". Created once in init() and
// kept alive for the whole process: freeglut accumulates native window handles across repeated
// create/destroy, so opening/closing one window per test hangs the window manager (Windows+Linux).
inline int glut_shared_window = 0;

// Initialize freeglut exactly once per process. glutInit() must be called only once; calling it
// again raises "illegal glutInit() reinitialization attempt" and can hang the test. ALL GL interop
// test files must go through EnsureGlutInitialized() (including the context-switch tests in
// hipGLContextSwitch.cc) so there is a single glutInit() for the whole process.
inline void GlutInitOnce() {
#ifdef __linux__
  // freeglut treats display failures as fatal. Check the prerequisites before entering it;
  // a returning error callback would let initialization continue with a null Display.
  Display* display = XOpenDisplay(nullptr);
  if (display == nullptr) {
    glut_init_skip_reason = "GLUT requires an accessible X display";
    return;
  }
  int error_base = 0;
  int event_base = 0;
  const bool has_glx = glXQueryExtension(display, &error_base, &event_base);
  XCloseDisplay(display);
  if (!has_glx) {
    glut_init_skip_reason = "GLUT requires the GLX extension on the X display";
    return;
  }
#endif

  static char proc_name[] = "";
  static std::array<char*, 2> glut_argv = {proc_name, nullptr};
  static int glut_argc = 1;
  glutInit(&glut_argc, glut_argv.data());
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
}

inline void EnsureGlutInitialized() {
  std::call_once(glut_init_flag, GlutInitOnce);
  if (glut_init_skip_reason != nullptr) {
    HIP_SKIP_TEST(glut_init_skip_reason);
  }
}

class GLUTContextScopeGuard : public IContextScopeGuard {
 public:
  GLUTContextScopeGuard() {
    EnsureGlutInitialized();
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
  EGLContextScopeGuard() : resources_{AcquireDisplay()} {
    REQUIRE(eglBindAPI(EGL_OPENGL_API));
    resources_.surface = eglCreatePbufferSurface(resources_.display->handle,
                                                 resources_.display->config, kPbufferAttribs);
    REQUIRE(resources_.surface != EGL_NO_SURFACE);
    resources_.context = eglCreateContext(resources_.display->handle, resources_.display->config,
                                          EGL_NO_CONTEXT, nullptr);
    REQUIRE(resources_.context != EGL_NO_CONTEXT);
    makeCurrent();
  }

  void makeCurrent() {
    REQUIRE(eglBindAPI(EGL_OPENGL_API));
    REQUIRE(eglMakeCurrent(resources_.display->handle, resources_.surface, resources_.surface,
                           resources_.context));
  }

  std::uintptr_t id() const { return reinterpret_cast<std::uintptr_t>(resources_.context); }

  EGLContextScopeGuard(const EGLContextScopeGuard&) = delete;
  EGLContextScopeGuard& operator=(const EGLContextScopeGuard&) = delete;

  EGLContextScopeGuard(EGLContextScopeGuard&&) = delete;
  EGLContextScopeGuard& operator=(EGLContextScopeGuard&&) = delete;

 private:
  struct DisplayState {
    EGLDisplay handle = EGL_NO_DISPLAY;
    EGLConfig config = nullptr;

    ~DisplayState() {
      if (handle != EGL_NO_DISPLAY) {
        eglTerminate(handle);
      }
    }
  };

  struct Resources {
    std::shared_ptr<DisplayState> display;
    EGLSurface surface = EGL_NO_SURFACE;
    EGLContext context = EGL_NO_CONTEXT;

    ~Resources() {
      if (context != EGL_NO_CONTEXT) {
        if (eglGetCurrentContext() == context) {
          eglMakeCurrent(display->handle, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }
        eglDestroyContext(display->handle, context);
      }
      if (surface != EGL_NO_SURFACE) {
        eglDestroySurface(display->handle, surface);
      }
    }
  };

  static std::shared_ptr<DisplayState> AcquireDisplay() {
    // Catch runs these tests serially. All simultaneously live contexts share one display;
    // terminating it when any one context is destroyed would invalidate the others.
    static std::weak_ptr<DisplayState> shared_display;
    if (auto display = shared_display.lock()) {
      return display;
    }

    const auto query_devices =
        reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
    REQUIRE(query_devices != nullptr);
    std::array<EGLDeviceEXT, 8> devices{};
    EGLint device_count = 0;
    REQUIRE(query_devices(devices.size(), devices.data(), &device_count));
    REQUIRE(device_count > 0);

    const auto get_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
        eglGetProcAddress("eglGetPlatformDisplayEXT"));
    REQUIRE(get_display != nullptr);
    const EGLDisplay handle = get_display(EGL_PLATFORM_DEVICE_EXT, devices.front(), nullptr);
    REQUIRE(handle != EGL_NO_DISPLAY);
    auto display = std::make_shared<DisplayState>();
    REQUIRE(eglInitialize(handle, nullptr, nullptr));
    display->handle = handle;
    EGLint config_count = 0;
    REQUIRE(eglChooseConfig(handle, kConfigAttribs, &display->config, 1, &config_count));
    REQUIRE(config_count == 1);
    shared_display = display;
    return display;
  }

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

  Resources resources_;
};
#endif

enum class GLContextType { Glut, Egl };

inline GLContextType GetGLContextType() {
  const char* value = std::getenv("GL_CONTEXT_TYPE");
  const std::string type = value == nullptr ? "" : value;
  if (type.empty() || type == "GLUT") {
    return GLContextType::Glut;
  }
#ifdef USE_EGL
  if (type == "EGL") {
    return GLContextType::Egl;
  }
#endif
  INFO("Unsupported GL_CONTEXT_TYPE value '" << type << "'");
  INFO("Supported values are ['GLUT'"
#ifdef USE_EGL
       << ", 'EGL'"
#endif
       << "]");
  REQUIRE(false);
  return GLContextType::Glut;
}

class GLContextScopeGuard {
 public:
  using GLContextScopeGuardPtr = std::unique_ptr<IContextScopeGuard>;

  GLContextScopeGuard() {
    if (GetGLContextType() == GLContextType::Glut) {
      gl_context_ = std::make_unique<GLUTContextScopeGuard>();
#ifdef USE_EGL
    } else {
      gl_context_ = std::make_unique<EGLContextScopeGuard>();
#endif
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
