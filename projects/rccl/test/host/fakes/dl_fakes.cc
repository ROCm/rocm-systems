/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "fakes/dl_fakes.h"

// Puts the micro_* prototypes in scope so the compiler checks them against the
// definitions below; the defaults call real libdl, so undo the renames at once.
#include "fakes/dl_seam.h"
#include "fakes/dl_seam_undef.h"

static void* DefaultDlopen(const char* file, int mode) { return dlopen(file, mode); }
static void* DefaultDlsym(void* handle, const char* name) { return dlsym(handle, name); }
static void* DefaultDlvsym(void* handle, const char* name, const char* version) {
  return dlvsym(handle, name, version);
}
static int DefaultDlclose(void* handle) { return dlclose(handle); }
static char* DefaultDlerror() { return dlerror(); }

std::function<void*(const char*, int)> g_dlopen = DefaultDlopen;
std::function<void*(void*, const char*)> g_dlsym = DefaultDlsym;
std::function<void*(void*, const char*, const char*)> g_dlvsym = DefaultDlvsym;
std::function<int(void*)> g_dlclose = DefaultDlclose;
std::function<char*()> g_dlerror = DefaultDlerror;

void ResetDlFakes() {
  g_dlopen = DefaultDlopen;
  g_dlsym = DefaultDlsym;
  g_dlvsym = DefaultDlvsym;
  g_dlclose = DefaultDlclose;
  g_dlerror = DefaultDlerror;
}

extern "C" {
void* micro_dlopen(const char* file, int mode) { return g_dlopen(file, mode); }
void* micro_dlsym(void* handle, const char* name) { return g_dlsym(handle, name); }
void* micro_dlvsym(void* handle, const char* name, const char* version) { return g_dlvsym(handle, name, version); }
int micro_dlclose(void* handle) { return g_dlclose(handle); }
char* micro_dlerror(void) { return g_dlerror(); }
}  // extern "C"

ScriptedDl::ScriptedDl() {
  g_dlopen = [this](const char* file, int mode) -> void* {
    opens.push_back({file ? file : "", mode});
    for (auto& lib : libraries_) {
      if (file && lib->name == file) return lib.get();
    }
    return nullptr;
  };
  g_dlvsym = [this](void* handle, const char* name, const char* version) -> void* {
    for (auto& lib : libraries_) {
      if (lib.get() != handle) continue;
      auto it = lib->symbols.find({name, version});
      return it == lib->symbols.end() ? nullptr : it->second;
    }
    return nullptr;
  };
  g_dlsym = [this](void* handle, const char* name) -> void* {
    for (auto& lib : libraries_) {
      if (lib.get() != handle) continue;
      for (const auto& [key, addr] : lib->symbols) {
        if (key.first == name) return addr;
      }
    }
    return nullptr;
  };
  g_dlclose = [this](void* handle) {
    closes.push_back(handle);
    return 0;
  };
  g_dlerror = []() {
    static char msg[] = "scripted dlerror";
    return msg;
  };
}

ScriptedDl::~ScriptedDl() { ResetDlFakes(); }

void* ScriptedDl::AddLibrary(const std::string& name) {
  libraries_.push_back(std::make_unique<Library>());
  libraries_.back()->name = name;
  return libraries_.back().get();
}

void ScriptedDl::Export(void* handle, const std::string& symbol, const std::string& version, void* addr) {
  static_cast<Library*>(handle)->symbols[{symbol, version}] = addr;
}
