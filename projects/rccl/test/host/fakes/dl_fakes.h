/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_FAKES_DL_FAKES_H_
#define RCCL_TEST_HOST_FAKES_DL_FAKES_H_

// Controllable seams for dlopen, dlvsym, dlclose and dlerror, installed over a unit by
// fakes/dl_seam.h. Defaults call the real libdl.
//
// ScriptedDl swaps in an in-memory loader of named libraries and versioned symbols. It
// models versioned libraries only: real dlvsym also matches any version in a library
// with no version script. Its dlerror always returns a message, so call it only after
// a failure, as real code does.

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

extern std::function<void*(const char* /*file*/, int /*mode*/)> g_dlopen;
extern std::function<void*(void* /*handle*/, const char* /*name*/, const char* /*version*/)> g_dlvsym;
extern std::function<int(void* /*handle*/)> g_dlclose;
extern std::function<char*()> g_dlerror;

void ResetDlFakes();

class ScriptedDl {
 public:
  ScriptedDl();
  ~ScriptedDl();
  ScriptedDl(const ScriptedDl&) = delete;
  ScriptedDl& operator=(const ScriptedDl&) = delete;

  // A library dlopen(name) returns; the handle is stable for the object's life.
  void* AddLibrary(const std::string& name);
  void Export(void* handle, const std::string& symbol, const std::string& version, void* addr);

  struct Open {
    std::string file;
    int mode;
  };
  std::vector<Open> opens;
  std::vector<void*> closes;

 private:
  struct Library {
    std::string name;
    std::map<std::pair<std::string, std::string>, void*> symbols;
  };
  std::vector<std::unique_ptr<Library>> libraries_;
};

#endif  // RCCL_TEST_HOST_FAKES_DL_FAKES_H_
