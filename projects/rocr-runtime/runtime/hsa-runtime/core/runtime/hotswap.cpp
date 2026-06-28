////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in the
//    documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/runtime/hotswap.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "core/runtime/hotswap_gfx_query.hpp"
#include "core/util/os.h"

namespace rocr {
namespace hotswap {
namespace {

std::mutex g_retained_elfs_mutex;
std::unordered_map<uint64_t, std::vector<OwnedElf>> g_retained_elfs;

bool EnvTruthy(const char* name) {
  if (!os::IsEnvVarSet(name)) {
    return false;
  }

  std::string value = os::GetEnvVar(name);
  if (value.empty()) {
    return false;
  }

  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value != "0" && value != "off" && value != "false" &&
         value != "no" && value != "n" && value != "f";
}

bool Disabled() { return EnvTruthy("HSA_HOTSWAP_DISABLE"); }

bool Verbose() {
  static const bool verbose = EnvTruthy("HSA_HOTSWAP_VERBOSE");
  return verbose;
}

#define HOTSWAP_LOG(...)                 \
  do {                                   \
    if (Verbose()) {                     \
      fprintf(stderr, __VA_ARGS__);      \
    }                                    \
  } while (false)

struct ComgrData {
  uint64_t handle;
};

constexpr int kComgrStatusSuccess = 0;
constexpr int kComgrDataKindExecutable = 0x8;

struct Comgr {
  os::LibHandle lib = nullptr;
  int (*create_data)(int kind, ComgrData* data) = nullptr;
  int (*release_data)(ComgrData data) = nullptr;
  int (*set_data)(ComgrData data, size_t size, const char* bytes) = nullptr;
  int (*get_data)(ComgrData data, size_t* size, char* bytes) = nullptr;
  int (*get_data_isa_name)(ComgrData data, size_t* size, char* isa_name) =
      nullptr;
  int (*hotswap_rewrite)(ComgrData input, const char* source_isa_name,
                         const char* target_isa_name, ComgrData* output) =
      nullptr;
};

template <typename T>
bool LoadComgrSymbol(os::LibHandle lib, const char* name, T* out) {
  *out = reinterpret_cast<T>(os::GetExportAddress(lib, name));
  return *out != nullptr;
}

bool LoadComgrFrom(os::LibHandle lib, Comgr* comgr) {
  comgr->lib = lib;
  return LoadComgrSymbol(lib, "amd_comgr_create_data", &comgr->create_data) &&
         LoadComgrSymbol(lib, "amd_comgr_release_data",
                         &comgr->release_data) &&
         LoadComgrSymbol(lib, "amd_comgr_set_data", &comgr->set_data) &&
         LoadComgrSymbol(lib, "amd_comgr_get_data", &comgr->get_data) &&
         LoadComgrSymbol(lib, "amd_comgr_get_data_isa_name",
                         &comgr->get_data_isa_name) &&
         LoadComgrSymbol(lib, "amd_comgr_hotswap_rewrite",
                         &comgr->hotswap_rewrite);
}

std::string RuntimeLibraryDir() {
#if defined(_WIN32) || defined(_WIN64)
  HMODULE module = nullptr;
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(&RuntimeLibraryDir),
                          &module)) {
    return {};
  }

  char path[MAX_PATH] = {};
  const DWORD len = GetModuleFileNameA(module, path, sizeof(path));
  if (len == 0 || len >= sizeof(path)) {
    return {};
  }

  std::string path_str(path);
  const std::string::size_type slash = path_str.find_last_of("\\/");
  return slash == std::string::npos ? std::string{} : path_str.substr(0, slash);
#else
  Dl_info info = {};
  if (dladdr(reinterpret_cast<void*>(&RuntimeLibraryDir), &info) == 0 ||
      !info.dli_fname || info.dli_fname[0] == '\0') {
    return {};
  }

  std::string path(info.dli_fname);
  const std::string::size_type slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string{} : path.substr(0, slash);
#endif
}

std::vector<std::string> ComgrCandidateNames() {
  std::vector<std::string> names;
  const std::string runtime_dir = RuntimeLibraryDir();
  if (!runtime_dir.empty()) {
#if defined(_WIN32) || defined(_WIN64)
    names.push_back(runtime_dir + "\\amd_comgr.dll");
#else
    names.push_back(runtime_dir + "/libamd_comgr.so.3");
    names.push_back(runtime_dir + "/libamd_comgr.so");
#endif
  }

#if defined(_WIN32) || defined(_WIN64)
  names.push_back("amd_comgr.dll");
#else
  names.push_back("libamd_comgr.so.3");
  names.push_back("libamd_comgr.so");
#endif
  return names;
}

Comgr* GetComgr() {
  static std::once_flag once;
  static Comgr comgr;
  static bool ready = false;

  std::call_once(once, [] {
    auto try_load_comgr = [](const char* name) {
      if (!name || name[0] == '\0') {
        return false;
      }

      os::LibHandle lib = os::LoadLib(name);
      if (!lib) {
        return false;
      }

      if (LoadComgrFrom(lib, &comgr)) {
        ready = true;
        HOTSWAP_LOG("hotswap: loaded COMGR from %s\n", name);
        return true;
      }

      os::CloseLib(lib);
      comgr = Comgr{};
      return false;
    };

    for (const std::string& name : ComgrCandidateNames()) {
      if (try_load_comgr(name.c_str())) {
        return;
      }
    }
    HOTSWAP_LOG("hotswap: COMGR hotswap entry points unavailable\n");
  });

  return ready ? &comgr : nullptr;
}

std::string GetDataIsaName(const void* elf_data, size_t elf_size) {
  Comgr* comgr = GetComgr();
  if (!comgr || !elf_data || elf_size == 0) {
    return {};
  }

  ComgrData data = {};
  if (comgr->create_data(kComgrDataKindExecutable, &data) !=
      kComgrStatusSuccess) {
    return {};
  }

  std::string isa;
  if (comgr->set_data(data, elf_size, static_cast<const char*>(elf_data)) ==
      kComgrStatusSuccess) {
    size_t isa_len = 0;
    if (comgr->get_data_isa_name(data, &isa_len, nullptr) ==
            kComgrStatusSuccess &&
        isa_len > 0) {
      isa.resize(isa_len);
      if (comgr->get_data_isa_name(data, &isa_len, isa.data()) ==
          kComgrStatusSuccess) {
        if (!isa.empty() && isa.back() == '\0') {
          isa.pop_back();
        }
      } else {
        isa.clear();
      }
    }
  }

  comgr->release_data(data);
  return isa;
}

bool GateAllowsAgent(hsa_agent_t agent) {
  const AgentGfxRevision gfx = QueryAgentGfxRevision(agent);
  HOTSWAP_LOG("hotswap: agent gfx=%s asic_revision=%u (valid=%s)\n",
              gfx.gfx_target.empty() ? "?" : gfx.gfx_target.c_str(),
              gfx.asic_revision, gfx.revision_valid ? "yes" : "no");
  return GateAllowsHotswap(gfx);
}

bool RetargetCodeObject(const void* elf_data, size_t elf_size,
                        const char* source_isa, const char* target_isa,
                        OwnedElf* out_elf, size_t* out_elf_size) {
  Comgr* comgr = GetComgr();
  if (!comgr || !elf_data || elf_size == 0 || !source_isa || !target_isa ||
      !out_elf || !out_elf_size) {
    return false;
  }

  ComgrData input = {};
  if (comgr->create_data(kComgrDataKindExecutable, &input) !=
      kComgrStatusSuccess) {
    return false;
  }

  if (comgr->set_data(input, elf_size, static_cast<const char*>(elf_data)) !=
      kComgrStatusSuccess) {
    comgr->release_data(input);
    return false;
  }

  ComgrData output = {};
  const int status =
      comgr->hotswap_rewrite(input, source_isa, target_isa, &output);
  comgr->release_data(input);
  if (status != kComgrStatusSuccess) {
    HOTSWAP_LOG("hotswap: COMGR rewrite failed for %s -> %s (rc=%d)\n",
                source_isa, target_isa, status);
    return false;
  }

  size_t output_size = 0;
  if (comgr->get_data(output, &output_size, nullptr) != kComgrStatusSuccess ||
      output_size == 0) {
    comgr->release_data(output);
    return false;
  }

  OwnedElf output_buf(std::malloc(output_size), &std::free);
  if (!output_buf) {
    comgr->release_data(output);
    return false;
  }

  size_t copy_size = output_size;
  if (comgr->get_data(output, &copy_size,
                      static_cast<char*>(output_buf.get())) !=
      kComgrStatusSuccess) {
    comgr->release_data(output);
    return false;
  }

  comgr->release_data(output);
  *out_elf = std::move(output_buf);
  *out_elf_size = output_size;
  return true;
}

}  // namespace

bool TryRetargetCodeObject(amd::hsa::loader::CodeObjectReaderImpl* reader,
                           hsa_agent_t agent, OwnedElf* out_elf,
                           size_t* out_elf_size) {
  if (Disabled() || !reader || !GateAllowsAgent(agent)) {
    return false;
  }

  const void* input = reader->GetCodeObjectMemory();
  const size_t input_size = reader->GetCodeObjectSize();
  const std::string source_isa = GetDataIsaName(input, input_size);
  const std::string target_isa = GetAgentIsaName(agent);
  if (source_isa.empty() || target_isa.empty()) {
    HOTSWAP_LOG("hotswap: rewrite skipped, empty isa (src='%s' tgt='%s')\n",
                source_isa.c_str(), target_isa.c_str());
    return false;
  }

  const bool rewritten = RetargetCodeObject(input, input_size,
                                            source_isa.c_str(),
                                            target_isa.c_str(), out_elf,
                                            out_elf_size);
  HOTSWAP_LOG("hotswap: rewrite src=%s tgt=%s in=%zu out=%zu changed=%d\n",
              source_isa.c_str(), target_isa.c_str(), input_size,
              rewritten ? *out_elf_size : 0, rewritten ? 1 : 0);
  return rewritten;
}

void RetainElf(hsa_executable_t executable, OwnedElf elf) {
  try {
    std::scoped_lock lock(g_retained_elfs_mutex);
    g_retained_elfs[executable.handle].push_back(std::move(elf));
  } catch (const std::bad_alloc&) {
    // If the keepalive container cannot grow, preserve the loaded code object's
    // raw ELF pointer by intentionally leaking this allocation.
    (void)elf.release();
  }
}

void ReleaseElfs(hsa_executable_t executable) {
  std::scoped_lock lock(g_retained_elfs_mutex);
  g_retained_elfs.erase(executable.handle);
}

void LogRewrittenLoadFailure(hsa_status_t status) {
  HOTSWAP_LOG("hotswap: rewritten load failed (status=%d), falling back to "
              "original code object\n",
              static_cast<int>(status));
}

}  // namespace hotswap
}  // namespace rocr
