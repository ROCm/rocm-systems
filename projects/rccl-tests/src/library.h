#ifndef RCCL_TESTS_LIBRARY_H_
#define RCCL_TESTS_LIBRARY_H_

#if defined(NCCL_OS_WINDOWS)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

inline void* rcclTestGetSymbol(const char* name) {
#if defined(NCCL_OS_WINDOWS)
  HMODULE library = nullptr;
  // Headers without dllimport expose an executable thunk instead of the DLL address.
#if defined(RCCL_TESTS_LIBRARY_NAME)
  library = GetModuleHandleA(RCCL_TESTS_LIBRARY_NAME);
#else
  GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCSTR>(&ncclGetVersion), &library);
#endif
  if (!library) return nullptr;
  return reinterpret_cast<void*>(GetProcAddress(library, name));
#else
  return dlsym(RTLD_DEFAULT, name);
#endif
}

#endif
