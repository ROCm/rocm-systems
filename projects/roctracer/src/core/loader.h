#ifndef SRC_CORE_LOADER_H_
#define SRC_CORE_LOADER_H_

#include <mutex>
#include <dlfcn.h>

namespace roctracer {

class Loader {
  public:
  typedef std::mutex mutex_t;

  Loader(const char* lib_name) {
    handle_ = dlopen(lib_name, RTLD_LAZY|RTLD_NODELETE);
    if (handle_ == NULL) {
      fprintf(stderr, "roctracer: Loading '%s' failed, %s\n", lib_name, dlerror());
      abort();
    }
  }

  ~Loader() {
    if (handle_ != NULL) dlclose(handle_);
  }

  template <class fun_t>
  fun_t* GetFun(const char* fun_name) { return (fun_t*) dlsym(handle_, fun_name); }

  protected:
  static mutex_t mutex_;

  private:
  void* handle_;
};

class HipLoader : protected Loader {
  public:
  typedef decltype(hipRegisterApiCallback) RegisterApiCallback_t;
  typedef decltype(hipRemoveApiCallback) RemoveApiCallback_t;
  typedef decltype(hipRegisterActivityCallback) RegisterActivityCallback_t;
  typedef decltype(hipRemoveActivityCallback) RemoveActivityCallback_t;
  typedef decltype(hipKernelNameRef) KernelNameRef_t;

  static HipLoader& Instance() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_ == NULL) {
      instance_ = new HipLoader();
    }
    return *instance_;
  }

  HipLoader() : Loader("libhip_hcc.so") {
    RegisterApiCallback = GetFun<RegisterApiCallback_t>("hipRegisterApiCallback");
    RemoveApiCallback = GetFun<RemoveApiCallback_t>("hipRemoveApiCallback");
    RegisterActivityCallback = GetFun<RegisterActivityCallback_t>("hipRegisterActivityCallback");
    RemoveActivityCallback = GetFun<RemoveActivityCallback_t>("hipRemoveActivityCallback");
    KernelNameRef = GetFun<KernelNameRef_t>("hipKernelNameRef");
  }

  RegisterApiCallback_t* RegisterApiCallback;
  RemoveApiCallback_t* RemoveApiCallback;
  RegisterActivityCallback_t* RegisterActivityCallback;
  RemoveActivityCallback_t* RemoveActivityCallback;
  KernelNameRef_t* KernelNameRef;

  private:
  static HipLoader* instance_;
};

class HccLoader : protected Loader {
  public:
  typedef std::mutex mutex_t;

  typedef decltype(Kalmar::CLAMP::SetActivityCallback) SetActivityCallback_t;
  typedef decltype(Kalmar::CLAMP::SetActivityIdCallback) SetActivityIdCallback_t;
  typedef decltype(Kalmar::CLAMP::GetCmdName) GetCmdName_t;

  static HccLoader& Instance() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_ == NULL) {
      instance_ = new HccLoader();
    }
    return *instance_;
  }

  HccLoader() : Loader("libmcwamp.so") {
    // Kalmar::CLAMP::SetActivityCallback
    // _ZN6Kalmar5CLAMP19SetActivityCallbackEjPvS1_
    SetActivityCallback = GetFun<SetActivityCallback_t>("_ZN6Kalmar5CLAMP19SetActivityCallbackEjPvS1_");
    // Kalmar::CLAMP::SetActivityIdCallback
    // _ZN6Kalmar5CLAMP21SetActivityIdCallbackEPv
    SetActivityIdCallback = GetFun<SetActivityIdCallback_t>("_ZN6Kalmar5CLAMP21SetActivityIdCallbackEPv");
    // Kalmar::CLAMP::GetCmdName
    // _ZN6Kalmar5CLAMP10GetCmdNameEj
    GetCmdName = GetFun<GetCmdName_t>("_ZN6Kalmar5CLAMP10GetCmdNameEj");
  }

  SetActivityCallback_t* SetActivityCallback;
  SetActivityIdCallback_t* SetActivityIdCallback;
  GetCmdName_t* GetCmdName;

  private:
  static HccLoader* instance_;
};

} // namespace roctracer

#if 0
namespace Kalmar {
namespace CLAMP {
extern bool SetActivityCallback(unsigned, void*, void*) __attribute__((weak_import));
extern void SetActivityIdCallback(void*) __attribute__((weak_import));
extern const char* GetCmdName(unsigned) __attribute__((weak_impot));
}}

namespace roctracer {
bool HccSetActivityCallback(unsigned op, void* fun, void* arg) {
  printf("HccSetActivityCallback(%p)\n", Kalmar::CLAMP::SetActivityCallback);
  return (Kalmar::CLAMP::SetActivityCallback != NULL) ? Kalmar::CLAMP::SetActivityCallback(op, fun, arg) : true;
}
void HccSetActivityIdCallback(void* fun) {
  if (Kalmar::CLAMP::SetActivityIdCallback != NULL) Kalmar::CLAMP::SetActivityIdCallback(fun);
}
const char* HccGetCmdName(unsigned op) {
  if (Kalmar::CLAMP::GetCmdName != NULL) Kalmar::CLAMP::GetCmdName(op);
}
} // namespace roctracer
#endif
#endif // SRC_CORE_LOADER_H_
