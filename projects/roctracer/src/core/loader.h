#ifndef SRC_CORE_LOADER_H_
#define SRC_CORE_LOADER_H_

#include <mutex>
#include <dlfcn.h>

namespace roctracer {

class Loader {
  public:
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

  private:
  void* handle_;
};

class HipLoader : protected Loader {
  public:
  typedef std::mutex mutex_t;

  typedef decltype(hipRegisterApiCallback) hipRegisterApiCallback_t;
  typedef decltype(hipRemoveApiCallback) hipRemoveApiCallback_t;
  typedef decltype(hipRegisterActivityCallback) hipRegisterActivityCallback_t;
  typedef decltype(hipRemoveActivityCallback) hipRemoveActivityCallback_t;
  typedef decltype(hipKernelNameRef) hipKernelNameRef_t;

  static HipLoader& Instance() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_ == NULL) {
      instance_ = new HipLoader();
    }
    return *instance_;
  }

  HipLoader() : Loader("libhip_hcc.so") {
    hipRegisterApiCallback = GetFun<hipRegisterApiCallback_t>("hipRegisterApiCallback");
    hipRemoveApiCallback = GetFun<hipRemoveApiCallback_t>("hipRemoveApiCallback");
    hipRegisterActivityCallback = GetFun<hipRegisterActivityCallback_t>("hipRegisterActivityCallback");
    hipRemoveActivityCallback = GetFun<hipRemoveActivityCallback_t>("hipRemoveActivityCallback");
    hipKernelNameRef = GetFun<hipKernelNameRef_t>("hipKernelNameRef");
  }

  hipRegisterApiCallback_t* hipRegisterApiCallback;
  hipRemoveApiCallback_t* hipRemoveApiCallback;
  hipRegisterActivityCallback_t* hipRegisterActivityCallback;
  hipRemoveActivityCallback_t* hipRemoveActivityCallback;
  hipKernelNameRef_t* hipKernelNameRef;

  private:
  static HipLoader* instance_;
  static mutex_t mutex_;
};

} // namespace roctracer

#if 0
namespace roctracer {
class HccLoader : protected Loader {
  public:
  typedef std::mutex mutex_t;

  typedef decltype(Kalmar::CLAMP::SetActivityCallback) hccSetActivityCallback_t;
  typedef decltype(Kalmar::CLAMP::SetActivityIdCallback) hccSetActivityIdCallback_t;
  typedef decltype(Kalmar::CLAMP::GetCmdName) hccGetCmdName_t;

  static HccLoader& Instance() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_ == NULL) {
      instance_ = new HccLoader();
    }
    return *instance_;
  }

  HccLoader() : Loader("libmcwamp.so") {
    // _ZN6Kalmar5CLAMP19SetActivityCallbackEjPvS1_
    hccSetActivityCallback = GetFun<hccSetActivityCallback_t>("Kalmar::CLAMP::SetActivityCallback");
    // _ZN6Kalmar5CLAMP21SetActivityIdCallbackEPv
    hccSetActivityIdCallback = GetFun<hccSetActivityIdCallback_t>("Kalmar::CLAMP::SetActivityIdCallback");
    // _ZN6Kalmar5CLAMP10GetCmdNameEj
    hccGetCmdName = GetFun<hccGetCmdName_t>("Kalmar::CLAMP::GetCmdName");

    printf("HccLoader hccSetActivityCallback %p\n", hccSetActivityCallback);
  }

  hccSetActivityCallback_t* hccSetActivityCallback;
  hccSetActivityIdCallback_t* hccSetActivityIdCallback;
  hccGetCmdName_t* hccGetCmdName;

  private:
  static HccLoader* instance_;
  static mutex_t mutex_;
};
} // namespace roctracer

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
