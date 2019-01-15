#ifndef SRC_CORE_LOADER_H_
#define SRC_CORE_LOADER_H_

#include <mutex>
#include <dlfcn.h>

namespace roctracer {

// Base runtime loader class
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

// HIP runtime library loader class
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

// HCC runtime library loader class
class HccLoader : protected Loader {
  public:
  typedef std::mutex mutex_t;

  typedef decltype(Kalmar::CLAMP::InitActivityCallback) InitActivityCallback_t;
  typedef decltype(Kalmar::CLAMP::EnableActivityCallback) EnableActivityCallback_t;
  typedef decltype(Kalmar::CLAMP::GetCmdName) GetCmdName_t;

  static HccLoader* GetRef() { return instance_; }

  static HccLoader& Instance() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_ == NULL) {
      instance_ = new HccLoader();
    }
    return *instance_;
  }

  HccLoader() : Loader("libmcwamp.so") {
    // Kalmar::CLAMP::InitActivityCallback
    InitActivityCallback = GetFun<InitActivityCallback_t>("_ZN6Kalmar5CLAMP20InitActivityCallbackEPvS1_S1_");
    // Kalmar::CLAMP::EnableActivityIdCallback
    EnableActivityCallback = GetFun<EnableActivityCallback_t>("_ZN6Kalmar5CLAMP22EnableActivityCallbackEjb");
    // Kalmar::CLAMP::GetCmdName
    GetCmdName = GetFun<GetCmdName_t>("_ZN6Kalmar5CLAMP10GetCmdNameEj");
  }

  InitActivityCallback_t* InitActivityCallback;
  EnableActivityCallback_t* EnableActivityCallback;
  GetCmdName_t* GetCmdName;

  private:
  static HccLoader* instance_;
};

} // namespace roctracer

#endif // SRC_CORE_LOADER_H_
