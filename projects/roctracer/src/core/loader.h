#ifndef SRC_CORE_LOADER_H_
#define SRC_CORE_LOADER_H_

#include <mutex>
#include <dlfcn.h>

namespace roctracer {

class Loader {
  public:
  Loader(const char* lib_name) {
    handle_ = dlopen(lib_name, RTLD_NOW);
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

#endif // SRC_CORE_LOADER_H_
