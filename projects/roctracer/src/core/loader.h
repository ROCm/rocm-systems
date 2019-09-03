#ifndef SRC_CORE_LOADER_H_
#define SRC_CORE_LOADER_H_

#include <atomic>
#include <mutex>
#include <dlfcn.h>

namespace roctracer {

// Base runtime loader class
template <class T>
class BaseLoader : public T {
  public:
  typedef std::mutex mutex_t;
  typedef BaseLoader<T> loader_t;

  bool Enabled() const { return (handle_ != NULL); }

  template <class fun_t>
  fun_t* GetFun(const char* fun_name) {
    if (handle_ == NULL) return NULL;

    fun_t *f = (fun_t*) dlsym(handle_, fun_name);
    if (f == NULL) {
      fprintf(stderr, "roctracer: symbol lookup '%s' failed: \"%s\"\n", fun_name, dlerror());
      abort();
    }
    dlerror();
    return f;
  }

  static inline loader_t& Instance(const bool& preload = false) {
    loader_t* obj = instance_.load(std::memory_order_acquire);
    if (obj == NULL) {
      std::lock_guard<mutex_t> lck(mutex_);
      if (instance_.load(std::memory_order_relaxed) == NULL) {
        obj = new loader_t(preload);
        instance_.store(obj, std::memory_order_release);
      }
    }
    return *instance_;
  }

  static loader_t* GetRef() { return instance_; }

  private:
  BaseLoader(bool preload) {
    const int flags = (preload) ? RTLD_LAZY : RTLD_LAZY|RTLD_NOLOAD;
    handle_ = dlopen(lib_name_, flags);
    if ((handle_ == NULL) && (strong_ld_check_)) {
      fprintf(stderr, "roctracer: Loading '%s' failed, preload(%d), %s\n", lib_name_, (int)preload, dlerror());
      abort();
    }
    dlerror();

    T::init(this);
  }

  ~BaseLoader() {
    if (handle_ != NULL) dlclose(handle_);
  }

  static mutex_t mutex_;
  static const char* lib_name_;
  static std::atomic<loader_t*> instance_;
  static const bool strong_ld_check_;
  void* handle_;
};

// HIP runtime library loader class
class HipApi {
  public:
  typedef BaseLoader<HipApi> Loader;

  typedef decltype(hipRegisterApiCallback) RegisterApiCallback_t;
  typedef decltype(hipRemoveApiCallback) RemoveApiCallback_t;
  typedef decltype(hipRegisterActivityCallback) RegisterActivityCallback_t;
  typedef decltype(hipRemoveActivityCallback) RemoveActivityCallback_t;
  typedef decltype(hipKernelNameRef) KernelNameRef_t;
  typedef decltype(hipApiName) ApiName_t;

  RegisterApiCallback_t* RegisterApiCallback;
  RemoveApiCallback_t* RemoveApiCallback;
  RegisterActivityCallback_t* RegisterActivityCallback;
  RemoveActivityCallback_t* RemoveActivityCallback;
  KernelNameRef_t* KernelNameRef;
  ApiName_t* ApiName;

  protected:
  void init(Loader* loader) {
    RegisterApiCallback = loader->GetFun<RegisterApiCallback_t>("hipRegisterApiCallback");
    RemoveApiCallback = loader->GetFun<RemoveApiCallback_t>("hipRemoveApiCallback");
    RegisterActivityCallback = loader->GetFun<RegisterActivityCallback_t>("hipRegisterActivityCallback");
    RemoveActivityCallback = loader->GetFun<RemoveActivityCallback_t>("hipRemoveActivityCallback");
    KernelNameRef = loader->GetFun<KernelNameRef_t>("hipKernelNameRef");
    ApiName = loader->GetFun<ApiName_t>("hipApiName");
  }
};

// HCC runtime library loader class
class HccApi {
  public:
  typedef BaseLoader<HccApi> Loader;

  typedef decltype(Kalmar::CLAMP::InitActivityCallback) InitActivityCallback_t;
  typedef decltype(Kalmar::CLAMP::EnableActivityCallback) EnableActivityCallback_t;
  typedef decltype(Kalmar::CLAMP::GetCmdName) GetCmdName_t;

  InitActivityCallback_t* InitActivityCallback;
  EnableActivityCallback_t* EnableActivityCallback;
  GetCmdName_t* GetCmdName;

  protected:
  void init(Loader* loader) {
    // Kalmar::CLAMP::InitActivityCallback
    InitActivityCallback = loader->GetFun<InitActivityCallback_t>("InitActivityCallbackImpl");
    // Kalmar::CLAMP::EnableActivityIdCallback
    EnableActivityCallback = loader->GetFun<EnableActivityCallback_t>("EnableActivityCallbackImpl");
    // Kalmar::CLAMP::GetCmdName
    GetCmdName = loader->GetFun<GetCmdName_t>("GetCmdNameImpl");
  }
};

// KFD runtime library loader class
class KfdApi {
  public:
  typedef BaseLoader<KfdApi> Loader;

  typedef bool (RegisterApiCallback_t)(uint32_t op, void* callback, void* arg);
  typedef bool (RemoveApiCallback_t)(uint32_t op);

  RegisterApiCallback_t* RegisterApiCallback;
  RemoveApiCallback_t* RemoveApiCallback;

  protected:
  void init(Loader* loader) {
    RegisterApiCallback = loader->GetFun<RegisterApiCallback_t>("RegisterApiCallback");
    RemoveApiCallback = loader->GetFun<RemoveApiCallback_t>("RemoveApiCallback");
  }
};

// rocTX runtime library loader class
class RocTxApi {
  public:
  typedef BaseLoader<RocTxApi> Loader;

  typedef bool (RegisterApiCallback_t)(uint32_t op, void* callback, void* arg);
  typedef bool (RemoveApiCallback_t)(uint32_t op);

  RegisterApiCallback_t* RegisterApiCallback;
  RemoveApiCallback_t* RemoveApiCallback;

  protected:
  void init(Loader* loader) {
    RegisterApiCallback = loader->GetFun<RegisterApiCallback_t>("RegisterApiCallback");
    RemoveApiCallback = loader->GetFun<RemoveApiCallback_t>("RemoveApiCallback");
  }
};

typedef BaseLoader<HipApi> HipLoader;
typedef BaseLoader<HccApi> HccLoader;
typedef BaseLoader<KfdApi> KfdLoader;
typedef BaseLoader<RocTxApi> RocTxLoader;

} // namespace roctracer

#define LOADER_INSTANTIATE() \
  template<class T> typename roctracer::BaseLoader<T>::mutex_t roctracer::BaseLoader<T>::mutex_; \
  template<class T> std::atomic<roctracer::BaseLoader<T>*> roctracer::BaseLoader<T>::instance_{}; \
  template<class T> const bool roctracer::BaseLoader<T>::strong_ld_check_ = false;
  template<> const char* roctracer::HipLoader::lib_name_ = "libhip_hcc.so"; \
  template<> const char* roctracer::HccLoader::lib_name_ = "libmcwamp_hsa.so"; \
  template<> const char* roctracer::KfdLoader::lib_name_ = "libkfdwrapper64.so"; \
  template<> const char* roctracer::RocTxLoader::lib_name_ = "libroctx64.so"; \
  template<> const bool roctracer::RocTxLoader::strong_ld_check_ = false;

#endif // SRC_CORE_LOADER_H_
