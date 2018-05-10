#include "inc/roctracer.h"

#include <atomic>
#include <hip/hip_runtime.h>
#include <mutex>
#include <string.h>
#include <pthread.h>

#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"
#include "util/logger.h"

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

#define PTHREAD_CALL(call)                                                                         \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      errno = err;                                                                                 \
      perror(#call);                                                                               \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

#define HSART_CALL(call)                                                                           \
  do {                                                                                             \
    hsa_status_t status = call;                                                                    \
    if (status != HSA_STATUS_SUCCESS) {                                                            \
      std::cerr << "HSA-rt call '" << #call << "' error(" << std::hex << status << ")"             \
        << std::dec << std::endl << std::flush;                                                    \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

#define API_METHOD_PREFIX                                                                          \
  int err = 0;                                                                                     \
  try {

#define API_METHOD_SUFFIX                                                                          \
  }                                                                                                \
  catch (std::exception & e) {                                                                     \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
    err = roctracer::GetExcStatus(e);                                                              \
  }                                                                                                \
  return err;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Internal library methods
//
namespace roctracer {

int GetExcStatus(const std::exception& e) {
  const util::exception* roctracer_exc_ptr = dynamic_cast<const util::exception*>(&e);
  return (roctracer_exc_ptr) ? static_cast<int>(roctracer_exc_ptr->status()) : 1;
}

class GlobalCounter {
  public:
  typedef std::mutex mutex_t;
  typedef uint64_t counter_t;

  static counter_t Increment() {
    std::lock_guard<mutex_t> lock(mutex_);
    return ++counter_;
  }

  private:
  static mutex_t mutex_;
  static counter_t counter_;
};
GlobalCounter::mutex_t GlobalCounter::mutex_;
GlobalCounter::counter_t GlobalCounter::counter_ = 0;

class MemoryPool {
  public:
  typedef std::mutex mutex_t;

  static void allocator_default(char** ptr, size_t size, void* arg) {
    (void)arg;
    if (*ptr == NULL) {
      *ptr = reinterpret_cast<char*>(malloc(size));
    } else if (size != 0) {
      *ptr = reinterpret_cast<char*>(realloc(ptr, size));
    } else {
      free(*ptr); 
      *ptr = NULL;
    }
  }

  MemoryPool(const roctracer_properties_t& properties) { 
    // Assigning pool allocator
    alloc_fun_ = allocator_default;
    alloc_arg_ = NULL;
    if (properties.alloc_fun != NULL) {
      alloc_fun_ = properties.alloc_fun;
      alloc_arg_ = properties.alloc_arg;
    }

    // Pool definition
    buffer_size_ = properties.buffer_size;
    const size_t pool_size = 2 * buffer_size_;
    pool_begin_ = NULL;
    alloc_fun_(&pool_begin_, pool_size, alloc_arg_);
    if (pool_begin_ == NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "pool allocator failed");
    pool_end_ = pool_begin_ + pool_size;
    buffer_begin_ = pool_begin_;
    buffer_end_ = buffer_begin_ + buffer_size_;
    write_ptr_ = buffer_begin_;

    // Consuming read thread
    read_callback_fun_ = properties.buffer_callback_fun;
    read_callback_arg_ = properties.buffer_callback_arg;
    consumer_arg_ = consumer_arg_t{this, true, NULL, NULL};
    PTHREAD_CALL(pthread_mutex_init(&read_mutex_, NULL));
    PTHREAD_CALL(pthread_cond_init(&read_cond_, NULL));
    PTHREAD_CALL(pthread_create(&consumer_thread_, NULL, reader_fun, &consumer_arg_));
  }

  ~MemoryPool() {
    Flush();
    allocator_default(&pool_begin_, 0, alloc_arg_);
  }

  template <typename Record>
  void* Write(const Record& record) {
    std::lock_guard<mutex_t> lock(write_mutex_);
    char* next = write_ptr_ + sizeof(Record);
    if (next > buffer_end_) {
      if (write_ptr_ == buffer_begin_) EXC_ABORT(ROCTRACER_STATUS_ERROR, "buffer size(" << buffer_size_ << ") is less then the record(" << sizeof(Record) << ")");
      spawn_reader(buffer_begin_, buffer_end_);
      buffer_begin_ = (buffer_end_ == pool_end_) ? pool_begin_ : buffer_end_;
      buffer_end_ = buffer_begin_ + buffer_size_;
      write_ptr_ = buffer_begin_;
      next = write_ptr_ + sizeof(Record);
    }
    Record* ptr = reinterpret_cast<Record*>(write_ptr_);
    *ptr = record;
    write_ptr_ = next;
    return reinterpret_cast<void*>(ptr);
  }

  void Flush() {
    if (write_ptr_ > buffer_begin_) {
      spawn_reader(buffer_begin_, write_ptr_);
      sync_reader(&consumer_arg_);
      buffer_begin_ = write_ptr_;
    }
  }

  private:
  struct consumer_arg_t {
    MemoryPool* obj;
    bool valid;
    const char* begin;
    const char* end;
  };

  static void reset_reader(consumer_arg_t* arg) {
    reinterpret_cast<std::atomic<bool>*>(&(arg->valid))->store(false, std::memory_order_release);
  }

  static void sync_reader(const consumer_arg_t* arg) {
    while(arg->valid) PTHREAD_CALL(pthread_yield());
  }

  static void* reader_fun(void* consumer_arg) {
    consumer_arg_t* arg = reinterpret_cast<consumer_arg_t*>(consumer_arg);
    roctracer::MemoryPool* obj = arg->obj;

    reset_reader(arg);

    while (1) {
      PTHREAD_CALL(pthread_mutex_lock(&(obj->read_mutex_)));
      while (arg->valid == false) {
        PTHREAD_CALL(pthread_cond_wait(&(obj->read_cond_), &(obj->read_mutex_)));
      }
      obj->read_callback_fun_(arg->begin, arg->end, obj->read_callback_arg_);
      reset_reader(arg);
      PTHREAD_CALL(pthread_mutex_unlock(&(obj->read_mutex_)));
    }

    return NULL;
  }

  void spawn_reader(const char* data_begin, const char* data_end) {
    sync_reader(&consumer_arg_);
    PTHREAD_CALL(pthread_mutex_lock(&read_mutex_));
    consumer_arg_ = consumer_arg_t{this, true, data_begin, data_end};
    PTHREAD_CALL(pthread_cond_signal(&read_cond_));
    PTHREAD_CALL(pthread_mutex_unlock(&read_mutex_));
  }

  // pool allocator
  roctracer_allocator_t alloc_fun_;
  void* alloc_arg_;

  // Pool definition
  size_t buffer_size_;
  char* pool_begin_;
  char* pool_end_;
  char* buffer_begin_;
  char* buffer_end_;
  char* write_ptr_;
  mutex_t write_mutex_;

  // Consuming read thread
  roctracer_buffer_callback_t read_callback_fun_;
  void* read_callback_arg_;
  consumer_arg_t consumer_arg_;
  pthread_t consumer_thread_;
  pthread_mutex_t read_mutex_;
  pthread_cond_t read_cond_;
};

class Timer {
  public:
  typedef uint64_t timestamp_t;
  typedef long double freq_t;
  
  Timer() {
    timestamp_t timestamp_hz = 0;
    HSART_CALL(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_hz));
    timestamp_factor_ = (freq_t)1000000000 / (freq_t)timestamp_hz;
  }

  // Return timestamp in 'ns'
  timestamp_t timestamp_ns() {
    timestamp_t timestamp;
    HSART_CALL(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &timestamp));
    return timestamp_t((freq_t)timestamp * timestamp_factor_);
  }

  private:
  // Timestamp frequency factor
  freq_t timestamp_factor_;
};

CONSTRUCTOR_API void constructor() {
  util::Logger::Create();
}

DESTRUCTOR_API void destructor() {
  util::HsaRsrcFactory::Destroy();
  util::Logger::Destroy();
}

// Activity callback to generate an activity record
void ActivityCallback(
    roctracer_record_t* record,
    uint32_t activity_kind,
    const void* callback_data,
    void* arg)
{
  static Timer timer;

  const hip_cb_data_t* data = reinterpret_cast<const hip_cb_data_t*>(callback_data);
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(arg);
  if (pool == NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "ActivityCallback pool is NULL");
  if (data->phase == ROCTRACER_API_PHASE_ENTER) {
    *record = {};
    record->name = data->name;
    record->activity_kind = activity_kind;
    record->begin_ns = timer.timestamp_ns();
    // Correlation ID generating
    const auto correlation_id = GlobalCounter::Increment();
    record->correlation_id = correlation_id;
    const_cast<hip_cb_data_t*>(data)->correlation_id = correlation_id;
  } else {
    record->end_ns = timer.timestamp_ns();
    pool->Write<roctracer_record_t>(*record);
  }
}

util::Logger::mutex_t util::Logger::mutex_;
util::Logger* util::Logger::instance_ = NULL;
MemoryPool* memory_pool = NULL;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public library methods
//
extern "C" {

// Returns library vesrion
PUBLIC_API uint32_t roctracer_version_major() { return ROCTRACER_VERSION_MAJOR; }
PUBLIC_API uint32_t roctracer_version_minor() { return ROCTRACER_VERSION_MINOR; }

// Returns the last error
PUBLIC_API const char* roctracer_error_string() {
  return strdup(roctracer::util::Logger::LastMessage().c_str());
}

// Return method name by given API domain and call ID
// NULL returned on the error and the library errno is set
PUBLIC_API const char* roctracer_get_api_name(roctracer_api_domain_t domain, roctracer_hip_api_cid_t cid) {
  return NULL;
}

// Enable runtime API callbacks
PUBLIC_API int roctracer_enable_api_callback(
    roctracer_api_domain_t domain,
    uint32_t cid,
    roctracer_api_callback_t callback,
    void* user_data)
{
  API_METHOD_PREFIX
  switch (domain) {
    case ROCTRACER_API_DOMAIN_HIP: {
      hipError_t hip_err = hipRegisterApiCallback(cid, callback, user_data);
      if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "hipRegisterApiCallback error(" << hip_err << ")");
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  API_METHOD_SUFFIX
}

// Enable runtime API callbacks
PUBLIC_API int roctracer_disable_api_callback(
    roctracer_api_domain_t domain,
    uint32_t cid)
{
  API_METHOD_PREFIX
  switch (domain) {
    case ROCTRACER_API_DOMAIN_HIP: {
      hipError_t hip_err = hipRemoveApiCallback(cid);
      if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "hipRemoveApiCallback error(" << hip_err << ")");
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  API_METHOD_SUFFIX
}

// Return default pool and set new one if parameter pool is not NULL.
roctracer_pool_t* roctracer_default_pool(roctracer_pool_t* pool) {
  roctracer_pool_t* p = reinterpret_cast<roctracer_pool_t*>(roctracer::memory_pool);
  if (pool != NULL) roctracer::memory_pool = reinterpret_cast<roctracer::MemoryPool*>(pool);
  if (p == NULL) EXC_RAISING(ROCTRACER_STATUS_UNINIT, "default pool is not initialized");
  return p;
}

// Open memory pool
PUBLIC_API int roctracer_open_pool(
    const roctracer_properties_t* properties,
    roctracer_pool_t** pool)
{
  API_METHOD_PREFIX
  if ((pool == NULL) && (roctracer::memory_pool != NULL)) {
    EXC_RAISING(ROCTRACER_STATUS_ERROR, "default pool already set");
  }
  roctracer::MemoryPool* p = new roctracer::MemoryPool(*properties);
  if (p == NULL) EXC_RAISING(ROCTRACER_STATUS_ERROR, "MemoryPool() error");
  if (pool != NULL) *pool = p;
  else roctracer::memory_pool = p;
  API_METHOD_SUFFIX
}

// Close memory pool
PUBLIC_API int roctracer_close_pool(roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  roctracer_pool_t* ptr = (pool == NULL) ? roctracer_default_pool() : pool;
  roctracer::MemoryPool* memory_pool = reinterpret_cast<roctracer::MemoryPool*>(ptr);
  delete(memory_pool);
  if (pool == NULL) roctracer::memory_pool = NULL;
  API_METHOD_SUFFIX
}

// Enable activity records logging
PUBLIC_API int roctracer_enable_api_activity(
    roctracer_api_domain_t domain,
    uint32_t activity_kind,
    roctracer_pool_t* pool)
{
  API_METHOD_PREFIX
  if (pool == NULL) pool = roctracer_default_pool();
  switch (domain) {
    case ROCTRACER_API_DOMAIN_HIP: {
      const hipError_t hip_err = hipRegisterActivityCallback(activity_kind, roctracer::ActivityCallback, pool);
      if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "hipRegisterActivityCallback error(" << hip_err << ")");
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  API_METHOD_SUFFIX
}

// Disable activity records logging
PUBLIC_API int roctracer_disable_api_activity(
    roctracer_api_domain_t domain,
    uint32_t activity_kind)
{
  API_METHOD_PREFIX
  switch (domain) {
    case ROCTRACER_API_DOMAIN_HIP: {
      const hipError_t hip_err = hipRemoveActivityCallback(activity_kind);
      if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "hipRemoveActivityCallback error(" << hip_err << ")");
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  API_METHOD_SUFFIX
}

// Flush available activity records
PUBLIC_API int roctracer_flush_api_activity(roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  if (pool == NULL) pool = roctracer_default_pool();
  roctracer::MemoryPool* memory_pool = reinterpret_cast<roctracer::MemoryPool*>(pool);
  memory_pool->Flush();
  API_METHOD_SUFFIX
}

}  // extern "C"
