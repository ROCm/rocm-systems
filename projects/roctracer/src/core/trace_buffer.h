#ifndef SRC_CORE_TRACE_BUFFER_H_
#define SRC_CORE_TRACE_BUFFER_H_

#include <atomic>
#include <iostream>
#include <list>
#include <mutex>
#include <sstream>

#include <pthread.h>
#include <string.h>
#include <unistd.h>

#define FATAL(stream)                                                                              \
  do {                                                                                             \
    std::ostringstream oss;                                                                        \
    oss << __FUNCTION__ << "(), " << stream;                                                       \
    std::cout << oss.str() << std::endl;                                                           \
    abort();                                                                                       \
  } while (0)

#define PTHREAD_CALL(call)                                                                         \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      errno = err;                                                                                 \
      perror(#call);                                                                               \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

namespace roctracer {
enum {
  TRACE_ENTRY_INV = 0,
  TRACE_ENTRY_INIT = 1,
  TRACE_ENTRY_COMPL = 2
};

enum {
  API_ENTRY_TYPE,
  COPY_ENTRY_TYPE,
  KERNEL_ENTRY_TYPE
};

struct trace_entry_t {
  std::atomic<uint32_t> valid;
  uint32_t type;
  uint64_t dispatch;
  uint64_t begin;                                      // kernel begin timestamp, ns
  uint64_t end;                                        // kernel end timestamp, ns
  uint64_t complete;
  hsa_agent_t agent;
  uint32_t dev_index;
  hsa_signal_t orig;
  hsa_signal_t signal;
  union {
    struct {
    } copy;
    struct {
      const char* name;
      hsa_agent_t agent;
      uint32_t tid;
    } kernel;
  };
};

template <class T>
struct push_element_fun {
  T* const elem_;
  void fun(T* node) { if (node->next_elem_ == NULL) node->next_elem_ = elem_; }
  push_element_fun(T* elem) : elem_(elem) {}
};

template <class T>
struct call_element_fun {
  void (T::*fptr_)();
  void fun(T* node) { (node->*fptr_)(); }
  call_element_fun(void (T::*f)()) : fptr_(f) {}
};

struct TraceBufferBase {
  typedef std::mutex mutex_t;

  virtual void StartWorkerThread() = 0;
  virtual void Flush() = 0;

  static void StartWorkerThreadAll() { foreach(call_element_fun<TraceBufferBase>(&TraceBufferBase::StartWorkerThread)); }
  static void FlushAll() { foreach(call_element_fun<TraceBufferBase>(&TraceBufferBase::Flush)); }

  static void Push(TraceBufferBase* elem) {
    if (head_elem_ == NULL) head_elem_ = elem;
    else foreach(push_element_fun<TraceBufferBase>(elem));
  }

  TraceBufferBase() : next_elem_(NULL) {}

  template<class F>
  static void foreach(const F& f_in) {
    std::lock_guard<mutex_t> lck(mutex_);
    F f = f_in;
    TraceBufferBase* p = head_elem_;
    while (p != NULL) {
      TraceBufferBase* next = p->next_elem_;
      f.fun(p);
      p = next;
    }
  }

  TraceBufferBase* next_elem_;
  static TraceBufferBase* head_elem_;
  static mutex_t mutex_;
};

template <typename Entry>
class TraceBuffer : protected TraceBufferBase {
  public:
  typedef void (*callback_t)(Entry*);
  typedef TraceBuffer<Entry> Obj;
  typedef uint64_t pointer_t;
  typedef std::mutex mutex_t;

  struct flush_prm_t {
    uint32_t type;
    callback_t fun;
  };

  TraceBuffer(const char* name, uint32_t size, flush_prm_t* flush_prm_arr, uint32_t flush_prm_count) :
    is_flushed_(false),
    work_thread_started_(false)
  {
    name_ = strdup(name);
    size_ = size;
    data_ = allocate_fun();
    next_ = NULL;
    read_pointer_ = 0;
    end_pointer_ = size;
    buf_list_.push_back(data_);

    flush_prm_arr_ = flush_prm_arr;
    flush_prm_count_ = flush_prm_count;

    TraceBufferBase::Push(this);
  }

  ~TraceBuffer() {
    StopWorkerThread();
    Flush();
  }

  void StartWorkerThread() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (work_thread_started_ == false) {
      PTHREAD_CALL(pthread_mutex_init(&work_mutex_, NULL));
      PTHREAD_CALL(pthread_cond_init(&work_cond_, NULL));
      PTHREAD_CALL(pthread_create(&work_thread_, NULL, allocate_worker, this));
      work_thread_started_ = true;
    }
  }

  void StopWorkerThread() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (work_thread_started_ == true) {
      PTHREAD_CALL(pthread_cancel(work_thread_));
      void *res;
      PTHREAD_CALL(pthread_join(work_thread_, &res));
      if (res != PTHREAD_CANCELED) FATAL("consumer thread wasn't stopped correctly");
      work_thread_started_ = false;
    }
  }

  Entry* GetEntry() {
    const pointer_t pointer = read_pointer_.fetch_add(1);
    if (pointer >= end_pointer_) wrap_buffer(pointer);
    if (pointer >= end_pointer_) FATAL("pointer >= end_pointer_ after buffer wrap");
    return data_ + (pointer + size_ - end_pointer_);
  }

  void Flush() { flush_buf(); }

  private:
  void flush_buf() {
    std::lock_guard<mutex_t> lck(mutex_);
    const bool is_flushed = is_flushed_.exchange(true, std::memory_order_acquire);

    if (is_flushed == false) {
      for (flush_prm_t* prm = flush_prm_arr_; prm < flush_prm_arr_ + flush_prm_count_; prm++) {
        // Flushed entries type
        uint32_t type = prm->type;
        // Flushing function
        callback_t fun = prm->fun;
        if (fun == NULL) FATAL("flush function is not set");

        pointer_t pointer = 0;
        for (Entry* ptr : buf_list_) {
          Entry* end = ptr + size_;
          while ((ptr < end) && (pointer < read_pointer_)) {
            if (ptr->type == type) {
              if (ptr->valid == TRACE_ENTRY_COMPL) {
                fun(ptr);
              }
            }
            ptr++;
            pointer++;
          }
        }
      }
    }
  }

  inline Entry* allocate_fun() {
    Entry* ptr = (Entry*) malloc(size_ * sizeof(Entry));
    if (ptr == NULL) FATAL("malloc failed");
    //memset(ptr, 0, size_ * sizeof(Entry));
    return ptr;
  }

  static void* allocate_worker(void* arg) {
    Obj* obj = (Obj*)arg;

    while (1) {
      PTHREAD_CALL(pthread_mutex_lock(&(obj->work_mutex_)));
      while (obj->next_ != NULL) {
        PTHREAD_CALL(pthread_cond_wait(&(obj->work_cond_), &(obj->work_mutex_)));
      }
      obj->next_ = obj->allocate_fun();
      PTHREAD_CALL(pthread_mutex_unlock(&(obj->work_mutex_)));
    }

    return NULL;
  }

  void wrap_buffer(const pointer_t pointer) {
    std::lock_guard<mutex_t> lck(mutex_);
    if (work_thread_started_ == false) StartWorkerThread();

    PTHREAD_CALL(pthread_mutex_lock(&work_mutex_));
    if (pointer >= end_pointer_) {
      data_ = next_;
      next_ = NULL;
      PTHREAD_CALL(pthread_cond_signal(&work_cond_));
      end_pointer_ += size_;
      if (end_pointer_ == 0) FATAL("pointer overflow");
      buf_list_.push_back(data_);
    }
    PTHREAD_CALL(pthread_mutex_unlock(&work_mutex_));
  }

  const char* name_;
  uint32_t size_;
  Entry* data_;
  Entry* next_;
  volatile std::atomic<pointer_t> read_pointer_;
  volatile std::atomic<pointer_t> end_pointer_;
  std::list<Entry*> buf_list_;

  flush_prm_t* flush_prm_arr_;
  uint32_t flush_prm_count_;
  volatile std::atomic<bool> is_flushed_;

  pthread_t work_thread_;
  pthread_mutex_t work_mutex_;
  pthread_cond_t work_cond_;
  bool work_thread_started_;

  mutex_t mutex_;
};
}  // namespace roctracer

#define TRACE_BUFFER_INSTANTIATE() \
  roctracer::TraceBufferBase* roctracer::TraceBufferBase::head_elem_ = NULL; \
  roctracer::TraceBufferBase::mutex_t roctracer::TraceBufferBase::mutex_;

#endif  // SRC_CORE_TRACE_BUFFER_H_
