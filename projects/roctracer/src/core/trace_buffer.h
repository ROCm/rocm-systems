/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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

enum entry_type_t {
  DFLT_ENTRY_TYPE = 0,
  API_ENTRY_TYPE = 1,
  COPY_ENTRY_TYPE = 2,
  KERNEL_ENTRY_TYPE = 3,
  NUM_ENTRY_TYPE = 4
};

struct trace_entry_t {
  std::atomic<uint32_t> valid;
  entry_type_t type;
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
  T** prev_;
  bool fun(T* node) {
    if (node->priority_ > elem_->priority_) {
      *prev_ = elem_;
      elem_->next_elem_ = node;
    } else if (node->next_elem_ == NULL) {
      node->next_elem_ = elem_;
    } else {
      prev_ = &(node->next_elem_);
      return false;
    }
    return true;
  }
  push_element_fun(T* elem, T** prev) : elem_(elem), prev_(prev) {}
};

template <class T>
struct call_element_fun {
  void (T::*fptr_)();
  bool fun(T* node) const { (node->*fptr_)(); return false; }
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
    else foreach(push_element_fun<TraceBufferBase>(elem, &head_elem_));
  }

  TraceBufferBase(const uint32_t& prior) : priority_(prior), next_elem_(NULL) {}

  template<class F>
  static void foreach(const F& f_in) {
    std::lock_guard<mutex_t> lck(mutex_);
    F f = f_in;
    TraceBufferBase* p = head_elem_;
    while (p != NULL) {
      TraceBufferBase* next = p->next_elem_;
      if (f.fun(p) == true) break;
      p = next;
    }
  }

  const uint32_t priority_;
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
  typedef std::recursive_mutex mutex_t;
  typedef typename std::list<Entry*> buf_list_t;
  typedef typename buf_list_t::iterator buf_list_it_t;

  struct flush_prm_t {
    entry_type_t type;
    callback_t fun;
  };

  TraceBuffer(const char* name, uint32_t size, const flush_prm_t* flush_prm_arr, uint32_t flush_prm_count, uint32_t prior = 0) :
    TraceBufferBase(prior),
    size_(size),
    work_thread_started_(false)
  {
    name_ = strdup(name);
    data_ = allocate_fun();
    next_ = allocate_fun();
    read_pointer_ = 0;
    write_pointer_ = 0;
    end_pointer_ = size;
    buf_list_.push_back(data_);

    memset(f_array_, 0, sizeof(f_array_));
    for (const flush_prm_t* prm = flush_prm_arr; prm < flush_prm_arr + flush_prm_count; prm++) {
      const entry_type_t type = prm->type;
      if (type >= NUM_ENTRY_TYPE) FATAL("out of f_array bounds (" << type << ")");
      if (f_array_[type] != NULL) FATAL("handler function ptr redefinition (" << type << ")");
      f_array_[type] = prm->fun;
    }

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
    const pointer_t pointer = write_pointer_.fetch_add(1);
    if (pointer >= end_pointer_) wrap_buffer(pointer);
    if (pointer >= end_pointer_) FATAL("pointer >= end_pointer_ after buffer wrap");
    Entry* entry = data_ + (size_ + pointer - end_pointer_);
    entry->valid = TRACE_ENTRY_INV;
    entry->type = DFLT_ENTRY_TYPE;
    return entry;
  }

  void Flush() { flush_buf(); }

  private:
  void flush_buf() {
    std::lock_guard<mutex_t> lck(mutex_);

    pointer_t pointer = read_pointer_;
    pointer_t curr_pointer = write_pointer_.load(std::memory_order_relaxed);
    buf_list_it_t it = buf_list_.begin();
    buf_list_it_t end_it = buf_list_.end();
    while(it != end_it) {
      Entry* buf = *it;
      Entry* ptr = buf + (pointer % size_);
      Entry* end_ptr = buf + size_;
      while ((ptr < end_ptr) && (pointer < curr_pointer)) {
        if (ptr->valid != TRACE_ENTRY_COMPL) break;

        entry_type_t type = ptr->type;
        if (type >= NUM_ENTRY_TYPE) FATAL("out of f_array bounds (" << type << ")");
        callback_t f_ptr = f_array_[type];
        if (f_ptr == NULL) FATAL("f_ptr == NULL");
        (*f_ptr)(ptr);

        ptr++;
        pointer++;
      }

      buf_list_it_t prev = it;
      it++;
      if (ptr == end_ptr) {
        free_fun(*prev);
        buf_list_.erase(prev);
      }
      if (pointer == curr_pointer) break;
    }

    read_pointer_ = pointer;
  }

  inline Entry* allocate_fun() {
    Entry* ptr = (Entry*) malloc(size_ * sizeof(Entry));
    if (ptr == NULL) FATAL("malloc failed");
    //memset(ptr, 0, size_ * sizeof(Entry));
    return ptr;
  }

  inline void free_fun(void* ptr) {
    free(ptr);
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
  const uint32_t size_;
  Entry* data_;
  Entry* next_;
  pointer_t read_pointer_;
  volatile std::atomic<pointer_t> write_pointer_;
  volatile std::atomic<pointer_t> end_pointer_;
  buf_list_t buf_list_;
  callback_t f_array_[NUM_ENTRY_TYPE];

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
