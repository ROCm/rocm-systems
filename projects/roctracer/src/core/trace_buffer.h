#ifndef SRC_CORE_TRACE_BUFFER_H_
#define SRC_CORE_TRACE_BUFFER_H_

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

template <typename Entry>
class TraceBuffer {
  public:
  typedef void (*callabck_t)(Entry*);

  TraceBuffer(uint32_t size) {
    size_ = size;
    data_ = new Entry[size_];;
    memset(data_, 0, size_ * sizeof(Entry));
    read_pointer_ = data_;
  }

  Entry* GetEntry() {
    Entry* ptr = read_pointer_.fetch_add(1);
    if (ptr >= (data_ + size_))  {
      fprintf(stderr, "GetEntry: trace buffer is out of range\n");
      abort();
    }
    return ptr;
  }

  void Flush(uint32_t type, callabck_t fun) {
    Entry* ptr = data_;
    for (; (ptr < read_pointer_) && (ptr < (data_ + size_)); ptr++) {
      if (ptr->type == type) {
        if (ptr->valid == TRACE_ENTRY_COMPL) {
          fun(ptr);
        }
      }
    }
    if (ptr >= (data_ + size_))  {
      fprintf(stderr, "Flush: trace buffer is out of range\n");
    }
  }

  private:
  Entry* data_;
  uint32_t size_;
  std::atomic<Entry*> read_pointer_;
};
}  // namespace roctracer

#endif  // SRC_CORE_TRACE_BUFFER_H_
