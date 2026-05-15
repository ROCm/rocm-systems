//
// Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc. All rights reserved.
//

#include <CL/opencl.h>
#include <vdi_agent_amd.h>

#if defined(CL_VERSION_2_0)
/* Deprecated in OpenCL 2.0 */
#define CL_DEVICE_QUEUE_PROPERTIES 0x102A
#define CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035
#endif

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdint>

#ifdef _MSC_VER
#include <windows.h>
#include <intrin.h>
#include <process.h>
#else
#include <unistd.h>
#include <pthread.h>
#endif

#define CASE(x)                                                                                    \
  case x:                                                                                          \
    return #x;

std::ofstream clTraceLog;
std::streambuf* cerrStreamBufSave;

// A call record with links for the checker
struct Rec {
  Rec* next;
  Rec* prev;
  std::ostringstream* sp;
  int visits;

  Rec() : sp(0) {}
  Rec(std::ostringstream* ps) : sp(ps), visits(0) {}
};

// This is the head of the checker Rec list
static Rec recs;

// About how many times per second the checker runs
static const int checks_per_second = 10;

// Some OS independent synchronization for the checker Rec list
#ifdef _MSC_VER
#define CHECKERTYPE static void
#define CHECKERRETURN return
static CRITICAL_SECTION recsCS[1];

static inline void initRecs(void) {
  InitializeCriticalSection(recsCS);
  recs.next = &recs;
  recs.prev = &recs;
}

static inline void lockRecs(void) { EnterCriticalSection(recsCS); }

static inline void unlockRecs(void) { LeaveCriticalSection(recsCS); }

static inline void waitRecs(void) { Sleep(1000 / checks_per_second); }
#else
#define CHECKERTYPE static void*
#define CHECKERRETURN return NULL
static pthread_mutex_t recsMtx = PTHREAD_MUTEX_INITIALIZER;

static inline void initRecs(void) {
  recs.next = &recs;
  recs.prev = &recs;
}

static inline void lockRecs(void) { pthread_mutex_lock(&recsMtx); }

static inline void unlockRecs(void) { pthread_mutex_unlock(&recsMtx); }

static inline void waitRecs(void) { usleep(1000000 / checks_per_second); }
#endif

// Link into checker Rec list
static inline void addRec(Rec* r) {
  lockRecs();
  r->next = recs.next;
  r->prev = &recs;
  recs.next->prev = r;
  recs.next = r;
  unlockRecs();
}

// unlink from checker Rec list
static inline void delRec(Rec* r) {
  lockRecs();
  r->next->prev = r->prev;
  r->prev->next = r->next;
  unlockRecs();
}

// This is the checker thread function
CHECKERTYPE
checker(void*) {
  Rec* b;
  Rec* e = &recs;

  for (;;) {
    // Wait for a while
    waitRecs();

    std::ostringstream ss;
    int go = 0;

    lockRecs();
    for (b = recs.next; b != e; b = b->next) {
      ++b->visits;
      if (b->visits == 2) {
        // This record has been on the list for a while
        // we'll log it in case the thread has hung
        ss << "Waiting for " << b->sp->str() << std::endl;
        go = 1;
      }
    }
    unlockRecs();

    if (go) std::cerr << ss.str();
  }
  CHECKERRETURN;
}

#ifdef _MSC_VER
static cl_int startChecker(void) {
  uintptr_t h = _beginthread(checker, 0, NULL);
  return h == 0;
}
#else
static cl_int startChecker(void) {
  int e;
  pthread_t tid;
  pthread_attr_t pa;

  e = pthread_attr_init(&pa);
  if (e) return e;

  e = pthread_attr_setdetachstate(&pa, PTHREAD_CREATE_DETACHED);
  if (e) return e;

  e = pthread_create(&tid, &pa, checker, NULL);
  return e;
}
#endif

template <typename T> std::string getDecimalString(T value) {
  std::ostringstream ss;
  ss << value;
  return ss.str();
}

template <typename T> std::string getDecimalString(T* value) {
  if (value == NULL) {
    return "NULL";
  }

  std::ostringstream ss;
  ss << '&' << *value;
  return ss.str();
}

template <typename T> std::string getHexString(T value) {
  std::ostringstream ss;
  ss << "0x" << std::hex << value;
  return ss.str();
}

template <typename T> std::string getHexString(T* value) {
  if (value == NULL) {
    return "NULL";
  }

  std::ostringstream ss;
  ss << "&0x" << std::hex << *value;
  return ss.str();
}

template <typename T> std::string getHexString(T** value) {
  if (value == NULL) {
    return "NULL";
  }

  std::ostringstream ss;
  ss << "&" << *value;
  return ss.str();
}

template <> std::string getHexString(void* value) {
  return getHexString(reinterpret_cast<intptr_t>(value));
}

static std::string getMemoryString(const void* ptr, size_t size) {
  switch (size) {
    case 1:
      return getHexString((const char*)ptr);
    case 2:
      return getHexString((const short*)ptr);
    case 4:
      return getHexString((const int*)ptr);
    case 8:
      return getHexString((const long long*)ptr);
    default:
      break;
  }
  std::ostringstream ss;
  ss << "&" << ptr;
  return ss.str();
}

static std::string getBoolString(cl_bool b) { return (b == CL_TRUE) ? "CL_TRUE" : "CL_FALSE"; }

static std::string getNDimString(const size_t* nd, size_t dims) {
  if (nd == NULL) {
    return "NULL";
  }
  if (dims == 0) {
    return "[]";
  }

  std::ostringstream ss;
  ss << '[' << nd[0];
  if (dims > 1) {
    ss << ',' << nd[1];
    if (dims > 2) {
      ss << ',' << nd[2];
    }
  }
  ss << ']';
  return ss.str();
}

static std::string getErrorString(cl_int errcode) {
  switch (errcode) {
    CASE(CL_SUCCESS);
    CASE(CL_DEVICE_NOT_FOUND);
    CASE(CL_DEVICE_NOT_AVAILABLE);
    CASE(CL_COMPILER_NOT_AVAILABLE);
    CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    CASE(CL_OUT_OF_RESOURCES);
    CASE(CL_OUT_OF_HOST_MEMORY);
    CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
    CASE(CL_MEM_COPY_OVERLAP);
    CASE(CL_IMAGE_FORMAT_MISMATCH);
    CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    CASE(CL_BUILD_PROGRAM_FAILURE);
    CASE(CL_MAP_FAILURE);
    CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    CASE(CL_INVALID_VALUE);
    CASE(CL_INVALID_DEVICE_TYPE);
    CASE(CL_INVALID_PLATFORM);
    CASE(CL_INVALID_DEVICE);
    CASE(CL_INVALID_CONTEXT);
    CASE(CL_INVALID_QUEUE_PROPERTIES);
    CASE(CL_INVALID_COMMAND_QUEUE);
    CASE(CL_INVALID_HOST_PTR);
    CASE(CL_INVALID_MEM_OBJECT);
    CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    CASE(CL_INVALID_IMAGE_SIZE);
    CASE(CL_INVALID_SAMPLER);
    CASE(CL_INVALID_BINARY);
    CASE(CL_INVALID_BUILD_OPTIONS);
    CASE(CL_INVALID_PROGRAM);
    CASE(CL_INVALID_PROGRAM_EXECUTABLE);
    CASE(CL_INVALID_KERNEL_NAME);
    CASE(CL_INVALID_KERNEL_DEFINITION);
    CASE(CL_INVALID_KERNEL);
    CASE(CL_INVALID_ARG_INDEX);
    CASE(CL_INVALID_ARG_VALUE);
    CASE(CL_INVALID_ARG_SIZE);
    CASE(CL_INVALID_KERNEL_ARGS);
    CASE(CL_INVALID_WORK_DIMENSION);
    CASE(CL_INVALID_WORK_GROUP_SIZE);
    CASE(CL_INVALID_WORK_ITEM_SIZE);
    CASE(CL_INVALID_GLOBAL_OFFSET);
    CASE(CL_INVALID_EVENT_WAIT_LIST);
    CASE(CL_INVALID_EVENT);
    CASE(CL_INVALID_OPERATION);
    CASE(CL_INVALID_GL_OBJECT);
    CASE(CL_INVALID_BUFFER_SIZE);
    CASE(CL_INVALID_MIP_LEVEL);
    CASE(CL_INVALID_GLOBAL_WORK_SIZE);
    default:
      return getDecimalString(errcode);
  }
}

static std::string getMemObjectTypeString(cl_mem_object_type type) {
  switch (type) {
    CASE(CL_MEM_OBJECT_BUFFER);
    CASE(CL_MEM_OBJECT_IMAGE2D);
    CASE(CL_MEM_OBJECT_IMAGE3D);
    default:
      return getHexString(type);
  }
}

static std::string getMemInfoString(cl_mem_info param_name) {
  switch (param_name) {
    CASE(CL_MEM_TYPE);
    CASE(CL_MEM_FLAGS);
    CASE(CL_MEM_SIZE);
    CASE(CL_MEM_HOST_PTR);
    CASE(CL_MEM_MAP_COUNT);
    CASE(CL_MEM_REFERENCE_COUNT);
    CASE(CL_MEM_CONTEXT);
    CASE(CL_MEM_ASSOCIATED_MEMOBJECT);
    CASE(CL_MEM_OFFSET);
    default:
      return getHexString(param_name);
  }
}

static std::string getImageInfoString(cl_image_info param_name) {
  switch (param_name) {
    CASE(CL_IMAGE_FORMAT);
    CASE(CL_IMAGE_ELEMENT_SIZE);
    CASE(CL_IMAGE_ROW_PITCH);
    CASE(CL_IMAGE_SLICE_PITCH);
    CASE(CL_IMAGE_WIDTH);
    CASE(CL_IMAGE_HEIGHT);
    CASE(CL_IMAGE_DEPTH);
    default:
      return getHexString(param_name);
  }
}

static std::string getErrorString(cl_int* errcode_ret) {
  if (errcode_ret == NULL) {
    return "NULL";
  }

  std::ostringstream ss;
  ss << '&' << getErrorString(*errcode_ret);
  return ss.str();
}

static std::string getHandlesString(const void* handles, cl_uint num_handles) {
  if (handles == NULL) {
    return "NULL";
  }
  if (num_handles == 0) {
    return "[]";
  }

  const cl_event* p = reinterpret_cast<const cl_event*>(handles);

  std::ostringstream ss;
  ss << '[';
  while (true) {
    ss << *p++;
    if (--num_handles == 0) {
      break;
    }
    ss << ',';
  }
  ss << ']';
  return ss.str();
}

static std::string getContextPropertyString(cl_context_properties cprop) {
  switch (cprop) {
    CASE(CL_CONTEXT_PLATFORM);
    default:
      return getHexString(cprop);
  }
}

static std::string getContextPropertiesString(const cl_context_properties* cprops) {
  if (cprops == NULL) {
    return "NULL";
  }

  std::ostringstream ss;
  ss << '{';
  while (*cprops != 0) {
    ss << getContextPropertyString(cprops[0]) << ',' << getHexString(cprops[1]) << ",";
    cprops += 2;
  }
  ss << "NULL}";
  return ss.str();
}

static std::string getCommandQueuePropertyString(cl_command_queue_properties property) {
  if (property == 0) {
    return "0";
  }

  std::ostringstream ss;
  while (property) {
    if (property & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
      ss << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE";
      property &= ~CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    } else if (property & CL_QUEUE_PROFILING_ENABLE) {
      ss << "CL_QUEUE_PROFILING_ENABLE";
      property &= ~CL_QUEUE_PROFILING_ENABLE;
    } else {
      ss << "0x" << std::hex << (int)property;
      property = 0;
    }
    if (property != 0) {
      ss << '|';
    }
  }

  return ss.str();
}

static std::string getQueuePropertyString(const cl_queue_properties* qprops) {
  if (qprops == NULL) {
    return "NULL";
  }

  std::ostringstream ss;
  cl_command_queue_properties property = 0;
  unsigned int queueSize = 0;

  const struct QueueProperty {
    cl_queue_properties name;
    union {
      cl_queue_properties raw;
      cl_uint size;
    } value;
  }* p = reinterpret_cast<const QueueProperty*>(qprops);

  if (p != NULL)
    while (p->name != 0) {
      switch (p->name) {
        case CL_QUEUE_PROPERTIES:
          property = static_cast<cl_command_queue_properties>(p->value.raw);

          if (property & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
            ss << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE";
            property &= ~CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
          } else if (property & CL_QUEUE_PROFILING_ENABLE) {
            ss << "CL_QUEUE_PROFILING_ENABLE";
            property &= ~CL_QUEUE_PROFILING_ENABLE;
          } else {
            ss << "0x" << std::hex << (int)property;
            property = 0;
          }
          if (property != 0) {
            ss << '|';
          }
          break;
        case CL_QUEUE_SIZE:  // Unimplemented
          queueSize = p->value.size;
          ss << "QUEUE_SIZE " << queueSize;
          break;
#define CL_QUEUE_REAL_TIME_COMPUTE_UNITS_AMD 0x404f
        case CL_QUEUE_REAL_TIME_COMPUTE_UNITS_AMD:
          queueSize = p->value.size;
          ss << " RT_QUEUE " << queueSize;
          break;
#define CL_QUEUE_MEDIUM_PRIORITY_AMD 0x4050
        case CL_QUEUE_MEDIUM_PRIORITY_AMD:
          queueSize = p->value.size;
          ss << " MEDIUM_PRIORITY " << queueSize;
          break;
        default:
          break;
      }
      ++p;
    }

  return ss.str();
}

static std::string getMemFlagsString(cl_mem_flags flags) {
  if (flags == 0) {
    return "0";
  }

  std::ostringstream ss;
  while (flags) {
    if (flags & CL_MEM_READ_WRITE) {
      ss << "CL_MEM_READ_WRITE";
      flags &= ~CL_MEM_READ_WRITE;
    } else if (flags & CL_MEM_WRITE_ONLY) {
      ss << "CL_MEM_WRITE_ONLY";
      flags &= ~CL_MEM_WRITE_ONLY;
    } else if (flags & CL_MEM_READ_ONLY) {
      ss << "CL_MEM_READ_ONLY";
      flags &= ~CL_MEM_READ_ONLY;
    } else if (flags & CL_MEM_USE_HOST_PTR) {
      ss << "CL_MEM_USE_HOST_PTR";
      flags &= ~CL_MEM_USE_HOST_PTR;
    } else if (flags & CL_MEM_ALLOC_HOST_PTR) {
      ss << "CL_MEM_ALLOC_HOST_PTR";
      flags &= ~CL_MEM_ALLOC_HOST_PTR;
    } else if (flags & CL_MEM_COPY_HOST_PTR) {
      ss << "CL_MEM_COPY_HOST_PTR";
      flags &= ~CL_MEM_COPY_HOST_PTR;
    } else {
      ss << "0x" << std::hex << (int)flags;
      flags = 0;
    }
    if (flags != 0) {
      ss << '|';
    }
  }

  return ss.str();
}

static std::string getMapFlagsString(cl_map_flags flags) {
  if (flags == 0) {
    return "0";
  }

  std::ostringstream ss;
  while (flags) {
    if (flags & CL_MAP_READ) {
      ss << "CL_MAP_READ";
      flags &= ~CL_MAP_READ;
    } else if (flags & CL_MAP_WRITE) {
      ss << "CL_MAP_WRITE";
      flags &= ~CL_MAP_WRITE;
    } else {
      ss << "0x" << std::hex << (int)flags;
      flags = 0;
    }
    if (flags != 0) {
      ss << '|';
    }
  }

  return ss.str();
}

static std::string getBufferCreateString(cl_buffer_create_type type, const void* info) {
  std::ostringstream ss;

  if (type == CL_BUFFER_CREATE_TYPE_REGION) {
    const cl_buffer_region* region = (const cl_buffer_region*)info;
    ss << "CL_BUFFER_CREATE_TYPE_REGION,{";
    ss << region->origin << ',' << region->size << '}';
  } else {
    ss << getHexString(type) << ',' << info;
  }
  return ss.str();
}

static std::string getChannelOrderString(cl_channel_order order) {
  switch (order) {
    CASE(CL_R);
    CASE(CL_A);
    CASE(CL_RG);
    CASE(CL_RA);
    CASE(CL_RGB);
    CASE(CL_RGBA);
    CASE(CL_BGRA);
    CASE(CL_ARGB);
    CASE(CL_INTENSITY);
    CASE(CL_LUMINANCE);
    CASE(CL_Rx);
    CASE(CL_RGx);
    CASE(CL_RGBx);
    default:
      return getHexString(order);
  }
}

static std::string getChannelTypeString(cl_channel_type type) {
  switch (type) {
    CASE(CL_SNORM_INT8);
    CASE(CL_SNORM_INT16);
    CASE(CL_UNORM_INT8);
    CASE(CL_UNORM_INT16);
    CASE(CL_UNORM_SHORT_565);
    CASE(CL_UNORM_SHORT_555);
    CASE(CL_UNORM_INT_101010);
    CASE(CL_SIGNED_INT8);
    CASE(CL_SIGNED_INT16);
    CASE(CL_SIGNED_INT32);
    CASE(CL_UNSIGNED_INT8);
    CASE(CL_UNSIGNED_INT16);
    CASE(CL_UNSIGNED_INT32);
    CASE(CL_HALF_FLOAT);
    CASE(CL_FLOAT);
    default:
      return getHexString(type);
  }
}

static std::string getImageFormatsString(const cl_image_format* format, size_t num_entries) {
  if (format == NULL) {
    return "NULL";
  }

  std::ostringstream ss;
  ss << '[';
  while (true) {
    ss << '{' << getChannelOrderString(format->image_channel_order) << ',';
    ss << getChannelTypeString(format->image_channel_data_type) << '}';
    if (--num_entries == 0) {
      break;
    }
    ss << ',';
  }
  ss << ']';
  return ss.str();
}

static std::string getImageDescString(const cl_image_desc* image_desc) {
  if (image_desc == NULL) {
    return "NULL";
  }

  std::ostringstream ss;
  ss << '{' << getMemObjectTypeString(image_desc->image_type) << ',';
  ss << image_desc->image_width << ',';
  ss << image_desc->image_height << ',';
  ss << image_desc->image_depth << ',';
  ss << image_desc->image_array_size << ',';
  ss << image_desc->image_row_pitch << ',';
  ss << image_desc->image_slice_pitch << ',';
  ss << image_desc->num_mip_levels << ',';
  ss << image_desc->num_samples << ',';
  ss << image_desc->mem_object << '}';
  return ss.str();
}


static std::string getAddressingModeString(cl_addressing_mode mode) {
  switch (mode) {
    CASE(CL_ADDRESS_NONE);
    CASE(CL_ADDRESS_CLAMP_TO_EDGE);
    CASE(CL_ADDRESS_CLAMP);
    CASE(CL_ADDRESS_REPEAT);
    CASE(CL_ADDRESS_MIRRORED_REPEAT);
    default:
      return getHexString(mode);
  }
}

std::string getFilterModeString(cl_filter_mode mode) {
  switch (mode) {
    CASE(CL_FILTER_NEAREST);
    CASE(CL_FILTER_LINEAR);
    default:
      return getHexString(mode);
  }
}

static std::string getSamplerInfoString(cl_sampler_info param_name) {
  switch (param_name) {
    CASE(CL_SAMPLER_REFERENCE_COUNT);
    CASE(CL_SAMPLER_CONTEXT);
    CASE(CL_SAMPLER_NORMALIZED_COORDS);
    CASE(CL_SAMPLER_ADDRESSING_MODE);
    CASE(CL_SAMPLER_FILTER_MODE);
    default:
      return getHexString(param_name);
  }
}


std::string getDeviceTypeString(cl_device_type type) {
  if (type == CL_DEVICE_TYPE_ALL) {
    return "CL_DEVICE_TYPE_ALL";
  }

  std::ostringstream ss;
  while (type) {
    if (type & CL_DEVICE_TYPE_CPU) {
      ss << "CL_DEVICE_TYPE_CPU";
      type &= ~CL_DEVICE_TYPE_CPU;
    } else if (type & CL_DEVICE_TYPE_GPU) {
      ss << "CL_DEVICE_TYPE_GPU";
      type &= ~CL_DEVICE_TYPE_GPU;
    } else if (type & CL_DEVICE_TYPE_ACCELERATOR) {
      ss << "CL_DEVICE_TYPE_ACCELERATOR";
      type &= ~CL_DEVICE_TYPE_ACCELERATOR;
    } else {
      ss << "0x" << std::hex << (int)type;
      type = 0;
    }
    if (type != 0) {
      ss << '|';
    }
  }

  return ss.str();
}

static std::string getPlatformInfoString(cl_platform_info param_name) {
  switch (param_name) {
    CASE(CL_PLATFORM_PROFILE);
    CASE(CL_PLATFORM_VERSION);
    CASE(CL_PLATFORM_NAME);
    CASE(CL_PLATFORM_VENDOR);
    CASE(CL_PLATFORM_EXTENSIONS);
    CASE(CL_PLATFORM_ICD_SUFFIX_KHR);
    default:
      return getHexString(param_name);
  }
}


static std::string getKernelArgInfoString(cl_kernel_arg_info param_name) {
  switch (param_name) {
    CASE(CL_KERNEL_ARG_ADDRESS_QUALIFIER);
    CASE(CL_KERNEL_ARG_ACCESS_QUALIFIER);
    CASE(CL_KERNEL_ARG_TYPE_NAME);
    CASE(CL_KERNEL_ARG_TYPE_QUALIFIER);
    CASE(CL_KERNEL_ARG_NAME);
    default:
      return getHexString(param_name);
  }
}

static std::string getDeviceInfoString(cl_device_info param_name) {
  switch (param_name) {
    CASE(CL_DEVICE_TYPE);
    CASE(CL_DEVICE_VENDOR_ID);
    CASE(CL_DEVICE_MAX_COMPUTE_UNITS);
    CASE(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    CASE(CL_DEVICE_MAX_WORK_GROUP_SIZE);
    CASE(CL_DEVICE_MAX_WORK_ITEM_SIZES);
    CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
    CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
    CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
    CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
    CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
    CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
    CASE(CL_DEVICE_MAX_CLOCK_FREQUENCY);
    CASE(CL_DEVICE_ADDRESS_BITS);
    CASE(CL_DEVICE_MAX_READ_IMAGE_ARGS);
    CASE(CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
    CASE(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    CASE(CL_DEVICE_IMAGE2D_MAX_WIDTH);
    CASE(CL_DEVICE_IMAGE2D_MAX_HEIGHT);
    CASE(CL_DEVICE_IMAGE3D_MAX_WIDTH);
    CASE(CL_DEVICE_IMAGE3D_MAX_HEIGHT);
    CASE(CL_DEVICE_IMAGE3D_MAX_DEPTH);
    CASE(CL_DEVICE_IMAGE_SUPPORT);
    CASE(CL_DEVICE_MAX_PARAMETER_SIZE);
    CASE(CL_DEVICE_MAX_SAMPLERS);
    CASE(CL_DEVICE_MEM_BASE_ADDR_ALIGN);
    CASE(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE);
    CASE(CL_DEVICE_SINGLE_FP_CONFIG);
    CASE(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE);
    CASE(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
    CASE(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
    CASE(CL_DEVICE_GLOBAL_MEM_SIZE);
    CASE(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    CASE(CL_DEVICE_MAX_CONSTANT_ARGS);
    CASE(CL_DEVICE_LOCAL_MEM_TYPE);
    CASE(CL_DEVICE_LOCAL_MEM_SIZE);
    CASE(CL_DEVICE_ERROR_CORRECTION_SUPPORT);
    CASE(CL_DEVICE_PROFILING_TIMER_RESOLUTION);
    CASE(CL_DEVICE_ENDIAN_LITTLE);
    CASE(CL_DEVICE_AVAILABLE);
    CASE(CL_DEVICE_COMPILER_AVAILABLE);
    CASE(CL_DEVICE_EXECUTION_CAPABILITIES);
    CASE(CL_DEVICE_QUEUE_PROPERTIES);
    CASE(CL_DEVICE_NAME);
    CASE(CL_DEVICE_VENDOR);
    CASE(CL_DRIVER_VERSION);
    CASE(CL_DEVICE_PROFILE);
    CASE(CL_DEVICE_VERSION);
    CASE(CL_DEVICE_EXTENSIONS);
    CASE(CL_DEVICE_PLATFORM);
    CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF);
    CASE(CL_DEVICE_HOST_UNIFIED_MEMORY);
    CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR);
    CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT);
    CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT);
    CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG);
    CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT);
    CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE);
    CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF);
    CASE(CL_DEVICE_OPENCL_C_VERSION);
    default:
      return getHexString(param_name);
  }
}

static std::string getContextInfoString(cl_context_info param_name) {
  switch (param_name) {
    CASE(CL_CONTEXT_REFERENCE_COUNT);
    CASE(CL_CONTEXT_DEVICES);
    CASE(CL_CONTEXT_PROPERTIES);
    CASE(CL_CONTEXT_NUM_DEVICES);
    default:
      return getHexString(param_name);
  }
}

static std::string getCommandQueueInfoString(cl_command_queue_info param_name) {
  switch (param_name) {
    CASE(CL_QUEUE_CONTEXT);
    CASE(CL_QUEUE_DEVICE);
    CASE(CL_QUEUE_REFERENCE_COUNT);
    CASE(CL_QUEUE_PROPERTIES);
    default:
      return getHexString(param_name);
  }
}

static std::string getProgramInfoString(cl_program_info param_name) {
  switch (param_name) {
    CASE(CL_PROGRAM_REFERENCE_COUNT);
    CASE(CL_PROGRAM_CONTEXT);
    CASE(CL_PROGRAM_NUM_DEVICES);
    CASE(CL_PROGRAM_DEVICES);
    CASE(CL_PROGRAM_SOURCE);
    CASE(CL_PROGRAM_BINARY_SIZES);
    CASE(CL_PROGRAM_BINARIES);
    default:
      return getHexString(param_name);
  }
}

static std::string getKernelInfoString(cl_kernel_info param_name) {
  switch (param_name) {
    CASE(CL_KERNEL_FUNCTION_NAME);
    CASE(CL_KERNEL_NUM_ARGS);
    CASE(CL_KERNEL_REFERENCE_COUNT);
    CASE(CL_KERNEL_CONTEXT);
    CASE(CL_KERNEL_PROGRAM);
    default:
      return getHexString(param_name);
  }
}

static std::string getKernelExecInfoString(cl_kernel_exec_info param_name) {
  switch (param_name) {
    CASE(CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM);
    CASE(CL_KERNEL_EXEC_INFO_SVM_PTRS);
    CASE(CL_KERNEL_EXEC_INFO_NEW_VCOP_AMD);
    CASE(CL_KERNEL_EXEC_INFO_PFPA_VCOP_AMD);
    default:
      return getHexString(param_name);
  }
}


static std::string getKernelWorkGroupInfoString(cl_kernel_work_group_info param_name) {
  switch (param_name) {
    CASE(CL_KERNEL_WORK_GROUP_SIZE);
    CASE(CL_KERNEL_COMPILE_WORK_GROUP_SIZE);
    CASE(CL_KERNEL_LOCAL_MEM_SIZE);
    CASE(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
    CASE(CL_KERNEL_PRIVATE_MEM_SIZE);
    default:
      return getHexString(param_name);
  }
}

static std::string getProgramBuildInfoString(cl_program_build_info param_name) {
  switch (param_name) {
    CASE(CL_PROGRAM_BUILD_STATUS);
    CASE(CL_PROGRAM_BUILD_OPTIONS);
    CASE(CL_PROGRAM_BUILD_LOG);
    default:
      return getHexString(param_name);
  }
}

static std::string getEventInfoString(cl_event_info param_name) {
  switch (param_name) {
    CASE(CL_EVENT_COMMAND_QUEUE);
    CASE(CL_EVENT_COMMAND_TYPE);
    CASE(CL_EVENT_REFERENCE_COUNT);
    CASE(CL_EVENT_COMMAND_EXECUTION_STATUS);
    CASE(CL_EVENT_CONTEXT);
    default:
      return getHexString(param_name);
  }
}

static std::string getProfilingInfoString(cl_profiling_info param_name) {
  switch (param_name) {
    CASE(CL_PROFILING_COMMAND_QUEUED);
    CASE(CL_PROFILING_COMMAND_SUBMIT);
    CASE(CL_PROFILING_COMMAND_START);
    CASE(CL_PROFILING_COMMAND_END);
    default:
      return getHexString(param_name);
  }
}

static std::string getCommandExecutionStatusString(cl_int param_name) {
  switch (param_name) {
    CASE(CL_COMPLETE);
    CASE(CL_RUNNING);
    CASE(CL_SUBMITTED);
    CASE(CL_QUEUED);
    default:
      return getHexString(param_name);
  }
}

static std::string getStringString(const char* src) {
  if (src == NULL) {
    return "NULL";
  }

  std::string str(src);

  if (str.length() > 6000) {
    str = str.substr(0, 6000).append("...");
  }

  size_t found = 0;
  while (true) {
    found = str.find_first_of("\n\r\t\"", found);
    if (found == std::string::npos) {
      break;
    }
    char subst[] = {'\\', '\0', '\0'};
    switch (str[found]) {
      case '\n':
        subst[1] = 'n';
        break;
      case '\r':
        subst[1] = 'r';
        break;
      case '\t':
        subst[1] = 't';
        break;
      case '\"':
        subst[1] = '\"';
        break;
      default:
        ++found;
        continue;
    }
    str.replace(found, 1, subst);
    found += 2;
  }

  str.insert(size_t(0), size_t(1), '\"').append(1, '\"');
  return str;
}

static std::string getProgramSourceString(const char** strings, const size_t* lengths,
                                          cl_uint count) {
  if (strings == NULL) {
    return "NULL";
  }
  if (count == 0) {
    return "[]";
  }
  std::ostringstream ss;
  ss << '[';

  for (cl_uint i = 0; i < count; ++i) {
    std::string src;
    if (lengths != NULL && lengths[i] != 0) {
      src = std::string(strings[i], lengths[i]);
    } else {
      src = strings[i];
    }
    if (i != 0) {
      ss << ',';
    }
    ss << getStringString(src.c_str());
  }

  ss << ']';
  return ss.str();
}

static cl_icd_dispatch_table original_dispatch;

static cl_int CL_API_CALL GetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms,
                                         cl_uint* num_platforms) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetPlatformIDs(" << num_entries << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetPlatformIDs(num_entries, platforms, num_platforms);
  delRec(&r);

  ss << getHandlesString(platforms, num_entries) << ',';
  ss << getHexString(num_platforms) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetPlatformInfo(cl_platform_id platform, cl_platform_info param_name,
                                          size_t param_value_size, void* param_value,
                                          size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetPlatformInfo(" << platform << ',';
  ss << getPlatformInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetPlatformInfo(platform, param_name, param_value_size,
                                                 param_value, param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
                                       cl_uint num_entries, cl_device_id* devices,
                                       cl_uint* num_devices) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetDeviceIDs(" << platform << ',';
  ss << getDeviceTypeString(device_type) << ',';
  ss << num_entries << ',';

  addRec(&r);
  cl_int ret =
      original_dispatch.GetDeviceIDs(platform, device_type, num_entries, devices, num_devices);
  delRec(&r);

  ss << getHandlesString(devices, num_entries) << ',';
  ss << getDecimalString(num_devices) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetDeviceInfo(cl_device_id device, cl_device_info param_name,
                                        size_t param_value_size, void* param_value,
                                        size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetDeviceInfo(" << device << ',';
  ss << getDeviceInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetDeviceInfo(device, param_name, param_value_size, param_value,
                                               param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_context CL_API_CALL CreateContext(const cl_context_properties* properties,
                                            cl_uint num_devices, const cl_device_id* devices,
                                            void(CL_CALLBACK* pfn_notify)(const char*, const void*,
                                                                          size_t, void*),
                                            void* user_data, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateContext(";
  ss << getContextPropertiesString(properties) << ',';
  ss << num_devices << ',';
  ss << getHandlesString(devices, num_devices) << ',';
  ss << pfn_notify << ',' << user_data << ',';

  addRec(&r);
  cl_context ret = original_dispatch.CreateContext(properties, num_devices, devices, pfn_notify,
                                                   user_data, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_context CL_API_CALL
CreateContextFromType(const cl_context_properties* properties, cl_device_type device_type,
                      void(CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*),
                      void* user_data, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateContextFromType(";
  ss << getContextPropertiesString(properties) << ',';
  ss << getDeviceTypeString(device_type) << ',';
  ss << pfn_notify << ',' << user_data << ',';

  addRec(&r);
  cl_context ret = original_dispatch.CreateContextFromType(properties, device_type, pfn_notify,
                                                           user_data, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL RetainContext(cl_context context) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clRetainContext(" << context;

  addRec(&r);
  cl_int ret = original_dispatch.RetainContext(context);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL ReleaseContext(cl_context context) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clReleaseContext(" << context;

  addRec(&r);
  cl_int ret = original_dispatch.ReleaseContext(context);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetContextInfo(cl_context context, cl_context_info param_name,
                                         size_t param_value_size, void* param_value,
                                         size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetContextInfo(" << context << ',';
  ss << getContextInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetContextInfo(context, param_name, param_value_size, param_value,
                                                param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_command_queue CL_API_CALL CreateCommandQueue(cl_context context, cl_device_id device,
                                                       cl_command_queue_properties properties,
                                                       cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateCommandQueue(" << context << ',' << device << ',';
  ss << getCommandQueuePropertyString(properties) << ',';

  addRec(&r);
  cl_command_queue ret =
      original_dispatch.CreateCommandQueue(context, device, properties, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_command_queue CL_API_CALL
CreateCommandQueueWithProperties(cl_context context, cl_device_id device,
                                 const cl_queue_properties* properties, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateCommandQueueWithProperties(" << context << ',' << device << ',';
  ss << getQueuePropertyString(properties) << ',';

  addRec(&r);
  cl_command_queue ret =
      original_dispatch.CreateCommandQueueWithProperties(context, device, properties, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL RetainCommandQueue(cl_command_queue command_queue) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clRetainCommandQueue(" << command_queue;

  addRec(&r);
  cl_int ret = original_dispatch.RetainCommandQueue(command_queue);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL ReleaseCommandQueue(cl_command_queue command_queue) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clReleaseCommandQueue(" << command_queue;

  addRec(&r);
  cl_int ret = original_dispatch.ReleaseCommandQueue(command_queue);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetCommandQueueInfo(cl_command_queue command_queue,
                                              cl_command_queue_info param_name,
                                              size_t param_value_size, void* param_value,
                                              size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetCommandQueueInfo(" << command_queue << ',';
  ss << getCommandQueueInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetCommandQueueInfo(command_queue, param_name, param_value_size,
                                                     param_value, param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL SetCommandQueueProperty(cl_command_queue command_queue,
                                                  cl_command_queue_properties properties,
                                                  cl_bool enable,
                                                  cl_command_queue_properties* old_properties) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clSetCommandQueueProperty(" << command_queue << ',';
  ss << getCommandQueuePropertyString(properties) << ',';
  ss << enable << ',';

  addRec(&r);
  cl_int ret =
      original_dispatch.SetCommandQueueProperty(command_queue, properties, enable, old_properties);
  delRec(&r);

  ss << getHexString(old_properties) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_mem CL_API_CALL CreateBuffer(cl_context context, cl_mem_flags flags, size_t size,
                                       void* host_ptr, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateBuffer(" << context << ',';
  ss << getMemFlagsString(flags) << ',' << size << ',' << host_ptr << ',';

  addRec(&r);
  cl_mem ret = original_dispatch.CreateBuffer(context, flags, size, host_ptr, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_mem CL_API_CALL CreateSubBuffer(cl_mem buffer, cl_mem_flags flags,
                                          cl_buffer_create_type buffer_create_type,
                                          const void* buffer_create_info, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateSubBuffer(" << buffer << ',';
  ss << getMemFlagsString(flags) << ',';
  ss << getBufferCreateString(buffer_create_type, buffer_create_info) << ',';

  addRec(&r);
  cl_mem ret = original_dispatch.CreateSubBuffer(buffer, flags, buffer_create_type,
                                                 buffer_create_info, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_mem CL_API_CALL CreateImage2D(cl_context context, cl_mem_flags flags,
                                        const cl_image_format* image_format, size_t image_width,
                                        size_t image_height, size_t image_row_pitch, void* host_ptr,
                                        cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateImage2D(" << context << ',';
  ss << getMemFlagsString(flags) << ',';
  ss << getImageFormatsString(image_format, 1) << ',';
  ss << image_width << ',' << image_height << ',' << image_row_pitch << ',';
  ss << host_ptr << ',';

  addRec(&r);
  cl_mem ret =
      original_dispatch.CreateImage2D(context, flags, image_format, image_width, image_height,
                                      image_row_pitch, host_ptr, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}


static cl_mem CL_API_CALL CreateImage3D(cl_context context, cl_mem_flags flags,
                                        const cl_image_format* image_format, size_t image_width,
                                        size_t image_height, size_t image_depth,
                                        size_t image_row_pitch, size_t image_slice_pitch,
                                        void* host_ptr, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateImage3D(" << context << ',';
  ss << getMemFlagsString(flags) << ',';
  ss << getImageFormatsString(image_format, 1) << ',';
  ss << image_width << ',' << image_height << ',' << image_depth << ',';
  ss << image_row_pitch << ',' << image_slice_pitch << ',';
  ss << host_ptr << ',';

  addRec(&r);
  cl_mem ret = original_dispatch.CreateImage3D(context, flags, image_format, image_width,
                                               image_height, image_depth, image_row_pitch,
                                               image_slice_pitch, host_ptr, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}


static cl_int CL_API_CALL RetainMemObject(cl_mem memobj) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clRetainMemObject(" << memobj;

  addRec(&r);
  cl_int ret = original_dispatch.RetainMemObject(memobj);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}


static cl_int CL_API_CALL ReleaseMemObject(cl_mem memobj) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clReleaseMemObject(" << memobj;

  addRec(&r);
  cl_int ret = original_dispatch.ReleaseMemObject(memobj);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetSupportedImageFormats(cl_context context, cl_mem_flags flags,
                                                   cl_mem_object_type image_type,
                                                   cl_uint num_entries,
                                                   cl_image_format* image_formats,
                                                   cl_uint* num_image_formats) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetSupportedImageFormats(" << context << ',';
  ss << getMemFlagsString(flags) << ',';
  ss << getMemObjectTypeString(image_type) << ',';
  ss << num_entries << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetSupportedImageFormats(context, flags, image_type, num_entries,
                                                          image_formats, num_image_formats);
  delRec(&r);

  ss << getImageFormatsString(image_formats, num_entries) << ',';
  ss << getDecimalString(num_image_formats);
  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetMemObjectInfo(cl_mem memobj, cl_mem_info param_name,
                                           size_t param_value_size, void* param_value,
                                           size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetMemObjectInfo(" << memobj << ',';
  ss << getMemInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetMemObjectInfo(memobj, param_name, param_value_size, param_value,
                                                  param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetImageInfo(cl_mem image, cl_image_info param_name,
                                       size_t param_value_size, void* param_value,
                                       size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetImageInfo(" << image << ',';
  ss << getImageInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetImageInfo(image, param_name, param_value_size, param_value,
                                              param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL SetMemObjectDestructorCallback(
    cl_mem memobj, void(CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data), void* user_data) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clSetMemObjectDestructorCallback(" << memobj << ',';
  ss << pfn_notify << ',' << user_data;

  addRec(&r);
  cl_int ret = original_dispatch.SetMemObjectDestructorCallback(memobj, pfn_notify, user_data);
  delRec(&r);

  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_sampler CL_API_CALL CreateSampler(cl_context context, cl_bool normalized_coords,
                                            cl_addressing_mode addressing_mode,
                                            cl_filter_mode filter_mode, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateSampler(" << context << ',';
  ss << normalized_coords << ',';
  ss << getAddressingModeString(addressing_mode) << ',';
  ss << getFilterModeString(filter_mode) << ',';

  addRec(&r);
  cl_sampler ret = original_dispatch.CreateSampler(context, normalized_coords, addressing_mode,
                                                   filter_mode, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL RetainSampler(cl_sampler sampler) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clRetainSampler(" << sampler;

  addRec(&r);
  cl_int ret = original_dispatch.RetainSampler(sampler);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL ReleaseSampler(cl_sampler sampler) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clReleaseSampler(" << sampler;

  addRec(&r);
  cl_int ret = original_dispatch.ReleaseSampler(sampler);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetSamplerInfo(cl_sampler sampler, cl_sampler_info param_name,
                                         size_t param_value_size, void* param_value,
                                         size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetSamplerInfo(" << sampler << ',';
  ss << getSamplerInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetSamplerInfo(sampler, param_name, param_value_size, param_value,
                                                param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_program CL_API_CALL CreateProgramWithSource(cl_context context, cl_uint count,
                                                      const char** strings, const size_t* lengths,
                                                      cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateProgramWithSource(" << context << ',' << count << ',';
  ss << getProgramSourceString(strings, lengths, count) << ',';
  ss << lengths << ',';

  addRec(&r);
  cl_program ret =
      original_dispatch.CreateProgramWithSource(context, count, strings, lengths, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_program CL_API_CALL CreateProgramWithBinary(cl_context context, cl_uint num_devices,
                                                      const cl_device_id* device_list,
                                                      const size_t* lengths,
                                                      const unsigned char** binaries,
                                                      cl_int* binary_status, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateProgramWithBinary(" << context << ',';
  ss << num_devices << ',' << getHandlesString(device_list, num_devices);
  ss << ',' << lengths << ',' << binaries << ',';
  ss << binary_status << ',';

  addRec(&r);
  cl_program ret = original_dispatch.CreateProgramWithBinary(
      context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL RetainProgram(cl_program program) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clRetainProgram(" << program;

  addRec(&r);
  cl_int ret = original_dispatch.RetainProgram(program);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL ReleaseProgram(cl_program program) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clReleaseProgram(" << program;

  addRec(&r);
  cl_int ret = original_dispatch.ReleaseProgram(program);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL BuildProgram(
    cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clBuildProgram(" << program << ',';
  ss << num_devices << ',' << getHandlesString(device_list, num_devices);
  ss << ',' << getStringString(options) << ',';
  ss << pfn_notify << ',' << user_data;

  addRec(&r);
  cl_int ret = original_dispatch.BuildProgram(program, num_devices, device_list, options,
                                              pfn_notify, user_data);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL UnloadCompiler(void) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clUnloadCompiler(";

  addRec(&r);
  cl_int ret = original_dispatch.UnloadCompiler();
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetProgramInfo(cl_program program, cl_program_info param_name,
                                         size_t param_value_size, void* param_value,
                                         size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetProgramInfo(" << program << ',';
  ss << getProgramInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetProgramInfo(program, param_name, param_value_size, param_value,
                                                param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetProgramBuildInfo(cl_program program, cl_device_id device,
                                              cl_program_build_info param_name,
                                              size_t param_value_size, void* param_value,
                                              size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetProgramBuildInfo(" << program << ',' << device << ',';
  ss << getProgramBuildInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetProgramBuildInfo(program, device, param_name, param_value_size,
                                                     param_value, param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_kernel CL_API_CALL CreateKernel(cl_program program, const char* kernel_name,
                                          cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateKernel(" << program << ',';
  ss << getStringString(kernel_name) << ',';

  addRec(&r);
  cl_kernel ret = original_dispatch.CreateKernel(program, kernel_name, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL CreateKernelsInProgram(cl_program program, cl_uint num_kernels,
                                                 cl_kernel* kernels, cl_uint* num_kernels_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateKernelInProgram(" << program << ',';
  ss << num_kernels << ',' << kernels << ',';

  addRec(&r);
  cl_int ret =
      original_dispatch.CreateKernelsInProgram(program, num_kernels, kernels, num_kernels_ret);
  delRec(&r);

  ss << getDecimalString(num_kernels_ret) << ',';
  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL RetainKernel(cl_kernel kernel) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clRetainKernel(" << kernel;

  addRec(&r);
  cl_int ret = original_dispatch.RetainKernel(kernel);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL ReleaseKernel(cl_kernel kernel) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clReleaseKernel(" << kernel;

  addRec(&r);
  cl_int ret = original_dispatch.ReleaseKernel(kernel);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL SetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size,
                                       const void* arg_value) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clSetKernelArg(" << kernel << ',';
  ss << arg_index << ',' << arg_size << ',';
  ss << getMemoryString(arg_value, arg_size);

  addRec(&r);
  cl_int ret = original_dispatch.SetKernelArg(kernel, arg_index, arg_size, arg_value);
  delRec(&r);

  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetKernelInfo(cl_kernel kernel, cl_kernel_info param_name,
                                        size_t param_value_size, void* param_value,
                                        size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetKernelInfo(" << kernel << ',';
  ss << getKernelInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetKernelInfo(kernel, param_name, param_value_size, param_value,
                                               param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device,
                                                 cl_kernel_work_group_info param_name,
                                                 size_t param_value_size, void* param_value,
                                                 size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetKernelWorkGroupInfo(" << kernel << ',' << device << ',';
  ss << getKernelWorkGroupInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetKernelWorkGroupInfo(
      kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL WaitForEvents(cl_uint num_events, const cl_event* event_list) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clWaitForEvents(" << num_events << ',';
  ss << getHandlesString(event_list, num_events);

  addRec(&r);
  cl_int ret = original_dispatch.WaitForEvents(num_events, event_list);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetEventInfo(cl_event event, cl_event_info param_name,
                                       size_t param_value_size, void* param_value,
                                       size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetEventInfo(" << event << ',';
  ss << getEventInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetEventInfo(event, param_name, param_value_size, param_value,
                                              param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_event CL_API_CALL CreateUserEvent(cl_context context, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateUserEvent(" << context << ',';

  addRec(&r);
  cl_event ret = original_dispatch.CreateUserEvent(context, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL RetainEvent(cl_event event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clRetainEvent(" << event;

  addRec(&r);
  cl_int ret = original_dispatch.RetainEvent(event);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL ReleaseEvent(cl_event event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clReleaseEvent(" << event;

  addRec(&r);
  cl_int ret = original_dispatch.ReleaseEvent(event);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL SetUserEventStatus(cl_event event, cl_int execution_status) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clSetUserEventStatus(" << event << ',' << execution_status;

  addRec(&r);
  cl_int ret = original_dispatch.SetUserEventStatus(event, execution_status);
  delRec(&r);

  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL SetEventCallback(cl_event event, cl_int command_exec_callback_type,
                                           void(CL_CALLBACK* pfn_notify)(cl_event, cl_int, void*),
                                           void* user_data) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clSetEventCallback(" << event << ',';
  ss << getCommandExecutionStatusString(command_exec_callback_type) << ',';
  ss << pfn_notify << ',' << user_data;

  addRec(&r);
  cl_int ret =
      original_dispatch.SetEventCallback(event, command_exec_callback_type, pfn_notify, user_data);
  delRec(&r);

  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetEventProfilingInfo(cl_event event, cl_profiling_info param_name,
                                                size_t param_value_size, void* param_value,
                                                size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetEventProfilingInfo(" << event << ',';
  ss << getProfilingInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetEventProfilingInfo(event, param_name, param_value_size,
                                                       param_value, param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL Flush(cl_command_queue command_queue) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clFlush(" << command_queue;

  addRec(&r);
  cl_int ret = original_dispatch.Flush(command_queue);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL Finish(cl_command_queue command_queue) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clFinish(" << command_queue;

  addRec(&r);
  cl_int ret = original_dispatch.Finish(command_queue);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer,
                                            cl_bool blocking_read, size_t offset, size_t cb,
                                            void* ptr, cl_uint num_events_in_wait_list,
                                            const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueReadBuffer(" << command_queue << ',';
  ss << buffer << ',' << getBoolString(blocking_read) << ',';
  ss << offset << ',' << cb << ',' << ptr << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret =
      original_dispatch.EnqueueReadBuffer(command_queue, buffer, blocking_read, offset, cb, ptr,
                                          num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueReadBufferRect(cl_command_queue command_queue, cl_mem buffer,
                                                cl_bool blocking_read, const size_t* buffer_offset,
                                                const size_t* host_offset, const size_t* region,
                                                size_t buffer_row_pitch, size_t buffer_slice_pitch,
                                                size_t host_row_pitch, size_t host_slice_pitch,
                                                void* ptr, cl_uint num_events_in_wait_list,
                                                const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueReadBufferRect(" << command_queue << ',';
  ss << buffer << ',' << getBoolString(blocking_read) << ',';
  ss << getNDimString(buffer_offset, 3) << ',';
  ss << getNDimString(host_offset, 3) << ',';
  ss << getNDimString(region, 3) << ',';
  ss << buffer_row_pitch << ',' << buffer_slice_pitch << ',';
  ss << host_row_pitch << ',' << host_slice_pitch << ',';
  ss << ptr << ',' << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueReadBufferRect(
      command_queue, buffer, blocking_read, buffer_offset, host_offset, region, buffer_row_pitch,
      buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list,
      event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer,
                                             cl_bool blocking_write, size_t offset, size_t cb,
                                             const void* ptr, cl_uint num_events_in_wait_list,
                                             const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueWriteBuffer(" << command_queue << ',';
  ss << buffer << ',' << getBoolString(blocking_write) << ',';
  ss << offset << ',' << cb << ',' << ptr << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret =
      original_dispatch.EnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, ptr,
                                           num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueWriteBufferRect(cl_command_queue command_queue, cl_mem buffer,
                                                 cl_bool blocking_write,
                                                 const size_t* buffer_offset,
                                                 const size_t* host_offset, const size_t* region,
                                                 size_t buffer_row_pitch, size_t buffer_slice_pitch,
                                                 size_t host_row_pitch, size_t host_slice_pitch,
                                                 const void* ptr, cl_uint num_events_in_wait_list,
                                                 const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueWriteBufferRect(" << command_queue << ',';
  ss << buffer << ',' << getBoolString(blocking_write) << ',';
  ss << getNDimString(buffer_offset, 3) << ',';
  ss << getNDimString(host_offset, 3) << ',';
  ss << getNDimString(region, 3) << ',';
  ss << buffer_row_pitch << ',' << buffer_slice_pitch << ',';
  ss << host_row_pitch << ',' << host_slice_pitch << ',';
  ss << ptr << ',' << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueWriteBufferRect(
      command_queue, buffer, blocking_write, buffer_offset, host_offset, region, buffer_row_pitch,
      buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list,
      event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer,
                                            cl_mem dst_buffer, size_t src_offset, size_t dst_offset,
                                            size_t cb, cl_uint num_events_in_wait_list,
                                            const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueCopyBuffer(" << command_queue << ',';
  ss << src_buffer << ',' << dst_buffer << ',';
  ss << src_offset << ',' << dst_offset << ',' << cb << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueCopyBuffer(command_queue, src_buffer, dst_buffer,
                                                   src_offset, dst_offset, cb,
                                                   num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueCopyBufferRect(cl_command_queue command_queue, cl_mem src_buffer,
                                                cl_mem dst_buffer, const size_t* src_origin,
                                                const size_t* dst_origin, const size_t* region,
                                                size_t src_row_pitch, size_t src_slice_pitch,
                                                size_t dst_row_pitch, size_t dst_slice_pitch,
                                                cl_uint num_events_in_wait_list,
                                                const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueCopyBufferRect(" << command_queue << ',';
  ss << src_buffer << ',' << dst_buffer << ',';
  ss << getNDimString(src_origin, 3) << ',';
  ss << getNDimString(dst_origin, 3) << ',';
  ss << getNDimString(region, 3) << ',';
  ss << src_row_pitch << ',' << src_slice_pitch << ',';
  ss << dst_row_pitch << ',' << dst_slice_pitch << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueCopyBufferRect(
      command_queue, src_buffer, dst_buffer, src_origin, dst_origin, region, src_row_pitch,
      src_slice_pitch, dst_row_pitch, dst_slice_pitch, num_events_in_wait_list, event_wait_list,
      event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueReadImage(cl_command_queue command_queue, cl_mem image,
                                           cl_bool blocking_read, const size_t* origin,
                                           const size_t* region, size_t row_pitch,
                                           size_t slice_pitch, void* ptr,
                                           cl_uint num_events_in_wait_list,
                                           const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueReadImage(" << command_queue << ',';
  ss << image << ',' << getBoolString(blocking_read) << ',';
  ss << getNDimString(origin, 3) << ',';
  ss << getNDimString(region, 3) << ',';
  ss << row_pitch << ',' << slice_pitch << ',';
  ss << ptr << ',' << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueReadImage(command_queue, image, blocking_read, origin,
                                                  region, row_pitch, slice_pitch, ptr,
                                                  num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueWriteImage(cl_command_queue command_queue, cl_mem image,
                                            cl_bool blocking_write, const size_t* origin,
                                            const size_t* region, size_t input_row_pitch,
                                            size_t input_slice_pitch, const void* ptr,
                                            cl_uint num_events_in_wait_list,
                                            const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueWriteImage(" << command_queue << ',';
  ss << image << ',' << getBoolString(blocking_write) << ',';
  ss << getNDimString(origin, 3) << ',';
  ss << getNDimString(region, 3) << ',';
  ss << input_row_pitch << ',' << input_slice_pitch << ',';
  ss << ptr << ',' << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueWriteImage(command_queue, image, blocking_write, origin,
                                                   region, input_row_pitch, input_slice_pitch, ptr,
                                                   num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueCopyImage(cl_command_queue command_queue, cl_mem src_image,
                                           cl_mem dst_image, const size_t* src_origin,
                                           const size_t* dst_origin, const size_t* region,
                                           cl_uint num_events_in_wait_list,
                                           const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueCopyImage(" << command_queue << ',';
  ss << src_image << ',' << dst_image << ',';
  ss << getNDimString(src_origin, 3) << ',';
  ss << getNDimString(dst_origin, 3) << ',';
  ss << getNDimString(region, 3) << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueCopyImage(command_queue, src_image, dst_image, src_origin,
                                                  dst_origin, region, num_events_in_wait_list,
                                                  event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueCopyImageToBuffer(cl_command_queue command_queue, cl_mem src_image,
                                                   cl_mem dst_buffer, const size_t* src_origin,
                                                   const size_t* region, size_t dst_offset,
                                                   cl_uint num_events_in_wait_list,
                                                   const cl_event* event_wait_list,
                                                   cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueCopyImageToBuffer(" << command_queue << ',';
  ss << src_image << ',' << dst_buffer << ',';
  ss << getNDimString(src_origin, 3) << ',';
  ss << getNDimString(region, 3) << ',';
  ss << dst_offset << ',' << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueCopyImageToBuffer(
      command_queue, src_image, dst_buffer, src_origin, region, dst_offset, num_events_in_wait_list,
      event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueCopyBufferToImage(
    cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image, size_t src_offset,
    const size_t* dst_origin, const size_t* region, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueCopyBufferToImage(" << command_queue << ',';
  ss << src_buffer << ',' << dst_image << ',' << src_offset << ',';
  ss << getNDimString(dst_origin, 3) << ',';
  ss << getNDimString(region, 3) << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueCopyBufferToImage(
      command_queue, src_buffer, dst_image, src_offset, dst_origin, region, num_events_in_wait_list,
      event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static void* CL_API_CALL EnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer,
                                          cl_bool blocking_map, cl_map_flags map_flags,
                                          size_t offset, size_t cb, cl_uint num_events_in_wait_list,
                                          const cl_event* event_wait_list, cl_event* event,
                                          cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueMapBuffer(" << command_queue << ',';
  ss << buffer << ',' << getBoolString(blocking_map) << ',';
  ss << getMapFlagsString(map_flags) << ',';
  ss << offset << ',' << cb << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  void* ret = original_dispatch.EnqueueMapBuffer(command_queue, buffer, blocking_map, map_flags,
                                                 offset, cb, num_events_in_wait_list,
                                                 event_wait_list, event, errcode_ret);
  delRec(&r);

  ss << getHexString(event) << ',' << getErrorString(errcode_ret);
  ss << ") = " << ret;
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static void* CL_API_CALL EnqueueMapImage(cl_command_queue command_queue, cl_mem image,
                                         cl_bool blocking_map, cl_map_flags map_flags,
                                         const size_t* origin, const size_t* region,
                                         size_t* image_row_pitch, size_t* image_slice_pitch,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event* event_wait_list, cl_event* event,
                                         cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueMapImage(" << command_queue << ',';
  ss << image << ',' << getBoolString(blocking_map) << ',';
  ss << getMapFlagsString(map_flags) << ',';
  ss << getNDimString(origin, 3) << ',';
  ss << getNDimString(region, 3) << ',';
  ss << image_row_pitch << ',' << image_slice_pitch << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  void* ret = original_dispatch.EnqueueMapImage(
      command_queue, image, blocking_map, map_flags, origin, region, image_row_pitch,
      image_slice_pitch, num_events_in_wait_list, event_wait_list, event, errcode_ret);
  delRec(&r);

  ss << getHexString(event) << ',' << getErrorString(errcode_ret);
  ss << ") = " << ret;
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj,
                                                void* mapped_ptr, cl_uint num_events_in_wait_list,
                                                const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueUnmapMemObject(" << command_queue << ',';
  ss << memobj << ',' << mapped_ptr << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueUnmapMemObject(
      command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel,
                                               cl_uint work_dim, const size_t* global_work_offset,
                                               const size_t* global_work_size,
                                               const size_t* local_work_size,
                                               cl_uint num_events_in_wait_list,
                                               const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueNDRangeKernel(" << command_queue << ',';
  ss << kernel << ',' << work_dim << ',';
  ss << getNDimString(global_work_offset, work_dim) << ',';
  ss << getNDimString(global_work_size, work_dim) << ',';
  ss << getNDimString(local_work_size, work_dim) << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueNDRangeKernel(
      command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size,
      num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueTask(cl_command_queue command_queue, cl_kernel kernel,
                                      cl_uint num_events_in_wait_list,
                                      const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueTask(" << command_queue << ',' << kernel << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueTask(command_queue, kernel, num_events_in_wait_list,
                                             event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueNativeKernel(cl_command_queue command_queue,
                                              void(CL_CALLBACK* user_func)(void*), void* args,
                                              size_t cb_args, cl_uint num_mem_objects,
                                              const cl_mem* mem_list, const void** args_mem_loc,
                                              cl_uint num_events_in_wait_list,
                                              const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueNativeKernel(" << command_queue << ',' << user_func << ',';
  ss << args << ',' << cb_args << ',' << num_mem_objects << ',';
  ss << getHandlesString(mem_list, num_mem_objects) << ',';
  ss << args_mem_loc << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueNativeKernel(
      command_queue, user_func, args, cb_args, num_mem_objects, mem_list, args_mem_loc,
      num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueMarker(cl_command_queue command_queue, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueMarker(" << command_queue << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueMarker(command_queue, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueWaitForEvents(cl_command_queue command_queue, cl_uint num_events,
                                               const cl_event* event_list) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueWaitForEvents(" << command_queue << ',';
  ss << num_events << ',';
  ss << getHandlesString(event_list, num_events);

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueWaitForEvents(command_queue, num_events, event_list);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueBarrier(cl_command_queue command_queue) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueBarrier(" << command_queue;

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueBarrier(command_queue);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static void* CL_API_CALL GetExtensionFunctionAddress(const char* func_name) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetExtensionFunctionAddress(" << func_name;

  addRec(&r);
  void* ret = original_dispatch.GetExtensionFunctionAddress(func_name);
  delRec(&r);

  ss << ") = " << ret;
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_mem CL_API_CALL CreateFromGLBuffer(cl_context context, cl_mem_flags flags,
                                             cl_GLuint bufobj, int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateFromGLBuffer(" << context << ',';
  ss << getMemFlagsString(flags) << ',' << bufobj << ',';

  addRec(&r);
  cl_mem ret = original_dispatch.CreateFromGLBuffer(context, flags, bufobj, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_mem CL_API_CALL CreateFromGLTexture2D(cl_context context, cl_mem_flags flags,
                                                cl_GLenum target, cl_GLint miplevel,
                                                cl_GLuint texture, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateFromGLTexture2D(" << context << ',';
  ss << getMemFlagsString(flags) << ',' << target << ',';
  ss << miplevel << ',' << texture << ',';

  addRec(&r);
  cl_mem ret = original_dispatch.CreateFromGLTexture2D(context, flags, target, miplevel, texture,
                                                       errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_mem CL_API_CALL CreateFromGLTexture3D(cl_context context, cl_mem_flags flags,
                                                cl_GLenum target, cl_GLint miplevel,
                                                cl_GLuint texture, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateFromGLTexture3D(" << context << ',';
  ss << getMemFlagsString(flags) << ',' << target << ',';
  ss << miplevel << ',' << texture << ',';

  addRec(&r);
  cl_mem ret = original_dispatch.CreateFromGLTexture3D(context, flags, target, miplevel, texture,
                                                       errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_mem CL_API_CALL CreateFromGLRenderbuffer(cl_context context, cl_mem_flags flags,
                                                   cl_GLuint renderbuffer, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateFromGLRenderbuffer(" << context << ',';
  ss << getMemFlagsString(flags) << ',' << renderbuffer << ',';

  addRec(&r);
  cl_mem ret =
      original_dispatch.CreateFromGLRenderbuffer(context, flags, renderbuffer, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetGLObjectInfo(cl_mem memobj, cl_gl_object_type* gl_object_type,
                                          cl_GLuint* gl_object_name) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetGLObjectInfo(" << memobj << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetGLObjectInfo(memobj, gl_object_type, gl_object_name);
  delRec(&r);

  ss << getHexString(gl_object_type) << ',';
  ss << getDecimalString(gl_object_name) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetGLTextureInfo(cl_mem memobj, cl_gl_texture_info param_name,
                                           size_t param_value_size, void* param_value,
                                           size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetGLTextureInfo(" << memobj << ',';
  ss << param_name << ',' << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetGLTextureInfo(memobj, param_name, param_value_size, param_value,
                                                  param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetGLContextInfoKHR(const cl_context_properties* properties,
                                              cl_gl_context_info param_name,
                                              size_t param_value_size, void* param_value,
                                              size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetGLContextInfoKHR(";
  ss << getContextPropertiesString(properties) << ',';
  ss << param_name << ',' << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetGLContextInfoKHR(properties, param_name, param_value_size,
                                                     param_value, param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueAcquireGLObjects(cl_command_queue command_queue,
                                                  cl_uint num_objects, const cl_mem* mem_objects,
                                                  cl_uint num_events_in_wait_list,
                                                  const cl_event* event_wait_list,
                                                  cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueAcquireGLObjects(" << command_queue << ',';
  ss << num_objects << ',' << getHandlesString(mem_objects, num_objects);
  ss << ',' << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueAcquireGLObjects(
      command_queue, num_objects, mem_objects, num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueReleaseGLObjects(cl_command_queue command_queue,
                                                  cl_uint num_objects, const cl_mem* mem_objects,
                                                  cl_uint num_events_in_wait_list,
                                                  const cl_event* event_wait_list,
                                                  cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueReleaseGLObjects(" << command_queue << ',';
  ss << num_objects << ',' << getHandlesString(mem_objects, num_objects);
  ss << ',' << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueReleaseGLObjects(
      command_queue, num_objects, mem_objects, num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL RetainDevice(cl_device_id device) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clRetainDevice(" << device;
  addRec(&r);
  cl_int ret = original_dispatch.RetainDevice(device);
  delRec(&r);

  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL ReleaseDevice(cl_device_id device) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "ReleaseDevice(" << device;
  addRec(&r);
  cl_int ret = original_dispatch.ReleaseDevice(device);
  delRec(&r);

  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_mem CL_API_CALL CreateImage(cl_context context, cl_mem_flags flags,
                                      const cl_image_format* image_format,
                                      const cl_image_desc* image_desc, void* host_ptr,
                                      cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "CreateImage(" << context << ',';
  ss << getMemFlagsString(flags) << ',';
  ss << getImageFormatsString(image_format, 1) << ',';
  ss << getImageDescString(image_desc) << ',';
  ss << host_ptr << ',';

  addRec(&r);
  cl_mem ret = original_dispatch.CreateImage(context, flags, image_format, image_desc, host_ptr,
                                             errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_program CL_API_CALL CreateProgramWithBuiltInKernels(cl_context context,
                                                              cl_uint num_devices,
                                                              const cl_device_id* device_list,
                                                              const char* kernel_names,
                                                              cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateProgramWithBuiltInKernels(" << context << ',';
  ss << num_devices << ',' << getHandlesString(device_list, num_devices);
  ss << ',' << kernel_names << ',';

  addRec(&r);
  cl_program ret = original_dispatch.CreateProgramWithBuiltInKernels(
      context, num_devices, device_list, kernel_names, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL CompileProgram(
    cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options,
    cl_uint num_input_headers, const cl_program* input_headers, const char** header_include_names,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCompileProgram(" << program << ',';
  ss << num_devices << ',' << getHandlesString(device_list, num_devices);
  ss << options << ',';
  ss << num_devices << ',' << getHandlesString(input_headers, num_input_headers);
  ss << header_include_names << ',';
  ss << pfn_notify << ',';

  addRec(&r);
  cl_int ret = original_dispatch.CompileProgram(program, num_devices, device_list, options,
                                                num_input_headers, input_headers,
                                                header_include_names, pfn_notify, user_data);
  delRec(&r);

  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_program CL_API_CALL
LinkProgram(cl_context context, cl_uint num_devices, const cl_device_id* device_list,
            const char* options, cl_uint num_input_programs, const cl_program* input_programs,
            void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data,
            cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clLinkProgram(" << context << ',';
  ss << num_devices << ',' << getHandlesString(device_list, num_devices);
  ss << options << ',';
  ss << getHandlesString(input_programs, num_input_programs);
  ss << pfn_notify << ',' << user_data << ',';

  addRec(&r);
  cl_program ret =
      original_dispatch.LinkProgram(context, num_devices, device_list, options, num_input_programs,
                                    input_programs, pfn_notify, user_data, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL UnloadPlatformCompiler(cl_platform_id platform) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clUnloadPlatformCompiler(" << platform << ',';

  addRec(&r);
  cl_int ret = original_dispatch.UnloadPlatformCompiler(platform);
  delRec(&r);

  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetKernelArgInfo(cl_kernel kernel, cl_uint arg_indx,
                                           cl_kernel_arg_info param_name, size_t param_value_size,
                                           void* param_value, size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetKernelArgInfo(" << kernel << ',';
  ss << arg_indx << ',';
  ss << getKernelArgInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetKernelArgInfo(kernel, arg_indx, param_name, param_value_size,
                                                  param_value, param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueFillBuffer(cl_command_queue command_queue, cl_mem buffer,
                                            const void* pattern, size_t pattern_size, size_t offset,
                                            size_t cb, cl_uint num_events_in_wait_list,
                                            const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueFillBuffer(" << command_queue << ',';
  ss << buffer << ',' << pattern << ',' << pattern_size << ',';
  ss << offset << ',' << cb << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret =
      original_dispatch.EnqueueFillBuffer(command_queue, buffer, pattern, pattern_size, offset, cb,
                                          num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueFillImage(cl_command_queue command_queue, cl_mem image,
                                           const void* fill_color, const size_t origin[3],
                                           const size_t region[3], cl_uint num_events_in_wait_list,
                                           const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueFillImage(" << command_queue << ',';
  ss << image << ',' << fill_color << ',';
  ss << getNDimString(origin, 3) << ',';
  ss << getNDimString(region, 3) << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueFillImage(command_queue, image, fill_color, origin, region,
                                                  num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event) << ',';
  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueMigrateMemObjects(
    cl_command_queue command_queue, cl_uint num_mem_objects, const cl_mem* mem_objects,
    cl_mem_migration_flags flags, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
    cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueMigrateMemObjects(" << command_queue << ',';
  ss << ',' << num_mem_objects << ',';
  ss << getHandlesString(mem_objects, num_mem_objects) << ',' << flags << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret =
      original_dispatch.EnqueueMigrateMemObjects(command_queue, num_mem_objects, mem_objects, flags,
                                                 num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event) << ',';
  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueMarkerWithWaitList(cl_command_queue command_queue,
                                                    cl_uint num_events_in_wait_list,
                                                    const cl_event* event_wait_list,
                                                    cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueMarkerWithWaitList(" << command_queue << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueMarkerWithWaitList(command_queue, num_events_in_wait_list,
                                                           event_wait_list, event);
  delRec(&r);

  ss << getHexString(event) << ',';
  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueBarrierWithWaitList(cl_command_queue command_queue,
                                                     cl_uint num_events_in_wait_list,
                                                     const cl_event* event_wait_list,
                                                     cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueBarrierWithWaitList(" << command_queue << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueBarrierWithWaitList(command_queue, num_events_in_wait_list,
                                                            event_wait_list, event);
  delRec(&r);

  ss << getHexString(event) << ',';
  ss << ") = " << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static void* CL_API_CALL GetExtensionFunctionAddressForPlatform(cl_platform_id platform,
                                                                const char* function_name) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetExtensionFunctionAddressForPlatform(" << platform << ',';
  ss << function_name << ',';

  addRec(&r);
  void* ret = original_dispatch.GetExtensionFunctionAddressForPlatform(platform, function_name);
  delRec(&r);

  ss << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_mem CL_API_CALL CreateFromGLTexture(cl_context context, cl_mem_flags flags,
                                              cl_GLenum target, cl_GLint miplevel,
                                              cl_GLuint texture, cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateFromGLTexture(" << context << ',';
  ss << getMemFlagsString(flags) << ',' << target << ',';
  ss << miplevel << ',' << texture << ',';

  addRec(&r);
  cl_mem ret =
      original_dispatch.CreateFromGLTexture(context, flags, target, miplevel, texture, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_mem CL_API_CALL CreatePipe(cl_context context, cl_mem_flags flags, cl_uint pipePacketSize,
                                     cl_uint pipeMaxPackets, const cl_pipe_properties* props,
                                     cl_int* errcode_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreatePipe(" << context << ',';
  ss << getMemFlagsString(flags) << ',' << pipePacketSize << ',' << pipeMaxPackets << ',' << props
     << ',';

  addRec(&r);
  cl_mem ret = original_dispatch.CreatePipe(context, flags, pipePacketSize, pipeMaxPackets, props,
                                            errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL GetPipeInfo(cl_mem memobj, cl_pipe_info param_name,
                                      size_t param_value_size, void* param_value,
                                      size_t* param_value_size_ret) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clGetPipeInfo(" << memobj << ',';
  ss << getMemInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret = original_dispatch.GetPipeInfo(memobj, param_name, param_value_size, param_value,
                                             param_value_size_ret);
  delRec(&r);

  ss << getHexString(param_value) << ',';
  ss << getHexString(param_value_size_ret) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static void* CL_API_CALL SVMAlloc(cl_context context, cl_svm_mem_flags flags, size_t size,
                                  cl_uint alignment) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clSVMAlloc(" << context << ',';
  ss << getHexString(flags) << ',';
  ss << getHexString(size) << ',';
  ss << getHexString(alignment) << ") = ";

  addRec(&r);
  void* ret = original_dispatch.SVMAlloc(context, flags, size, alignment);
  delRec(&r);

  ss << ret << std::endl;

  std::cerr << ss.str();
  return ret;
}

static void CL_API_CALL SVMFree(cl_context context, void* svm_pointer) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clSVMFree(" << context << ',';
  ss << svm_pointer << ')';

  addRec(&r);
  original_dispatch.SVMFree(context, svm_pointer);
  delRec(&r);

  ss << std::endl;

  std::cerr << ss.str();
}

static cl_int CL_API_CALL EnqueueSVMFree(
    cl_command_queue command_queue, cl_uint num_svm_pointers, void* svm_pointers[],
    void(CL_CALLBACK* pfn_free_func)(cl_command_queue /*queue */, cl_uint /* num_svm_pointers */,
                                     void*[] /* svm_pointers */, void* /* user_data */),
    void* user_data, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
    cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueSVMMap(" << command_queue << ',';
  ss << num_svm_pointers << ',';
  ss << '[';
  for (cl_uint i = 0; i < num_svm_pointers; ++i) {
    ss << svm_pointers[i] << ',';
  }
  ss << "],";
  ss << pfn_free_func << ',';
  ss << user_data << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret =
      original_dispatch.EnqueueSVMFree(command_queue, num_svm_pointers, svm_pointers, pfn_free_func,
                                       user_data, num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}


static cl_int CL_API_CALL EnqueueSVMMemcpy(cl_command_queue command_queue, cl_bool blocking_copy,
                                           void* dst_ptr, const void* src_ptr, size_t size,
                                           cl_uint num_events_in_wait_list,
                                           const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueSVMMemcpy(" << command_queue << ',';
  ss << getBoolString(blocking_copy) << ',';
  ss << dst_ptr << ',';
  ss << src_ptr << ',' << getHexString(size) << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret =
      original_dispatch.EnqueueSVMMemcpy(command_queue, blocking_copy, dst_ptr, src_ptr, size,
                                         num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueSVMMemFill(cl_command_queue command_queue, void* svm_ptr,
                                            const void* pattern, size_t pattern_size, size_t size,
                                            cl_uint num_events_in_wait_list,
                                            const cl_event* event_wait_list,
                                            cl_event* event) CL_API_SUFFIX__VERSION_2_0 {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueSVMMemFill(" << command_queue << ',';
  ss << svm_ptr << ',';
  ss << pattern << ',';
  ss << getHexString(pattern_size) << ',' << getHexString(size) << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret =
      original_dispatch.EnqueueSVMMemFill(command_queue, svm_ptr, pattern, pattern_size, size,
                                          num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueSVMMap(cl_command_queue command_queue, cl_bool blocking_map,
                                        cl_map_flags flags, void* svm_ptr, size_t size,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueSVMMap(" << command_queue << ',';
  ss << getBoolString(blocking_map) << ',';
  ss << getMapFlagsString(flags) << ',';
  ss << svm_ptr << ',' << getHexString(size) << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueSVMMap(command_queue, blocking_map, flags, svm_ptr, size,
                                               num_events_in_wait_list, event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL EnqueueSVMUnmap(cl_command_queue command_queue, void* svm_ptr,
                                          cl_uint num_events_in_wait_list,
                                          const cl_event* event_wait_list, cl_event* event) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clEnqueueSVMUnmap(" << command_queue << ',';
  ss << svm_ptr << ',';
  ss << num_events_in_wait_list << ',';
  ss << getHandlesString(event_wait_list, num_events_in_wait_list) << ',';

  addRec(&r);
  cl_int ret = original_dispatch.EnqueueSVMUnmap(command_queue, svm_ptr, num_events_in_wait_list,
                                                 event_wait_list, event);
  delRec(&r);

  ss << getHexString(event);
  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_sampler CL_API_CALL
CreateSamplerWithProperties(cl_context context, const cl_sampler_properties* sampler_properties,
                            cl_int* errcode_ret) CL_API_SUFFIX__VERSION_2_0 {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clCreateSamplerWithProperties(" << context << ',';
  ss << "[";

  const struct SamplerProperty {
    cl_sampler_properties name;
    union {
      cl_sampler_properties raw;
      cl_bool normalizedCoords;
      cl_addressing_mode addressingMode;
      cl_filter_mode filterMode;
      cl_float lod;
    } value;
  }* p = reinterpret_cast<const SamplerProperty*>(sampler_properties);

  if (p != NULL)
    while (p->name != 0) {
      ss << getSamplerInfoString((cl_sampler_info)p->name) << ':';
      switch (p->name) {
        case CL_SAMPLER_NORMALIZED_COORDS:
          ss << getBoolString(p->value.normalizedCoords) << ',';
          break;
        case CL_SAMPLER_ADDRESSING_MODE:
          ss << getAddressingModeString(p->value.addressingMode) << ',';
          break;
        case CL_SAMPLER_FILTER_MODE:
          ss << getFilterModeString(p->value.filterMode) << ',';
          break;
        case CL_SAMPLER_MIP_FILTER_MODE:
          ss << getFilterModeString(p->value.filterMode) << ',';
          break;
        case CL_SAMPLER_LOD_MIN:
          ss << p->value.lod << ',';
          break;
        case CL_SAMPLER_LOD_MAX:
          ss << p->value.lod << ',';
          break;
        default:
          break;
      }
      ++p;
    }

  addRec(&r);
  cl_sampler ret =
      original_dispatch.CreateSamplerWithProperties(context, sampler_properties, errcode_ret);
  delRec(&r);

  ss << getErrorString(errcode_ret) << ") = " << ret;
  ss << ret << std::endl;

  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL SetKernelArgSVMPointer(cl_kernel kernel, cl_uint arg_index,
                                                 const void* arg_value) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clSetKernelArgSVMPointer(" << kernel << ',';
  ss << arg_index << ',';
  ss << arg_value;

  addRec(&r);
  cl_int ret = original_dispatch.SetKernelArgSVMPointer(kernel, arg_index, arg_value);
  delRec(&r);

  ss << ") = " << getErrorString(ret);
  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_int CL_API_CALL SetKernelExecInfo(cl_kernel kernel, cl_kernel_exec_info param_name,
                                            size_t param_value_size, const void* param_value) {
  std::ostringstream ss;
  Rec r(&ss);

  ss << "clSetKernelExecInfo(" << kernel << ',';
  ss << getKernelExecInfoString(param_name) << ',';
  ss << param_value_size << ',';

  addRec(&r);
  cl_int ret =
      original_dispatch.SetKernelExecInfo(kernel, param_name, param_value_size, param_value);
  delRec(&r);

  ss << getHexString(const_cast<void*>(param_value)) << ") = ";
  ss << getErrorString(ret);

  ss << std::endl;
  std::cerr << ss.str();
  return ret;
}

static cl_icd_dispatch_table modified_dispatch = {
    /* OpenCL 1.0 */
    GetPlatformIDs,
    GetPlatformInfo,
    GetDeviceIDs,
    GetDeviceInfo,
    CreateContext,
    CreateContextFromType,
    RetainContext,
    ReleaseContext,
    GetContextInfo,
    CreateCommandQueue,
    RetainCommandQueue,
    ReleaseCommandQueue,
    GetCommandQueueInfo,
    SetCommandQueueProperty,
    CreateBuffer,
    CreateImage2D,
    CreateImage3D,
    RetainMemObject,
    ReleaseMemObject,
    GetSupportedImageFormats,
    GetMemObjectInfo,
    GetImageInfo,
    CreateSampler,
    RetainSampler,
    ReleaseSampler,
    GetSamplerInfo,
    CreateProgramWithSource,
    CreateProgramWithBinary,
    RetainProgram,
    ReleaseProgram,
    BuildProgram,
    UnloadCompiler,
    GetProgramInfo,
    GetProgramBuildInfo,
    CreateKernel,
    CreateKernelsInProgram,
    RetainKernel,
    ReleaseKernel,
    SetKernelArg,
    GetKernelInfo,
    GetKernelWorkGroupInfo,
    WaitForEvents,
    GetEventInfo,
    RetainEvent,
    ReleaseEvent,
    GetEventProfilingInfo,
    Flush,
    Finish,
    EnqueueReadBuffer,
    EnqueueWriteBuffer,
    EnqueueCopyBuffer,
    EnqueueReadImage,
    EnqueueWriteImage,
    EnqueueCopyImage,
    EnqueueCopyImageToBuffer,
    EnqueueCopyBufferToImage,
    EnqueueMapBuffer,
    EnqueueMapImage,
    EnqueueUnmapMemObject,
    EnqueueNDRangeKernel,
    EnqueueTask,
    EnqueueNativeKernel,
    EnqueueMarker,
    EnqueueWaitForEvents,
    EnqueueBarrier,
    GetExtensionFunctionAddress,
    CreateFromGLBuffer,
    CreateFromGLTexture2D,
    CreateFromGLTexture3D,
    CreateFromGLRenderbuffer,
    GetGLObjectInfo,
    GetGLTextureInfo,
    EnqueueAcquireGLObjects,
    EnqueueReleaseGLObjects,
    GetGLContextInfoKHR,
    {NULL, NULL, NULL, NULL, NULL, NULL}, /* _reservedForD3D10KHR[6] */

    /* OpenCL 1.1 */
    SetEventCallback,
    CreateSubBuffer,
    SetMemObjectDestructorCallback,
    CreateUserEvent,
    SetUserEventStatus,
    EnqueueReadBufferRect,
    EnqueueWriteBufferRect,
    EnqueueCopyBufferRect,
    {NULL, NULL, NULL}, /* _reservedForDeviceFissionEXT[3] */
    NULL,               /* CreateEventFromGLsyncKHR */

    /* OpenCL 1.2 */
    NULL, /* CreateSubDevices */
    RetainDevice,
    ReleaseDevice,
    CreateImage,
    CreateProgramWithBuiltInKernels,
    CompileProgram,
    LinkProgram,
    UnloadPlatformCompiler,
    GetKernelArgInfo,
    EnqueueFillBuffer,
    EnqueueFillImage,
    EnqueueMigrateMemObjects,
    EnqueueMarkerWithWaitList,
    EnqueueBarrierWithWaitList,
    GetExtensionFunctionAddressForPlatform,
    CreateFromGLTexture,
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* _reservedD3DExtensions[10] */
    {NULL, NULL, NULL, NULL},                                     /* _reservedEGLExtensions[4] */

    /* OpenCL 2.0 */
    CreateCommandQueueWithProperties,
    CreatePipe,
    GetPipeInfo,
    SVMAlloc,
    SVMFree,
    EnqueueSVMFree,
    EnqueueSVMMemcpy,
    EnqueueSVMMemFill,
    EnqueueSVMMap,
    EnqueueSVMUnmap,
    CreateSamplerWithProperties,
    SetKernelArgSVMPointer,
    SetKernelExecInfo,
    NULL, /* clGetKernelSubGroupInfoKHR */

    /* OpenCL 2.1 */
    NULL, /* clCloneKernel */
    NULL, /* clCreateProgramWithILKHR */
    NULL, /* clEnqueueSVMMigrateMem */
    NULL, /* clGetDeviceAndHostTimer */
    NULL, /* clGetHostTimer */
    NULL, /* clGetKernelSubGroupInfo */
    NULL, /* clSetDefaultDeviceCommandQueue */

    /* OpenCL 2.2 */
    NULL, /* clSetProgramReleaseCallback */
    NULL, /* clSetProgramSpecializationConstant */
};

static void cleanup(void) { std::cerr.rdbuf(cerrStreamBufSave); }

#define SET_ORIGINAL_EXTENSION(DISPATCH)                                                           \
  memcpy(modified_dispatch._reservedFor##DISPATCH, original_dispatch._reservedFor##DISPATCH,       \
         sizeof(original_dispatch._reservedFor##DISPATCH));

#define SET_ORIGINAL(DISPATCH) modified_dispatch.DISPATCH = original_dispatch.DISPATCH;

int32_t CL_CALLBACK vdiAgent_OnLoad(vdi_agent* agent) {
  char* clTraceLogEnv;

  int32_t err = agent->GetICDDispatchTable(agent, &original_dispatch, sizeof(original_dispatch));
  if (err != CL_SUCCESS) {
    return err;
  }

  clTraceLogEnv = getenv("CL_TRACE_OUTPUT");
  if (clTraceLogEnv != NULL) {
    std::string clTraceLogStr = clTraceLogEnv;
    const std::size_t pidPos = clTraceLogStr.find("%pid%");
    if (pidPos != std::string::npos) {
#if defined(_WIN32)
      const std::int32_t pid = _getpid();
#else
      const std::int32_t pid = getpid();
#endif
      clTraceLogStr.replace(pidPos, 5, std::to_string(pid));
    }
    clTraceLog.open(clTraceLogStr);
    cerrStreamBufSave = std::cerr.rdbuf(clTraceLog.rdbuf());
    std::atexit(cleanup);
  }

  cl_platform_id platform;
  err = agent->GetPlatform(agent, &platform);
  if (err != CL_SUCCESS) {
    return err;
  }

  char version[256];
  err = original_dispatch.GetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(version), version,
                                          NULL);
  if (err != CL_SUCCESS) {
    return err;
  }

  std::cerr << "!!!" << std::endl
            << "!!! API trace for \"" << version << "\"" << std::endl
            << "!!!" << std::endl;

  SET_ORIGINAL_EXTENSION(D3D10KHR);
  SET_ORIGINAL_EXTENSION(DeviceFissionEXT);
  SET_ORIGINAL(CreateEventFromGLsyncKHR);
  SET_ORIGINAL(CreateSubDevices);
  SET_ORIGINAL_EXTENSION(D3DExtensions);
  SET_ORIGINAL_EXTENSION(EGLExtensions);
  SET_ORIGINAL(GetKernelSubGroupInfoKHR);
  SET_ORIGINAL(CloneKernel);
  SET_ORIGINAL(CreateProgramWithILKHR);
  SET_ORIGINAL(EnqueueSVMMigrateMem);
  SET_ORIGINAL(GetDeviceAndHostTimer);
  SET_ORIGINAL(GetHostTimer);
  SET_ORIGINAL(GetKernelSubGroupInfo);
  SET_ORIGINAL(SetDefaultDeviceCommandQueue);
  SET_ORIGINAL(SetProgramReleaseCallback);
  SET_ORIGINAL(SetProgramSpecializationConstant);

  err = agent->SetICDDispatchTable(agent, &modified_dispatch, sizeof(modified_dispatch));
  if (err != CL_SUCCESS) {
    return err;
  }

  initRecs();
  err = startChecker();
  return err;
}

void CL_CALLBACK vdiAgent_OnUnload(vdi_agent* agent) { clTraceLog.close(); }
