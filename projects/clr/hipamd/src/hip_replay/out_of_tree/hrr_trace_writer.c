/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

/* Out-of-tree trace writer. Same .hrr format as in-tree hip_hrr.cpp.
 * Pure C for maximum portability (no C++ ABI issues in LD_PRELOAD). */

#include "hrr_trace_writer.h"
#include "hrr_code_object.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define HRR_MKDIR(p) _mkdir(p)
#define HRR_SEP "\\"
#define HRR_MUTEX CRITICAL_SECTION
#define HRR_MUTEX_INIT(m) InitializeCriticalSection(m)
#define HRR_MUTEX_LOCK(m) EnterCriticalSection(m)
#define HRR_MUTEX_UNLOCK(m) LeaveCriticalSection(m)
static uint64_t hrr_now_ns(void) {
  LARGE_INTEGER freq, count;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&count);
  return (uint64_t)(count.QuadPart * 1000000000LL / freq.QuadPart);
}
#else
#include <sys/stat.h>
#include <pthread.h>
#define HRR_MKDIR(p) mkdir(p, 0755)
#define HRR_SEP "/"
#define HRR_MUTEX pthread_mutex_t
#define HRR_MUTEX_INIT(m) pthread_mutex_init(m, NULL)
#define HRR_MUTEX_LOCK(m) pthread_mutex_lock(m)
#define HRR_MUTEX_UNLOCK(m) pthread_mutex_unlock(m)
static uint64_t hrr_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

/* Event types */
#define EVT_MALLOC        0x0001
#define EVT_FREE          0x0002
#define EVT_MEMCPY        0x0003
#define EVT_MEMSET        0x0004
#define EVT_MODULE_LOAD   0x0010
#define EVT_MODULE_UNLOAD 0x0011
#define EVT_KERNEL_LAUNCH 0x0020
#define EVT_DEVICE_SYNC   0x0050
#define EVT_STREAM_SYNC   0x0032

#define HRR_MAGIC   0x52524845
#define HRR_VERSION 1

#pragma pack(push, 1)
typedef struct {
  uint32_t magic;
  uint16_t version;
  uint16_t event_type;
  uint64_t sequence_id;
  uint64_t timestamp_ns;
  uint32_t stream_id;
  uint16_t device_id;
  uint16_t payload_length;
} event_header_t;
#pragma pack(pop)

/* Hash type (FNV-1a placeholder, same as in-tree) */
typedef struct { uint64_t lo, hi; } hash128_t;

static hash128_t hash_buffer(const void* data, size_t len) {
  uint64_t h1 = 0xcbf29ce484222325ULL, h2 = 0x100000001b3ULL;
  const uint8_t* p = (const uint8_t*)data;
  for (size_t i = 0; i < len; i++) {
    h1 ^= p[i]; h1 *= 0x100000001b3ULL;
    h2 ^= p[i]; h2 *= 0xcbf29ce484222325ULL;
  }
  return (hash128_t){h1, h2};
}

/* ---- Global state ---- */

#define MAX_ALLOCS 65536
#define MAX_MODULES 256
#define MAX_CO_KERNELS 512
#define MAX_FUNC_ENTRIES 4096
#define MAX_FUNC_NAME 256

typedef struct {
  uintptr_t ptr;
  uint64_t handle;
  size_t size;
} alloc_entry_t;

typedef struct {
  uintptr_t module;
  uint64_t handle;
  const void* image;
  size_t image_size;
  /* Parsed kernel metadata for this module */
  hrr_kernel_meta_t kernels[32];
  int num_kernels;
} module_entry_t;

typedef struct {
  uintptr_t handle;
  char name[MAX_FUNC_NAME];
} func_entry_t;

static struct {
  int active;
  int mode;  /* 0=timeline, 1=inputs, 2=full */
  char output_dir[512];
  char kernel_filter[256];
  size_t max_blob_mb;

  FILE* events_file;
  HRR_MUTEX mu;
  uint64_t seq_id;
  uint64_t next_handle;
  uint64_t next_mod_handle;

  alloc_entry_t allocs[MAX_ALLOCS];
  int num_allocs;

  module_entry_t modules[MAX_MODULES];
  int num_modules;

  /* Function handle → kernel name mapping (always active, not gated by HRR_RECORD) */
  func_entry_t funcs[MAX_FUNC_ENTRIES];
  int num_funcs;

  size_t blob_count;
  int verbose;  /* HRR_VERBOSE=1 — print diagnostic messages to stderr */
} g;

static void hash_hex(hash128_t h, char buf[33]) {
  snprintf(buf, 33, "%016llx%016llx",
           (unsigned long long)h.lo, (unsigned long long)h.hi);
}

static void write_blob(const void* data, size_t len, hash128_t* out_hash) {
  *out_hash = hash_buffer(data, len);
  char hex[33];
  hash_hex(*out_hash, hex);

  char subdir[600], path[640];
  snprintf(subdir, sizeof(subdir), "%s" HRR_SEP "blobs" HRR_SEP "%.2s",
           g.output_dir, hex);
  HRR_MKDIR(subdir);

  snprintf(path, sizeof(path), "%s" HRR_SEP "%s.blob", subdir, hex);
  FILE* f = fopen(path, "rb");
  if (f) { fclose(f); return; }  /* already exists */

  f = fopen(path, "wb");
  if (f) { fwrite(data, 1, len, f); fclose(f); g.blob_count++; }
}

static void write_event(uint16_t type, uint32_t stream_id,
                        const void* payload, uint16_t payload_len) {
  event_header_t hdr;
  hdr.magic = HRR_MAGIC;
  hdr.version = HRR_VERSION;
  hdr.event_type = type;
  hdr.sequence_id = g.seq_id++;
  hdr.timestamp_ns = hrr_now_ns();
  hdr.stream_id = stream_id;
  hdr.device_id = 0;
  hdr.payload_length = payload_len;

  HRR_MUTEX_LOCK(&g.mu);
  if (g.events_file) {
    fwrite(&hdr, sizeof(hdr), 1, g.events_file);
    if (payload && payload_len > 0)
      fwrite(payload, 1, payload_len, g.events_file);
  }
  HRR_MUTEX_UNLOCK(&g.mu);
}

static alloc_entry_t* find_alloc(uintptr_t ptr) {
  for (int i = 0; i < g.num_allocs; i++) {
    if (g.allocs[i].ptr == ptr) return &g.allocs[i];
  }
  return NULL;
}

static module_entry_t* find_module(uintptr_t mod) {
  for (int i = 0; i < g.num_modules; i++) {
    if (g.modules[i].module == mod) return &g.modules[i];
  }
  return NULL;
}

/* ---- Public API ---- */

int hrr_writer_init(void) {
  memset(&g, 0, sizeof(g));
  g.next_handle = 1;
  g.next_mod_handle = 1;
  HRR_MUTEX_INIT(&g.mu);

  const char* env = getenv("HRR_RECORD");
  if (!env || strcmp(env, "1") != 0) return 0;

  const char* out = getenv("HRR_OUTPUT");
  if (out && out[0]) {
    strncpy(g.output_dir, out, sizeof(g.output_dir) - 1);
  } else {
    time_t t = time(NULL);
    struct tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    snprintf(g.output_dir, sizeof(g.output_dir),
             "capture_%04d%02d%02d_%02d%02d%02d.hrr",
             tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec);
  }

  const char* mode = getenv("HRR_MODE");
  if (mode) {
    if (strcmp(mode, "timeline") == 0) g.mode = 0;
    else if (strcmp(mode, "full") == 0) g.mode = 2;
    else g.mode = 1;
  } else {
    g.mode = 1;
  }

  const char* filter = getenv("HRR_KERNEL_FILTER");
  if (filter) strncpy(g.kernel_filter, filter, sizeof(g.kernel_filter) - 1);

  const char* max_blob = getenv("HRR_MAX_BLOB_MB");
  if (max_blob) g.max_blob_mb = (size_t)atol(max_blob);

  const char* verbose = getenv("HRR_VERBOSE");
  if (verbose && verbose[0] == '1') g.verbose = 1;

  HRR_MKDIR(g.output_dir);
  char tmp[600];
  snprintf(tmp, sizeof(tmp), "%s" HRR_SEP "blobs", g.output_dir);
  HRR_MKDIR(tmp);
  snprintf(tmp, sizeof(tmp), "%s" HRR_SEP "code_objects", g.output_dir);
  HRR_MKDIR(tmp);

  snprintf(tmp, sizeof(tmp), "%s" HRR_SEP "events.bin", g.output_dir);
  g.events_file = fopen(tmp, "wb");
  if (!g.events_file) {
    fprintf(stderr, "[HRR] Cannot open %s\n", tmp);
    return 0;
  }

  g.active = 1;
  fprintf(stderr, "[HRR] Recording to %s (mode=%s)\n",
          g.output_dir, mode ? mode : "inputs");
  return 1;
}

void hrr_writer_shutdown(void) {
  if (!g.active) return;
  if (g.events_file) { fflush(g.events_file); fclose(g.events_file); }

  /* Write manifest */
  char path[600];
  snprintf(path, sizeof(path), "%s" HRR_SEP "manifest.json", g.output_dir);
  FILE* f = fopen(path, "w");
  if (f) {
    const char* modes[] = {"timeline", "inputs", "full"};
    fprintf(f, "{\n  \"version\": 1,\n  \"format\": \"hrr-v1\",\n"
               "  \"capture_mode\": \"%s\",\n  \"event_count\": %llu,\n"
               "  \"blob_count\": %zu\n}\n",
            modes[g.mode], (unsigned long long)g.seq_id, g.blob_count);
    fclose(f);
  }
  fprintf(stderr, "[HRR] Recording complete: %llu events, %zu blobs\n",
          (unsigned long long)g.seq_id, g.blob_count);
  g.active = 0;
}

int hrr_writer_enabled(void) { return g.active; }

void hrr_record_malloc(const void* ptr, size_t size, unsigned int flags) {
  if (!g.active) return;
  uint64_t handle;
  HRR_MUTEX_LOCK(&g.mu);
  handle = g.next_handle++;
  if (g.num_allocs < MAX_ALLOCS) {
    g.allocs[g.num_allocs++] = (alloc_entry_t){(uintptr_t)ptr, handle, size};
  }
  HRR_MUTEX_UNLOCK(&g.mu);

#pragma pack(push,1)
  struct { uint64_t h; uint64_t s; uint32_t f; } pl = {handle, size, flags};
#pragma pack(pop)
  write_event(EVT_MALLOC, 0, &pl, sizeof(pl));
}

void hrr_record_free(const void* ptr) {
  if (!g.active) return;
  uint64_t handle = 0;
  HRR_MUTEX_LOCK(&g.mu);
  alloc_entry_t* e = find_alloc((uintptr_t)ptr);
  if (e) { handle = e->handle; *e = g.allocs[--g.num_allocs]; }
  HRR_MUTEX_UNLOCK(&g.mu);
  write_event(EVT_FREE, 0, &handle, sizeof(handle));
}

void hrr_record_memcpy(void* dst, const void* src, size_t size,
                       unsigned int kind, const void* stream) {
  if (!g.active) return;
  hash128_t blob_hash = {0, 0};
  if (g.mode > 0 && kind == 1 && src && size > 0) {
    if (g.max_blob_mb == 0 || size <= g.max_blob_mb * 1024 * 1024)
      write_blob(src, size, &blob_hash);
  }
#pragma pack(push,1)
  struct { uint64_t d,s,sz; uint32_t k; uint64_t hl,hh; } pl = {
    (uint64_t)(uintptr_t)dst, (uint64_t)(uintptr_t)src, size, kind,
    blob_hash.lo, blob_hash.hi};
#pragma pack(pop)
  write_event(EVT_MEMCPY, (uint32_t)(uintptr_t)stream, &pl, sizeof(pl));
}

void hrr_record_memset(void* dst, int value, size_t size, const void* stream) {
  if (!g.active) return;
#pragma pack(push,1)
  struct { uint64_t d; uint32_t v; uint64_t s; } pl = {
    (uint64_t)(uintptr_t)dst, (uint32_t)value, size};
#pragma pack(pop)
  write_event(EVT_MEMSET, (uint32_t)(uintptr_t)stream, &pl, sizeof(pl));
}

void hrr_record_module_load(void* module, const void* image, size_t image_size) {
  if (!g.active) return;
  uint64_t mod_handle;
  HRR_MUTEX_LOCK(&g.mu);
  mod_handle = g.next_mod_handle++;
  if (g.num_modules < MAX_MODULES) {
    module_entry_t* me = &g.modules[g.num_modules++];
    me->module = (uintptr_t)module;
    me->handle = mod_handle;
    me->image = image;
    me->image_size = image_size;
    me->num_kernels = hrr_parse_code_object(image, image_size,
                                            me->kernels, 32);
  }
  HRR_MUTEX_UNLOCK(&g.mu);

  /* Save code object */
  if (image && image_size > 0) {
    hash128_t h = hash_buffer(image, image_size);
    char hex[33]; hash_hex(h, hex);
    char path[640];
    snprintf(path, sizeof(path), "%s" HRR_SEP "code_objects" HRR_SEP "%s.hsaco",
             g.output_dir, hex);
    FILE* check = fopen(path, "rb");
    if (!check) {
      FILE* f = fopen(path, "wb");
      if (f) { fwrite(image, 1, image_size, f); fclose(f); }
    } else { fclose(check); }

#pragma pack(push,1)
    struct { uint64_t hl,hh,mh; } pl = {h.lo, h.hi, mod_handle};
#pragma pack(pop)
    write_event(EVT_MODULE_LOAD, 0, &pl, sizeof(pl));
  }
}

void hrr_record_module_unload(void* module) {
  if (!g.active) return;
  uint64_t handle = 0;
  HRR_MUTEX_LOCK(&g.mu);
  module_entry_t* me = find_module((uintptr_t)module);
  if (me) { handle = me->handle; *me = g.modules[--g.num_modules]; }
  HRR_MUTEX_UNLOCK(&g.mu);
  write_event(EVT_MODULE_UNLOAD, 0, &handle, sizeof(handle));
}

void hrr_record_kernel_launch(const char* kernel_name,
                              const void* code_object_image,
                              size_t code_object_size,
                              uint32_t gx, uint32_t gy, uint32_t gz,
                              uint32_t bx, uint32_t by, uint32_t bz,
                              uint32_t shared_mem,
                              const void* stream,
                              void** kernel_args) {
  if (!g.active) return;
  if (g.kernel_filter[0] != '\0' && kernel_name) {
    size_t flen = strlen(g.kernel_filter);
    if (g.kernel_filter[flen-1] == '*') {
      if (strncmp(kernel_name, g.kernel_filter, flen-1) != 0) return;
    } else {
      if (strcmp(kernel_name, g.kernel_filter) != 0) return;
    }
  }

  /* Find kernel metadata for arg introspection */
  const hrr_kernel_meta_t* meta = NULL;
  if (kernel_name) {
    HRR_MUTEX_LOCK(&g.mu);
    for (int i = 0; i < g.num_modules && !meta; i++) {
      meta = hrr_find_kernel(g.modules[i].kernels,
                             g.modules[i].num_kernels, kernel_name);
    }
    HRR_MUTEX_UNLOCK(&g.mu);
  }

  /* Build payload */
  const char* name = kernel_name ? kernel_name : "<unknown>";
  uint16_t name_len = (uint16_t)strlen(name);
  uint16_t num_args = meta ? (uint16_t)meta->num_args : 0;
  uint16_t num_snaps = 0;  /* TODO: buffer snapshots in out-of-tree */

  size_t pl_size = 2 + name_len + 12 + 12 + 4 + 2 + 2;
  /* Add args */
  for (uint32_t i = 0; i < num_args; i++) {
    pl_size += 1 + 2 + (meta->args[i].kind == HRR_ARG_GLOBAL_BUFFER ? 8 :
                         meta->args[i].size);
  }

  uint8_t* pl = (uint8_t*)malloc(pl_size);
  if (!pl) return;
  uint8_t* p = pl;

  memcpy(p, &name_len, 2); p += 2;
  memcpy(p, name, name_len); p += name_len;
  memcpy(p, &gx, 4); p += 4;
  memcpy(p, &gy, 4); p += 4;
  memcpy(p, &gz, 4); p += 4;
  memcpy(p, &bx, 4); p += 4;
  memcpy(p, &by, 4); p += 4;
  memcpy(p, &bz, 4); p += 4;
  memcpy(p, &shared_mem, 4); p += 4;
  memcpy(p, &num_args, 2); p += 2;
  memcpy(p, &num_snaps, 2); p += 2;

  for (uint32_t i = 0; i < num_args; i++) {
    const hrr_arg_desc_t* ad = &meta->args[i];
    uint8_t vk = (uint8_t)ad->kind;
    *p++ = vk;

    if (ad->kind == HRR_ARG_GLOBAL_BUFFER) {
      uint16_t sz = 8;
      memcpy(p, &sz, 2); p += 2;
      if (kernel_args && kernel_args[i]) {
        /* Translate raw device pointer to allocation handle so the replay
         * can remap it to a live GPU address on the target machine. */
        uint64_t dev_ptr = 0;
        memcpy(&dev_ptr, kernel_args[i], 8);
        uint64_t handle = 0;
        HRR_MUTEX_LOCK(&g.mu);
        alloc_entry_t* ae = find_alloc((uintptr_t)dev_ptr);
        if (ae) handle = ae->handle;
        HRR_MUTEX_UNLOCK(&g.mu);
        memcpy(p, &handle, 8);
      } else {
        memset(p, 0, 8);
      }
      p += 8;
    } else {
      uint16_t sz = ad->size;
      memcpy(p, &sz, 2); p += 2;
      if (kernel_args && kernel_args[i] && ad->kind != HRR_ARG_HIDDEN) {
        memcpy(p, kernel_args[i], ad->size);
      } else {
        memset(p, 0, ad->size);
      }
      p += ad->size;
    }
  }

  uint16_t pl_len = (uint16_t)(pl_size > 65535 ? 65535 : pl_size);
  write_event(EVT_KERNEL_LAUNCH, (uint32_t)(uintptr_t)stream, pl, pl_len);
  free(pl);
}

void hrr_record_kernel_launch_packed(const char* kernel_name,
                                      uint32_t gx, uint32_t gy, uint32_t gz,
                                      uint32_t bx, uint32_t by, uint32_t bz,
                                      uint32_t shared_mem,
                                      const void* stream,
                                      const void* packed_buf,
                                      size_t packed_size) {
  if (!g.active) return;
  if (g.kernel_filter[0] != '\0' && kernel_name) {
    size_t flen = strlen(g.kernel_filter);
    if (g.kernel_filter[flen-1] == '*') {
      if (strncmp(kernel_name, g.kernel_filter, flen-1) != 0) return;
    } else {
      if (strcmp(kernel_name, g.kernel_filter) != 0) return;
    }
  }

  /* Find kernel metadata for arg introspection */
  const hrr_kernel_meta_t* meta = NULL;
  if (kernel_name) {
    HRR_MUTEX_LOCK(&g.mu);
    for (int i = 0; i < g.num_modules && !meta; i++) {
      meta = hrr_find_kernel(g.modules[i].kernels,
                             g.modules[i].num_kernels, kernel_name);
    }
    HRR_MUTEX_UNLOCK(&g.mu);
  }

  const char* name = kernel_name ? kernel_name : "<unknown>";
  uint16_t name_len = (uint16_t)strlen(name);
  uint16_t num_args = meta ? (uint16_t)meta->num_args : 0;
  uint16_t num_snaps = 0;

  size_t pl_size = 2 + name_len + 12 + 12 + 4 + 2 + 2;
  for (uint32_t i = 0; i < num_args; i++) {
    pl_size += 1 + 2 + (meta->args[i].kind == HRR_ARG_GLOBAL_BUFFER ? 8 :
                         meta->args[i].size);
  }

  uint8_t* pl = (uint8_t*)malloc(pl_size);
  if (!pl) return;
  uint8_t* p = pl;

  memcpy(p, &name_len, 2); p += 2;
  memcpy(p, name, name_len); p += name_len;
  memcpy(p, &gx, 4); p += 4;
  memcpy(p, &gy, 4); p += 4;
  memcpy(p, &gz, 4); p += 4;
  memcpy(p, &bx, 4); p += 4;
  memcpy(p, &by, 4); p += 4;
  memcpy(p, &bz, 4); p += 4;
  memcpy(p, &shared_mem, 4); p += 4;
  memcpy(p, &num_args, 2); p += 2;
  memcpy(p, &num_snaps, 2); p += 2;

  for (uint32_t i = 0; i < num_args; i++) {
    const hrr_arg_desc_t* ad = &meta->args[i];
    uint8_t vk = (uint8_t)ad->kind;
    *p++ = vk;

    const char* arg_data = packed_buf ?
        (const char*)packed_buf + ad->offset : NULL;
    int arg_in_bounds = packed_buf &&
        (size_t)(ad->offset + (ad->kind == HRR_ARG_GLOBAL_BUFFER ? 8 : ad->size))
        <= packed_size;

    if (ad->kind == HRR_ARG_GLOBAL_BUFFER) {
      uint16_t sz = 8;
      memcpy(p, &sz, 2); p += 2;
      if (arg_in_bounds) {
        uint64_t dev_ptr = 0;
        memcpy(&dev_ptr, arg_data, 8);
        uint64_t handle = 0;
        HRR_MUTEX_LOCK(&g.mu);
        alloc_entry_t* ae = find_alloc((uintptr_t)dev_ptr);
        if (ae) handle = ae->handle;
        HRR_MUTEX_UNLOCK(&g.mu);
        if (!ae && dev_ptr != 0) {
          /* Pointer not tracked — likely allocated via an API we don't intercept
           * (e.g. HSA-level, hipGraph, or internal runtime pool). Emit a synthetic
           * MALLOC event so the replayer can create a backing buffer for it.
           * We don't know the true size; use 1MB as a safe upper bound. */
          if (g.verbose) {
            fprintf(stderr, "[HRR] kernel '%s' arg[%u]: ptr=0x%llx untracked, "
                    "registering synthetic alloc (1MB)\n",
                    name, i, (unsigned long long)dev_ptr);
          }
          hrr_record_malloc((const void*)(uintptr_t)dev_ptr, 1024*1024, 0);
          HRR_MUTEX_LOCK(&g.mu);
          ae = find_alloc((uintptr_t)dev_ptr);
          if (ae) handle = ae->handle;
          HRR_MUTEX_UNLOCK(&g.mu);
        }
        memcpy(p, &handle, 8);
      } else {
        memset(p, 0, 8);
      }
      p += 8;
    } else {
      uint16_t sz = ad->size;
      memcpy(p, &sz, 2); p += 2;
      if (arg_in_bounds && ad->kind != HRR_ARG_HIDDEN) {
        memcpy(p, arg_data, ad->size);
      } else {
        memset(p, 0, ad->size);
      }
      p += ad->size;
    }
  }

  uint16_t pl_len = (uint16_t)(pl_size > 65535 ? 65535 : pl_size);
  write_event(EVT_KERNEL_LAUNCH, (uint32_t)(uintptr_t)stream, pl, pl_len);
  free(pl);
}

void hrr_record_device_sync(void) {
  if (!g.active) return;
  write_event(EVT_DEVICE_SYNC, 0, NULL, 0);
}

void hrr_record_stream_sync(const void* stream) {
  if (!g.active) return;
  write_event(EVT_STREAM_SYNC, (uint32_t)(uintptr_t)stream, NULL, 0);
}

void hrr_register_function(const void* func_handle, const char* kernel_name) {
  if (!func_handle || !kernel_name) return;
  HRR_MUTEX_LOCK(&g.mu);
  /* Check if already registered (update in place) */
  uintptr_t h = (uintptr_t)func_handle;
  for (int i = 0; i < g.num_funcs; i++) {
    if (g.funcs[i].handle == h) {
      strncpy(g.funcs[i].name, kernel_name, MAX_FUNC_NAME - 1);
      g.funcs[i].name[MAX_FUNC_NAME - 1] = '\0';
      HRR_MUTEX_UNLOCK(&g.mu);
      return;
    }
  }
  if (g.num_funcs < MAX_FUNC_ENTRIES) {
    g.funcs[g.num_funcs].handle = h;
    strncpy(g.funcs[g.num_funcs].name, kernel_name, MAX_FUNC_NAME - 1);
    g.funcs[g.num_funcs].name[MAX_FUNC_NAME - 1] = '\0';
    g.num_funcs++;
  }
  HRR_MUTEX_UNLOCK(&g.mu);
}

const char* hrr_lookup_function_name(const void* func_handle) {
  if (!func_handle) return NULL;
  uintptr_t h = (uintptr_t)func_handle;
  HRR_MUTEX_LOCK(&g.mu);
  for (int i = 0; i < g.num_funcs; i++) {
    if (g.funcs[i].handle == h) {
      const char* name = g.funcs[i].name;
      HRR_MUTEX_UNLOCK(&g.mu);
      return name;
    }
  }
  HRR_MUTEX_UNLOCK(&g.mu);
  return NULL;
}
