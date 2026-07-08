#include <dlfcn.h>
#include <link.h>
#include <cstring>
#include <string>

#include "impl/hsa/hsa.h"
#include "impl/hsa/hsa_ven_amd_loader.h"

namespace {

constexpr const char *kRequiredHsaRuntimeSymbols[] = {
    "hsa_signal_load_relaxed",
    "hsa_signal_wait_relaxed",
    "hsa_signal_store_screlease",
    "hsa_system_get_extension_table",
};

bool hsa_runtime_symbols_available() {
  for (const char *sym : kRequiredHsaRuntimeSymbols) {
    if (dlsym(RTLD_DEFAULT, sym) == nullptr)
      return false;
  }
  return true;
}

constexpr int kHsaRuntimeDlopenFlags = RTLD_NOW | RTLD_GLOBAL;

bool dlopen_promote_global(const char *name, std::string *last_error) {
  void *handle = dlopen(name, kHsaRuntimeDlopenFlags);
  if (handle == nullptr) {
    const char *err = dlerror();
    pr_debug("dlopen %s failed - %s\n", name, err ? err : "(null)");
    if (last_error && err)
      *last_error = std::string(name) + ": " + err;
    return false;
  }

  dlclose(handle);
  return true;
}

const char *kHsaRuntimeSonames[] = {
    "libhsa-runtime64.so.1",
    "libhsa-runtime64.so",
    nullptr,
};

struct MappedHsaRuntimeSearch {
  bool promoted = false;
  std::string last_error;
};

int find_mapped_hsa_runtime_cb(struct dl_phdr_info *info, size_t /*size*/,
                               void *data) {
  auto *search = static_cast<MappedHsaRuntimeSearch *>(data);

  if (info->dlpi_name == nullptr || info->dlpi_name[0] == '\0')
    return 0;

  // Match by basename to avoid accidentally hitting unrelated paths.
  const char *base = std::strrchr(info->dlpi_name, '/');
  base = base ? base + 1 : info->dlpi_name;
  if (std::strncmp(base, "libhsa-runtime64.so", 19) != 0)
    return 0;

  if (dlopen_promote_global(info->dlpi_name, &search->last_error)) {
    search->promoted = true;
    return 1;
  }
  return 0;
}

bool dlopen_promote_mapped_hsa_runtime(std::string *last_error) {
  MappedHsaRuntimeSearch search;
  dl_iterate_phdr(find_mapped_hsa_runtime_cb, &search);
  if (!search.promoted && last_error && !search.last_error.empty())
    *last_error = search.last_error;
  return search.promoted;
}

} // namespace

static std::mutex* lock_ = new std::mutex();

#if 1
#define _HSAKMT_LOOKUP_SYMS(_sym)                                              \
if (fn_##_sym == nullptr) {                                                    \
    std::lock_guard<std::mutex> gard(*lock_);                                  \
    if (fn_##_sym == nullptr) {                                                \
      fn_##_sym =                                                              \
        reinterpret_cast<decltype(fn_##_sym)>(dlsym(RTLD_DEFAULT, #_sym));     \
      if (!fn_##_sym) {                                                        \
        pr_err("%s not found - %s\n", #_sym, dlerror());                       \
      }                                                                        \
    }                                                                          \
}

#define _HSAKMT_EXEC_API(_sym, ...) \
do { \
    if (fn_##_sym != nullptr) {    \
        return fn_##_sym(__VA_ARGS__);   \
    } \
} while(0);

bool hsakmt_hsa_loader_init() {
  if (hsa_runtime_symbols_available())
    return true;

  std::string last_error;
  for (const char *const *name = kHsaRuntimeSonames; *name != nullptr; ++name) {
    if (dlopen_promote_global(*name, &last_error))
      return true;
  }

  if (dlopen_promote_mapped_hsa_runtime(&last_error))
    return true;

  pr_err("failed to promote libhsa-runtime64 symbols (last error: %s)\n",
         last_error.empty() ? "(none)" : last_error.c_str());
  return false;
}

hsa_signal_value_t hsakmt_hsa_signal_load_relaxed(hsa_signal_t signal) {
  static hsa_signal_value_t (*fn_hsa_signal_load_relaxed)(hsa_signal_t signal) = nullptr;

  _HSAKMT_LOOKUP_SYMS(hsa_signal_load_relaxed);
  _HSAKMT_EXEC_API(hsa_signal_load_relaxed, signal);

  return 0;
}

hsa_signal_value_t hsakmt_hsa_signal_wait_relaxed(
    hsa_signal_t signal, hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value, uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint) {
static hsa_signal_value_t (*fn_hsa_signal_wait_relaxed)(
    hsa_signal_t signal, hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value, uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint) = nullptr;

  _HSAKMT_LOOKUP_SYMS(hsa_signal_wait_relaxed);
  _HSAKMT_EXEC_API(hsa_signal_wait_relaxed, signal, condition, compare_value,
                   timeout_hint, wait_state_hint);

  return 0;
}

void hsakmt_hsa_signal_store_screlease(hsa_signal_t hsa_signal,
                                      hsa_signal_value_t value){
static void (*fn_hsa_signal_store_screlease)(hsa_signal_t hsa_signal,
                                      hsa_signal_value_t value) = nullptr;

  _HSAKMT_LOOKUP_SYMS(hsa_signal_store_screlease);
  _HSAKMT_EXEC_API(hsa_signal_store_screlease, hsa_signal, value);
}

hsa_status_t hsakmt_hsa_ven_amd_loader_query_host_address(
    const void *device_address, const void **host_address) {
  static hsa_status_t (*fn_hsa_ven_amd_loader_query_host_address)(
    const void *device_address, const void **host_address) = nullptr;

  if (fn_hsa_ven_amd_loader_query_host_address == nullptr) {
    std::lock_guard<std::mutex> gard(*lock_);
    if (fn_hsa_ven_amd_loader_query_host_address == nullptr) {
      hsa_status_t (*fn_hsa_system_get_extension_table)(
      uint16_t extension, uint16_t version_major, uint16_t version_minor, void *table);
      fn_hsa_system_get_extension_table =
        reinterpret_cast<decltype(fn_hsa_system_get_extension_table)>(dlsym(RTLD_DEFAULT, "hsa_system_get_extension_table"));
      if (fn_hsa_system_get_extension_table == nullptr) {
        pr_err("%s not found - %s\n", "hsa_system_get_extension_table", dlerror());
        return HSA_STATUS_ERROR;
      }

      hsa_ven_amd_loader_1_03_pfn_t table;
      fn_hsa_system_get_extension_table(HSA_EXTENSION_AMD_LOADER, 1, 3, &table);
      fn_hsa_ven_amd_loader_query_host_address =
          table.hsa_ven_amd_loader_query_host_address;
    }
  }

  _HSAKMT_EXEC_API(hsa_ven_amd_loader_query_host_address, device_address, host_address);
  return HSA_STATUS_ERROR;
}

#else
hsa_signal_value_t hsakmt_hsa_signal_load_relaxed(hsa_signal_t signal) {
  return hsa_signal_load_relaxed(signal);
}

hsa_signal_value_t hsakmt_hsa_signal_wait_relaxed(
    hsa_signal_t signal, hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value, uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint) {
  return hsa_signal_wait_relaxed(signal, condition, compare_value, timeout_hint,
                                 wait_state_hint);
}

void hsakmt_hsa_signal_store_screlease(hsa_signal_t hsa_signal,
                                      hsa_signal_value_t value) {
  hsa_signal_store_screlease(hsa_signal, value);
}

hsa_status_t hsakmt_hsa_ven_amd_loader_query_host_address(
    const void *device_address, const void **host_address) {
  static hsa_status_t (*fn_hsa_ven_amd_loader_query_host_address)(
    const void *device_address, const void **host_address) = nullptr;

  if (fn_hsa_ven_amd_loader_query_host_address == nullptr) {
    std::lock_guard<std::mutex> gard(*lock_);
    if (fn_hsa_ven_amd_loader_query_host_address == nullptr) {
      hsa_ven_amd_loader_1_03_pfn_t table;
      hsa_system_get_extension_table(HSA_EXTENSION_AMD_LOADER, 1, 3, &table);
      fn_hsa_ven_amd_loader_query_host_address =
          table.hsa_ven_amd_loader_query_host_address;
    }
  }

  if (fn_hsa_ven_amd_loader_query_host_address)
    return fn_hsa_ven_amd_loader_query_host_address(device_address, host_address);

  return HSA_STATUS_ERROR;
}
#endif
