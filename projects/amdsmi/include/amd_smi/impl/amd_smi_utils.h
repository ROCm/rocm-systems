// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_AMD_SMI_UTILS_H_
#define AMD_SMI_INCLUDE_AMD_SMI_UTILS_H_

#include <dirent.h>
#include <sys/utsname.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"

extern "C" {
void amdsmi_free_name_value_pairs(void* p);
}

amdsmi_status_t smi_amdgpu_find_hwmon_dir(amd::smi::AMDSmiGPUDevice* device,
                                          std::string* full_path);
amdsmi_status_t smi_amdgpu_get_board_info(amd::smi::AMDSmiGPUDevice* device,
                                          amdsmi_board_info_t* info);
amdsmi_status_t smi_amdgpu_get_power_cap(amd::smi::AMDSmiGPUDevice* device, uint32_t sensor_ind,
                                         int* cap);
amdsmi_status_t smi_amdgpu_get_ranges(amd::smi::AMDSmiGPUDevice* device, amdsmi_clk_type_t domain,
                                      int* max_freq, int* min_freq, int* num_dpm,
                                      int* sleep_state_freq);
amdsmi_status_t smi_amdgpu_get_enabled_blocks(amd::smi::AMDSmiGPUDevice* device,
                                              uint64_t* enabled_blocks);
amdsmi_status_t smi_amdgpu_get_bad_page_info(amd::smi::AMDSmiGPUDevice* device, uint32_t* num_pages,
                                             amdsmi_retired_page_record_t* info);
amdsmi_status_t smi_amdgpu_get_bad_page_threshold(amd::smi::AMDSmiGPUDevice* device,
                                                  uint32_t* threshold);
amdsmi_status_t smi_amdgpu_validate_ras_eeprom(amd::smi::AMDSmiGPUDevice* device);
amdsmi_status_t smi_amdgpu_get_driver_version(amd::smi::AMDSmiGPUDevice* device, int* length,
                                              char* version);
amdsmi_status_t smi_amdgpu_get_pcie_speed_from_pcie_type(uint16_t pcie_type, uint32_t* pcie_speed);
// True when KFD reports the GPU's link to the CPU as xGMI: the GPU has no PCIe host link, and
// the PCIe link in its config space is emulated. False otherwise, including when the lookup fails.
bool smi_amdgpu_has_xgmi_host_link(amd::smi::AMDSmiGPUDevice* device);
// PLX temperature is carried in gpu_metrics rather than hwmon. Reports
// NOT_SUPPORTED when the reading holds the not-applicable sentinel (an APU has
// no such sensor), otherwise writes *temperature.
amdsmi_status_t smi_amdgpu_plx_temp_from_metrics(const amdsmi_gpu_metrics_t& metrics,
                                                 int64_t* temperature);
amdsmi_status_t smi_amdgpu_get_market_name_from_dev_id(amd::smi::AMDSmiGPUDevice* device,
                                                       char* market_name);
// ids_flags must come from a successful native DRM query; zero is valid.
amdsmi_status_t smi_amdgpu_is_apu(uint64_t ids_flags, bool* is_apu);
amdsmi_status_t smi_amdgpu_is_gpu_power_management_enabled(amd::smi::AMDSmiGPUDevice* device,
                                                           bool* enabled);
amdsmi_status_t smi_amdgpu_get_vcn_busy_percent(amd::smi::AMDSmiGPUDevice* device,
                                                uint32_t* vcn_busy_percent);
std::string smi_amdgpu_split_string(std::string str, char delim);
std::vector<std::string> split_string(const std::string& line, char delim);
std::string smi_amdgpu_get_status_string(amdsmi_status_t ret, bool fullStatus);

std::string smi_read_sysfs_string(const std::string& folder, const std::string& file_name);

amdsmi_status_t smi_clear_char_and_reinitialize(char buffer[], uint32_t len, std::string newString);

/**
 *  @brief Get the device index given the processor handle.
 *
 *  @details Given a processor handle @p processor_handle
 *  and a pointer to a uint32_t @p device_index will be returned.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[inout] device_index a pointer to uint32_t to which the matching device
 *  index will be stored
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS is returned upon successful call.
 *          ::AMDSMI_STATUS_INVAL is returned if user provides a null pointer
 *          for device_index.
 *          ::AMDSMI_STATUS_API_FAILED is returned if the corresponding device
 *          index for the processor handle cannot be found.
 */
amdsmi_status_t smi_amdgpu_get_device_index(amdsmi_processor_handle processor_handle,
                                            uint32_t* device_index);

/**
 *  @brief Get total number of devices
 *
 *  @details Given a pointer to a uint32_t @p total_num_devices will be returned
 *
 *  @param[inout] total_num_devices a pointer to uint32_t to which the total number
 *  of devices will be stored
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS is returned upon successful call.
 *          ::AMDSMI_STATUS_INVAL is returned if user provides a null pointer
 *          for total_num_devices.
 */
amdsmi_status_t smi_amdgpu_get_device_count(uint32_t* total_num_devices);

/**
 *  @brief Get the ainic processor handle given the device index.
 *
 *  @details Given a uint32_t @p device_index and a pointer to
 *  a ainic processor handle @p processor_handle, the device index will be used to
 *  find the processor handle of the device and store it in the provided pointer
 *
 *  @param[in] device_index a uint32_t to value to help find the corresponding
 *  ainic processor handle
 *
 *  @param[inout] processor_handle a pointer to amdsmi_processor_handle
 *  which the corresponding processor_handle will be stored
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS is returned upon successful call.
 *          ::AMDSMI_STATUS_INVAL is returned if user provides a null pointer
 *          for processor_handle.
 *          ::AMDSMI_STATUS_API_FAILED is returned if the device_index is cannot
 *          be found.
 */
amdsmi_status_t smi_amdgpu_get_ainic_processor_handle_by_index(
    uint32_t device_index, amdsmi_processor_handle* processor_handle);

/**
 *  @brief Get the processor handle given the device index.
 *
 *  @details Given a uint32_t @p device_index and a pointer to
 *  a processor handle @p processor_handle, the device index will be used to
 *  find the processor handle of the device and store it in the provided pointer
 *
 *  @param[in] device_index a uint32_t to value to help find the corresponding
 *  processor handle
 *
 *  @param[inout] processor_handle a pointer to amdsmi_processor_handle
 *  which the corresponding processor_handle will be stored
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS is returned upon successful call.
 *          ::AMDSMI_STATUS_INVAL is returned if user provides a null pointer
 *          for processor_handle.
 *          ::AMDSMI_STATUS_API_FAILED is returned if the device_index is cannot
 *          be found.
 */
amdsmi_status_t smi_amdgpu_get_processor_handle_by_index(uint32_t device_index,
                                                         amdsmi_processor_handle* processor_handle);

amdsmi_status_t get_gpu_device_from_handle(amdsmi_processor_handle processor_handle,
                                           amd::smi::AMDSmiGPUDevice** gpudevice);

/**
 *  @brief Get an int environment var or return default if does not exist
 *
 *  @details Given a const char* @p name and a default int @p def
 *  and call getenv with name. On any error, return default int
 *
 *  @param[in] name a const char* containing ENV var name
 *
 *  @param[in] def default int in case of error
 *
 *  @retval int of environment variable
 */
int read_env_ms(const char* name, int def);

template <typename>
constexpr bool is_dependent_false_v = false;

template <typename T>
inline constexpr bool is_supported_type_v =
    (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::uint8_t> ||
     std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::uint16_t> ||
     std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::uint32_t> ||
     std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::uint64_t>);

template <typename T>
constexpr T get_std_num_limit() {
  if constexpr (is_supported_type_v<T>) {
    return std::numeric_limits<T>::max();
  } else {
    return std::numeric_limits<T>::min();
    static_assert(is_dependent_false_v<T>, "Error: Type not supported...");
  }
}

template <typename T>
constexpr bool is_std_num_limit(T value) {
  return (value == get_std_num_limit<T>());
}

template <typename T, typename U, typename V = T>
constexpr T translate_umax_or_assign_value(U source_value, V target_value) {
  T result{};
  if constexpr (is_supported_type_v<T> && is_supported_type_v<U>) {
    // If the source value is uint<U>::max(), then return is uint<T>::max()
    if (is_std_num_limit(source_value)) {
      result = get_std_num_limit<T>();
    } else {
      result = static_cast<T>(target_value);
    }

    return result;
  } else {
    static_assert(is_dependent_false_v<T>, "Error: Type not supported...");
  }

  return result;
}

template <typename A, typename T>
void fill_2d_array(A& arr, T value) {
  for (auto& row : arr) {
    std::fill(std::begin(row), std::end(row), value);
  }
}

std::string_view trim(std::string_view str);

template <typename Tp, typename = std::enable_if_t<std::is_integral_v<Tp>>>
std::optional<Tp> parse_number_from_string(std::string_view str) {
  auto trimmed_str = trim(str);
  if (trimmed_str.empty()) {
    return std::nullopt;
  }

  auto value = Tp{};
  auto [ptr, error_code] =
      std::from_chars(trimmed_str.data(), (trimmed_str.data() + trimmed_str.size()), value);
  if ((error_code == std::errc{}) && (ptr == (trimmed_str.data() + trimmed_str.size()))) {
    return value;
  }

  return std::nullopt;
}

/**
 *  @brief Get the product serial number given the processor handle.
 *
 *  @param[in] processor_handle a pointer to amdsmi_processor_handle
 *  which the corresponding processor_handle will be stored
 *
 *  @retval ::The serial number
 *          ::0 if it cannot be determined
 */
uint64_t get_product_serial_number(amdsmi_processor_handle processor_handle);

/*
 *  Note:
 *      - A full 128-bit UUID is 16 bytes
 *      - Therefore, the total number of hexadecimal digits needed to represent a full 128-bit UUID
 * is 32
 *          - (16 bytes * 2 hexadecimal digits per byte)
 */
constexpr auto HIP_UUID_BYTES_SIZE = size_t(16);
constexpr auto HIP_UUID_STRING_FULL_SIZE = size_t(HIP_UUID_BYTES_SIZE * 2);
typedef struct hipUUID_t {
  // Pointer to raw UUID bytes (length = HIP_UUID_BYTES_SIZE), not null-terminated
  char bytes[HIP_UUID_BYTES_SIZE];
} hipUUID_t;

const char* from_uuid_to_cstring(const hipUUID_t& uuid) noexcept;
std::optional<amdsmi_bdf_t> from_cstring_to_bdf(const char* bdf_str) noexcept;
std::optional<hipUUID_t> from_cstring_to_uuid(const char* uuid_str) noexcept;
std::string stringify_bdf(const amdsmi_bdf_t& bdf);

/**
 *  @brief Tokenize bdfid into components.
 *
 *  @param[in] bdfid a uint64_t containing the bdfid
 *
 *  @retval ::Tuple of domain, bus, device, function
 */
std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> parse_bdfid(uint64_t bdfid);

/**
 *  @brief Read a pp_dpm_* sysfs file and populate an amdsmi_frequencies_t.
 *
 *  @details Parses the kernel's per-domain DPM table (e.g. pp_dpm_vclk,
 *  pp_dpm_dclk) for the given device. Lines look like:
 *      0: 200Mhz
 *      1: 400Mhz *
 *      2: 800Mhz
 *  The '*' marker is recorded into @p f->current.
 *
 *  @param[in]  device       Device whose render node sysfs is read.
 *  @param[in]  pp_dpm_file  Basename of the sysfs file (e.g. "pp_dpm_vclk").
 *  @param[out] f            Frequencies struct to populate.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS on success.
 *          ::AMDSMI_STATUS_INVAL if @p f is null.
 *          ::AMDSMI_STATUS_NOT_SUPPORTED if the sysfs file is missing or empty.
 */
amdsmi_status_t smi_amdgpu_read_clk_freq_from_pp_dpm(amd::smi::AMDSmiGPUDevice* device,
                                                     const char* pp_dpm_file,
                                                     amdsmi_frequencies_t* f);

/**
 *  @brief Map a VCLK/DCLK clock type to its pp_dpm_* sysfs filename.
 *
 *  rsmi does not expose these clock types via gpu_metrics, so amdsmi reads
 *  them directly from sysfs.
 *
 *  @param[in] clk_type Clock type.
 *
 *  @retval Pointer to a static filename string, or nullptr for clock types
 *          that are not VCLK0/VCLK1/DCLK0/DCLK1.
 */
const char* smi_amdgpu_pp_dpm_filename_for_clk_type(amdsmi_clk_type_t clk_type);

/**
 *  @brief Reset amdsmi_asic_info_t to its not-supported state.
 *
 *  Backends call this before filling in what they can, so a field a backend
 *  does not know about reports N/A rather than a plausible zero.
 *
 *  @param[out] info Structure to reset to its not-supported state.
 */
void init_asic_info_defaults(amdsmi_asic_info_t* info);

/**
 *  @brief Format a DRM driver date such as ``20150101`` as ``2015/01/01 00:00``.
 *
 *  Returns an empty string (N/A) for ``0``, which kernels without a DRM driver
 *  date report.
 */
auto smi_amdgpu_format_driver_date(std::string_view drm_date) -> std::string;

inline constexpr auto kAmdgpuDkmsRoot = std::string_view{"/var/lib/dkms/amdgpu"};
inline constexpr auto kRpmPath = std::string_view{"/usr/bin/rpm"};
inline constexpr auto kRpmQueryTimeout = std::chrono::milliseconds{5000};
inline constexpr auto kKmodRetryInterval = std::chrono::seconds{5};

/**
 *  @brief Select the DKMS package built for a running kernel.
 *
 *  @details Reads ``kernel-<release>-<machine>`` under @p dkms_root, then
 *  validates only that PACKAGE_VERSION directory through the dkms.conf behind
 *  its source symlink. Does not scan sibling packages.
 *
 *  @param[in] dkms_root Directory that contains kernel-* symlinks (for example
 *             /var/lib/dkms/amdgpu).
 *  @param[in] release Kernel release from uname, such as 6.8.0-124-generic.
 *  @param[in] machine Hardware name from uname, such as x86_64.
 *  @param[out] active_version Cleared on entry. Set to PACKAGE_VERSION, such as
 *              6.19.14-2370381.24.04, when the kernel symlink points at a
 *              validated package.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS when a package is found.
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED when there is none.
 *  @retval ::AMDSMI_STATUS_INVAL if @p active_version is null.
 */
auto smi_amdgpu_get_active_dkms_version(std::string_view dkms_root, std::string_view release,
                                        std::string_view machine, std::string* active_version)
    -> amdsmi_status_t;

/**
 *  @brief List the amdgpu-kmod RPM versions installed for a running kernel.
 *
 *  @details Runs ``<rpm_path> --eval '%{_dbpath}'`` and then, only when that
 *  directory holds a packages file (Packages, Packages.db or rpmdb.sqlite),
 *  ``<rpm_path> -qa --qf '%{VERSION}-%{RELEASE}\n' amdgpu-kmod-<release>``,
 *  which lists nothing, and exits 0, when that package is not installed.
 *  The first command is a check before the query, which makes rpm create a
 *  database where it finds none; it does not guarantee that rpm writes
 *  nothing. Both run without a shell, with LC_ALL=C and HOME=/ as their whole
 *  environment so that rpm reads no configuration of the caller, and without
 *  the file descriptors the caller has open.
 *
 *  @param[in] rpm_path rpm executable.
 *  @param[in] release Kernel release from uname; kmod package names embed it.
 *  @param[out] kmod_versions Cleared on entry. Set to one ``VERSION-<build>``
 *              for each installed version, where build is the leading digits
 *              of RELEASE (6.16.13.000000-2412345). Empty when the package
 *              is not installed.
 *  @param[in] timeout Limit for both commands together, up to the exit of
 *             rpm. An rpm that is still running then is left to end by
 *             itself and is not signalled or killed. While it runs, this
 *             function starts no other rpm and reports
 *             ::AMDSMI_STATUS_NOT_SUPPORTED; the first call after it has
 *             ended reaps it.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS when rpm listed the installed versions, also
 *          when it listed none.
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED when a listed version has no build,
 *          or rpm failed or gave no complete answer in time.
 *  @retval ::AMDSMI_STATUS_INVAL if @p kmod_versions is null.
 */
auto smi_amdgpu_get_kmod_versions(std::string_view rpm_path, std::string_view release,
                                  std::vector<std::string>* kmod_versions,
                                  std::chrono::milliseconds timeout = kRpmQueryTimeout)
    -> amdsmi_status_t;

/**
 *  @brief Keeps the answer of smi_amdgpu_get_kmod_versions() for one kernel
 *  release, asking rpm again only when that is worth it.
 *
 *  @details A list that rpm gave is kept until the object ends: the package
 *  of a running kernel does not change, and rpm is slow to start. That holds
 *  for an empty list too: no package installed is an answer. A failed lookup,
 *  such as an rpm that was busy, is asked again after @p retry_after.
 *  Thread safe. rpm runs once at a time and without the lock. A caller that
 *  has no answer for @p release waits for it. One that has only a stale
 *  failure gets that at once while another caller asks rpm, and asks rpm
 *  itself otherwise.
 */
class KmodVersionCache {
 public:
  explicit KmodVersionCache(std::string_view rpm_path,
                            std::chrono::milliseconds retry_after = kKmodRetryInterval);

  /** @brief The versions for @p release, empty when rpm gave none. */
  auto versions(std::string_view release) -> std::vector<std::string>;

 private:
  std::mutex mutex_;
  std::condition_variable refreshed_;
  std::string rpm_path_;
  std::chrono::milliseconds retry_after_;
  std::string release_;
  std::vector<std::string> versions_;
  std::chrono::steady_clock::time_point asked_at_{};
  bool has_answer_ = false;
  bool is_found_ = false;
  bool is_refreshing_ = false;
};

/**
 *  @brief Fill the version and package fields of @p info.
 *
 *  @details The package is the first of DKMS, kmod that has the kernel version
 *  (A.B.C) of the module, and its build is added to the full version. A package
 *  for another module is not reported, and neither is one whose build does not
 *  fit behind the module version in driver_full_version. driver_version,
 *  driver_date and driver_name are left alone.
 *
 *  @param[in] module_version /sys/module/amdgpu/version, or N/A.
 *  @param[in] dkms_version Output of smi_amdgpu_get_active_dkms_version().
 *  @param[in] kmod_version One version of smi_amdgpu_get_kmod_versions().
 *  @param[out] info Its version and package fields are overwritten.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS, or ::AMDSMI_STATUS_INVAL if @p info is null.
 */
auto smi_amdgpu_parse_driver_versions(std::string_view module_version,
                                      std::string_view dkms_version, std::string_view kmod_version,
                                      amdsmi_driver_info_v2_t* info) -> amdsmi_status_t;

/**
 *  @brief Look up the driver package of the loaded module, then fill the
 *  version and package fields of @p info.
 *
 *  @details Reads the DKMS package under @p dkms_root for the kernel in
 *  @p uts. @p kmod_lookup is called with the kernel release, and only when the
 *  module has an amdgpu version (A.B.C or A.B.C.D) that the DKMS package does
 *  not match. The kmod package is the only version it lists that has the
 *  kernel version of the module; when it lists several different ones, none
 *  is reported.
 *
 *  @param[in] module_version /sys/module/amdgpu/version, or N/A.
 *  @param[in] dkms_root Directory that contains kernel-* symlinks.
 *  @param[in] uts Kernel release and hardware name from uname.
 *  @param[in] kmod_lookup Returns the amdgpu-kmod versions for a kernel
 *             release, as smi_amdgpu_get_kmod_versions() reports them.
 *  @param[out] info Its version and package fields are overwritten.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS, or ::AMDSMI_STATUS_INVAL if @p info is null.
 */
auto smi_amdgpu_get_driver_versions(
    std::string_view module_version, std::string_view dkms_root, const utsname& uts,
    const std::function<std::vector<std::string>(std::string_view)>& kmod_lookup,
    amdsmi_driver_info_v2_t* info) -> amdsmi_status_t;

#endif  // AMD_SMI_INCLUDE_AMD_SMI_UTILS_H_
