// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/thread_trace/kfd_resource.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/platform/agent.hpp"

#include <hsa/amd_hsa_queue.h>

#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace thread_trace
{
namespace
{
constexpr size_t   KFD_PAGE_SIZE            = 4096;
constexpr size_t   KFD_HUGE_PAGE_SIZE       = 2 * 1024 * 1024;
constexpr size_t   AQL_PACKET_SIZE          = sizeof(hsa_ext_amd_aql_pm4_packet_t);
constexpr size_t   AQL_QUEUE_PACKETS        = 512;
constexpr size_t   AQL_QUEUE_SIZE           = AQL_PACKET_SIZE * AQL_QUEUE_PACKETS;
constexpr size_t   SDMA_QUEUE_MIN_SIZE      = 64 * 1024;
constexpr size_t   SDMA_MAX_COPY_SIZE       = 0x3FFFFFFF;
constexpr uint32_t SDMA_OP_COPY             = 1;
constexpr uint32_t SDMA_OP_FENCE            = 5;
constexpr uint32_t SDMA_OP_GCR              = 17;
constexpr uint32_t SDMA_SUBOP_USER_GCR      = 1;
constexpr uint32_t SDMA_MEMORY_SCOPE_SYSTEM = 3;
constexpr uint32_t KFD_QUEUE_PRIORITY_MAX   = 15;

// Upper bounds on the per-CU wave state, used only when KFD does not report the
// context-save size. They are deliberately the largest values across supported GPUs
// rather than a per-ASIC table: over-declaring only wastes memory, under-declaring
// makes KFD reject the queue.
constexpr uint64_t MAX_VGPR_BYTES_PER_CU  = 0x80000;
constexpr uint64_t MAX_SGPR_BYTES_PER_CU  = 0x8000;
constexpr uint64_t MIN_HWREG_BYTES_PER_CU = 0x1000;

static_assert(AQL_PACKET_SIZE == 64, "KFD AQL queue assumes the standard 64-byte packet size");

template <typename PredicateT, typename DiagnosticT>
void
wait_with_diagnostics(PredicateT&& pending, DiagnosticT&& diagnostic)
{
    constexpr auto interval        = std::chrono::seconds{30};
    auto           next_diagnostic = std::chrono::steady_clock::now() + interval;
    while(pending())
    {
        const auto now = std::chrono::steady_clock::now();
        if(now >= next_diagnostic)
        {
            diagnostic();
            next_diagnostic = now + interval;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        std::this_thread::yield();
    }
}

size_t
align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

size_t
next_power_of_two(size_t value)
{
    size_t result = 1;
    while(result < value)
    {
        if(result > std::numeric_limits<size_t>::max() / 2)
            throw std::overflow_error{"KFD queue size overflow"};
        result <<= 1;
    }
    return result;
}

int
kfd_ioctl(int fd, unsigned long request, void* args)
{
    // TODO: Deduplicate KFD descriptor and ioctl handling with the SDK's other KFD users.
    int result = 0;
    do
    {
        result = ::ioctl(fd, request, args);
    } while(result == -1 && (errno == EINTR || errno == EAGAIN));
    return result;
}

void
check_ioctl(int result, const char* operation)
{
    if(result == 0) return;
    const auto error = errno;
    throw std::runtime_error{std::string{operation} + " failed: " + std::strerror(error)};
}

int
duplicate_open_device(const std::string& path)
{
    struct stat target_stat
    {};
    if(::stat(path.c_str(), &target_stat) != 0) return -1;

    auto* directory = ::opendir("/proc/self/fd");
    if(directory == nullptr) return -1;

    int duplicate = -1;
    while(auto* entry = ::readdir(directory))
    {
        char* end       = nullptr;
        long  candidate = std::strtol(entry->d_name, &end, 10);
        if(end == entry->d_name || *end != '\0' || candidate < 0) continue;

        struct stat candidate_stat
        {};
        if(::fstat(static_cast<int>(candidate), &candidate_stat) != 0) continue;
        if(candidate_stat.st_rdev != target_stat.st_rdev ||
           (candidate_stat.st_mode & S_IFMT) != (target_stat.st_mode & S_IFMT))
            continue;

        duplicate = ::fcntl(static_cast<int>(candidate), F_DUPFD_CLOEXEC, 3);
        if(duplicate >= 0) break;
    }

    ::closedir(directory);
    return duplicate;
}

void*
reserve_aligned(size_t size, size_t alignment)
{
    alignment          = std::max(alignment, KFD_PAGE_SIZE);
    auto  reserve_size = size + alignment;
    void* mapping      = ::mmap(
        nullptr, reserve_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if(mapping == MAP_FAILED) return nullptr;

    auto begin   = reinterpret_cast<uintptr_t>(mapping);
    auto aligned = align_up(begin, alignment);
    auto end     = aligned + size;
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    if(aligned > begin) ::munmap(reinterpret_cast<void*>(begin), aligned - begin);
    if(end < begin + reserve_size)
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        ::munmap(reinterpret_cast<void*>(end), begin + reserve_size - end);
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(aligned);
}

uint32_t
gfx_major(uint32_t gfx_target_version)
{
    return (gfx_target_version / 10000) % 100;
}

uint32_t
gfx_minor(uint32_t gfx_target_version)
{
    return (gfx_target_version / 100) % 100;
}

uint32_t
gfx_stepping(uint32_t gfx_target_version)
{
    return gfx_target_version % 100;
}

bool
sdma_extended_copy_supported(uint32_t gfx_target_version)
{
    const auto major    = gfx_major(gfx_target_version);
    const auto minor    = gfx_minor(gfx_target_version);
    const auto stepping = gfx_stepping(gfx_target_version);

    return major != 9 || minor >= 4 || (minor == 0 && stepping == 10);
}

uint32_t
low32(const void* ptr)
{
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
}

uint32_t
high32(const void* ptr)
{
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr) >> 32);
}

uint64_t
load_acquire(const volatile uint64_t* ptr)
{
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

void
store_release(volatile uint64_t* ptr, uint64_t value)  // NOLINT(readability-non-const-parameter)
{
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

int64_t
load_signal_value(const amd_signal_t* signal)
{
    return __atomic_load_n(&signal->value, __ATOMIC_ACQUIRE);
}

void
store_signal_value(amd_signal_t* signal, int64_t value)
{
    __atomic_store_n(&signal->value, value, __ATOMIC_RELEASE);
}

class kfd_runtime_t
{
public:
    kfd_runtime_t()
    {
        _kfd_fd = ::open("/dev/kfd", O_RDWR | O_CLOEXEC);
        if(_kfd_fd < 0) throw std::runtime_error{"could not open /dev/kfd"};
    }

    ~kfd_runtime_t()
    {
        _render_fds.wlock([](auto& render_fds) {
            for(const auto& [_, fd] : render_fds)
                if(fd >= 0) ::close(fd);
            render_fds.clear();
        });
        if(_kfd_fd >= 0) ::close(_kfd_fd);
    }

    int kfd_fd() const { return _kfd_fd; }

    int render_fd(uint32_t render_minor)
    {
        return _render_fds.wlock([&](auto& render_fds) {
            auto itr = render_fds.find(render_minor);
            if(itr != render_fds.end()) return itr->second;

            const auto path = std::string{"/dev/dri/renderD"} + std::to_string(render_minor);
            int        fd   = duplicate_open_device(path);
            if(fd < 0)
                throw std::runtime_error{"could not duplicate ROCr's " + path + " descriptor"};
            return render_fds.emplace(render_minor, fd).first->second;
        });
    }

private:
    int                                                     _kfd_fd{-1};
    common::Synchronized<std::unordered_map<uint32_t, int>> _render_fds{};
};

std::shared_ptr<kfd_runtime_t>
get_kfd_runtime()
{
    static std::mutex mutex{};
    static auto*      weak   = new std::weak_ptr<kfd_runtime_t>{};
    auto              lock   = std::unique_lock{mutex};
    auto              result = weak->lock();
    if(!result)
    {
        result = std::make_shared<kfd_runtime_t>();
        *weak  = result;
    }
    return result;
}

struct sdma_builder_t
{
    explicit sdma_builder_t(std::vector<uint32_t>& storage)
    : words{storage}
    {}

    void append(uint32_t value)
    {
        if(offset >= words.size()) throw std::runtime_error{"KFD SDMA command buffer overflow"};
        words[offset++] = value;
    }

    void append_copy(void* dst, const void* src, size_t size, bool scope_fields)
    {
        if(size == 0 || size > SDMA_MAX_COPY_SIZE)
            throw std::runtime_error{"invalid KFD SDMA linear-copy size"};
        append(SDMA_OP_COPY | (scope_fields ? (1u << 28) : 0u));
        append(static_cast<uint32_t>(size - 1));
        append(scope_fields ? ((SDMA_MEMORY_SCOPE_SYSTEM << 18) | (SDMA_MEMORY_SCOPE_SYSTEM << 26))
                            : 0u);
        append(low32(src));
        append(high32(src));
        append(low32(dst));
        append(high32(dst));
    }

    void append_fence(void* address, uint32_t value, uint32_t major, bool scope_fields)
    {
        uint32_t header = SDMA_OP_FENCE;
        if(major >= 12)
        {
            header |= (3u << 16);  // uncached MTYPE
            header |= (1u << 20);  // system memory
            if(scope_fields) header |= (SDMA_MEMORY_SCOPE_SYSTEM << 24);
        }
        else if(major >= 10)
        {
            header |= (3u << 16);  // uncached MTYPE
        }
        append(header);
        append(low32(address));
        append(high32(address));
        append(value);
    }

    void append_gcr(bool invalidate)
    {
        append(SDMA_OP_GCR | (SDMA_SUBOP_USER_GCR << 8));
        append(0);
        uint32_t control = (1u << 31) | (1u << 22);  // GL2 + GLK writeback
        if(invalidate) control |= (1u << 30) | (1u << 25) | (1u << 24) | (1u << 23);
        append(control);
        append(0);
        append(0);
    }

    size_t      size_bytes() const { return offset * sizeof(uint32_t); }
    const void* data() const { return words.data(); }

    std::vector<uint32_t>& words;
    size_t                 offset{0};
};

}  // namespace

struct kfd_memory_pool_t::impl
{
    struct allocation_t
    {
        size_t            size{0};
        uint64_t          handle{0};
        kfd_memory_kind_t kind{kfd_memory_kind_t::host};
    };

    explicit impl(const rocprofiler_agent_t& agent)
    : runtime{get_kfd_runtime()}
    , gpu_id{static_cast<uint32_t>(agent.gpu_id)}
    , gfx_target_version{agent.gfx_target_version}
    , num_xcc{std::max(agent.num_xcc, 1u)}
    {
        if(gpu_id == 0) throw std::runtime_error{"KFD thread trace requires a GPU agent"};

        render_fd = runtime->render_fd(agent.drm_render_minor);

        // KFD sizes the context-save buffer as (cwsr_size + debug area) * num_xcc and
        // rejects any other size, so all three values below have to agree with the driver.
        const auto major = gfx_major(gfx_target_version);
        const auto cus   = (agent.simd_per_cu == 0) ? 0 : agent.simd_count / agent.simd_per_cu;

        uint64_t waves = 0;
        if(major >= 10)
            waves = (agent.simd_count / num_xcc) * agent.max_waves_per_simd;
        else if(agent.simd_arrays_per_engine != 0)
            waves =
                std::min<uint64_t>((cus / num_xcc) * 40,
                                   (agent.num_shader_banks / agent.simd_arrays_per_engine) * 512);

        if(waves == 0 || cus == 0)
            throw std::runtime_error{"KFD topology does not describe the GPU's wave capacity"};

        max_cu_id   = cus - 1;
        max_wave_id = (agent.max_waves_per_simd * agent.simd_per_cu) - 1;

        debug_memory_size = static_cast<uint32_t>(align_up(waves * 32, 64));

        const auto* info = rocprofiler::agent::get_agent_info(agent.id);

        // KFD only reports the context-save geometry from ABI 1.20 on. Both values are
        // derived from the topology when it does not, which costs memory for the wave
        // state but keeps the driver's layout requirements satisfied on older kernels.
        ctl_stack_size =
            (info != nullptr && info->ctl_stack_size != 0)
                ? info->ctl_stack_size
                : static_cast<uint32_t>(align_up(
                      sizeof(kfd_context_save_area_header) + (waves * ((major >= 10) ? 12 : 8)) + 8,
                      KFD_PAGE_SIZE));

        // The driver only requires the declared area to be large enough, so bound the
        // per-CU wave state instead of tracking each ASIC's register file sizes.
        const auto wave_state_bound =
            (cus / num_xcc) * (MAX_VGPR_BYTES_PER_CU + MAX_SGPR_BYTES_PER_CU +
                               (static_cast<uint64_t>(agent.lds_size_in_kb) << 10) +
                               std::max<uint64_t>(MIN_HWREG_BYTES_PER_CU,
                                                  static_cast<uint64_t>(agent.max_waves_per_simd) *
                                                      agent.simd_per_cu * 512));

        cwsr_size =
            (info != nullptr && info->cwsr_size != 0)
                ? info->cwsr_size
                : static_cast<uint32_t>(ctl_stack_size + align_up(wave_state_bound, KFD_PAGE_SIZE));

        if(info == nullptr || info->cwsr_size == 0 || info->ctl_stack_size == 0)
        {
            static auto once = std::once_flag{};
            std::call_once(once, [&]() {
                ROCP_WARNING << "KFD does not report the queue context-save geometry (added in "
                                "KFD ABI 1.20); deriving it instead, which reserves "
                             << ((static_cast<size_t>(cwsr_size) + debug_memory_size) * num_xcc) /
                                    (1024 * 1024)
                             << " MiB per thread-trace queue";
            });
        }
    }

    void release(void* ptr, allocation_t allocation)
    {
        auto gpu                  = gpu_id;
        auto args                 = kfd_ioctl_unmap_memory_from_gpu_args{};
        args.handle               = allocation.handle;
        args.device_ids_array_ptr = reinterpret_cast<uintptr_t>(&gpu);
        args.n_devices            = 1;
        const auto unmap_result =
            kfd_ioctl(runtime->kfd_fd(), AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &args);
        ROCP_WARNING_IF(unmap_result != 0)
            << "AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU failed: " << std::strerror(errno);

        auto       free_args = kfd_ioctl_free_memory_of_gpu_args{allocation.handle};
        const auto result = kfd_ioctl(runtime->kfd_fd(), AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args);
        ROCP_WARNING_IF(result != 0)
            << "AMDKFD_IOC_FREE_MEMORY_OF_GPU failed: " << std::strerror(errno);
        ::munmap(ptr, allocation.size);
    }

    std::shared_ptr<kfd_runtime_t>                                runtime{};
    uint32_t                                                      gpu_id{0};
    uint32_t                                                      gfx_target_version{0};
    uint32_t                                                      num_xcc{1};
    uint32_t                                                      cwsr_size{0};
    uint32_t                                                      ctl_stack_size{0};
    uint32_t                                                      debug_memory_size{0};
    uint32_t                                                      max_cu_id{0};
    uint32_t                                                      max_wave_id{0};
    int                                                           render_fd{-1};
    common::Synchronized<std::unordered_map<void*, allocation_t>> allocations{};
};

kfd_memory_pool_t::kfd_memory_pool_t(const rocprofiler_agent_t& agent)
: _impl{std::make_unique<impl>(agent)}
{}

kfd_memory_pool_t::~kfd_memory_pool_t()
{
    ROCP_INFO << "Deleting kfd pool";

    auto allocations =
        _impl->allocations.wlock([](auto& entries) { return std::exchange(entries, {}); });
    for(const auto& [ptr, allocation] : allocations)
        _impl->release(ptr, allocation);
}

void*
kfd_memory_pool_t::allocate(size_t size, kfd_memory_kind_t kind, size_t alignment)
{
    if(size == 0) return nullptr;
    const auto allocation_size = align_up(size, KFD_PAGE_SIZE);
    void*      ptr             = reserve_aligned(allocation_size, alignment);
    if(ptr == nullptr) throw std::bad_alloc{};

    auto args    = kfd_ioctl_alloc_memory_of_gpu_args{};
    args.va_addr = reinterpret_cast<uintptr_t>(ptr);
    args.size    = allocation_size;
    args.gpu_id  = _impl->gpu_id;
    args.flags   = KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;
    args.flags |= (kind == kfd_memory_kind_t::device)
                      ? (KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)
                      : (KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT);

    if(kfd_ioctl(_impl->runtime->kfd_fd(), AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args) != 0)
    {
        const auto error = errno;
        ::munmap(ptr, allocation_size);
        throw std::runtime_error{std::string{"AMDKFD_IOC_ALLOC_MEMORY_OF_GPU failed: "} +
                                 std::strerror(error)};
    }

    if(kind == kfd_memory_kind_t::host)
    {
        auto* mapped = ::mmap(ptr,
                              allocation_size,
                              PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_FIXED,
                              _impl->render_fd,
                              args.mmap_offset);
        if(mapped == MAP_FAILED)
        {
            const auto error     = errno;
            auto       free_args = kfd_ioctl_free_memory_of_gpu_args{args.handle};
            kfd_ioctl(_impl->runtime->kfd_fd(), AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args);
            ::munmap(ptr, allocation_size);
            throw std::runtime_error{std::string{"mmap of KFD GTT allocation failed: "} +
                                     std::strerror(error)};
        }
        std::memset(ptr, 0, allocation_size);
        ::madvise(ptr, allocation_size, MADV_DONTFORK);
    }

    auto gpu                      = _impl->gpu_id;
    auto map_args                 = kfd_ioctl_map_memory_to_gpu_args{};
    map_args.handle               = args.handle;
    map_args.device_ids_array_ptr = reinterpret_cast<uintptr_t>(&gpu);
    map_args.n_devices            = 1;
    if(kfd_ioctl(_impl->runtime->kfd_fd(), AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map_args) != 0 ||
       map_args.n_success != 1)
    {
        const auto error     = errno;
        auto       free_args = kfd_ioctl_free_memory_of_gpu_args{args.handle};
        kfd_ioctl(_impl->runtime->kfd_fd(), AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args);
        ::munmap(ptr, allocation_size);
        throw std::runtime_error{std::string{"AMDKFD_IOC_MAP_MEMORY_TO_GPU failed: "} +
                                 std::strerror(error)};
    }

    _impl->allocations.wlock([&](auto& allocations) {
        allocations.emplace(ptr, impl::allocation_t{allocation_size, args.handle, kind});
    });
    return ptr;
}

void
kfd_memory_pool_t::deallocate(void* ptr)
{
    if(ptr == nullptr) return;

    auto allocation = _impl->allocations.wlock([&](auto& entries) { return entries.extract(ptr); });
    if(allocation.empty())
    {
        ROCP_WARNING << "Ignoring unknown KFD allocation " << ptr;
        return;
    }
    _impl->release(ptr, allocation.mapped());
}

bool
kfd_memory_pool_t::is_device_pointer(const void* ptr) const
{
    if(ptr == nullptr) return false;

    const auto address = reinterpret_cast<uintptr_t>(ptr);
    return _impl->allocations.rlock([&](const auto& allocations) {
        for(const auto& [base_ptr, allocation] : allocations)
        {
            const auto base = reinterpret_cast<uintptr_t>(base_ptr);
            if(address >= base && address - base < allocation.size)
                return allocation.kind == kfd_memory_kind_t::device;
        }
        return false;
    });
}

uint32_t
kfd_memory_pool_t::gpu_id() const
{
    return _impl->gpu_id;
}
uint32_t
kfd_memory_pool_t::gfx_target_version() const
{
    return _impl->gfx_target_version;
}
uint32_t
kfd_memory_pool_t::cwsr_size() const
{
    return _impl->cwsr_size;
}
uint32_t
kfd_memory_pool_t::ctl_stack_size() const
{
    return _impl->ctl_stack_size;
}
uint32_t
kfd_memory_pool_t::num_xcc() const
{
    return _impl->num_xcc;
}
uint32_t
kfd_memory_pool_t::debug_memory_size() const
{
    return _impl->debug_memory_size;
}
uint32_t
kfd_memory_pool_t::max_cu_id() const
{
    return _impl->max_cu_id;
}
uint32_t
kfd_memory_pool_t::max_wave_id() const
{
    return _impl->max_wave_id;
}
int
kfd_memory_pool_t::kfd_fd() const
{
    return _impl->runtime->kfd_fd();
}

kfd_signal_t::kfd_signal_t(std::shared_ptr<kfd_memory_pool_t> memory)
: _memory{std::move(memory)}
{
    if(!_memory) throw std::invalid_argument{"KFD signal requires a memory pool"};
    _signal = static_cast<amd_signal_t*>(
        _memory->allocate(sizeof(amd_signal_t), kfd_memory_kind_t::host));

    // The value is the only part required by AQL and SDMA completion. Keep the
    // event fields empty: KFD owns one process-wide event page, which ROCr may
    // tear down before rocprofiler's global finalizer runs.
    _signal->kind              = AMD_SIGNAL_KIND_USER;
    _signal->event_id          = 0;
    _signal->event_mailbox_ptr = 0;
    store_signal_value(_signal, 0);
}

kfd_signal_t::~kfd_signal_t()
{
    wait();
    ROCP_INFO << "Deleting signal";
    _memory->deallocate(_signal);
}

hsa_signal_t
kfd_signal_t::handle() const
{
    return hsa_signal_t{.handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(_signal))};
}

void
kfd_signal_t::reset()
{
    store_signal_value(_signal, 1);
}

void
kfd_signal_t::wait() const
{
    ROCP_TRACE << "Waiting for KFD signal";
    auto t0 = std::chrono::system_clock::now();
    wait_with_diagnostics(
        [&]() { return load_signal_value(_signal) != 0; },
        [&]() {
            ROCP_INFO << "Still waiting in kfd_signal_t::wait (30-second diagnostic): gpu_id="
                      << _memory->gpu_id() << ", signal=" << _signal
                      << ", value=" << load_signal_value(_signal);
        });

    std::atomic_thread_fence(std::memory_order_acq_rel);
    auto t1 = std::chrono::system_clock::now();
    ROCP_TRACE << "KFD signal pass in " << (t1 - t0).count() / 1000.0f;
}

namespace
{
amd_queue_t*
init_aql_descriptor(void* storage, void* ring, size_t ring_size, const kfd_memory_pool_t& memory)
{
    const auto packets = ring_size / AQL_PACKET_SIZE;
    for(size_t i = 0; i < packets; ++i)
        static_cast<hsa_ext_amd_aql_pm4_packet_t*>(ring)[i].header = HSA_PACKET_TYPE_INVALID
                                                                     << HSA_PACKET_HEADER_TYPE;

    auto* descriptor                                    = static_cast<amd_queue_t*>(storage);
    descriptor->hsa_queue.type                          = HSA_QUEUE_TYPE_MULTI;
    descriptor->hsa_queue.features                      = HSA_QUEUE_FEATURE_KERNEL_DISPATCH;
    descriptor->hsa_queue.base_address                  = ring;
    descriptor->hsa_queue.size                          = static_cast<uint32_t>(packets);
    descriptor->hsa_queue.id                            = std::numeric_limits<uint64_t>::max();
    descriptor->read_dispatch_id_field_base_byte_offset = offsetof(amd_queue_t, read_dispatch_id);
    descriptor->max_cu_id                               = memory.max_cu_id();
    descriptor->max_wave_id                             = memory.max_wave_id();
    AMD_HSA_BITS_SET(descriptor->queue_properties, AMD_QUEUE_PROPERTIES_IS_PTR64, 1);
    return descriptor;
}

struct direct_queue_t
{
    direct_queue_t(std::shared_ptr<kfd_memory_pool_t> _memory,
                   uint32_t                           queue_type,
                   size_t                             requested_ring_size)
    : memory{std::move(_memory)}
    {
        try
        {
            ring_size = next_power_of_two(std::max(requested_ring_size, KFD_PAGE_SIZE));
            ring      = memory->allocate(ring_size, kfd_memory_kind_t::host);
            pointers  = memory->allocate(KFD_PAGE_SIZE, kfd_memory_kind_t::host);

            if(queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE_AQL)
            {
                auto* descriptor = init_aql_descriptor(pointers, ring, ring_size, *memory);
                rptr             = &descriptor->read_dispatch_id;
                wptr             = &descriptor->write_dispatch_id;
            }
            else
            {
                rptr = &static_cast<uint64_t*>(pointers)[0];
                wptr = &static_cast<uint64_t*>(pointers)[1];
            }

            auto args                  = kfd_ioctl_create_queue_args{};
            args.ring_base_address     = reinterpret_cast<uintptr_t>(ring);
            args.write_pointer_address = reinterpret_cast<uintptr_t>(wptr);
            args.read_pointer_address  = reinterpret_cast<uintptr_t>(rptr);
            args.ring_size             = static_cast<uint32_t>(ring_size);
            args.gpu_id                = memory->gpu_id();
            args.queue_type            = queue_type;
            args.queue_percentage      = 100;
            args.queue_priority        = KFD_QUEUE_PRIORITY_MAX;

            if(queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE_AQL)
            {
                // KFD does not use an EOP buffer for AQL queues on gfx94x.
                if(gfx_major(memory->gfx_target_version()) != 9 ||
                   gfx_minor(memory->gfx_target_version()) != 4)
                {
                    eop = memory->allocate(KFD_PAGE_SIZE, kfd_memory_kind_t::device);
                    args.eop_buffer_address = reinterpret_cast<uintptr_t>(eop);
                    args.eop_buffer_size    = KFD_PAGE_SIZE;
                }

                // KFD requires this buffer to be exactly this size, no more, no less.
                const auto total_cwsr = align_up(
                    (static_cast<size_t>(memory->cwsr_size()) + memory->debug_memory_size()) *
                        memory->num_xcc(),
                    KFD_PAGE_SIZE);
                cwsr = memory->allocate(total_cwsr, kfd_memory_kind_t::host, KFD_HUGE_PAGE_SIZE);

                for(uint32_t i = 0; i < memory->num_xcc(); ++i)
                {
                    auto* header = reinterpret_cast<kfd_context_save_area_header*>(
                        static_cast<char*>(cwsr) + (i * memory->cwsr_size()));
                    header->debug_offset = (memory->num_xcc() - i) * memory->cwsr_size();
                    header->debug_size   = memory->debug_memory_size() * memory->num_xcc();
                }

                args.ctx_save_restore_address = reinterpret_cast<uintptr_t>(cwsr);
                args.ctx_save_restore_size    = memory->cwsr_size();
                args.ctl_stack_size           = memory->ctl_stack_size();
            }

            check_ioctl(kfd_ioctl(memory->kfd_fd(), AMDKFD_IOC_CREATE_QUEUE, &args),
                        "AMDKFD_IOC_CREATE_QUEUE");
            queue_id = args.queue_id;
            created  = true;

            doorbell_size = (memory->gfx_target_version() >= 90000) ? 8192 : KFD_PAGE_SIZE;
            const auto doorbell_base  = args.doorbell_offset & ~(doorbell_size - 1);
            const auto doorbell_index = args.doorbell_offset & (doorbell_size - 1);
            doorbell_mapping          = ::mmap(nullptr,
                                      doorbell_size,
                                      PROT_READ | PROT_WRITE,
                                      MAP_SHARED,
                                      memory->kfd_fd(),
                                      doorbell_base);
            if(doorbell_mapping == MAP_FAILED)
            {
                doorbell_mapping = nullptr;
                throw std::runtime_error{std::string{"KFD doorbell mmap failed: "} +
                                         std::strerror(errno)};
            }
            doorbell = reinterpret_cast<volatile uint64_t*>(static_cast<char*>(doorbell_mapping) +
                                                            doorbell_index);
        } catch(...)
        {
            destroy();
            throw;
        }
    }

    ~direct_queue_t() { destroy(); }

    void destroy()
    {
        if(created)
        {
            auto args   = kfd_ioctl_destroy_queue_args{queue_id, 0};
            auto result = kfd_ioctl(memory->kfd_fd(), AMDKFD_IOC_DESTROY_QUEUE, &args);
            ROCP_WARNING_IF(result != 0)
                << "AMDKFD_IOC_DESTROY_QUEUE failed: " << std::strerror(errno);
            created = false;
        }
        if(doorbell_mapping)
        {
            ::munmap(doorbell_mapping, doorbell_size);
            doorbell_mapping = nullptr;
        }
        memory->deallocate(eop);
        eop = nullptr;
        memory->deallocate(cwsr);
        cwsr = nullptr;
        memory->deallocate(pointers);
        pointers = nullptr;
        memory->deallocate(ring);
        ring = nullptr;
    }

    void wait_for_space(uint64_t end_index) const
    {
        wait_with_diagnostics(
            [&]() { return end_index - load_acquire(rptr) >= ring_size; },
            [&]() {
                ROCP_INFO
                    << "Still waiting in direct_queue_t::wait_for_space (30-second diagnostic): "
                    << "gpu_id=" << memory->gpu_id() << ", queue_id=" << queue_id
                    << ", rptr=" << load_acquire(rptr) << ", wptr=" << load_acquire(wptr)
                    << ", end_index=" << end_index << ", ring_size=" << ring_size;
            });
    }

    std::shared_ptr<kfd_memory_pool_t> memory{};
    void*                              ring{nullptr};
    size_t                             ring_size{0};
    void*                              pointers{nullptr};
    volatile uint64_t*                 rptr{nullptr};
    volatile uint64_t*                 wptr{nullptr};
    void*                              eop{nullptr};
    void*                              cwsr{nullptr};
    void*                              doorbell_mapping{nullptr};
    size_t                             doorbell_size{0};
    volatile uint64_t*                 doorbell{nullptr};
    uint32_t                           queue_id{0};
    bool                               created{false};
};
}  // namespace

struct kfd_aql_queue_t::impl
{
    explicit impl(std::shared_ptr<kfd_memory_pool_t> memory)
    : queue{std::move(memory), KFD_IOC_QUEUE_TYPE_COMPUTE_AQL, AQL_QUEUE_SIZE}
    {
        write_index = load_acquire(queue.wptr);
    }

    ~impl() { ROCP_INFO << "Deleting queue impl"; }

    void submit(const hsa_ext_amd_aql_pm4_packet_t& packet, hsa_signal_t completion)
    {
        while((write_index + 1) - load_acquire(queue.rptr) >= AQL_QUEUE_PACKETS)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            std::this_thread::yield();
        }

        const auto  slot      = write_index & (AQL_QUEUE_PACKETS - 1);
        auto*       dst       = static_cast<hsa_ext_amd_aql_pm4_packet_t*>(queue.ring) + slot;
        const auto* src_words = reinterpret_cast<const uint32_t*>(&packet);
        auto*       dst_words = reinterpret_cast<uint32_t*>(dst);

        std::memcpy(&dst_words[1], &src_words[1], AQL_PACKET_SIZE - sizeof(uint32_t));
        dst->completion_signal = completion;
        reinterpret_cast<std::atomic<uint32_t>*>(dst_words)->store(src_words[0],
                                                                   std::memory_order_release);

        store_release(queue.wptr, write_index + 1);
        std::atomic_thread_fence(std::memory_order_release);
        *queue.doorbell = write_index;
        ++write_index;
    }

    direct_queue_t queue;
    uint64_t       write_index{0};
    std::mutex     mutex{};
};

kfd_aql_queue_t::kfd_aql_queue_t(std::shared_ptr<kfd_memory_pool_t> memory)
: _impl{std::make_unique<impl>(std::move(memory))}
{}

kfd_aql_queue_t::~kfd_aql_queue_t() { ROCP_INFO << "Deleting queue"; }

void
kfd_aql_queue_t::submit(const hsa_ext_amd_aql_pm4_packet_t& packet, hsa_signal_t completion)
{
    auto lock = std::unique_lock{_impl->mutex};
    _impl->submit(packet, completion);
}

namespace
{
class sdma_queue_t
{
public:
    sdma_queue_t(const std::shared_ptr<kfd_memory_pool_t>& memory, size_t max_copy_size);
    ~sdma_queue_t();

    sdma_queue_t(const sdma_queue_t&) = delete;
    sdma_queue_t& operator=(const sdma_queue_t&) = delete;

    void copy(void* dst, const void* src, size_t size, kfd_signal_t& completion);

private:
    struct impl;
    std::unique_ptr<impl> _impl;
};

struct sdma_queue_t::impl
{
    impl(const std::shared_ptr<kfd_memory_pool_t>& memory, size_t max_copy_size)
    : queue{memory,
            KFD_IOC_QUEUE_TYPE_SDMA,
            next_power_of_two(std::max(
                SDMA_QUEUE_MIN_SIZE,
                ((((max_copy_size + SDMA_MAX_COPY_SIZE - 1) / SDMA_MAX_COPY_SIZE) * 7 + 20) *
                 sizeof(uint32_t)) +
                    KFD_PAGE_SIZE))}
    , major{gfx_major(memory->gfx_target_version())}
    , minor{gfx_minor(memory->gfx_target_version())}
    , use_gcr{major >= 10 && !(major == 12 && minor >= 5)}
    , scope_fields{major == 12 && minor >= 5}
    {
        const size_t max_copy_packets =
            std::max<size_t>(1, (max_copy_size + SDMA_MAX_COPY_SIZE - 1) / SDMA_MAX_COPY_SIZE);
        commands.resize(max_copy_packets * 7 + (use_gcr ? 10 : 0) + 10);
        write_index = load_acquire(queue.wptr);
    }

    void publish(const void* data, size_t bytes)
    {
        if(bytes == 0 || bytes >= queue.ring_size)
            throw std::runtime_error{"invalid KFD SDMA submission size"};

        size_t offset = write_index & (queue.ring_size - 1);
        if(offset + bytes > queue.ring_size)
        {
            const size_t padding = queue.ring_size - offset;
            queue.wait_for_space(write_index + padding);
            std::memset(static_cast<char*>(queue.ring) + offset, 0, padding);
            write_index += padding;
            store_release(queue.wptr, write_index);
            std::atomic_thread_fence(std::memory_order_release);
            *queue.doorbell = write_index;
            offset          = 0;
        }

        queue.wait_for_space(write_index + bytes);
        std::memcpy(static_cast<char*>(queue.ring) + offset, data, bytes);
        write_index += bytes;
        store_release(queue.wptr, write_index);
        std::atomic_thread_fence(std::memory_order_release);
        *queue.doorbell = write_index;
    }

    direct_queue_t        queue;
    uint64_t              write_index{0};
    uint32_t              major{0};
    uint32_t              minor{0};
    bool                  use_gcr{false};
    bool                  scope_fields{false};
    std::vector<uint32_t> commands{};
    std::mutex            mutex{};
};

sdma_queue_t::sdma_queue_t(const std::shared_ptr<kfd_memory_pool_t>& memory, size_t max_copy_size)
: _impl{std::make_unique<impl>(memory, max_copy_size)}
{}

sdma_queue_t::~sdma_queue_t() { ROCP_INFO << "Deleting SDMA"; };

void
sdma_queue_t::copy(void* dst, const void* src, size_t size, kfd_signal_t& completion)
{
    if(size == 0) return;
    auto lock    = std::unique_lock{_impl->mutex};
    auto builder = sdma_builder_t{_impl->commands};

    if(_impl->use_gcr) builder.append_gcr(true);

    size_t copied = 0;
    while(copied < size)
    {
        const size_t chunk = std::min(size - copied, SDMA_MAX_COPY_SIZE);
        builder.append_copy(static_cast<char*>(dst) + copied,
                            static_cast<const char*>(src) + copied,
                            chunk,
                            _impl->scope_fields);
        copied += chunk;
    }

    if(_impl->use_gcr) builder.append_gcr(false);
    builder.append_fence(
        const_cast<int64_t*>(&completion.abi()->value), 0, _impl->major, _impl->scope_fields);

    completion.reset();
    _impl->publish(builder.data(), builder.size_bytes());
    completion.wait();
}
}  // namespace

struct kfd_copy_queue_t::impl
{
    impl(const std::shared_ptr<kfd_memory_pool_t>& _memory, size_t max_copy_size)
    : aql_queue{std::make_shared<kfd_aql_queue_t>(_memory)}
    , completion{_memory}
    {
        if(!sdma_extended_copy_supported(_memory->gfx_target_version()))
            throw std::runtime_error{"extended SDMA copy packets are unavailable"};
        sdma_queue = std::make_unique<sdma_queue_t>(_memory, max_copy_size);
    }

    std::shared_ptr<kfd_aql_queue_t> aql_queue{};
    std::unique_ptr<sdma_queue_t>    sdma_queue{};
    kfd_signal_t                     completion;
};

kfd_copy_queue_t::kfd_copy_queue_t(const std::shared_ptr<kfd_memory_pool_t>& memory,
                                   size_t                                    max_copy_size)
: _impl{std::make_unique<impl>(memory, max_copy_size)}
{}

kfd_copy_queue_t::~kfd_copy_queue_t() { ROCP_INFO << "Destroy copy queue"; }

void
kfd_copy_queue_t::submit(const hsa_ext_amd_aql_pm4_packet_t& packet, hsa_signal_t completion)
{
    _impl->aql_queue->submit(packet, completion);
}

void
kfd_copy_queue_t::copy(void* dst, const void* src, size_t size)
{
    _impl->sdma_queue->copy(dst, src, size, _impl->completion);
}

}  // namespace thread_trace
}  // namespace rocprofiler
