//
//
//

#include "lib/aqlprofile/hsa_includes.h"
#include "lib/aqlprofile/aqlprofile.hpp"
#include "lib/aqlprofile/core/spm_common.hpp"
#include "lib/aqlprofile/core/memorymanager.hpp"
#include "lib/aqlprofile/core/commandbuffermgr.hpp"

#include "lib/aqlprofile/core/logger.hpp"
#include "lib/aqlprofile/core/pm4_factory.h"
#include "lib/common/static_object.hpp"
#include "lib/common/environment.hpp"

#include <thread>
#include <condition_variable>
#include <map>
#include <array>
#include <shared_mutex>
#include <filesystem>

// On Windows the .def file controls symbol export; no declspec needed.
// On Linux use visibility("default") to mark public API symbols.
#ifdef _WIN32
#    define PUBLIC_API
#else
#    define PUBLIC_API __attribute__((visibility("default")))
#endif

static void
producer(std::shared_ptr<class spm_state_t> s);
static void
consumer(std::shared_ptr<class spm_state_t> s,
         aqlprofile_spm_data_callback_t     callback,
         void*                              userdata);

#define CHECKHSA(x, action)                                                                        \
    {                                                                                              \
        auto _status = (x);                                                                        \
        if(_status != HSA_STATUS_SUCCESS)                                                          \
        {                                                                                          \
            std::cerr << __FILE__ << ':' << __LINE__ << " error:" << _status << std::endl;         \
            action;                                                                                \
        }                                                                                          \
    }

struct spm_set_dest_buffer_args
{
    hsa_agent_t hsa_agent{0};
    size_t      buf_size{0};
    uint32_t    timeout{0};
    uint32_t    size_copied{0};
    void*       dest_buf{nullptr};
    bool        is_data_loss{false};
};

struct spm_state_t : public spm_set_dest_buffer_args
{
    static constexpr size_t NUM_SLOTS = 6;

    struct slot_t
    {
        void*  buf         = nullptr;
        size_t size_copied = 0;
        bool   data_loss   = false;
    };

    aqlprofile_agent_handle_t aql_agent{};
    std::thread*              manager_thread{nullptr};
    std::mutex                work_mutex{};
    std::condition_variable   work_cond{};

    std::atomic<int>   signal_data_loss{};
    std::atomic<bool>  stop_prod_thread{};
    std::atomic<bool>  stop_cons_thread{};
    slot_t             slots[NUM_SLOTS];
    std::atomic<size_t> ring_head{0};
    std::atomic<size_t> ring_tail{0};
    uint32_t           num_xcc{0};
    size_t             buf_size_xcc{0};

    void*                                                  output_buffer_ptr{nullptr};
    size_t                                                 output_buffer_size{0};
    std::unique_ptr<SPMMemoryManager>                      memory{nullptr};
    std::array<size_t, AQLPROFILE_SPM_PARAMETER_TYPE_LAST> parameters;

    bool ring_empty() const
    {
        return ring_head.load(std::memory_order_acquire) ==
               ring_tail.load(std::memory_order_relaxed);
    }
};

inline static hsa_status_t
HsaSpmSetDestBuffer(spm_set_dest_buffer_args& args)
{
    if(args.hsa_agent.handle == 0) throw std::runtime_error("Invalid hsa agent");
    return rocprofiler::aqlprofile::get_amd_ext_table()->hsa_amd_spm_set_dest_buffer_fn(
        args.hsa_agent,
        args.buf_size,
        &args.timeout,
        &args.size_copied,
        args.dest_buf,
        &args.is_data_loss);
}

class ManagerThread
{
public:
    ManagerThread(std::shared_ptr<spm_state_t>   _s,
                  aqlprofile_spm_data_callback_t cb,
                  void*                          userdata)
    : s(_s)
    , agent(_s->hsa_agent)
    {
        if(agent.handle == 0) throw std::runtime_error("Invalid hsa agent");
        s->stop_cons_thread = false;
        s->stop_prod_thread = false;

        status = rocprofiler::aqlprofile::get_amd_ext_table()->hsa_amd_spm_acquire_fn(s->hsa_agent);
        CHECKHSA(status, return );

        // This non-blocking (timeout = 0) HsaSpmSetDestBuffer() call will clear up all the
        // residual counters from previous SPM runs. Most of the time, nothing will be copied.
        // This call will also trigger KFD to call spm_start() function. We must make sure
        // spm_start() is finished before we give back the control to caller of
        // start_spm_threads().
        spm_set_dest_buffer_args args = *s;
        args.size_copied              = 0;
        args.timeout                  = 0;
        if(HsaSpmSetDestBuffer(args) != HSA_STATUS_SUCCESS)
            throw std::runtime_error("hsa_amd_spm_set_dest_buffer() init error");

        producer_thread = std::thread(producer, s);
        consumer_thread = std::thread(consumer, s, cb, userdata);
    }

    ~ManagerThread()
    {
        s->stop_prod_thread.store(true);

        if(producer_thread.joinable()) producer_thread.join();
        if(consumer_thread.joinable()) consumer_thread.join();

        rocprofiler::aqlprofile::get_amd_ext_table()->hsa_amd_spm_release_fn(this->agent);
    }

    hsa_status_t status = HSA_STATUS_ERROR;

private:
    std::thread                  producer_thread{};
    std::thread                  consumer_thread{};
    std::shared_ptr<spm_state_t> s{nullptr};

    hsa_agent_t agent;
};

namespace aqlprofile
{
namespace spm
{
bool
is_virtualization_enabled()
{
    // Check if GPU virtualization (SR-IOV) is enabled by looking for virtual function indicators
    //
    // In SR-IOV GPU virtualization:
    // - Physical Function (PF): The actual GPU hardware device
    // - Virtual Function (VF): Virtualized GPU instances derived from the PF
    //
    // The /sys/class/drm/card*/device/physfn symlink exists ONLY on VF devices
    // and points back to their corresponding PF device. If this link exists,
    // the GPU is running as a virtual function (virtualization enabled).

    try
    {
        for(const auto& entry : std::filesystem::directory_iterator("/sys/class/drm"))
        {
            if(entry.path().filename().string().substr(0, 4) == "card" &&
               std::filesystem::exists(entry.path() / "device" / "physfn"))
            {
                return true;
            }
        }
    } catch(...)
    {
        // If filesystem access fails, assume no virtualization; fall-through
    }
    return false;
}

bool
is_agent_supported_for_spm(const AgentInfo* agentInfo)
{
    // Check value, not just presence (must be non-empty and non-zero to override)
    auto env_val = rocprofiler::common::get_env_optional("AQLPROFILE_SPM_OVERRIDE_AGENT_CHECK");
    if(env_val && !env_val->empty() && env_val->front() != '0') return true;

    // if the device is gfx90a, then spm is not supported
    if(strncmp(agentInfo->gfxip, "gfx90a", 6) == 0)
    {
        printf("Streaming Performance Monitor (SPM) is not supported on gfx90a devices\n");
        return false;
    }
    else if(strncmp(agentInfo->gfxip, "gfx942", 6) == 0)
    {
        // if the device is gfx942, check if virtualization is enabled
        if(is_virtualization_enabled())
        {
            printf("Streaming Performance Monitor (SPM) is not supported on gfx942 devices "
                   "when GPU virtualization (SR-IOV) is enabled\n");
            return false;
        }
    }
    return true;
}

std::vector<aqlprofile_spm_parameter_t> default_spm_params = {
    {AQLPROFILE_SPM_PARAMETER_TYPE_BUFFER_SIZE, 1 << 27},      // 128MB
    {AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL, 1 << 13},  // 4us
    {AQLPROFILE_SPM_PARAMETER_TYPE_TIMEOUT, 0},                // 0ms
    {AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_MODE, AQLPROFILE_SPM_PARAMETER_SAMPLE_MODE_SCLK}};
static_assert(AQLPROFILE_SPM_PARAMETER_TYPE_LAST == 4 && "Dont forget to add default param!");

counter_des_t
GetCounter(aql_profile::Pm4Factory*                       pm4_factory,
           const aqlprofile_pmc_event_t&                  event,
           std::map<block_des_t, uint32_t, lt_block_des>& index_map)
{
    const GpuBlockInfo* block_info = pm4_factory->GetBlockInfo(event.block_name);
    const block_des_t   block_des  = {block_info->id, event.block_index};
    const auto          ret        = index_map.insert({block_des, 0});
    auto                reg_index  = ret.first->second;

    if(reg_index >= block_info->counter_count)
        throw std::runtime_error("Event is out of block counter registers number limit");

    ret.first->second++;
    return {event.event_id, reg_index, block_des, block_info};
}

pm4_builder::counters_vector
CountersVec(const aqlprofile_pmc_event_t* events,
            size_t                        num_events,
            aql_profile::Pm4Factory*      pm4_factory)
{
    pm4_builder::counters_vector                  vec;
    std::map<block_des_t, uint32_t, lt_block_des> index_map;

    for(size_t i = 0; i < num_events; i++)
        vec.push_back(GetCounter(pm4_factory, events[i], index_map));

    return vec;
}

class SpmStateMap
{
public:
    std::shared_ptr<spm_state_t> query(aqlprofile_handle_t handle)
    {
        auto lock = std::shared_lock{mut};
        auto it   = map.find(handle);
        if(it != map.end()) return it->second;
        return nullptr;
    }
    void insert(aqlprofile_handle_t handle, std::shared_ptr<spm_state_t> state)
    {
        auto lock = std::unique_lock{mut};
        map.emplace(handle, std::move(state));
    }
    void remove(aqlprofile_handle_t handle)
    {
        auto lock = std::unique_lock{mut};
        try
        {
            map.at(handle)->manager_thread = nullptr;
            map.at(handle)->memory         = nullptr;
            map.erase(handle);
        } catch(...)
        {}
    }
    bool setthread(aqlprofile_handle_t handle, std::unique_ptr<ManagerThread>&& thread)
    {
        auto lock       = std::unique_lock{mut};
        bool bret       = threads.find(handle) != threads.end();
        threads[handle] = std::move(thread);
        return bret;
    }

private:
    std::shared_mutex                                             mut;
    std::map<aqlprofile_handle_t, std::shared_ptr<spm_state_t>>   map{};
    std::map<aqlprofile_handle_t, std::unique_ptr<ManagerThread>> threads{};
};

// Lazy-init via rocprofiler::common::static_object; destroyed in destroy_static_objects() at exit.
SpmStateMap*
spm_state_map()
{
    static auto*& _v = rocprofiler::common::static_object<SpmStateMap>::construct();
    return _v;
}

hsa_status_t
_internal_aqlprofile_spm_create_packets(aqlprofile_handle_t*          handle,
                                        aqlprofile_spm_buffer_desc_t* out_desc,
                                        aqlprofile_spm_aql_packets_t* packets,
                                        aqlprofile_spm_profile_t      profile,
                                        size_t                        flags)
{
    if(!is_agent_supported_for_spm(aql_profile::GetAgentInfo(profile.aql_agent)))
        return HSA_STATUS_ERROR_INVALID_AGENT;

    auto s       = std::make_shared<spm_state_t>();
    s->aql_agent = profile.aql_agent;
    s->hsa_agent = profile.hsa_agent;

    auto& params = s->parameters;
    for(auto& p : default_spm_params)
        params.at(p.type) = p.value;  // Set default params

    try
    {
        for(size_t i = 0; i < profile.parameter_count; i++)
            params.at(profile.parameters[i].type) = profile.parameters[i].value;
    } catch(...)
    {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    s->memory    = std::make_unique<SPMMemoryManager>(profile.aql_agent,
                                                   profile.hsa_agent,
                                                   profile.alloc_cb,
                                                   profile.dealloc_cb,
                                                   profile.userdata);
    auto& memory = s->memory;

    try
    {
        memory->CreateOutputBuf(params.at(AQLPROFILE_SPM_PARAMETER_TYPE_BUFFER_SIZE) +
                                SPM_DESC_SIZE);
    } catch(...)
    {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    // Populate user output
    handle->handle = memory->GetHandler();
    out_desc->data = memory->GetOutputBuf();
    out_desc->size = SPM_DESC_SIZE;
    spm_state_map()->insert(*handle, s);

    {
        aql_profile::Pm4Factory* pm4_factory = nullptr;
        try
        {
            pm4_factory = aql_profile::Pm4Factory::Create(profile.aql_agent);
            if(!pm4_factory) throw std::exception();
        } catch(...)
        {
            return HSA_STATUS_ERROR_INVALID_AGENT;
        }

        const pm4_builder::counters_vector countersVec =
            CountersVec(profile.events, profile.event_count, pm4_factory);

        pm4_builder::TraceConfig& trace_config = memory->config;

        trace_config.spm_sq_32bit_mode = true;
        trace_config.spm_has_core1     = (pm4_factory->GetGpuId() == aql_profile::MI100_GPU_ID) ||
                                     (pm4_factory->GetGpuId() == aql_profile::MI200_GPU_ID);
        trace_config.spm_sample_delay_max = pm4_factory->GetSpmSampleDelayMax();
        trace_config.sampleRate =
            (s->parameters.at(AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL) + 16) & ~31ul;
        if(trace_config.sampleRate == 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

        if(s->parameters.at(AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_MODE) !=
           AQLPROFILE_SPM_PARAMETER_SAMPLE_MODE_SCLK)
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;

        trace_config.xcc_number = pm4_factory->GetXccNumber();
        trace_config.se_number  = pm4_factory->GetShaderEnginesNumber() / trace_config.xcc_number;
        trace_config.sa_number  = pm4_factory->GetGpuId() >= aql_profile::GFX10_GPU_ID ? 2 : 0;

        trace_config.data_buffer_ptr  = memory->GetOutputBuf();
        trace_config.data_buffer_size = memory->GetOutputBufSize();

        pm4_builder::CmdBuffer start_cmd;
        pm4_builder::CmdBuffer stop_cmd;

        pm4_builder::SpmBuilder* spm_builder = pm4_factory->GetSpmBuilder();
        // Generate commands
        spm_builder->Begin(&start_cmd, &trace_config, countersVec);
        spm_builder->End(&stop_cmd, &trace_config);

        // Copy generated commands
        size_t start_size = aql_profile::CommandBufferMgr::Align(start_cmd.Size());
        size_t stop_size  = aql_profile::CommandBufferMgr::Align(stop_cmd.Size());

        try
        {
            memory->CreateCmdBuf(start_size + stop_size);
        } catch(...)
        {
            return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
        }

        pm4_builder::CmdBuilder* cmd_writer = pm4_factory->GetCmdBuilder();
        uint8_t*                 cmdbuf     = reinterpret_cast<uint8_t*>(memory->GetCmdBuf());

        profile.memcpy_cb(cmdbuf, start_cmd.Data(), start_cmd.Size(), profile.userdata);
        aql_profile::PopulateAql(cmdbuf, start_cmd.Size(), cmd_writer, &packets->start_packet);
        cmdbuf += start_size;
        profile.memcpy_cb(cmdbuf, stop_cmd.Data(), stop_cmd.Size(), profile.userdata);
        aql_profile::PopulateAql(cmdbuf, stop_cmd.Size(), cmd_writer, &packets->stop_packet);
    }

    s->output_buffer_ptr  = memory->GetOutputBuf();
    s->output_buffer_size = memory->GetOutputBufSize();

    return HSA_STATUS_SUCCESS;
}

}  // namespace spm
}  // namespace aqlprofile

PUBLIC_API hsa_status_t
aqlprofile_spm_create_packets(aqlprofile_handle_t*          handle,
                              aqlprofile_spm_buffer_desc_t* out_desc,
                              aqlprofile_spm_aql_packets_t* packets,
                              aqlprofile_spm_profile_t      profile,
                              size_t                        flags)
{
    try
    {
        return aqlprofile::spm::_internal_aqlprofile_spm_create_packets(
            handle, out_desc, packets, profile, flags);
    } catch(...)
    {
        return HSA_STATUS_ERROR;
    }
    return HSA_STATUS_SUCCESS;
}

PUBLIC_API hsa_status_t
aqlprofile_spm_start(aqlprofile_handle_t            handle,
                     aqlprofile_spm_data_callback_t data_cb,
                     void*                          userdata)
{
    auto s = aqlprofile::spm::spm_state_map()->query(handle);
    if(!s) return HSA_STATUS_ERROR_NOT_INITIALIZED;

    // The first page of output_buffer is reserved for SpmBufferDesc
    char*          buf_ptr  = (char*) (s->output_buffer_ptr) + SPM_DESC_SIZE;
    size_t         buf_size = (s->output_buffer_size - SPM_DESC_SIZE) / spm_state_t::NUM_SLOTS;
    SpmBufferDesc* desc     = (SpmBufferDesc*) s->output_buffer_ptr;
    size_t         seg_size = (desc->global_num_line + desc->se_num_line * desc->num_se) * 32;
    // Align buf_size to the exact multiples of segments, so that every HsaSpmSetDestBuffer
    // will always return complete segments
    if(!desc->num_xcc) desc->num_xcc = 1;

    buf_size /= desc->num_xcc;
    if(seg_size)
    {
        buf_size = (buf_size - sizeof(kfd_ioctl_spm_buffer_header)) / seg_size * seg_size +
                   sizeof(kfd_ioctl_spm_buffer_header);
    }
    buf_size *= desc->num_xcc;

    // Args for hsa_amd_spm_set_dest_buffer
    s->buf_size = buf_size;
    s->timeout  = s->parameters.at(AQLPROFILE_SPM_PARAMETER_TYPE_TIMEOUT);
    s->num_xcc      = desc->num_xcc;
    s->buf_size_xcc = s->buf_size / desc->num_xcc;

    // Initialize ring buffer slots
    for(size_t i = 0; i < spm_state_t::NUM_SLOTS; i++)
        s->slots[i].buf = buf_ptr + i * buf_size;
    s->ring_head.store(0, std::memory_order_relaxed);
    s->ring_tail.store(0, std::memory_order_relaxed);
    s->dest_buf = s->slots[0].buf;

    try
    {
        auto manager = std::make_unique<ManagerThread>(s, data_cb, userdata);

        CHECKHSA(manager->status, return manager->status);
        aqlprofile::spm::spm_state_map()->setthread(handle, std::move(manager));
    } catch(...)
    {
        return HSA_STATUS_ERROR;
    }
    return HSA_STATUS_SUCCESS;
}

PUBLIC_API hsa_status_t
aqlprofile_spm_stop(aqlprofile_handle_t handle)
{
    bool b = aqlprofile::spm::spm_state_map()->setthread(handle, nullptr);
    return b ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_NOT_INITIALIZED;
}

PUBLIC_API void
aqlprofile_spm_delete_packets(aqlprofile_handle_t handle)
{
    if(auto* map = rocprofiler::common::static_object<aqlprofile::spm::SpmStateMap>::get())
        map->remove(handle);
}

struct consumer_thread_handle_t
{
    consumer_thread_handle_t(std::shared_ptr<spm_state_t> _s)
    : s(std::move(_s)){};
    ~consumer_thread_handle_t()
    {
        std::unique_lock<std::mutex> lock(s->work_mutex);
        s->stop_cons_thread = true;
        s->work_cond.notify_one();
    }
    std::shared_ptr<spm_state_t> s;
};

static void
producer(std::shared_ptr<spm_state_t> s)
{
    spm_set_dest_buffer_args args = *s;
    bool                     exiting    = false;
    int                      count_down = 0;
    size_t                   write_idx  = 0;

    consumer_thread_handle_t consumer_handle(s);

    args.timeout = s->timeout;
    while(true)
    {
        // s->stop_prod_thread should be set after SPM End() sequence is submitted, this is the
        // handshake protocol between app/library and aqlprofile.
        // If s->stop_prod_thread is set in current loop, producer thread will exit after all
        // SPM counters are drained (args.size_copied == 0) which could be at least one
        // HsaSpmSetDestBuffer() call or maybe more than one.
        if(s->stop_prod_thread) exiting = true;

        // Get the next write slot
        size_t next = (write_idx + 1) % spm_state_t::NUM_SLOTS;
        if(next == s->ring_tail.load(std::memory_order_acquire))
        {
            // Ring full — yield and retry
            sched_yield();
            continue;
        }

        auto& slot       = s->slots[write_idx];
        args.size_copied = 0;
        args.dest_buf    = slot.buf;

        if(HsaSpmSetDestBuffer(args) != HSA_STATUS_SUCCESS)
        {
            std::cerr << "hsa_amd_spm_set_dest_buffer() error" << std::endl;
            return;
        }

        slot.size_copied = 0;
        slot.data_loss   = false;
        // In the initial XCC SPM design, 'size_copied' and 'is_data_loss' are stored in
        // kfd_ioctl_spm_buffer_header. They are no longer stored in kfd_ioctl_spm_args.
        // But we still need accumulated version for some quick checks and KFD will add
        // them back to kfd_ioctl_spm_args.
        // This is only a temporary patch as KFD will fix this in ROCm 6.5
        char* base       = (char*) slot.buf;
        for(uint32_t i = 0; i < s->num_xcc; i++)
        {
            auto* buf_info = (kfd_ioctl_spm_buffer_header*) base;
            slot.size_copied += buf_info->bytes_copied;
            slot.data_loss |= buf_info->has_data_loss;
            base += s->buf_size_xcc;
        }
        s->signal_data_loss.fetch_or(slot.data_loss);

        // Publish slot to consumer
        write_idx = next;
        s->ring_head.store(next, std::memory_order_release);
        s->work_cond.notify_one();

        // Forced exit: This happens when we want to stop SPM but not the app. This should be
        // improved by getting the hint from caller instead of a hardcoded number. Will consider
        // this in the new SPM api design.
        if(exiting)
        {
            if(slot.size_copied)
            {
                if(count_down++ < 5) continue;
                printf("Forced exit after %d extra hsa_amd_spm_set_dest_buffer() calls\n",
                       count_down);
            }
            // We cannot directly use s->stop_prod_thread here, otherwise we might miss the last
            // HsaSpmSetDestBuffer() call if s->stop_prod_thread is set after the
            // HsaSpmSetDestBuffer() call from this loop.
            break;
        }
        if(s->stop_cons_thread) break;
    }
}

static void
consumer(std::shared_ptr<spm_state_t> s, aqlprofile_spm_data_callback_t callback, void* userdata)
{
    // Wait for data from producer or shutdown signal.
    // Producer destructor sets stop_cons_thread; exit without processing stale work.
    while(true)
    {
        {
            std::unique_lock<std::mutex> lock(s->work_mutex);
            s->work_cond.wait(lock, [&s]() {
                return !s->ring_empty() || s->stop_cons_thread;
            });
            if(s->ring_empty() && s->stop_cons_thread) return;
        }

        // Drain all available slots
        while(!s->ring_empty())
        {
            size_t t    = s->ring_tail.load(std::memory_order_relaxed);
            auto&  slot = s->slots[t];
            int    flags =
                s->signal_data_loss.exchange(0) << AQLPROFILE_SPM_DATA_FLAGS_DATA_LOSS;

            char* base = (char*) slot.buf;
            for(uint32_t i = 0; i < s->num_xcc; i++)
            {
                auto* buf_info = (kfd_ioctl_spm_buffer_header*) base;
                if(buf_info->bytes_copied)
                    callback(
                        i, (void*) (buf_info + 1), buf_info->bytes_copied, flags, userdata);
                base += s->buf_size_xcc;
            }

            s->ring_tail.store((t + 1) % spm_state_t::NUM_SLOTS, std::memory_order_release);
        }
    }
}

PUBLIC_API bool
aqlprofile_spm_is_event_supported(aqlprofile_agent_handle_t agent, aqlprofile_pmc_event_t event)
{
    aql_profile::Pm4Factory* pm4_factory = nullptr;
    try
    {
        pm4_factory = aql_profile::Pm4Factory::Create(agent);
        if(!pm4_factory) return false;
    } catch(...)
    {
        return false;
    }

    if(pm4_factory->GetGpuId() < aql_profile::MI200_GPU_ID ||
       pm4_factory->GetGpuId() > aql_profile::MI350_GPU_ID)
        return false;

    static auto blocks = []() {
        std::array<bool, AQLPROFILE_BLOCKS_NUMBER> valid_blocks{};
        valid_blocks[HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC] = true;
        valid_blocks[HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF] = true;
        valid_blocks[HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ]  = true;
        valid_blocks[HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI] = true;
        valid_blocks[HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC] = true;
        valid_blocks[HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA] = true;
        valid_blocks[HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP] = true;
        valid_blocks[HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA]  = true;
        valid_blocks[HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD]  = true;
        valid_blocks[HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI] = true;
        return valid_blocks;
    }();

    if(event.flags.spm_flags.depth != AQLPROFILE_SPM_DEPTH_NONE) return false;
    if(event.block_name >= blocks.size()) return false;

    return blocks.at(event.block_name);
}

PUBLIC_API hsa_status_t
aqlprofile_spm_query_agent_configurations(aqlprofile_agent_handle_t                    agent,
                                          aqlprofile_spm_available_configurations_cb_t cb,
                                          void*                                        userdata)
{
    if(!cb) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    aql_profile::Pm4Factory* pm4_factory = nullptr;
    try
    {
        pm4_factory = aql_profile::Pm4Factory::Create(agent);
        if(!pm4_factory) return HSA_STATUS_ERROR_INVALID_AGENT;
    } catch(...)
    {
        return HSA_STATUS_ERROR_INVALID_AGENT;
    }

    if(!aqlprofile::spm::is_agent_supported_for_spm(aql_profile::GetAgentInfo(agent)))
        return HSA_STATUS_ERROR_INVALID_AGENT;

    auto configs = std::vector<aqlprofile_spm_available_configuration_t>{};

    auto& interval_config = configs.emplace_back();
    interval_config.type  = AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL;
    interval_config.interval.min_interval =
        32;  // Sample interval time in sclks. Minimum is 32 clocks by HW design
    interval_config.interval.max_interval =
        (1 << 16) - 32;  // Maximum value by HW design. Must be multiples of 32.
    interval_config.interval.mode = AQLPROFILE_SPM_PARAMETER_SAMPLE_MODE_SCLK;

    return cb(configs.data(), configs.size(), userdata);
}
