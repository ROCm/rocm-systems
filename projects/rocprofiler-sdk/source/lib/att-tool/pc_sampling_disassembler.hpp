#pragma once

#include "att_lib_wrapper.hpp"
#include "code.hpp"
#include "util.hpp"

#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/output/generator.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace att_wrapper
{
namespace pc_sampling
{
class PcSamplingDisassembler
{
public:
    using Fspath          = rocprofiler::common::filesystem::path;
    using CodeFile        = rocprofiler::att_wrapper::CodeFile;
    using CodeLine        = rocprofiler::att_wrapper::CodeLine;
    using CodeobjLoadInfo = rocprofiler::att_wrapper::CodeobjLoadInfo;
    using AddressTable    = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;

    template <typename PcSamplingToolRecordT>
    using generator = rocprofiler::tool::generator<PcSamplingToolRecordT>;

    explicit PcSamplingDisassembler(Fspath output_dir);
    ~PcSamplingDisassembler() = default;

    void load_code_objects(const Fspath& input_dir, const std::vector<CodeobjLoadInfo>& codeobjs);

    void disassemble_all();

    std::shared_ptr<CodeFile>     get_codefile() const { return codefile; }
    std::shared_ptr<AddressTable> get_address_table() const { return table; }

    template <typename PcSamplingToolRecordT>
    void aggregate_pc_samples(const generator<PcSamplingToolRecordT>& pc_sampling_gen)
    {
        using rocprofiler_tool_pc_sampling_stochastic_record_t =
            rocprofiler::tool::rocprofiler_tool_pc_sampling_stochastic_record_t;

        for(auto pitr : pc_sampling_gen)
        {
            for(auto itr : pc_sampling_gen.get(pitr))
            {
                auto  record = itr.pc_sample_record;
                auto& cline = get(pcinfo_t{record.pc.code_object_offset, record.pc.code_object_id});
                cline.hitcount++;
                cline.service_type = code_line_service_type_host_trap_pc_sampling;

                if constexpr(std::is_same_v<PcSamplingToolRecordT,
                                            rocprofiler_tool_pc_sampling_stochastic_record_t>)
                {
                    // Consider issue VS stalls for stochastic sampling.
                    cline.service_type = code_line_service_type_stochastic_pc_sampling;
                    if(record.wave_issued)
                    {
                        cline.issued++;
                    }
                    else
                    {
                        cline.stalled++;
                        // No available slot for
                        // ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NONE, so we
                        // subtract 1 from the no issue reason value to determine the index of a
                        // counter.
                        cline.not_issued_reasons[record.snapshot.reason_not_issued - 1]++;
                    }
                }
            }
        }
    }

private:
    struct LoadedObj
    {
        uint64_t id{};
        uint64_t load_addr{};
        uint64_t size{};
    };

    Fspath                        out_dir;
    std::shared_ptr<AddressTable> table;
    std::shared_ptr<CodeFile>     codefile;
    std::vector<LoadedObj>        loaded;

    void add_decoder_bytes(uint64_t    id,
                           uint64_t    load_addr,
                           uint64_t    memsize,
                           const char* data,
                           size_t      sz);

    void disassemble_one(const LoadedObj& obj);

    CodeLine& get(pcinfo_t pc);
};

}  // namespace pc_sampling
}  // namespace att_wrapper
}  // namespace rocprofiler
