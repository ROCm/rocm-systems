#pragma once

#include "att_lib_wrapper.hpp"
#include "code.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "util.hpp"

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

    explicit PcSamplingDisassembler(Fspath output_dir);
    ~PcSamplingDisassembler() = default;

    void load_code_objects(const Fspath& input_dir, const std::vector<CodeobjLoadInfo>& codeobjs);

    void disassemble_all();

    std::shared_ptr<CodeFile>     get_codefile() const { return codefile; }
    std::shared_ptr<AddressTable> get_address_table() const { return table; }

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
};

}  // namespace pc_sampling
}  // namespace att_wrapper
}  // namespace rocprofiler
