#include "pc_sampling_disassembler.hpp"

#include <fstream>
#include <stdexcept>

namespace rocprofiler
{
namespace att_wrapper
{
namespace pc_sampling
{
PcSamplingDisassembler::PcSamplingDisassembler(Fspath output_dir)
: out_dir(std::move(output_dir))
{
    rocprofiler::common::filesystem::create_directories(out_dir);
    table    = std::make_shared<AddressTable>();
    codefile = std::make_shared<CodeFile>(out_dir, table);

    auto& formats = GlobalDefs::get().output_formats;
    formats       = "json,csv";
    std::transform(formats.begin(), formats.end(), formats.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
}

void
PcSamplingDisassembler::add_decoder_bytes(uint64_t    id,
                                          uint64_t    load_addr,
                                          uint64_t    memsize,
                                          const char* data,
                                          size_t      sz)
{
    // Register raw code object bytes for address translation/disassembly
    table->addDecoder(data, sz, id, load_addr, memsize);
    loaded.push_back({id, load_addr, memsize});
}

void
PcSamplingDisassembler::load_code_objects(const Fspath&                       input_dir,
                                          const std::vector<CodeobjLoadInfo>& codeobjs)
{
    for(const auto& obj : codeobjs)
    {
        if(obj.id == 0 && obj.name.empty()) continue;
        if(obj.name.find("memory://") == 0)
        {
            ROCP_WARNING << "Skipping in-memory code object stub: " << obj.name;
            continue;
        }

        auto          full_path = input_dir / obj.name;
        std::ifstream ifs(full_path, std::ios::binary);
        if(!ifs.is_open())
        {
            ROCP_WARNING << "Unable to open code object file: " << full_path.string();
            continue;
        }

        ifs.seekg(0, std::ios::end);
        auto sz = static_cast<size_t>(ifs.tellg());
        ifs.seekg(0, std::ios::beg);

        std::vector<char> buffer(sz);
        if(sz > 0) ifs.read(buffer.data(), sz);

        if(buffer.empty())
        {
            ROCP_WARNING << "Empty code object file: " << full_path.string();
            continue;
        }

        try
        {
            add_decoder_bytes(obj.id, obj.addr, obj.size, buffer.data(), buffer.size());
        } catch(const std::exception& e)
        {
            ROCP_WARNING << "Failed adding decoder bytes for " << obj.name << ": " << e.what();
        }
    }
}

void
PcSamplingDisassembler::disassemble_one(const LoadedObj& obj)
{
    // Extract symbol map for this code object
    auto symbol_map = table->getSymbolMap(obj.id);

    // Iterate over each symbol's address range
    for(const auto& [vaddr, symbol_info] : symbol_map)
    {
        uint64_t       pc  = vaddr;
        const uint64_t end = vaddr + symbol_info.mem_size;

        while(pc < end)
        {
            auto inst_ptr = table->get(obj.id, pc);
            if(!inst_ptr) break;  // Cannot decode further

            if(inst_ptr->size == 0)
            {
                ROCP_WARNING << "Instruction size zero at code_object_id=" << obj.id << " pc=0x"
                             << std::hex << pc << std::dec << ". Aborting this symbol.";
                break;
            }

            pcinfo_t key{pc, obj.id};  // offset within code object
            // Update PC before changing the ownership of the inst_ptr
            pc += inst_ptr->size;

            if(codefile->isa_map.find(key) == codefile->isa_map.end())
            {
                auto line         = std::make_unique<CodeLine>();
                line->line_number = 0;
                line->type        = 0;
                line->code_line   = std::move(inst_ptr);  // transfer ownership
                codefile->isa_map.emplace(key, std::move(line));
            }
        }
    }

    // Why this doesn't work?
    // // Disassemble the entire code object without using symbols
    // uint64_t pc = obj.load_addr;
    // const uint64_t end = obj.load_addr + obj.size;

    // while(pc < end)
    // {

    //     auto offset = pc - obj.load_addr;
    //     auto inst_ptr = table->get(obj.id, offset);
    //     if(!inst_ptr) break; // Cannot decode further

    //     if(inst_ptr->size == 0)
    //     {
    //         ROCP_WARNING << "Instruction size zero at code_object_id=" << obj.id
    //                      << " pc=0x" << std::hex << pc << std::dec << ". Stopping disassembly.";
    //         break;
    //     }

    //     pcinfo_t key{obj.id, offset};
    //     // Update PC before changing the ownership of the inst_ptr
    //     pc += inst_ptr->size;

    //     if(codefile->isa_map.find(key) == codefile->isa_map.end())
    //     {
    //         auto line = std::make_unique<CodeLine>();
    //         line->line_number = 0;
    //         line->type        = 0;
    //         line->code_line   = std::move(inst_ptr); // transfer ownership
    //         codefile->isa_map.emplace(key, std::move(line));
    //     }
    // }
}

void
PcSamplingDisassembler::disassemble_all()
{
    for(const auto& obj : loaded)
        disassemble_one(obj);
}

}  // namespace pc_sampling
}  // namespace att_wrapper
}  // namespace rocprofiler
