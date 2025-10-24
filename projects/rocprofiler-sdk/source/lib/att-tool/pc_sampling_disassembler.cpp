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

CodeLine&
PcSamplingDisassembler::get(pcinfo_t _pc)
{
    auto& isa_map = codefile->isa_map;
    if(isa_map.find(_pc) != isa_map.end()) return *isa_map.at(_pc);

    // Attempt to disassemble full kernel
    if(_pc.code_object_id != 0u) try
        {
            rocprofiler::sdk::codeobj::segment::CodeobjTableTranslator symbol_table;
            for(auto& [vaddr, symbol] : codefile->table->getSymbolMap(_pc.code_object_id))
                symbol_table.insert({symbol.vaddr, symbol.mem_size, _pc.code_object_id});

            auto addr_range = symbol_table.find_codeobj_in_range(_pc.address);
            try
            {
                auto symbol = codefile->table->getSymbolMap(_pc.code_object_id).at(addr_range.addr);
                // auto pair   = KernelName{symbol.name, demangle(symbol.name)};
                // TODO: cover demangling later
                auto pair = KernelName{symbol.name, symbol.name};
                codefile->kernel_names.emplace(pcinfo_t{addr_range.addr, _pc.code_object_id}, pair);
            } catch(...)
            {
                ROCP_INFO << "Missing kernelSymbol at " << _pc.code_object_id << ':'
                          << addr_range.addr;
            }

            for(auto addr = addr_range.addr; addr < addr_range.addr + addr_range.size;)
            {
                pcinfo_t info{.address = addr, .code_object_id = addr_range.id};
                auto& cline = *(isa_map.emplace(info, std::make_unique<CodeLine>()).first->second);

                cline.line_number            = isa_map.size() + codefile->kernel_names.size() - 1;
                codefile->line_numbers[info] = cline.line_number;

                cline.code_line = codefile->table->get(addr_range.id, addr);
                addr += cline.code_line->size;
                if(cline.code_line->size == 0u) throw std::invalid_argument("Line has 0 bytes!");
            }

            if(isa_map.find(_pc) != isa_map.end()) return *isa_map.at(_pc);
        } catch(std::exception& e)
        {}

    auto& cline = *(isa_map.emplace(_pc, std::make_unique<CodeLine>()).first->second);

    cline.line_number           = isa_map.size();
    codefile->line_numbers[_pc] = cline.line_number;

    cline.code_line = codefile->table->get(_pc.code_object_id, _pc.address);

    return cline;
}

}  // namespace pc_sampling
}  // namespace att_wrapper
}  // namespace rocprofiler
