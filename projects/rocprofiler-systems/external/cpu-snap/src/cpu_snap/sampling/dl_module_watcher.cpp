// dl_module_watcher.cpp
//
// Implementation of cpu_snap::sampling::dl_module_watcher. See the header
// for the contract.

#include "cpu_snap/sampling/dl_module_watcher.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <elf.h>
#include <link.h>
#include <sys/types.h>
#include <unistd.h>

#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/self_exe.hpp"
#include "cpu_snap/sampling/time.hpp"

namespace cpu_snap::sampling
{

namespace
{

// NOLINTBEGIN(readability-identifier-naming): codebase k_*-prefix convention
constexpr std::uint32_t      k_nt_gnu_build_id      = 3;
constexpr std::size_t        k_note_align           = 4;
constexpr std::array<char, 4> k_gnu_note_name       = { 'G', 'N', 'U', '\0' };
constexpr const char*        k_vdso_path            = "[vdso]";
constexpr const char*        k_main_exe_path        = "[main-exe]";
// NOLINTEND(readability-identifier-naming)

/**
 * Round @p n up to the next multiple of k_note_align. ELF notes pad
 * their name[] and desc[] fields to 4-byte boundaries.
 */
[[nodiscard]] constexpr std::size_t
align_up_note(std::size_t n) noexcept
{
    return (n + k_note_align - 1) & ~(k_note_align - 1);
}

/**
 * Scan a PT_NOTE segment for an NT_GNU_BUILD_ID note and copy the
 * descriptor bytes out.
 *
 * Layout: each note is `Elf64_Nhdr` (12 B) + name[n_namesz] padded
 * up to 4-byte alignment + desc[n_descsz] padded up to 4-byte
 * alignment. We only recognise notes whose name field is exactly the
 * string "GNU\0" (4 bytes including the terminator).
 */
[[nodiscard]] std::vector<std::uint8_t>
read_build_id_from_note(const std::uint8_t* base, std::size_t size) noexcept
{
    std::size_t offset = 0;
    while(offset + sizeof(ElfW(Nhdr)) <= size)
    {
        ElfW(Nhdr) nhdr{};
        std::memcpy(&nhdr, base + offset, sizeof(nhdr));

        const std::size_t name_off  = offset + sizeof(nhdr);
        const std::size_t name_size = align_up_note(nhdr.n_namesz);
        const std::size_t desc_off  = name_off + name_size;
        const std::size_t desc_size = align_up_note(nhdr.n_descsz);
        const std::size_t next_off  = desc_off + desc_size;

        if(next_off > size) break;

        const bool is_gnu =
            nhdr.n_namesz == k_gnu_note_name.size() &&
            std::memcmp(base + name_off, k_gnu_note_name.data(),
                        k_gnu_note_name.size()) == 0;

        if(is_gnu && nhdr.n_type == k_nt_gnu_build_id && nhdr.n_descsz > 0)
        {
            std::vector<std::uint8_t> out(nhdr.n_descsz);
            std::memcpy(out.data(), base + desc_off, nhdr.n_descsz);
            return out;
        }
        offset = next_off;
    }
    return {};
}

struct phdr_collect_ctx
{
    std::vector<core::module_info>* out                 = nullptr;
    const std::string*        main_exe_path       = nullptr;
    bool                      saw_main_executable = false;
};

[[nodiscard]] core::module_info
build_module_from_phdr(const dl_phdr_info& info, const std::string& path_override)
{
    core::module_info mod{};
    mod.base_va = info.dlpi_addr;
    mod.path    = path_override.empty() ? std::string(info.dlpi_name) : path_override;

    const auto page_size = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));

    bool found_text = false;
    for(std::uint16_t i = 0; i < info.dlpi_phnum; ++i)
    {
        const auto& phdr = info.dlpi_phdr[i];
        if(phdr.p_type == PT_LOAD && (phdr.p_flags & PF_X) != 0 && !found_text)
        {
            const std::uint64_t seg_start = (info.dlpi_addr + phdr.p_vaddr) & ~(page_size - 1);
            const std::uint64_t seg_end =
                (info.dlpi_addr + phdr.p_vaddr + phdr.p_memsz + page_size - 1) &
                ~(page_size - 1);
            mod.base_va     = seg_start;
            mod.size        = seg_end - seg_start;
            mod.file_offset = phdr.p_offset & ~(page_size - 1);
            found_text      = true;
        }
        if(phdr.p_type == PT_NOTE && mod.build_id.empty())
        {
            // dl_phdr_info reports addresses as integers; the segment lives in
            // process memory, so reinterpreting the sum as a pointer is safe.
            // NOLINTBEGIN(performance-no-int-to-ptr)
            const auto* note_base =
                reinterpret_cast<const std::uint8_t*>(info.dlpi_addr + phdr.p_vaddr);
            // NOLINTEND(performance-no-int-to-ptr)
            mod.build_id = read_build_id_from_note(note_base, phdr.p_memsz);
        }
    }
    return mod;
}

int
dl_iterate_callback(struct dl_phdr_info* info, std::size_t /*size*/, void* data)
{
    auto* ctx = static_cast<phdr_collect_ctx*>(data);

    std::string path_override;
    const bool  name_empty = info->dlpi_name == nullptr || info->dlpi_name[0] == '\0';
    if(name_empty)
    {
        // dl_iterate_phdr reports the main executable with an empty
        // name on the very first callback. vDSO and other anonymous
        // mappings also arrive nameless; the main exe is the one whose
        // base_va matches dlpi_addr of the first iteration, which is
        // also the one we have not yet labelled.
        if(!ctx->saw_main_executable)
        {
            path_override = ctx->main_exe_path != nullptr && !ctx->main_exe_path->empty()
                                ? *ctx->main_exe_path
                                : std::string{ k_main_exe_path };
            ctx->saw_main_executable = true;
        }
        else
        {
            path_override = k_vdso_path;
        }
    }

    core::module_info mod = build_module_from_phdr(*info, path_override);
    if(mod.size != 0 || mod.base_va != 0 || !mod.path.empty())
    {
        ctx->out->push_back(std::move(mod));
    }
    return 0;
}

using generation = std::pair<unsigned long long, unsigned long long>;

int
read_generation_callback(struct dl_phdr_info* info, std::size_t size, void* data)
{
    constexpr std::size_t k_needed = offsetof(dl_phdr_info, dlpi_subs) + sizeof(info->dlpi_subs);
    if(size < k_needed)
    {
        return 0;
    }
    auto* gen = static_cast<generation*>(data);
    *gen = { info->dlpi_adds, info->dlpi_subs };
    return 1;
}

[[nodiscard]] std::optional<generation>
read_generation()
{
    generation gen{};
    if(dl_iterate_phdr(&read_generation_callback, &gen) == 0)
    {
        return std::nullopt;
    }
    return gen;
}

}  // namespace

dl_module_watcher::dl_module_watcher(clock_kind clock) noexcept
: m_clock{ clock }
{}

dl_module_watcher::~dl_module_watcher() = default;

std::uint64_t
dl_module_watcher::now_wall_ns() const noexcept
{
    return utils::clock_now_ns(m_clock);
}

void
dl_module_watcher::set_iterator_for_tests(iterator_fn fn)
{
    m_iterator = std::move(fn);
}

std::vector<core::module_info>
dl_module_watcher::collect_current_modules() const
{
    std::vector<core::module_info> out;

    if(m_iterator)
    {
        m_iterator([&out](const injected_module& im) {
            core::module_info mod{};
            mod.base_va     = im.base_va;
            mod.size        = im.size;
            mod.file_offset = im.file_offset;
            mod.path        = im.path;
            out.push_back(std::move(mod));
        });
        return out;
    }

    const std::string main_exe = core::self_exe_path();
    phdr_collect_ctx  ctx{ &out, &main_exe, false };
    dl_iterate_phdr(&dl_iterate_callback, &ctx);
    return out;
}

core::maps_init_record
dl_module_watcher::snapshot()
{
    m_last_observed_ns = now_wall_ns();
    if(!m_iterator)
    {
        m_generation = read_generation();
    }
    ++m_full_walks;
    auto                   current = collect_current_modules();
    core::maps_init_record rec;
    rec.wall_ns = m_last_observed_ns;
    rec.modules = current;
    m_known     = std::move(current);
    return rec;
}

std::vector<core::maps_delta_record>
dl_module_watcher::poll()
{
    const std::uint64_t previous_observed_ns = m_last_observed_ns;
    m_last_observed_ns                       = now_wall_ns();

    if(!m_iterator)
    {
        const auto generation_now = read_generation();
        if(generation_now && m_generation && *generation_now == *m_generation)
        {
            return {};
        }
        m_generation = generation_now;
    }

    ++m_full_walks;
    auto current = collect_current_modules();

    // Identity is (base_va, path); fold both into one hash key so the
    // diff runs in O(N) instead of the two O(N^2) any_of scans this
    // function used to do.
    struct module_key
    {
        std::uint64_t base_va;
        std::string   path;

        bool operator==(const module_key& other) const noexcept
        {
            return base_va == other.base_va && path == other.path;
        }
    };
    struct module_key_hash
    {
        std::size_t operator()(const module_key& k) const noexcept
        {
            const std::size_t h1 = std::hash<std::uint64_t>{}(k.base_va);
            const std::size_t h2 = std::hash<std::string>{}(k.path);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    std::unordered_set<module_key, module_key_hash> current_keys;
    current_keys.reserve(current.size());
    for(const auto& cur : current)
    {
        current_keys.insert(module_key{ cur.base_va, cur.path });
    }

    std::unordered_set<module_key, module_key_hash> known_keys;
    known_keys.reserve(m_known.size());
    for(const auto& prev : m_known)
    {
        known_keys.insert(module_key{ prev.base_va, prev.path });
    }

    std::vector<core::maps_delta_record> deltas;
    const std::uint64_t                  wall = now_wall_ns();
    std::unordered_set<std::uint64_t>    removed_bases;

    // Removes first, then adds: preserves the emit order of the old
    // any_of-pair implementation for callers that read deltas in order.
    for(const auto& prev : m_known)
    {
        if(current_keys.find(module_key{ prev.base_va, prev.path }) ==
           current_keys.end())
        {
            core::maps_delta_record delta;
            delta.wall_ns   = wall;
            delta.operation = core::maps_delta_record::op::remove;
            delta.module    = prev;
            removed_bases.insert(prev.base_va);
            deltas.push_back(std::move(delta));
        }
    }

    for(const auto& cur : current)
    {
        if(known_keys.find(module_key{ cur.base_va, cur.path }) == known_keys.end())
        {
            core::maps_delta_record delta;
            delta.wall_ns   = removed_bases.contains(cur.base_va) ? wall : previous_observed_ns;
            delta.operation = core::maps_delta_record::op::add;
            delta.module    = cur;
            deltas.push_back(std::move(delta));
        }
    }

    m_known = std::move(current);
    return deltas;
}

std::uint64_t
dl_module_watcher::full_walk_count() const noexcept
{
    return m_full_walks;
}

std::size_t
dl_module_watcher::known_module_count() const noexcept
{
    return m_known.size();
}

}  // namespace cpu_snap::sampling
