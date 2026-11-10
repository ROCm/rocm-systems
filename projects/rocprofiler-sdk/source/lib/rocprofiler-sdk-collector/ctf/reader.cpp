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

#include "lib/rocprofiler-sdk-collector/ctf/reader.hpp"
#include "lib/rocprofiler-sdk-collector/ctf/metadata.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <queue>
#include <sstream>
#include <string_view>
#include <utility>

namespace rocprofiler
{
namespace collector
{
namespace ctf
{
namespace
{
namespace fs = std::filesystem;

constexpr uint32_t packet_magic = 0xC1FC1FC1;

// A read-only memory map of a stream file.
class mapped_file
{
public:
    mapped_file() = default;
    ~mapped_file() { reset(); }

    mapped_file(const mapped_file&) = delete;
    mapped_file& operator=(const mapped_file&) = delete;

    bool open(const std::string& path, std::string* error)
    {
        auto fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if(fd < 0)
        {
            *error = "cannot open " + path + ": " + std::strerror(errno);
            return false;
        }
        struct stat info = {};
        if(::fstat(fd, &info) != 0)
        {
            *error = "cannot stat " + path + ": " + std::strerror(errno);
            ::close(fd);
            return false;
        }
        m_size = static_cast<size_t>(info.st_size);
        if(m_size > 0)
        {
            auto* addr = ::mmap(nullptr, m_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if(addr == MAP_FAILED)
            {
                *error = "cannot map " + path + ": " + std::strerror(errno);
                ::close(fd);
                m_size = 0;
                return false;
            }
            m_data = static_cast<const unsigned char*>(addr);
        }
        ::close(fd);
        return true;
    }

    void reset()
    {
        if(m_data != nullptr) ::munmap(const_cast<unsigned char*>(m_data), m_size);  // NOLINT
        m_data = nullptr;
        m_size = 0;
    }

    const unsigned char* data() const { return m_data; }
    size_t               size() const { return m_size; }

private:
    const unsigned char* m_data = nullptr;
    size_t               m_size = 0;
};

class decode_error : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// What decoding a section records besides the flattened values.
struct section_state
{
    bool      track_clock = false;  // event header: integers mapped to the clock
    uint64_t* clock       = nullptr;
};

// Decodes the fields of one packet (bit positions are absolute in the file).
class decoder
{
public:
    decoder(const unsigned char* data, uint64_t limit_bits)
    : m_data{data}
    , m_limit{limit_bits}
    {}

    uint64_t position() const { return m_pos; }
    void     seek(uint64_t pos) { m_pos = pos; }
    void     set_limit(uint64_t limit_bits) { m_limit = limit_bits; }
    void     clear_scope() { m_scope.clear(); }

    void decode(const type&               declared,
                const std::string&        name,
                const std::string&        declared_name,
                std::vector<field_value>* out,
                const section_state&      state)
    {
        switch(declared.kind)
        {
            case type_kind::integer:
                decode_integer(declared, name, declared_name, out, state);
                break;
            case type_kind::enumeration:
                decode_enumeration(declared, name, declared_name, out);
                break;
            case type_kind::floating_point: decode_float(declared, name, out); break;
            case type_kind::string: decode_string(name, out); break;
            case type_kind::structure: decode_struct(declared, name, out, state); break;
            case type_kind::variant: decode_variant(declared, name, out, state); break;
            case type_kind::array:
            case type_kind::sequence: decode_array(declared, name, out); break;
        }
    }

private:
    struct scope_entry
    {
        const std::string* name  = nullptr;  // declared
        uint64_t           value = 0;
        const std::string* label = nullptr;  // enumeration label
    };

    const unsigned char*     m_data  = nullptr;
    uint64_t                 m_pos   = 0;
    uint64_t                 m_limit = 0;
    std::vector<scope_entry> m_scope = {};

    void align(uint32_t bits)
    {
        if(bits > 1) m_pos = (m_pos + bits - 1) / bits * bits;
    }

    void need(uint64_t bits) const
    {
        if(m_pos + bits > m_limit) throw decode_error{"event extends past the packet content"};
    }

    uint64_t read_bits(uint32_t size)
    {
        need(size);
        auto value = uint64_t{0};
        if(m_pos % 8 == 0 && size % 8 == 0)
        {
            const auto* src = m_data + (m_pos / 8);
            for(uint32_t i = 0; i < size / 8; ++i)
                value |= static_cast<uint64_t>(src[i]) << (8 * i);
        }
        else
        {
            auto got = uint32_t{0};
            auto pos = m_pos;
            while(got < size)
            {
                auto byte   = static_cast<uint32_t>(m_data[pos / 8]);
                auto offset = static_cast<uint32_t>(pos % 8);
                auto take   = std::min<uint32_t>(8 - offset, size - got);
                auto bits   = (byte >> offset) & ((1U << take) - 1U);
                value |= static_cast<uint64_t>(bits) << got;
                got += take;
                pos += take;
            }
        }
        m_pos += size;
        return value;
    }

    static int64_t sign_extend(uint64_t value, uint32_t size)
    {
        if(size < 64 && (value & (uint64_t{1} << (size - 1))) != 0) value |= ~uint64_t{0} << size;
        return static_cast<int64_t>(value);
    }

    const scope_entry* lookup(const std::string& declared_name) const
    {
        for(auto itr = m_scope.rbegin(); itr != m_scope.rend(); ++itr)
            if(*itr->name == declared_name) return &*itr;
        return nullptr;
    }

    uint64_t read_integer(const type& declared)
    {
        align(declared.alignment);
        return read_bits(declared.size);
    }

    static void update_clock(uint64_t* clock, uint64_t value, uint32_t size)
    {
        if(size >= 64)
        {
            *clock = value;
            return;
        }
        // A compact timestamp holds the low bits; a smaller value than the clock's low
        // bits means they wrapped once since the previous event.
        auto mask = (uint64_t{1} << size) - 1;
        auto next = (*clock & ~mask) | value;
        if(value < (*clock & mask)) next += uint64_t{1} << size;
        *clock = next;
    }

    void decode_integer(const type&               declared,
                        const std::string&        name,
                        const std::string&        declared_name,
                        std::vector<field_value>* out,
                        const section_state&      state)
    {
        auto raw = read_integer(declared);
        if(state.track_clock && declared.maps_clock && state.clock != nullptr)
            update_clock(state.clock, raw, declared.size);
        m_scope.push_back({&declared_name, raw, nullptr});

        auto value = field_value{};
        value.name = name;
        if(declared.is_signed)
        {
            value.kind = value_kind::signed_integer;
            value.i64  = sign_extend(raw, declared.size);
            value.u64  = static_cast<uint64_t>(value.i64);
        }
        else
        {
            value.kind = value_kind::unsigned_integer;
            value.u64  = raw;
            value.i64  = static_cast<int64_t>(raw);
        }
        out->push_back(std::move(value));
    }

    void decode_enumeration(const type&               declared,
                            const std::string&        name,
                            const std::string&        declared_name,
                            std::vector<field_value>* out)
    {
        const auto& container = *declared.container;
        auto        raw       = read_integer(container);
        auto        value     = field_value{};
        value.name            = name;
        if(container.is_signed)
        {
            value.kind = value_kind::signed_integer;
            value.i64  = sign_extend(raw, container.size);
            value.u64  = static_cast<uint64_t>(value.i64);
        }
        else
        {
            value.kind = value_kind::unsigned_integer;
            value.u64  = raw;
            value.i64  = static_cast<int64_t>(raw);
        }
        const std::string* label = nullptr;
        for(const auto& itr : declared.enumerators)
        {
            auto in_range = container.is_signed ? (value.i64 >= itr.lo && value.i64 <= itr.hi)
                                                : (value.u64 >= static_cast<uint64_t>(itr.lo) &&
                                                   value.u64 <= static_cast<uint64_t>(itr.hi));
            if(in_range)
            {
                label = &itr.label;
                break;
            }
        }
        if(label != nullptr) value.enum_label = *label;
        m_scope.push_back({&declared_name, raw, label});
        out->push_back(std::move(value));
    }

    void decode_float(const type& declared, const std::string& name, std::vector<field_value>* out)
    {
        align(declared.alignment);
        auto raw   = read_bits(declared.size);
        auto value = field_value{};
        value.name = name;
        value.kind = value_kind::floating_point;
        if(declared.size == 32)
        {
            auto bits   = static_cast<uint32_t>(raw);
            auto single = float{};
            std::memcpy(&single, &bits, sizeof(single));
            value.f64 = static_cast<double>(single);
        }
        else
        {
            std::memcpy(&value.f64, &raw, sizeof(value.f64));
        }
        out->push_back(std::move(value));
    }

    void decode_string(const std::string& name, std::vector<field_value>* out)
    {
        align(8);
        auto value = field_value{};
        value.name = name;
        value.kind = value_kind::string;
        while(true)
        {
            auto ch = static_cast<char>(read_bits(8));
            if(ch == '\0') break;
            value.str += ch;
        }
        out->push_back(std::move(value));
    }

    void decode_struct(const type&               declared,
                       const std::string&        name,
                       std::vector<field_value>* out,
                       const section_state&      state)
    {
        align(declared.alignment);
        for(const auto& itr : declared.members)
        {
            // A length is folded into its sequence: skip its value, keep it in scope.
            auto is_length = std::any_of(
                declared.members.begin(), declared.members.end(), [&itr](const member& other) {
                    return other.type->kind == type_kind::sequence &&
                           other.type->length_name == itr.name;
                });
            auto child = name.empty() ? field_name(itr.name) : name + "." + field_name(itr.name);
            if(is_length)
            {
                auto ignored = std::vector<field_value>{};
                decode(*itr.type, child, itr.name, &ignored, state);
            }
            else
            {
                decode(*itr.type, child, itr.name, out, state);
            }
        }
    }

    void decode_variant(const type&               declared,
                        const std::string&        name,
                        std::vector<field_value>* out,
                        const section_state&      state)
    {
        const auto* tag = lookup(declared.tag);
        if(tag == nullptr) throw decode_error{"variant tag '" + declared.tag + "' not found"};
        if(tag->label == nullptr)
            throw decode_error{"variant tag '" + declared.tag + "' value " +
                               std::to_string(tag->value) + " has no label"};
        for(const auto& itr : declared.members)
        {
            if(itr.name == *tag->label)
            {
                auto child =
                    name.empty() ? field_name(itr.name) : name + "." + field_name(itr.name);
                decode(*itr.type, child, itr.name, out, state);
                return;
            }
        }
        throw decode_error{"variant has no option '" + *tag->label + "'"};
    }

    void decode_array(const type& declared, const std::string& name, std::vector<field_value>* out)
    {
        auto length = declared.length;
        if(declared.kind == type_kind::sequence)
        {
            const auto* entry = lookup(declared.length_name);
            if(entry == nullptr)
                throw decode_error{"sequence length '" + declared.length_name + "' not found"};
            length = entry->value;
        }
        const auto& element = *declared.element;
        align(declared.alignment);

        auto value = field_value{};
        value.name = name;
        if(element.kind == type_kind::integer && element.size == 8 && element.is_text)
        {
            // Text: up to the first NUL.
            need(length * 8);
            value.kind      = value_kind::string;
            const auto* src = reinterpret_cast<const char*>(m_data + (m_pos / 8));  // NOLINT
            value.str.assign(src, strnlen(src, length));
            m_pos += length * 8;
        }
        else if(element.kind == type_kind::integer)
        {
            value.kind = value_kind::unsigned_array;
            value.elements.reserve(length);
            for(uint64_t i = 0; i < length; ++i)
                value.elements.push_back(read_integer(element));
        }
        else
        {
            // Not written by LTTng-UST for the providers this reads; decode elements as
            // separate fields so their size is still honored.
            for(uint64_t i = 0; i < length; ++i)
                decode(element, name + "[" + std::to_string(i) + "]", name, out, {});
            return;
        }
        out->push_back(std::move(value));
    }
};

uint64_t
to_ns(const clock_description& clock, uint64_t cycles)
{
    if(clock.frequency == 1000000000) return cycles;
    auto seconds = cycles / clock.frequency;
    auto rest    = cycles % clock.frequency;
    return (seconds * 1000000000) + (rest * 1000000000 / clock.frequency);
}

const field_value*
find_value(const std::vector<field_value>& values, std::string_view name)
{
    for(const auto& itr : values)
        if(itr.name == name) return &itr;
    return nullptr;
}

// Iterates over the events of one stream file, packet by packet.
class stream_reader
{
public:
    stream_reader(const trace* source, std::string path)
    : m_source{source}
    , m_path{std::move(path)}
    {}

    const std::string&  path() const { return m_path; }
    uint64_t            timestamp() const { return m_record.timestamp; }
    const event_record& record() const { return m_record; }

    bool open(std::string* error) { return m_file.open(m_path, error); }

    // Read the packet headers only.
    bool scan(const discarded_callback& on_discarded, read_statistics* stats, std::string* error)
    {
        try
        {
            while(m_packet_end < static_cast<uint64_t>(m_file.size()) * 8)
                open_packet(on_discarded, stats);
            return true;
        } catch(const decode_error& e)
        {
            *error = m_path + " at byte " + std::to_string(m_dec.position() / 8) + ": " + e.what();
            return false;
        }
    }

    // Decode the next event; false at the end of the stream (error set on failure).
    bool next(const discarded_callback& on_discarded, read_statistics* stats, std::string* error)
    {
        try
        {
            while(m_dec.position() >= m_content_end)
            {
                if(m_packet_end >= static_cast<uint64_t>(m_file.size()) * 8) return false;
                open_packet(on_discarded, stats);
            }
            decode_event();
            ++stats->events;
            return true;
        } catch(const decode_error& e)
        {
            *error = m_path + " at byte " + std::to_string(m_dec.position() / 8) + ": " + e.what();
            return false;
        }
    }

private:
    const trace*             m_source      = nullptr;
    std::string              m_path        = {};
    mapped_file              m_file        = {};
    decoder                  m_dec         = {nullptr, 0};
    uint64_t                 m_packet_end  = 0;  // bits
    uint64_t                 m_content_end = 0;  // bits
    uint64_t                 m_clock       = 0;
    uint64_t                 m_discarded   = 0;
    uint64_t                 m_last_end    = 0;
    bool                     m_has_packet  = false;
    event_record             m_record      = {};
    std::vector<field_value> m_header      = {};

    void open_packet(const discarded_callback& on_discarded, read_statistics* stats)
    {
        const auto& desc  = m_source->description;
        auto        start = m_packet_end;
        auto        size  = static_cast<uint64_t>(m_file.size()) * 8;
        m_dec             = decoder{m_file.data(), size};
        m_dec.seek(start);
        m_dec.clear_scope();

        auto values = std::vector<field_value>{};
        if(desc.packet_header) m_dec.decode(*desc.packet_header, "", "", &values, {});
        if(const auto* magic = find_value(values, "magic");
           magic != nullptr && magic->u64 != packet_magic)
            throw decode_error{"bad packet magic"};
        auto stream_id = uint64_t{0};
        if(const auto* id = find_value(values, "stream_id"); id != nullptr) stream_id = id->u64;
        auto stream = desc.streams.find(stream_id);
        if(stream == desc.streams.end())
            throw decode_error{"unknown stream id " + std::to_string(stream_id)};
        m_stream = &stream->second;

        values.clear();
        if(m_stream->packet_context) m_dec.decode(*m_stream->packet_context, "", "", &values, {});

        auto packet_bits  = size - start;
        auto content_bits = packet_bits;
        if(const auto* v = find_value(values, "packet_size"); v != nullptr) packet_bits = v->u64;
        if(const auto* v = find_value(values, "content_size"); v != nullptr) content_bits = v->u64;
        if(packet_bits == 0 || content_bits > packet_bits || start + packet_bits > size)
            throw decode_error{"packet sizes out of range"};
        m_packet_end  = start + packet_bits;
        m_content_end = start + content_bits;
        m_dec.set_limit(m_content_end);

        if(const auto* v = find_value(values, "timestamp_begin"); v != nullptr) m_clock = v->u64;
        auto packet_end_ts = m_clock;
        if(const auto* v = find_value(values, "timestamp_end"); v != nullptr)
            packet_end_ts = v->u64;
        if(const auto* v = find_value(values, "events_discarded"); v != nullptr)
        {
            // A running count per stream (it can wrap at the field's width).
            if(v->u64 != m_discarded)
            {
                auto record   = discarded_record{};
                record.source = m_source;
                record.stream = m_path;
                record.count  = v->u64 - m_discarded;
                record.begin  = to_ns(desc.clock, m_has_packet ? m_last_end : m_clock);
                record.end    = to_ns(desc.clock, packet_end_ts);
                stats->discarded += record.count;
                if(on_discarded) on_discarded(record);
                m_discarded = v->u64;
            }
        }
        m_last_end   = packet_end_ts;
        m_has_packet = true;
        ++stats->packets;
    }

    void decode_event()
    {
        const auto& desc = m_source->description;
        m_dec.clear_scope();
        m_header.clear();
        auto state        = section_state{};
        state.track_clock = true;
        state.clock       = &m_clock;
        if(m_stream->event_header) m_dec.decode(*m_stream->event_header, "", "", &m_header, state);

        // The id is the last "id" of the header (an extended header's id follows the
        // compact one).
        auto id = uint64_t{0};
        for(const auto& itr : m_header)
        {
            const auto& name = itr.name;
            if(name == "id" || (name.size() > 3 && name.compare(name.size() - 3, 3, ".id") == 0))
                id = itr.u64;
        }
        auto event = desc.events.find(id);
        if(event == desc.events.end()) throw decode_error{"unknown event id " + std::to_string(id)};

        m_record.source      = m_source;
        m_record.description = &event->second;
        m_record.timestamp   = to_ns(desc.clock, m_clock);
        m_record.context.clear();
        m_record.fields.clear();
        if(m_stream->event_context)
            m_dec.decode(*m_stream->event_context, "", "", &m_record.context, {});
        if(event->second.context)
            m_dec.decode(*event->second.context, "", "", &m_record.context, {});
        if(event->second.fields) m_dec.decode(*event->second.fields, "", "", &m_record.fields, {});
    }

    const stream_description* m_stream = nullptr;
};

bool
read_file(const std::string& path, std::string* contents, std::string* error)
{
    auto input = std::ifstream{path, std::ios::binary};
    if(!input)
    {
        *error = "cannot open " + path;
        return false;
    }
    auto buffer = std::ostringstream{};
    buffer << input.rdbuf();
    *contents = buffer.str();
    return true;
}

bool
open_streams(const std::vector<trace_ptr>&                traces,
             std::vector<std::unique_ptr<stream_reader>>* streams,
             read_statistics*                             stats,
             std::string*                                 error)
{
    for(const auto& source : traces)
    {
        ++stats->traces;
        auto files = std::vector<std::string>{};
        auto ec    = std::error_code{};
        for(const auto& entry : fs::directory_iterator{source->path, ec})
        {
            auto name = entry.path().filename().string();
            if(entry.is_regular_file(ec) && name != "metadata" && name.front() != '.')
                files.push_back(entry.path().string());
        }
        if(ec)
        {
            *error = "cannot list " + source->path + ": " + ec.message();
            return false;
        }
        std::sort(files.begin(), files.end());
        for(auto& file : files)
        {
            auto stream = std::make_unique<stream_reader>(source.get(), std::move(file));
            if(!stream->open(error)) return false;
            ++stats->streams;
            streams->push_back(std::move(stream));
        }
    }
    return true;
}
}  // namespace

std::vector<std::string>
find_traces(const std::string& root)
{
    auto result = std::vector<std::string>{};
    auto ec     = std::error_code{};
    if(fs::is_regular_file(fs::path{root} / "metadata", ec)) result.push_back(root);
    for(auto itr = fs::recursive_directory_iterator{root, ec};
        !ec && itr != fs::recursive_directory_iterator{};
        itr.increment(ec))
    {
        if(itr->is_directory(ec) && fs::is_regular_file(itr->path() / "metadata", ec))
            result.push_back(itr->path().string());
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

bool
load_trace(const std::string& path, trace* result, std::string* error)
{
    auto contents = std::string{};
    auto text     = std::string{};
    if(!read_file((fs::path{path} / "metadata").string(), &contents, error)) return false;
    if(!metadata_text(contents, &text, error) || !parse_metadata(text, &result->description, error))
    {
        *error = path + ": " + *error;
        return false;
    }
    result->path = path;
    return true;
}

bool
read_events(const std::vector<trace_ptr>& traces,
            const event_callback&         on_event,
            const discarded_callback&     on_discarded,
            read_statistics*              statistics,
            std::string*                  error)
{
    error->clear();
    auto stats   = read_statistics{};
    auto streams = std::vector<std::unique_ptr<stream_reader>>{};
    if(!open_streams(traces, &streams, &stats, error)) return false;

    // Merge by (timestamp, stream index).
    using entry  = std::pair<uint64_t, size_t>;
    auto pending = std::priority_queue<entry, std::vector<entry>, std::greater<>>{};
    for(size_t i = 0; i < streams.size(); ++i)
    {
        if(streams[i]->next(on_discarded, &stats, error))
            pending.emplace(streams[i]->timestamp(), i);
        else if(!error->empty())
            return false;
    }
    while(!pending.empty())
    {
        auto index = pending.top().second;
        pending.pop();
        auto& stream = *streams[index];
        if(on_event) on_event(stream.record());
        if(stream.next(on_discarded, &stats, error))
            pending.emplace(stream.timestamp(), index);
        else if(!error->empty())
            return false;
    }
    if(statistics != nullptr) *statistics = stats;
    return true;
}

bool
scan_discarded(const std::vector<trace_ptr>& traces,
               const discarded_callback&     on_discarded,
               std::string*                  error)
{
    error->clear();
    auto stats   = read_statistics{};
    auto streams = std::vector<std::unique_ptr<stream_reader>>{};
    if(!open_streams(traces, &streams, &stats, error)) return false;
    for(auto& stream : streams)
        if(!stream->scan(on_discarded, &stats, error)) return false;
    return true;
}

const field_value*
find_field(const std::vector<field_value>& values, const std::string& name)
{
    return find_value(values, name);
}
}  // namespace ctf
}  // namespace collector
}  // namespace rocprofiler
