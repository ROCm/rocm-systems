// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kineto_loader.hpp"
#include "sql_cancel.hpp"
#include "utility.hpp"

#include <nlohmann/json.hpp>
#include <zlib.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <functional>
#include <istream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <streambuf>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace profiler_hub::pytorch
{
namespace
{
using json_t = nlohmann::json;

class input_buf final : public std::streambuf
{
public:
    explicit input_buf(const std::string&    path,
                       std::function<bool()> cancelled = {},
                       std::size_t           chunk     = 1 << 20)
    : m_fp{ std::fopen(path.c_str(), "rb") }
    , m_cancelled{ std::move(cancelled) }
    , m_in(chunk)
    , m_out(chunk)
    {
        if(!m_fp) throw std::runtime_error("cannot open " + path);

        unsigned char magic[2] = {};
        const auto    n        = std::fread(magic, 1, sizeof(magic), m_fp.get());
        std::fseek(m_fp.get(), 0, SEEK_SET);
        m_gzip = n == 2 && magic[0] == 0x1f && magic[1] == 0x8b;
        if(m_gzip)
        {
            if(inflateInit2(&m_zs, 15 + 16) != Z_OK)
            {
                throw std::runtime_error("inflateInit2 failed");
            }
            m_zs_ready = true;
        }
        setg(m_out.data(), m_out.data(), m_out.data());
    }

    ~input_buf() override
    {
        if(m_zs_ready) inflateEnd(&m_zs);
    }

    input_buf(const input_buf&)            = delete;
    input_buf& operator=(const input_buf&) = delete;
    input_buf(input_buf&&)                 = delete;
    input_buf& operator=(input_buf&&)      = delete;

    [[nodiscard]] bool               gzip() const { return m_gzip; }
    [[nodiscard]] const std::string& error() const { return m_error; }
    [[nodiscard]] std::uint64_t      inflate_ms() const { return m_inflate_ms; }

protected:
    int_type underflow() override
    {
        if(gptr() < egptr()) return traits_type::to_int_type(*gptr());
        if(m_done) return traits_type::eof();
        if(m_cancelled && m_cancelled()) throw operation_cancelled{};

        const auto        t0 = common::steady_clock_t::now();
        const std::size_t n  = m_gzip ? fill_gzip() : fill_plain();
        if(m_gzip) m_inflate_ms += common::elapsed_ms(t0);

        if(n == 0)
        {
            m_done = true;
            return traits_type::eof();
        }
        setg(m_out.data(), m_out.data(), m_out.data() + n);
        return traits_type::to_int_type(*gptr());
    }

private:
    std::size_t fill_plain()
    {
        return std::fread(m_out.data(), 1, m_out.size(), m_fp.get());
    }

    std::size_t fill_gzip()
    {
        while(true)
        {
            if(m_zs.avail_in == 0)
            {
                m_zs.avail_in = static_cast<uInt>(
                    std::fread(m_in.data(), 1, m_in.size(), m_fp.get()));
                m_zs.next_in = m_in.data();
                if(m_zs.avail_in == 0)
                {
                    if(!m_member_ended) m_error = "unexpected end of gzip stream";
                    return 0;
                }
            }
            if(m_member_ended)
            {
                while(m_zs.avail_in != 0 && *m_zs.next_in == 0)
                {
                    ++m_zs.next_in;
                    --m_zs.avail_in;
                }
                if(m_zs.avail_in == 0) continue;
                inflateReset(&m_zs);
                m_member_ended = false;
            }

            m_zs.next_out  = reinterpret_cast<Bytef*>(m_out.data());
            m_zs.avail_out = static_cast<uInt>(m_out.size());
            const int r    = inflate(&m_zs, Z_NO_FLUSH);
            if(r != Z_OK && r != Z_STREAM_END)
            {
                m_error = m_zs.msg != nullptr ? m_zs.msg : "inflate failed";
                return 0;
            }
            if(r == Z_STREAM_END) m_member_ended = true;

            const std::size_t n = m_out.size() - m_zs.avail_out;
            if(n != 0) return n;
        }
    }

    std::unique_ptr<std::FILE, common::file_closer> m_fp;
    std::function<bool()>                           m_cancelled;
    std::vector<unsigned char>                      m_in;
    std::vector<char>                               m_out;
    z_stream                                        m_zs{};
    bool                                            m_zs_ready     = false;
    bool                                            m_gzip         = false;
    bool                                            m_member_ended = false;
    bool                                            m_done         = false;
    std::string                                     m_error;
    std::uint64_t                                   m_inflate_ms = 0;
};

bool
icontains_launch(std::string_view s)
{
    constexpr std::string_view needle = "launch";
    return std::search(
               s.begin(), s.end(), needle.begin(), needle.end(), [](char a, char b) {
                   return std::tolower(static_cast<unsigned char>(a)) == b;
               }) != s.end();
}

// Bound on a ts or dur in ns, so that start + duration cannot wrap.
constexpr std::int64_t k_max_time_ns = std::numeric_limits<std::int64_t>::max() / 2;

std::int64_t
signed_ns(std::uint64_t magnitude, bool negative)
{
    if(magnitude > static_cast<std::uint64_t>(k_max_time_ns))
    {
        throw std::runtime_error("JSON: timestamp out of range");
    }
    return negative ? -static_cast<std::int64_t>(magnitude)
                    : static_cast<std::int64_t>(magnitude);
}

std::int64_t
integer_us_to_ns(std::uint64_t magnitude, bool negative)
{
    if(magnitude > static_cast<std::uint64_t>(k_max_time_ns) / 1000)
    {
        throw std::runtime_error("JSON: timestamp out of range");
    }
    return signed_ns(magnitude * 1000, negative);
}

// round-half-away-from-zero(us * 1000), read off the digits of the number's source text:
// the first (integer digits + exponent + 3) digits are whole nanoseconds, the next one
// decides the rounding.
std::int64_t
text_us_to_ns(std::string_view s)
{
    std::size_t pos      = 0;
    bool        negative = false;
    if(pos < s.size() && (s[pos] == '-' || s[pos] == '+')) negative = s[pos++] == '-';

    const auto exp_at = s.find_first_of("eE", pos);
    const auto mant =
        s.substr(pos, exp_at == std::string_view::npos ? exp_at : exp_at - pos);

    long exponent = 0;
    if(exp_at != std::string_view::npos)
    {
        bool exp_negative = false;
        auto i            = exp_at + 1;
        if(i < s.size() && (s[i] == '-' || s[i] == '+')) exp_negative = s[i++] == '-';
        for(; i < s.size(); ++i)
        {
            exponent = std::min(exponent * 10 + (s[i] - '0'), 100000L);
        }
        if(exp_negative) exponent = -exponent;
    }

    const auto dot = mant.find('.');
    const long integer_digits =
        static_cast<long>(dot == std::string_view::npos ? mant.size() : dot);
    const long whole_digits = integer_digits + exponent + 3;

    std::uint64_t ns       = 0;
    long          taken    = 0;
    bool          round_up = false;
    for(const char c : mant)
    {
        if(c == '.') continue;
        ++taken;
        if(taken <= whole_digits)
        {
            const auto digit = static_cast<std::uint64_t>(c - '0');
            if(ns > (static_cast<std::uint64_t>(k_max_time_ns) - digit) / 10)
            {
                throw std::runtime_error("JSON: timestamp out of range");
            }
            ns = ns * 10 + digit;
        }
        else
        {
            round_up = taken == whole_digits + 1 && c >= '5';
            break;
        }
    }
    for(; taken < whole_digits && ns != 0; ++taken)
    {
        if(ns > static_cast<std::uint64_t>(k_max_time_ns) / 10)
        {
            throw std::runtime_error("JSON: timestamp out of range");
        }
        ns *= 10;
    }
    return signed_ns(ns + (round_up ? 1 : 0), negative);
}

// Re-serialises the SAX events of one JSON value as compact text. Keys keep their order
// and a number keeps the token the file wrote.
class json_text_t
{
public:
    void begin(std::string& out)
    {
        m_out  = &out;
        m_done = false;
        m_open.clear();
        m_out->clear();
    }
    void end() { m_out = nullptr; }

    [[nodiscard]] bool active() const { return m_out != nullptr; }
    [[nodiscard]] bool complete() const { return m_done; }

    void token(std::string_view text)
    {
        prefix();
        m_out->append(text);
        m_done = m_open.empty();
    }
    void string(std::string_view text)
    {
        prefix();
        quote(text);
        m_done = m_open.empty();
    }
    void key(std::string_view text)
    {
        auto& top = m_open.back();
        if(!top.first) m_out->push_back(',');
        top.first = false;
        quote(text);
        m_out->push_back(':');
    }
    void open(char bracket)
    {
        prefix();
        m_out->push_back(bracket);
        m_open.push_back({ bracket == '{', true });
    }
    void close()
    {
        m_out->push_back(m_open.back().object ? '}' : ']');
        m_open.pop_back();
        m_done = m_open.empty();
    }

private:
    struct level_t
    {
        bool object;
        bool first;
    };

    void prefix()
    {
        if(m_open.empty() || m_open.back().object) return;
        if(!m_open.back().first) m_out->push_back(',');
        m_open.back().first = false;
    }

    void quote(std::string_view text) { append_json_string(*m_out, text); }

    std::string*         m_out  = nullptr;
    bool                 m_done = false;
    std::vector<level_t> m_open;
};

// Depth map: 1 root members, 2 traceEvents / deviceProperties array, 3 an event or device
// object, 4 args members.
class sax_handler
{
public:
    using number_integer_t  = json_t::number_integer_t;
    using number_unsigned_t = json_t::number_unsigned_t;
    using number_float_t    = json_t::number_float_t;
    using string_t          = json_t::string_t;
    using binary_t          = json_t::binary_t;

    explicit sax_handler(loaded_trace_t& t)
    : m_t{ t }
    {}

    const std::string& parse_error_message() const { return m_parse_error; }

    bool null()
    {
        if(m_text.active())
            m_text.token("null");
        else if(member_wanted())
            add_member("null");
        return true;
    }
    bool boolean(bool val)
    {
        if(m_text.active())
            m_text.token(val ? "true" : "false");
        else if(member_wanted())
            add_member(val ? "true" : "false");
        return true;
    }
    static bool binary(binary_t& /*val*/) { return true; }

    bool number_integer(number_integer_t v)
    {
        if(m_text.active())
            m_text.token(std::to_string(v));
        else if(member_wanted())
            add_member(std::to_string(v));
        if(at_time_key())
        {
            const auto magnitude = v < 0 ? ~static_cast<std::uint64_t>(v) + 1
                                         : static_cast<std::uint64_t>(v);
            set_time(integer_us_to_ns(magnitude, v < 0));
            return true;
        }
        on_number(static_cast<std::int64_t>(v));
        return true;
    }
    bool number_unsigned(number_unsigned_t v)
    {
        if(m_text.active())
            m_text.token(std::to_string(v));
        else if(member_wanted())
            add_member(std::to_string(v));
        if(at_time_key())
        {
            set_time(integer_us_to_ns(v, false));
            return true;
        }
        if(m_depth == 1 && m_k1 == "baseTimeNanoseconds") m_t.base_time_ns = v;
        on_number(static_cast<std::int64_t>(v));
        return true;
    }
    bool number_float(number_float_t v, const string_t& text)
    {
        if(m_text.active())
            m_text.token(text);
        else if(member_wanted())
            add_member(text);
        if(at_time_key())
        {
            set_time(text_us_to_ns(text));
            return true;
        }
        on_number(static_cast<std::int64_t>(v));
        return true;
    }

    bool string(string_t& s)
    {
        if(m_text.active())
        {
            m_text.string(s);
        }
        else if(member_wanted())
        {
            std::string quoted;
            append_json_string(quoted, s);
            add_member(quoted);
        }
        if(m_depth == 3 && m_in_event)
        {
            if(m_k3 == "name")
            {
                m_cur.name = s;
            }
            else if(m_k3 == "ph")
            {
                m_cur.ph = s.empty() ? '\0' : s[0];
            }
            else if(m_k3 == "cat")
            {
                m_cur.cat = s;
            }
            else if(m_k3 == "pid")
            {
                m_cur.pid = lane_id(s);
            }
            else if(m_k3 == "tid")
            {
                m_cur.tid = lane_id(s);
            }
        }
        else if(m_depth == 4 && m_in_event && m_k3 == "args" && m_k4 == "name")
        {
            m_cur.args_name = s;
        }
        else if(m_depth == 3 && m_in_device && m_k3 == "name")
        {
            m_dev_name = s;
        }
        else if(m_depth == 1 && m_k1 == "host_name")
        {
            m_t.host_name = s;
        }
        return true;
    }

    bool start_object(std::size_t /*elements*/)
    {
        const bool nested = m_text.active();
        if(nested)
            m_text.open('{');
        else if(member_wanted())
            begin_member('{');
        ++m_depth;
        if(m_depth == 3 && m_k1 == "traceEvents")
        {
            m_in_event = true;
            m_cur      = {};
        }
        else if(!nested && m_depth == 4 && m_in_event && m_k3 == "args")
        {
            m_owner = capture_t::args;
            m_text.begin(m_args_text);
            m_text.open('{');
        }
        else if(m_depth == 3 && m_k1 == "deviceProperties")
        {
            m_in_device = true;
            m_dev_id    = -1;
            m_dev_name.clear();
            m_owner = capture_t::device;
            m_text.begin(m_dev_text);
            m_text.open('{');
        }
        return true;
    }

    bool end_object()
    {
        if(close_text()) finish_capture();
        if(m_depth == 3 && m_in_event)
        {
            finish_event();
            m_in_event = false;
        }
        else if(m_depth == 3 && m_in_device)
        {
            finish_device();
            m_in_device = false;
        }
        --m_depth;
        return true;
    }

    bool start_array(std::size_t /*elements*/)
    {
        if(m_text.active())
            m_text.open('[');
        else if(member_wanted())
            begin_member('[');
        ++m_depth;
        return true;
    }
    bool end_array()
    {
        if(close_text()) finish_capture();
        --m_depth;
        return true;
    }

    bool key(string_t& s)
    {
        if(m_text.active()) m_text.key(s);
        if(m_depth == 1)
        {
            m_k1 = s;
        }
        else if(m_depth == 3)
        {
            m_k3 = s;
        }
        else if(m_depth == 4)
        {
            m_k4 = s;
        }
        return true;
    }

    bool parse_error(std::size_t position,
                     const std::string& /*last_token*/,
                     const json_t::exception& e)
    {
        m_parse_error =
            std::string{ e.what() } + " (byte " + std::to_string(position) + ")";
        return false;
    }

private:
    struct current_t
    {
        std::string     name, cat, args_name;
        char            ph     = '\0';
        bool            has_ts = false, has_dur = false;
        std::int64_t    ts_ns = 0, dur_ns = 0;
        std::int64_t    pid = 0, tid = 0, stream = -1, device = -1;
        std::int64_t    corr = -1, ext = -1;
        bool            has_args = false;
        memory_sample_t memory;
        std::int64_t    sort_index  = 0;
        bool            has_sort    = false;
        std::int64_t    flow_id     = 0;
        bool            has_flow_id = false;
    };

    enum class capture_t
    {
        none,
        args,    // an event's args object
        device,  // a deviceProperties entry
        member,  // a top-level member other than the ones read for something else
    };

    // The members that are not read for anything else are kept as JSON text, in file
    // order.
    [[nodiscard]] bool member_wanted() const
    {
        return m_depth == 1 && m_k1 != "traceEvents" && m_k1 != "deviceProperties" &&
               m_k1 != "schemaVersion" && m_k1 != "baseTimeNanoseconds" &&
               m_k1 != "host_name";
    }

    void add_member(std::string_view value)
    {
        if(!m_t.trace_info.empty()) m_t.trace_info.push_back(',');
        append_json_string(m_t.trace_info, m_k1);
        m_t.trace_info.push_back(':');
        m_t.trace_info.append(value);
    }

    void begin_member(char bracket)
    {
        m_owner = capture_t::member;
        m_text.begin(m_member_text);
        m_text.open(bracket);
    }

    void finish_capture()
    {
        if(m_owner == capture_t::args) m_cur.has_args = true;
        if(m_owner == capture_t::member) add_member(m_member_text);
        m_owner = capture_t::none;
    }

    // True when the value being re-serialised just ended.
    bool close_text()
    {
        if(!m_text.active()) return false;
        m_text.close();
        if(!m_text.complete()) return false;
        m_text.end();
        return true;
    }

    bool at_time_key() const
    {
        return m_depth == 3 && m_in_event && (m_k3 == "ts" || m_k3 == "dur");
    }

    void set_time(std::int64_t ns)
    {
        if(m_k3 == "ts")
        {
            m_cur.ts_ns  = ns;
            m_cur.has_ts = true;
        }
        else
        {
            m_cur.dur_ns  = ns;
            m_cur.has_dur = true;
        }
    }

    void on_number(std::int64_t i)
    {
        if(m_depth == 3 && m_in_event)
        {
            if(m_k3 == "pid")
            {
                m_cur.pid = i;
            }
            else if(m_k3 == "tid")
            {
                m_cur.tid = i;
            }
            else if(m_k3 == "id")
            {
                m_cur.flow_id     = i;
                m_cur.has_flow_id = true;
            }
        }
        else if(m_depth == 4 && m_in_event && m_k3 == "args")
        {
            if(m_k4 == "correlation")
            {
                m_cur.corr = i;
            }
            else if(m_k4 == "External id")
            {
                m_cur.ext = i;
            }
            else if(m_k4 == "stream")
            {
                m_cur.stream = i;
            }
            else if(m_k4 == "device")
            {
                m_cur.device = i;
            }
            else if(m_k4 == "Addr")
            {
                m_cur.memory.address = static_cast<std::uint64_t>(i);
            }
            else if(m_k4 == "Bytes")
            {
                m_cur.memory.bytes = i;
            }
            else if(m_k4 == "Device Id")
            {
                m_cur.memory.device_id = i;
            }
            else if(m_k4 == "Device Type")
            {
                m_cur.memory.device_type = i;
            }
            else if(m_k4 == "Total Allocated")
            {
                m_cur.memory.allocated = i;
            }
            else if(m_k4 == "Total Reserved")
            {
                m_cur.memory.reserved = i;
            }
            else if(m_k4 == "sort_index")
            {
                m_cur.sort_index = i;
                m_cur.has_sort   = true;
            }
        }
        else if(m_depth == 3 && m_in_device && m_k3 == "id")
        {
            m_dev_id = i;
        }
        else if(m_depth == 1 && m_k1 == "schemaVersion")
        {
            m_t.schema_version = static_cast<std::int32_t>(i);
        }
    }

    // A text-valued pid or tid ("Spans", "PyTorch Profiler") has no OS id. Each distinct
    // string gets the next number counting down from INT32_MAX, in order of first
    // appearance.
    std::int64_t lane_id(const std::string& text)
    {
        const auto [it, added] = m_lane_ids.try_emplace(text, m_next_lane);
        if(added)
        {
            m_t.lane_names.emplace(m_next_lane, text);
            --m_next_lane;
        }
        return it->second;
    }

    static std::uint32_t intern(std::unordered_map<std::string, std::uint32_t>& ids,
                                std::vector<std::string>&                       list,
                                const std::string&                              s)
    {
        const auto it = ids.find(s);
        if(it != ids.end()) return it->second;
        const auto id = static_cast<std::uint32_t>(list.size());
        ids.emplace(s, id);
        list.push_back(s);
        return id;
    }

    category_class_t class_of(const std::string& cat) const
    {
        const auto it = m_cat_ids.find(cat);
        return it != m_cat_ids.end() ? m_t.cat_classes[it->second]
                                     : classify_category(cat);
    }

    // Same link-key choice as TraceLens: the first launch event decides it for the whole
    // trace.
    void maybe_decide_link_key()
    {
        if(m_key_decided) return;
        if(!class_of(m_cur.cat).launch_site) return;
        if(!icontains_launch(m_cur.name)) return;

        m_key_decided           = true;
        m_t.link_by_correlation = m_cur.corr >= 0;
    }

    // Moves the current event's args into the arena and gives their place; an empty
    // object is not kept.
    std::pair<std::uint64_t, std::uint32_t> keep_args()
    {
        if(!m_cur.has_args || m_args_text.size() <= 2) return { 0, 0 };
        const std::pair<std::uint64_t, std::uint32_t> where{
            m_t.args.size(), static_cast<std::uint32_t>(m_args_text.size())
        };
        m_t.args.append(m_args_text);
        return where;
    }

    void add_flow()
    {
        const bool wanted = m_cur.cat == "fwdbwd" || m_cur.cat == "forward_backward" ||
                            m_cur.cat == "critical_path" ||
                            m_cur.cat.starts_with("async_");
        if(!wanted || !m_cur.has_flow_id || !m_cur.has_ts) return;
        flow_t flow;
        flow.phase                             = m_cur.ph;
        flow.id                                = m_cur.flow_id;
        flow.ts_ns                             = m_cur.ts_ns;
        flow.pid                               = m_cur.pid;
        flow.tid                               = m_cur.tid;
        flow.category                          = m_cur.cat;
        std::tie(flow.args_off, flow.args_len) = keep_args();
        m_t.flows.push_back(std::move(flow));
    }

    void add_marker()
    {
        marker_t marker;
        marker.ts_ns                               = m_cur.ts_ns;
        marker.pid                                 = m_cur.pid;
        marker.tid                                 = m_cur.tid;
        marker.name                                = m_cur.name;
        std::tie(marker.args_off, marker.args_len) = keep_args();
        m_t.markers.push_back(std::move(marker));
    }

    std::uint32_t add_event(phase_t phase)
    {
        event_t e{};
        e.name_id = intern(m_name_ids, m_t.names, m_cur.name);
        e.cat_id  = intern(m_cat_ids, m_t.cats, m_cur.cat);
        if(m_t.cat_classes.size() < m_t.cats.size())
        {
            m_t.cat_classes.push_back(classify_category(m_cur.cat));
        }
        e.pid                            = m_cur.pid;
        e.tid                            = m_cur.tid;
        e.stream                         = m_cur.stream;
        e.device                         = m_cur.device;
        e.ts_ns                          = m_cur.ts_ns;
        e.dur_ns                         = m_cur.dur_ns;
        e.corr                           = m_cur.corr;
        e.ext                            = m_cur.ext;
        e.phase                          = phase;
        std::tie(e.args_off, e.args_len) = keep_args();
        m_t.events.push_back(e);
        return static_cast<std::uint32_t>(m_t.events.size() - 1);
    }

    void finish_event()
    {
        auto& d = m_t.diag;
        ++d.total;
        switch(m_cur.ph)
        {
            case 'X':
            {
                ++d.x_events;
                maybe_decide_link_key();
                if(!m_cur.has_ts || !m_cur.has_dur)
                {
                    ++d.bad_events;
                    return;
                }
                add_event(phase_t::complete);
                break;
            }
            case 'i':
                if(m_cur.name == "[memory]" && m_cur.has_ts)
                {
                    ++d.memory_events;
                    m_cur.memory.event = add_event(phase_t::memory);
                    m_t.memory.push_back(m_cur.memory);
                }
                else
                {
                    ++d.other_events;
                    if(m_cur.has_ts) add_marker();
                }
                break;
            case 's':
                ++d.flow_s;
                add_flow();
                break;
            case 'f':
                ++d.flow_f;
                add_flow();
                break;
            case 'M':
                ++d.meta_events;
                if(m_cur.name == "thread_name")
                {
                    m_t.thread_names[{ m_cur.pid, m_cur.tid }] = m_cur.args_name;
                }
                else if(m_cur.name == "process_name")
                {
                    m_t.process_names[m_cur.pid] = m_cur.args_name;
                }
                else if(m_cur.name == "process_sort_index" && m_cur.has_sort)
                {
                    m_t.process_sort[m_cur.pid] = m_cur.sort_index;
                }
                else if(m_cur.name == "thread_sort_index" && m_cur.has_sort)
                {
                    m_t.thread_sort[{ m_cur.pid, m_cur.tid }] = m_cur.sort_index;
                }
                break;
            default: ++d.other_events; break;
        }
    }

    void finish_device()
    {
        if(m_dev_id < 0) return;
        m_t.device_names[m_dev_id] = m_dev_name;
        m_t.device_json[m_dev_id]  = m_dev_text;
    }

    loaded_trace_t& m_t;
    int             m_depth = 0;
    std::string     m_k1, m_k3, m_k4;
    bool            m_in_event  = false;
    bool            m_in_device = false;
    current_t       m_cur;
    std::int64_t    m_dev_id = -1;
    std::string     m_dev_name;
    std::string     m_parse_error;

    std::unordered_map<std::string, std::uint32_t> m_name_ids, m_cat_ids;
    bool                                           m_key_decided = false;
    json_text_t                                    m_text;
    std::string                                    m_args_text;
    std::string                                    m_dev_text;
    std::string                                    m_member_text;
    capture_t                                      m_owner = capture_t::none;
    std::unordered_map<std::string, std::int64_t>  m_lane_ids;
    std::int64_t m_next_lane = std::numeric_limits<std::int32_t>::max();
};
}  // namespace

void
append_json_string(std::string& out, std::string_view text)
{
    constexpr std::string_view hex = "0123456789abcdef";
    out.push_back('"');
    for(const char c : text)
    {
        switch(c)
        {
            case '"': out.append("\\\""); break;
            case '\\': out.append("\\\\"); break;
            case '\b': out.append("\\b"); break;
            case '\f': out.append("\\f"); break;
            case '\n': out.append("\\n"); break;
            case '\r': out.append("\\r"); break;
            case '\t': out.append("\\t"); break;
            default:
                if(static_cast<unsigned char>(c) < 0x20)
                {
                    out.append("\\u00");
                    out.push_back(hex[static_cast<unsigned char>(c) >> 4]);
                    out.push_back(hex[static_cast<unsigned char>(c) & 0xf]);
                }
                else
                {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

loaded_trace_t
load_kineto(const std::string& path, std::function<bool()> cancelled)
{
    const auto t0 = common::steady_clock_t::now();

    loaded_trace_t t{};
    input_buf      buf{ path, std::move(cancelled) };
    std::istream   is{ &buf };
    sax_handler    handler{ t };

    bool ok = false;
    try
    {
        ok = json_t::sax_parse(is, &handler);
    } catch(const std::runtime_error&)
    {
        throw;
    } catch(const operation_cancelled&)
    {
        throw;
    } catch(const std::exception& e)
    {
        throw std::runtime_error(std::string{ "JSON: " } + e.what());
    }

    if(!buf.error().empty()) throw std::runtime_error("gzip: " + buf.error());
    if(!ok) throw std::runtime_error("JSON: " + handler.parse_error_message());

    for(const auto& [id, text] : t.lane_names)
    {
        t.process_names.emplace(id, text);
    }

    const auto total_ms = common::elapsed_ms(t0);
    t.gzip              = buf.gzip();
    t.diag.gunzip_ms    = buf.gzip() ? buf.inflate_ms() : 0;
    t.diag.parse_ms     = total_ms - t.diag.gunzip_ms;
    return t;
}
}  // namespace profiler_hub::pytorch
