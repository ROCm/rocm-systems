// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kineto_loader.hpp"
#include "utility.hpp"

#include <nlohmann/json.hpp>
#include <zlib.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <istream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <streambuf>
#include <string_view>
#include <unordered_map>

namespace profiler_hub::pytorch
{
namespace
{
using json_t = nlohmann::json;

class input_buf final : public std::streambuf
{
public:
    explicit input_buf(const std::string& path, std::size_t chunk = 1 << 20)
    : m_fp{ std::fopen(path.c_str(), "rb") }
    , m_in(chunk)
    , m_out(chunk)
    {
        if(!m_fp) throw std::runtime_error("cannot open " + path);

        unsigned char magic[2] = {};
        const auto    n        = std::fread(magic, 1, sizeof(magic), m_fp.get());
        std::rewind(m_fp.get());
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

    bool               gzip() const { return m_gzip; }
    const std::string& error() const { return m_error; }
    std::uint64_t      inflate_ms() const { return m_inflate_ms; }

protected:
    int_type underflow() override
    {
        if(gptr() < egptr()) return traits_type::to_int_type(*gptr());
        if(m_done) return traits_type::eof();

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

    bool null() { return true; }
    bool boolean(bool) { return true; }
    bool binary(binary_t&) { return true; }

    bool number_integer(number_integer_t v)
    {
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
        }
        else if(m_depth == 4 && m_in_event && m_k3 == "args" && m_k4 == "name")
        {
            m_cur.args_name = s;
        }
        else if(m_depth == 3 && m_in_device && m_k3 == "name")
        {
            m_dev_name = s;
        }
        return true;
    }

    bool start_object(std::size_t)
    {
        ++m_depth;
        if(m_depth == 3 && m_k1 == "traceEvents")
        {
            m_in_event = true;
            m_cur      = {};
        }
        else if(m_depth == 3 && m_k1 == "deviceProperties")
        {
            m_in_device = true;
            m_dev_id    = -1;
            m_dev_name.clear();
        }
        return true;
    }

    bool end_object()
    {
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

    bool start_array(std::size_t)
    {
        ++m_depth;
        return true;
    }
    bool end_array()
    {
        --m_depth;
        return true;
    }

    bool key(string_t& s)
    {
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

    bool parse_error(std::size_t position, const std::string&, const json_t::exception& e)
    {
        m_parse_error =
            std::string{ e.what() } + " (byte " + std::to_string(position) + ")";
        return false;
    }

private:
    struct current_t
    {
        std::string  name, cat, args_name;
        char         ph     = '\0';
        bool         has_ts = false, has_dur = false;
        std::int64_t ts_ns = 0, dur_ns = 0;
        std::int32_t pid = 0, tid = 0, stream = -1, device = -1;
        std::int64_t corr = -1, ext = -1;
    };
    struct pending_t
    {
        std::size_t  index;
        std::int64_t ext;
    };

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
                m_cur.pid = static_cast<std::int32_t>(i);
            }
            else if(m_k3 == "tid")
            {
                m_cur.tid = static_cast<std::int32_t>(i);
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
                m_cur.stream = static_cast<std::int32_t>(i);
            }
            else if(m_k4 == "device")
            {
                m_cur.device = static_cast<std::int32_t>(i);
            }
        }
        else if(m_depth == 3 && m_in_device && m_k3 == "id")
        {
            m_dev_id = static_cast<std::int32_t>(i);
        }
        else if(m_depth == 1 && m_k1 == "schemaVersion")
        {
            m_t.schema_version = static_cast<std::int32_t>(i);
        }
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

        m_key_decided = true;
        m_use_corr    = m_cur.corr >= 0;
        if(!m_use_corr)
        {
            for(const auto& p : m_pending)
            {
                m_t.events[p.index].correlation = p.ext;
            }
        }
        m_pending.clear();
        m_pending.shrink_to_fit();
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
                if(m_t.cats.size() >= 256 && m_cat_ids.find(m_cur.cat) == m_cat_ids.end())
                {
                    throw std::runtime_error("more than 255 distinct event categories");
                }

                event_t e{};
                e.name_id = intern(m_name_ids, m_t.names, m_cur.name);
                e.cat_id =
                    static_cast<std::uint8_t>(intern(m_cat_ids, m_t.cats, m_cur.cat));
                if(m_t.cat_classes.size() < m_t.cats.size())
                {
                    m_t.cat_classes.push_back(classify_category(m_cur.cat));
                }
                e.pid    = m_cur.pid;
                e.tid    = m_cur.tid;
                e.stream = m_cur.stream;
                e.device = m_cur.device;
                e.ts_ns  = m_cur.ts_ns;
                e.dur_ns = m_cur.dur_ns;
                if(m_key_decided)
                {
                    e.correlation = m_use_corr ? m_cur.corr : m_cur.ext;
                }
                else
                {
                    e.correlation = m_cur.corr;
                    if(m_cur.corr >= 0 || m_cur.ext >= 0)
                    {
                        m_pending.push_back({ m_t.events.size(), m_cur.ext });
                    }
                }
                m_t.events.push_back(e);
                break;
            }
            case 's': ++d.flow_s; break;
            case 'f': ++d.flow_f; break;
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
                break;
            default: ++d.other_events; break;
        }
    }

    void finish_device()
    {
        const auto idx =
            m_dev_id >= 0 ? static_cast<std::size_t>(m_dev_id) : m_t.device_names.size();
        if(m_t.device_names.size() <= idx) m_t.device_names.resize(idx + 1);
        m_t.device_names[idx] = m_dev_name;
    }

    loaded_trace_t& m_t;
    int             m_depth = 0;
    std::string     m_k1, m_k3, m_k4;
    bool            m_in_event  = false;
    bool            m_in_device = false;
    current_t       m_cur;
    std::int32_t    m_dev_id = -1;
    std::string     m_dev_name;
    std::string     m_parse_error;

    std::unordered_map<std::string, std::uint32_t> m_name_ids, m_cat_ids;
    bool                                           m_key_decided = false;
    bool                                           m_use_corr    = true;
    std::vector<pending_t>                         m_pending;
};
}  // namespace

loaded_trace_t
load_kineto(const std::string& path)
{
    const auto t0 = common::steady_clock_t::now();

    loaded_trace_t t{};
    input_buf      buf{ path };
    std::istream   is{ &buf };
    sax_handler    handler{ t };

    bool ok = false;
    try
    {
        ok = json_t::sax_parse(is, &handler);
    } catch(const std::runtime_error&)
    {
        throw;
    } catch(const std::exception& e)
    {
        throw std::runtime_error(std::string{ "JSON: " } + e.what());
    }

    if(!buf.error().empty()) throw std::runtime_error("gzip: " + buf.error());
    if(!ok) throw std::runtime_error("JSON: " + handler.parse_error_message());

    const auto total_ms = common::elapsed_ms(t0);
    t.gzip              = buf.gzip();
    t.diag.gunzip_ms    = buf.gzip() ? buf.inflate_ms() : 0;
    t.diag.parse_ms     = total_ms - t.diag.gunzip_ms;
    return t;
}
}  // namespace profiler_hub::pytorch
