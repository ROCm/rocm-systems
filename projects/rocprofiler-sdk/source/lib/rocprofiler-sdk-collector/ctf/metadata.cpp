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

#include "lib/rocprofiler-sdk-collector/ctf/metadata.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace collector
{
namespace ctf
{
std::string
field_name(const std::string& declared)
{
    return (!declared.empty() && declared.front() == '_') ? declared.substr(1) : declared;
}

namespace
{
constexpr uint32_t metadata_packet_magic       = 0x75D11D57;
constexpr size_t   metadata_packet_header_size = 37;  // bytes

uint32_t
read_u32_le(std::string_view data, size_t offset)
{
    auto value = uint32_t{0};
    for(size_t i = 0; i < 4; ++i)
        value |= static_cast<uint32_t>(static_cast<unsigned char>(data[offset + i])) << (8 * i);
    return value;
}

// ------------------------------------------------------------------------------------ //
//  Tokenizer
// ------------------------------------------------------------------------------------ //

enum class token_kind
{
    identifier,
    number,
    string,
    punct,
    end,
};

struct token
{
    token_kind  kind = token_kind::end;
    std::string text = {};
    int         line = 0;
};

class parse_error : public std::runtime_error
{
public:
    parse_error(int line, const std::string& what)
    : std::runtime_error{"metadata line " + std::to_string(line) + ": " + what}
    {}
};

std::vector<token>
tokenize(std::string_view text)
{
    auto tokens = std::vector<token>{};
    auto line   = 1;
    auto i      = size_t{0};
    auto n      = text.size();
    while(i < n)
    {
        auto c = text[i];
        if(c == '\n')
        {
            ++line;
            ++i;
        }
        else if(std::isspace(static_cast<unsigned char>(c)) != 0)
        {
            ++i;
        }
        else if(c == '/' && i + 1 < n && text[i + 1] == '*')
        {
            auto end = text.find("*/", i + 2);
            if(end == std::string_view::npos) throw parse_error{line, "unterminated comment"};
            line += static_cast<int>(std::count(text.begin() + i, text.begin() + end, '\n'));
            i = end + 2;
        }
        else if(c == '/' && i + 1 < n && text[i + 1] == '/')
        {
            while(i < n && text[i] != '\n')
                ++i;
        }
        else if(std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_')
        {
            auto start = i;
            while(i < n &&
                  (std::isalnum(static_cast<unsigned char>(text[i])) != 0 || text[i] == '_'))
                ++i;
            tokens.push_back(
                {token_kind::identifier, std::string{text.substr(start, i - start)}, line});
        }
        else if(std::isdigit(static_cast<unsigned char>(c)) != 0 ||
                (c == '-' && i + 1 < n &&
                 std::isdigit(static_cast<unsigned char>(text[i + 1])) != 0))
        {
            auto start = i++;
            while(i < n && (std::isalnum(static_cast<unsigned char>(text[i])) != 0))
                ++i;
            tokens.push_back(
                {token_kind::number, std::string{text.substr(start, i - start)}, line});
        }
        else if(c == '"')
        {
            auto value = std::string{};
            ++i;
            while(i < n && text[i] != '"')
            {
                if(text[i] == '\\' && i + 1 < n) ++i;
                value += text[i++];
            }
            if(i >= n) throw parse_error{line, "unterminated string"};
            ++i;
            tokens.push_back({token_kind::string, value, line});
        }
        else if(text.substr(i, 3) == "...")
        {
            tokens.push_back({token_kind::punct, "...", line});
            i += 3;
        }
        else if(text.substr(i, 2) == ":=")
        {
            tokens.push_back({token_kind::punct, ":=", line});
            i += 2;
        }
        else if(std::strchr("{};=:[]<>,().", c) != nullptr)
        {
            tokens.push_back({token_kind::punct, std::string(1, c), line});
            ++i;
        }
        else
        {
            throw parse_error{line, std::string{"unexpected character '"} + c + "'"};
        }
    }
    tokens.push_back({token_kind::end, "", line});
    return tokens;
}

int64_t
parse_integer(const token& tok)
{
    if(tok.kind != token_kind::number) throw parse_error{tok.line, "expected a number"};
    try
    {
        auto pos   = size_t{0};
        auto value = int64_t{0};
        if(tok.text.rfind("0x", 0) == 0 || tok.text.rfind("0X", 0) == 0)
            value = static_cast<int64_t>(std::stoull(tok.text, &pos, 16));
        else if(tok.text.front() == '-')
            value = std::stoll(tok.text, &pos, 10);
        else
            value = static_cast<int64_t>(std::stoull(tok.text, &pos, 10));
        // Integer suffixes (U, L) are allowed by TSDL.
        while(pos < tok.text.size() && std::strchr("uUlL", tok.text[pos]) != nullptr)
            ++pos;
        if(pos != tok.text.size()) throw parse_error{tok.line, "bad number " + tok.text};
        return value;
    } catch(const std::logic_error&)
    {
        throw parse_error{tok.line, "bad number " + tok.text};
    }
}

// ------------------------------------------------------------------------------------ //
//  Parser
// ------------------------------------------------------------------------------------ //

bool
is_type_keyword(const std::string& word)
{
    return word == "integer" || word == "floating_point" || word == "string" || word == "struct" ||
           word == "variant" || word == "enum";
}

class parser
{
public:
    parser(std::vector<token> tokens, trace_description* trace)
    : m_tokens{std::move(tokens)}
    , m_trace{trace}
    {}

    void parse()
    {
        while(peek().kind != token_kind::end)
            parse_top_level();
        if(!m_has_trace) throw parse_error{peek().line, "no trace block"};
    }

private:
    std::vector<token>              m_tokens    = {};
    size_t                          m_pos       = 0;
    trace_description*              m_trace     = nullptr;
    bool                            m_has_trace = false;
    bool                            m_has_clock = false;
    std::map<std::string, type_ptr> m_aliases   = {};
    std::map<std::string, type_ptr> m_structs   = {};
    std::map<std::string, type_ptr> m_variants  = {};
    std::map<std::string, type_ptr> m_enums     = {};

    const token& peek(size_t ahead = 0) const
    {
        return m_tokens[std::min(m_pos + ahead, m_tokens.size() - 1)];
    }

    const token& next() { return m_tokens[std::min(m_pos++, m_tokens.size() - 1)]; }

    bool accept(const char* punct)
    {
        if(peek().kind == token_kind::punct && peek().text == punct)
        {
            ++m_pos;
            return true;
        }
        return false;
    }

    void expect(const char* punct)
    {
        if(!accept(punct))
            throw parse_error{peek().line,
                              std::string{"expected '"} + punct + "', found '" + peek().text + "'"};
    }

    std::string expect_identifier()
    {
        const auto& tok = next();
        if(tok.kind != token_kind::identifier)
            throw parse_error{tok.line, "expected an identifier, found '" + tok.text + "'"};
        return tok.text;
    }

    // ------------------------------------------------------------------ top level

    void parse_top_level()
    {
        const auto line    = peek().line;
        const auto keyword = expect_identifier();
        if(keyword == "typealias")
        {
            auto aliased = parse_type();
            expect(":=");
            auto name = parse_words_until(";");
            expect(";");
            m_aliases[name] = aliased;
        }
        else if(keyword == "trace" || keyword == "env" || keyword == "clock" ||
                keyword == "stream" || keyword == "event" || keyword == "callsite")
        {
            parse_block(keyword);
        }
        else if(keyword == "struct" || keyword == "variant" || keyword == "enum")
        {
            --m_pos;
            parse_type();  // a named definition
            expect(";");
        }
        else
        {
            throw parse_error{line, "unsupported top-level declaration '" + keyword + "'"};
        }
    }

    // Identifiers up to a punctuation token, joined with spaces ("unsigned long").
    std::string parse_words_until(const char* punct)
    {
        auto words = std::string{};
        while(peek().kind == token_kind::identifier)
        {
            if(!words.empty()) words += ' ';
            words += next().text;
        }
        if(words.empty() || !(peek().kind == token_kind::punct && peek().text == punct))
            throw parse_error{peek().line, std::string{"expected a name before '"} + punct + "'"};
        return words;
    }

    std::string parse_dotted_key()
    {
        auto key = expect_identifier();
        while(accept("."))
            key += "." + expect_identifier();
        return key;
    }

    void parse_block(const std::string& kind)
    {
        expect("{");
        auto event  = event_description{};
        auto stream = stream_description{};
        auto clock  = clock_description{};
        while(!accept("}"))
        {
            const auto line = peek().line;
            auto       key  = parse_dotted_key();
            if(accept(":="))
            {
                auto declared = parse_type();
                assign_type(kind, key, declared, &event, &stream, line);
            }
            else
            {
                expect("=");
                const auto& value = next();
                assign_value(kind, key, value, &event, &stream, &clock);
            }
            expect(";");
        }
        expect(";");

        if(kind == "trace")
        {
            m_has_trace = true;
        }
        else if(kind == "stream")
        {
            m_trace->streams[stream.id] = stream;
        }
        else if(kind == "event")
        {
            if(event.fields == nullptr) event.fields = std::make_shared<type>(empty_struct());
            m_trace->events[event.id] = event;
        }
        else if(kind == "clock")
        {
            if(m_has_clock) throw parse_error{peek().line, "more than one clock"};
            m_has_clock    = true;
            m_trace->clock = clock;
        }
    }

    static type empty_struct()
    {
        auto empty      = type{};
        empty.kind      = type_kind::structure;
        empty.alignment = 1;
        return empty;
    }

    void assign_type(const std::string&  kind,
                     const std::string&  key,
                     const type_ptr&     declared,
                     event_description*  event,
                     stream_description* stream,
                     int                 line)
    {
        if(kind == "trace" && key == "packet.header")
            m_trace->packet_header = declared;
        else if(kind == "stream" && key == "packet.context")
            stream->packet_context = declared;
        else if(kind == "stream" && key == "event.header")
            stream->event_header = declared;
        else if(kind == "stream" && key == "event.context")
            stream->event_context = declared;
        else if(kind == "event" && key == "fields")
            event->fields = declared;
        else if(kind == "event" && key == "context")
            event->context = declared;
        else
            throw parse_error{line, "unsupported type assignment " + kind + "." + key};
    }

    static std::string value_text(const token& value)
    {
        if(value.kind == token_kind::end) throw parse_error{value.line, "expected a value"};
        return value.text;
    }

    void assign_value(const std::string&  kind,
                      const std::string&  key,
                      const token&        value,
                      event_description*  event,
                      stream_description* stream,
                      clock_description*  clock)
    {
        if(kind == "trace")
        {
            if(key == "major") m_trace->major = static_cast<uint32_t>(parse_integer(value));
            if(key == "minor") m_trace->minor = static_cast<uint32_t>(parse_integer(value));
            if(key == "byte_order")
            {
                if(value.text != "le")
                    throw parse_error{value.line, "byte order " + value.text + " not supported"};
                m_trace->little_endian = true;
            }
        }
        else if(kind == "env")
        {
            m_trace->env[key] = value_text(value);
        }
        else if(kind == "clock")
        {
            if(key == "name") clock->name = value_text(value);
            if(key == "freq") clock->frequency = static_cast<uint64_t>(parse_integer(value));
            if(key == "offset_s") clock->offset_s = parse_integer(value);
            if(key == "offset") clock->offset = parse_integer(value);
        }
        else if(kind == "stream")
        {
            if(key == "id") stream->id = static_cast<uint64_t>(parse_integer(value));
        }
        else if(kind == "event")
        {
            if(key == "name") event->name = value_text(value);
            if(key == "id") event->id = static_cast<uint64_t>(parse_integer(value));
            if(key == "stream_id") event->stream_id = static_cast<uint64_t>(parse_integer(value));
            if(key == "loglevel") event->loglevel = parse_integer(value);
        }
        // Other keys (uuid, description, precision, absolute, model.emf.uri, ...) are not
        // needed to decode the trace.
    }

    // ------------------------------------------------------------------ types

    // A type specifier: a keyword type, or an alias name (possibly several words).
    type_ptr parse_type()
    {
        const auto& tok = peek();
        if(tok.kind != token_kind::identifier)
            throw parse_error{tok.line, "expected a type, found '" + tok.text + "'"};
        if(tok.text == "integer") return parse_integer_type();
        if(tok.text == "floating_point") return parse_float_type();
        if(tok.text == "string") return parse_string_type();
        if(tok.text == "struct") return parse_struct_type();
        if(tok.text == "variant") return parse_variant_type();
        if(tok.text == "enum") return parse_enum_type();

        // An alias: all identifiers up to the next punctuation.
        auto words = std::string{};
        while(peek().kind == token_kind::identifier)
        {
            if(!words.empty()) words += ' ';
            words += next().text;
        }
        return lookup_alias(words, tok.line);
    }

    type_ptr lookup_alias(const std::string& name, int line) const
    {
        auto itr = m_aliases.find(name);
        if(itr == m_aliases.end()) throw parse_error{line, "unknown type '" + name + "'"};
        return itr->second;
    }

    // name = value; pairs inside { } of integer, floating_point and string.
    std::vector<std::pair<std::string, token>> parse_attributes()
    {
        auto attributes = std::vector<std::pair<std::string, token>>{};
        expect("{");
        while(!accept("}"))
        {
            auto key = expect_identifier();
            expect("=");
            auto value = next();
            // map = clock.<name>.value
            while(value.kind == token_kind::identifier && accept("."))
                value.text += "." + expect_identifier();
            attributes.emplace_back(key, value);
            expect(";");
        }
        return attributes;
    }

    static bool truthy(const token& value)
    {
        if(value.text == "true" || value.text == "TRUE") return true;
        if(value.text == "false" || value.text == "FALSE") return false;
        return parse_integer(value) != 0;
    }

    type_ptr parse_integer_type()
    {
        next();  // integer
        auto result      = type{};
        result.kind      = type_kind::integer;
        auto align_given = false;
        for(const auto& [key, value] : parse_attributes())
        {
            if(key == "size")
                result.size = static_cast<uint32_t>(parse_integer(value));
            else if(key == "align")
            {
                result.alignment = static_cast<uint32_t>(parse_integer(value));
                align_given      = true;
            }
            else if(key == "signed")
                result.is_signed = truthy(value);
            else if(key == "base")
            {
                const auto& base = value.text;
                if(base == "16" || base == "x" || base == "X" || base == "hexadecimal")
                    result.base = 16;
                else if(base == "8" || base == "o" || base == "octal")
                    result.base = 8;
                else if(base == "2" || base == "b" || base == "binary")
                    result.base = 2;
                else
                    result.base = 10;
            }
            else if(key == "encoding")
                result.is_text = (value.text == "UTF8" || value.text == "ASCII");
            else if(key == "byte_order")
            {
                if(value.text != "le" && value.text != "native")
                    throw parse_error{value.line, "byte order " + value.text + " not supported"};
            }
            else if(key == "map")
                result.maps_clock = value.text.rfind("clock.", 0) == 0;
        }
        if(result.size == 0 || result.size > 64)
            throw parse_error{peek().line, "integer size must be 1..64 bits"};
        if(!align_given) result.alignment = (result.size % 8 == 0) ? 8 : 1;
        return std::make_shared<type>(result);
    }

    type_ptr parse_float_type()
    {
        next();  // floating_point
        auto result      = type{};
        result.kind      = type_kind::floating_point;
        auto align_given = false;
        for(const auto& [key, value] : parse_attributes())
        {
            if(key == "exp_dig") result.exp_dig = static_cast<uint32_t>(parse_integer(value));
            if(key == "mant_dig") result.mant_dig = static_cast<uint32_t>(parse_integer(value));
            if(key == "align")
            {
                result.alignment = static_cast<uint32_t>(parse_integer(value));
                align_given      = true;
            }
        }
        result.size = result.exp_dig + result.mant_dig;
        if(result.size != 32 && result.size != 64)
            throw parse_error{peek().line, "only 32- and 64-bit floating point are supported"};
        if(!align_given) result.alignment = 8;
        return std::make_shared<type>(result);
    }

    type_ptr parse_string_type()
    {
        next();  // string
        if(peek().kind == token_kind::punct && peek().text == "{") parse_attributes();
        auto result      = type{};
        result.kind      = type_kind::string;
        result.alignment = 8;
        return std::make_shared<type>(result);
    }

    uint32_t parse_align_suffix()
    {
        if(peek().kind == token_kind::identifier && peek().text == "align")
        {
            next();
            expect("(");
            auto align = static_cast<uint32_t>(parse_integer(next()));
            expect(")");
            return align;  // bits, as for integers (lttng-ust writes bytes * CHAR_BIT)
        }
        return 0;
    }

    type_ptr parse_struct_type()
    {
        next();  // struct
        auto name = std::string{};
        if(peek().kind == token_kind::identifier) name = next().text;

        if(!(peek().kind == token_kind::punct && peek().text == "{"))
        {
            auto itr = m_structs.find(name);
            if(itr == m_structs.end()) throw parse_error{peek().line, "unknown struct " + name};
            return itr->second;
        }

        auto result      = type{};
        result.kind      = type_kind::structure;
        result.alignment = 1;
        result.members   = parse_members();
        for(const auto& itr : result.members)
            result.alignment = std::max(result.alignment, itr.type->alignment);
        if(auto align = parse_align_suffix(); align > 0)
            result.alignment = std::max(result.alignment, align);

        auto declared = std::make_shared<type>(result);
        if(!name.empty()) m_structs[name] = declared;
        return declared;
    }

    type_ptr parse_variant_type()
    {
        next();  // variant
        auto name = std::string{};
        if(peek().kind == token_kind::identifier) name = next().text;
        auto tag = std::string{};
        if(accept("<"))
        {
            tag = parse_dotted_key();
            expect(">");
            // A path selects the last component in the enclosing struct (the only form
            // LTTng uses).
            if(auto dot = tag.rfind('.'); dot != std::string::npos) tag = tag.substr(dot + 1);
        }

        if(!(peek().kind == token_kind::punct && peek().text == "{"))
        {
            auto itr = m_variants.find(name);
            if(itr == m_variants.end()) throw parse_error{peek().line, "unknown variant " + name};
            auto copy = std::make_shared<type>(*itr->second);
            if(!tag.empty()) copy->tag = tag;
            return copy;
        }

        auto result      = type{};
        result.kind      = type_kind::variant;
        result.alignment = 1;
        result.tag       = tag;
        result.members   = parse_members();
        auto declared    = std::make_shared<type>(result);
        if(!name.empty()) m_variants[name] = declared;
        return declared;
    }

    type_ptr parse_enum_type()
    {
        next();  // enum
        auto name = std::string{};
        if(peek().kind == token_kind::identifier) name = next().text;

        auto container = type_ptr{};
        if(accept(":")) container = parse_type();

        if(!(peek().kind == token_kind::punct && peek().text == "{"))
        {
            auto itr = m_enums.find(name);
            if(itr == m_enums.end()) throw parse_error{peek().line, "unknown enum " + name};
            return itr->second;
        }
        if(!container)
        {
            auto itr = m_aliases.find("int");
            if(itr == m_aliases.end())
                throw parse_error{peek().line, "enum without a container type"};
            container = itr->second;
        }
        if(container->kind != type_kind::integer)
            throw parse_error{peek().line, "enum container must be an integer"};

        auto result      = type{};
        result.kind      = type_kind::enumeration;
        result.container = container;
        result.alignment = container->alignment;
        result.size      = container->size;
        result.is_signed = container->is_signed;

        expect("{");
        auto next_value = int64_t{0};
        while(!accept("}"))
        {
            const auto& label_tok = next();
            if(label_tok.kind != token_kind::identifier && label_tok.kind != token_kind::string)
                throw parse_error{label_tok.line, "expected an enumerator"};
            auto item  = enumerator{};
            item.label = label_tok.text;
            if(accept("="))
            {
                item.lo = parse_integer(next());
                item.hi = accept("...") ? parse_integer(next()) : item.lo;
            }
            else
            {
                item.lo = item.hi = next_value;
            }
            next_value = item.hi + 1;
            result.enumerators.push_back(item);
            if(!accept(",") && !(peek().kind == token_kind::punct && peek().text == "}"))
                throw parse_error{peek().line, "expected ',' or '}' in enum"};
        }

        auto declared = std::make_shared<type>(result);
        if(!name.empty()) m_enums[name] = declared;
        return declared;
    }

    // { <type> <name>[dims]; ... }
    std::vector<member> parse_members()
    {
        expect("{");
        auto members = std::vector<member>{};
        while(!accept("}"))
        {
            members.push_back(parse_member());
            expect(";");
        }
        return members;
    }

    member parse_member()
    {
        auto       field     = member{};
        const auto line      = peek().line;
        auto       base_type = type_ptr{};

        if(peek().kind == token_kind::identifier && is_type_keyword(peek().text))
        {
            base_type  = parse_type();
            field.name = expect_identifier();
        }
        else
        {
            // "<alias words> <name>": the longest leading word sequence that is an alias.
            auto words = std::vector<std::string>{};
            while(peek().kind == token_kind::identifier)
                words.push_back(next().text);
            if(words.size() < 2) throw parse_error{line, "expected a type and a field name"};
            auto alias = std::string{};
            for(size_t i = 0; i + 1 < words.size(); ++i)
                alias += (i > 0 ? " " : "") + words[i];
            base_type  = lookup_alias(alias, line);
            field.name = words.back();
        }

        // Array and sequence suffixes; [a][b] is an array of a arrays of b.
        auto dims = std::vector<token>{};
        while(accept("["))
        {
            auto dim = next();
            while(dim.kind == token_kind::identifier && accept("."))
                dim.text += "." + expect_identifier();
            dims.push_back(dim);
            expect("]");
        }
        auto declared = base_type;
        for(auto itr = dims.rbegin(); itr != dims.rend(); ++itr)
        {
            auto wrapped      = type{};
            wrapped.element   = declared;
            wrapped.alignment = declared->alignment;
            if(itr->kind == token_kind::number)
            {
                wrapped.kind   = type_kind::array;
                wrapped.length = static_cast<uint64_t>(parse_integer(*itr));
            }
            else if(itr->kind == token_kind::identifier)
            {
                wrapped.kind = type_kind::sequence;
                auto dot     = itr->text.rfind('.');
                wrapped.length_name =
                    dot == std::string::npos ? itr->text : itr->text.substr(dot + 1);
            }
            else
            {
                throw parse_error{itr->line, "bad array length"};
            }
            declared = std::make_shared<type>(wrapped);
        }
        field.type = declared;
        return field;
    }
};
}  // namespace

bool
metadata_text(std::string_view file_contents, std::string* text, std::string* error)
{
    text->clear();
    if(file_contents.size() < 4 || read_u32_le(file_contents, 0) != metadata_packet_magic)
    {
        *text = std::string{file_contents};
        return true;
    }

    auto offset = size_t{0};
    while(offset < file_contents.size())
    {
        if(file_contents.size() - offset < metadata_packet_header_size ||
           read_u32_le(file_contents, offset) != metadata_packet_magic)
        {
            *error = "malformed metadata packet at byte " + std::to_string(offset);
            return false;
        }
        auto content_bits = read_u32_le(file_contents, offset + 24);
        auto packet_bits  = read_u32_le(file_contents, offset + 28);
        auto compression  = static_cast<unsigned char>(file_contents[offset + 32]);
        auto encryption   = static_cast<unsigned char>(file_contents[offset + 33]);
        if(compression != 0 || encryption != 0)
        {
            *error = "compressed or encrypted metadata is not supported";
            return false;
        }
        auto content = content_bits / 8;
        auto packet  = packet_bits / 8;
        if(content < metadata_packet_header_size || packet < content ||
           offset + packet > file_contents.size())
        {
            *error = "metadata packet sizes out of range at byte " + std::to_string(offset);
            return false;
        }
        text->append(file_contents.substr(offset + metadata_packet_header_size,
                                          content - metadata_packet_header_size));
        offset += packet;
    }
    return true;
}

bool
parse_metadata(std::string_view text, trace_description* trace, std::string* error)
{
    try
    {
        *trace     = trace_description{};
        auto parse = parser{tokenize(text), trace};
        parse.parse();
        if(trace->major != 1 || trace->minor != 8)
            throw parse_error{0,
                              "CTF " + std::to_string(trace->major) + "." +
                                  std::to_string(trace->minor) + " is not supported (1.8 is)"};
        if(trace->streams.empty()) throw parse_error{0, "no stream"};
        return true;
    } catch(const std::exception& e)
    {
        *error = e.what();
        return false;
    }
}
}  // namespace ctf
}  // namespace collector
}  // namespace rocprofiler
