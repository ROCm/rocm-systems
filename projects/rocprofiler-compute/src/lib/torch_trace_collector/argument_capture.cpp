// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "argument_capture.h"

#include "wire_format.h"

#include <ATen/record_function.h>
#include <torch_abi.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace c10
{
struct alignas(torch_abi::kIValueAlignment) IValue
{
    void visit(const std::function<bool(const IValue&)>& visitor) const;
    bool isTensorList() const;

    std::byte     payload[8];
    std::uint32_t tag;
    std::byte     tail[4];
};

static_assert(std::is_standard_layout_v<IValue>);
static_assert(sizeof(IValue) == torch_abi::kIValueSize);
static_assert(alignof(IValue) == torch_abi::kIValueAlignment);
static_assert(offsetof(IValue, payload) == torch_abi::kIValuePayloadOff);
static_assert(offsetof(IValue, tag) == torch_abi::kIValueTagOff);

namespace impl
{
struct OperatorEntry
{
    std::string dumpState() const;
};
}  // namespace impl

struct OperatorHandle
{
    ~OperatorHandle();

    [[nodiscard]] const impl::OperatorEntry* entry() const noexcept
    {
        return static_cast<const impl::OperatorEntry*>(operator_def_);
    }

    void* operator_def_;
    void* operator_iterator_;
};

static_assert(std::is_standard_layout_v<OperatorHandle>);
static_assert(sizeof(OperatorHandle) == torch_abi::kOperatorHandleSize);
static_assert(alignof(OperatorHandle) == torch_abi::kOperatorHandleAlignment);
static_assert(offsetof(OperatorHandle, operator_def_) == torch_abi::kOperatorHandleDefinitionOff);
static_assert(offsetof(OperatorHandle, operator_iterator_) == torch_abi::kOperatorHandleIteratorOff);

class Dispatcher
{
public:
    static Dispatcher& realSingleton();
    OperatorHandle     findSchemaOrThrow(const char* name, const char* overload_name);
};

}  // namespace c10

extern "C"
{
using AtenTensorHandle = void*;
using AotiTorchError   = std::int32_t;

AotiTorchError aoti_torch_get_dim(AtenTensorHandle tensor, std::int64_t* dimension);
AotiTorchError aoti_torch_get_sizes(AtenTensorHandle tensor, std::int64_t** sizes);
AotiTorchError aoti_torch_get_dtype(AtenTensorHandle tensor, std::int32_t* dtype);
AotiTorchError aoti_torch_is_defined(AtenTensorHandle tensor, bool* defined);

std::uint64_t aoti_torch_abi_version();

std::int32_t aoti_torch_dtype_bfloat16();
std::int32_t aoti_torch_dtype_bool();
std::int32_t aoti_torch_dtype_complex128();
std::int32_t aoti_torch_dtype_complex32();
std::int32_t aoti_torch_dtype_complex64();
std::int32_t aoti_torch_dtype_float16();
std::int32_t aoti_torch_dtype_float32();
std::int32_t aoti_torch_dtype_float64();
std::int32_t aoti_torch_dtype_int16();
std::int32_t aoti_torch_dtype_int32();
std::int32_t aoti_torch_dtype_int64();
std::int32_t aoti_torch_dtype_int8();
std::int32_t aoti_torch_dtype_uint8();
}

namespace
{

using torch_trace_collector::detail::BoundedTextOutput;
using torch_trace_collector::detail::kUnavailable;

constexpr std::size_t kMaxArgsLen                = torch_trace_collector::detail::kMaxArgsLength;
constexpr std::size_t kMaxArgItems               = 32;
constexpr std::size_t kMaxNestedArgItems         = 8;
constexpr std::size_t kMaxInputCount             = std::size_t{1024} * 1024;
constexpr std::size_t kMaxIntegerChars           = std::numeric_limits<std::int64_t>::digits10 + 2;
constexpr std::size_t kSchemaCacheShardCount     = 64;
constexpr std::size_t kThreadSchemaCacheCount    = 64;
constexpr std::size_t kExpectedOperatorsPerShard = 64;
constexpr std::size_t kExpectedOverloadsPerOperator = 4;

struct InputArrayView
{
    const c10::IValue* data;
    std::size_t        size;
};

static_assert(sizeof(InputArrayView) == torch_abi::kInputArrayViewSize);
static_assert(alignof(InputArrayView) == torch_abi::kInputArrayViewAlignment);

std::atomic_flag g_capture_failure_warning = ATOMIC_FLAG_INIT;
std::atomic_flag g_schema_failure_warning  = ATOMIC_FLAG_INIT;

using SchemaNames     = std::vector<std::string>;
using SchemaOverloads = std::unordered_map<std::string, SchemaNames>;

struct SchemaCacheShard
{
    SchemaCacheShard() { operators.reserve(kExpectedOperatorsPerShard); }

    std::shared_mutex                                mutex;
    std::unordered_map<std::string, SchemaOverloads> operators;
};

struct SchemaCache
{
    std::array<SchemaCacheShard, kSchemaCacheShardCount> shards;
};

struct SchemaCacheLookup
{
    std::size_t        hash           = 0;
    const std::string* operator_name  = nullptr;
    const std::string* overload_name  = nullptr;
    const SchemaNames* argument_names = nullptr;
};

thread_local std::array<SchemaCacheLookup, kThreadSchemaCacheCount> g_thread_schema_cache{};

SchemaCache& schema_cache()
{
    // PyTorch retains the callback until process exit, so this cache must
    // remain alive during C++ static destruction.
    static auto* const cache = new SchemaCache{};
    return *cache;
}

std::size_t schema_hash(const c10::OperatorName& operator_name)
{
    const std::size_t name_hash     = std::hash<std::string_view>{}(operator_name.name);
    const std::size_t overload_hash = std::hash<std::string_view>{}(operator_name.overload_name);
    return name_hash ^ (overload_hash + std::size_t{0x9e3779b9U} + (name_hash * 64U) + (name_hash / 4U));
}

void warn_once(std::atomic_flag& warning, const char* message) noexcept
{
    if (!warning.test_and_set(std::memory_order_relaxed))
    {
        std::fputs(message, stderr);
    }
}

void warn_capture_failure_once() noexcept
{
    warn_once(g_capture_failure_warning,
              "rocprof-compute: PyTorch argument capture failed; using fallback values. "
              "The loaded build may not match the validated ABI shim.\n");
}

void warn_schema_failure_once() noexcept
{
    warn_once(g_schema_failure_warning,
              "rocprof-compute: PyTorch schema lookup failed; argument names may be "
              "unavailable.\n");
}

class BoundedArgumentBuffer
{
public:
    struct Checkpoint
    {
        std::size_t size;
        bool        truncated;
    };

    // Only the initialized [0, size_) prefix is exposed; avoid a per-event
    // zero-fill of the complete hot-path buffer.
    // cppcheck-suppress uninitMemberVar
    BoundedArgumentBuffer()                                        = default;
    BoundedArgumentBuffer(const BoundedArgumentBuffer&)            = delete;
    BoundedArgumentBuffer& operator=(const BoundedArgumentBuffer&) = delete;
    BoundedArgumentBuffer(BoundedArgumentBuffer&&)                 = delete;
    BoundedArgumentBuffer& operator=(BoundedArgumentBuffer&&)      = delete;
    ~BoundedArgumentBuffer()                                       = default;

    void append(std::string_view value) noexcept
    {
        const std::size_t available = kMaxArgsLen - size_;
        const std::size_t copied    = std::min(value.size(), available);
        if (copied > 0)
        {
            std::memcpy(data_.data() + size_, value.data(), copied);
            size_ += copied;
        }
        truncated_ = truncated_ || copied < value.size();
    }

    void append(char value) noexcept
    {
        if (size_ < kMaxArgsLen)
        {
            data_[size_++] = value;
        }
        else
        {
            truncated_ = true;
        }
    }

    BoundedArgumentBuffer& operator+=(std::string_view value) noexcept
    {
        append(value);
        return *this;
    }

    BoundedArgumentBuffer& operator+=(char value) noexcept
    {
        append(value);
        return *this;
    }

    [[nodiscard]] Checkpoint checkpoint() const noexcept { return {size_, truncated_}; }

    void restore(Checkpoint checkpoint) noexcept
    {
        size_      = std::min(checkpoint.size, kMaxArgsLen);
        truncated_ = checkpoint.truncated && size_ == kMaxArgsLen;
    }

    [[nodiscard]] bool truncated() const noexcept { return truncated_; }

    [[nodiscard]] std::string_view finish() noexcept
    {
        if (truncated_)
        {
            constexpr std::string_view suffix = "...)";
            std::memcpy(data_.data() + kMaxArgsLen, suffix.data(), suffix.size());
            return {data_.data(), kMaxArgsLen + suffix.size()};
        }
        return {data_.data(), size_};
    }

private:
    std::array<char, kMaxArgsLen + torch_trace_collector::detail::kTruncationSuffixLength> data_;
    std::size_t size_      = 0;
    bool        truncated_ = false;
};

std::size_t encode_to_output(std::string_view args, char* output, std::size_t capacity) noexcept
{
    BoundedTextOutput encoded{output, capacity};
    for (const char character : args)
    {
        switch (character)
        {
        case '%':
            encoded.append("%25");
            break;
        case '|':
            encoded.append("%7C");
            break;
        case ';':
            encoded.append("%3B");
            break;
        case '\r':
            encoded.append("%0D");
            break;
        case '\n':
            encoded.append("%0A");
            break;
        default:
            encoded.append(std::string_view{&character, 1});
            break;
        }
    }
    return encoded.finish();
}

std::string trim(std::string_view value)
{
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0)
    {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0)
    {
        value.remove_suffix(1);
    }
    return std::string{value};
}

class DelimiterState
{
public:
    [[nodiscard]] bool consume(char character) noexcept
    {
        if (quote_ != '\0')
        {
            if (escaped_)
            {
                escaped_ = false;
            }
            else if (character == '\\')
            {
                escaped_ = true;
            }
            else if (character == quote_)
            {
                quote_ = '\0';
            }
            return true;
        }

        if (character == '\'' || character == '"')
        {
            quote_ = character;
            return true;
        }

        switch (character)
        {
        case '(':
            ++round_;
            return true;
        case ')':
            return decrement(round_);
        case '[':
            ++square_;
            return true;
        case ']':
            return decrement(square_);
        case '{':
            ++curly_;
            return true;
        case '}':
            return decrement(curly_);
        default:
            return true;
        }
    }

    [[nodiscard]] bool at_top_level() const noexcept
    {
        return quote_ == '\0' && round_ == 0 && square_ == 0 && curly_ == 0;
    }

private:
    static bool decrement(std::size_t& depth) noexcept
    {
        if (depth == 0)
        {
            return false;
        }
        --depth;
        return true;
    }

    std::size_t round_   = 0;
    std::size_t square_  = 0;
    std::size_t curly_   = 0;
    char        quote_   = '\0';
    bool        escaped_ = false;
};

std::size_t matching_paren(std::string_view text, std::size_t open)
{
    if (open >= text.size() || text[open] != '(')
    {
        return std::string_view::npos;
    }

    DelimiterState state;
    for (std::size_t i = open; i < text.size(); ++i)
    {
        const char character = text[i];
        if (!state.consume(character))
        {
            return std::string_view::npos;
        }
        if (character == ')' && state.at_top_level())
        {
            return i;
        }
    }
    return std::string_view::npos;
}

std::optional<std::string> argument_name(std::string_view argument)
{
    const std::string cleaned = trim(argument);
    if (cleaned.empty() || cleaned == "*" || cleaned == "...")
    {
        return std::string{};
    }

    std::size_t    end = cleaned.size();
    DelimiterState state;
    for (std::size_t i = 0; i < cleaned.size(); ++i)
    {
        const char character = cleaned[i];
        if (character == '=' && state.at_top_level())
        {
            end = i;
            break;
        }
        if (!state.consume(character))
        {
            return {};
        }
    }

    while (end > 0 && std::isspace(static_cast<unsigned char>(cleaned[end - 1])) != 0)
    {
        --end;
    }
    const std::size_t name_end = end;
    while (end > 0)
    {
        const unsigned char c = static_cast<unsigned char>(cleaned[end - 1]);
        if (std::isalnum(c) == 0 && c != '_')
        {
            break;
        }
        --end;
    }
    if (end == name_end ||
        (std::isalpha(static_cast<unsigned char>(cleaned[end])) == 0 && cleaned[end] != '_'))
    {
        return std::nullopt;
    }
    return cleaned.substr(end, name_end - end);
}

std::vector<std::string> parse_schema_argument_names(std::string_view dump)
{
    constexpr std::string_view prefix     = "schema: ";
    const std::size_t          schema_pos = dump.find(prefix);
    if (schema_pos == std::string_view::npos)
    {
        return {};
    }
    const std::size_t open = dump.find('(', schema_pos + prefix.size());
    if (open == std::string_view::npos)
    {
        return {};
    }
    const std::size_t close = matching_paren(dump, open);
    if (close == std::string_view::npos)
    {
        return {};
    }

    const std::string_view   arguments = dump.substr(open + 1, close - open - 1);
    std::vector<std::string> result;
    result.reserve(kMaxArgItems);
    std::size_t    begin = 0;
    DelimiterState state;
    for (std::size_t i = 0; i <= arguments.size(); ++i)
    {
        const char character = i < arguments.size() ? arguments[i] : ',';
        if (character == ',' && state.at_top_level())
        {
            auto name = argument_name(arguments.substr(begin, i - begin));
            if (!name.has_value())
            {
                warn_schema_failure_once();
                return {};
            }
            if (!name->empty())
            {
                result.push_back(std::move(*name));
                if (result.size() == kMaxArgItems)
                {
                    break;
                }
            }
            begin = i + 1;
            continue;
        }
        if (!state.consume(character))
        {
            warn_schema_failure_once();
            return {};
        }
    }
    return result;
}

const SchemaNames& load_schema_names(const c10::OperatorName& operator_name)
{
    const std::size_t hash         = schema_hash(operator_name);
    auto&             thread_entry = g_thread_schema_cache[hash % g_thread_schema_cache.size()];
    if (thread_entry.argument_names != nullptr && thread_entry.operator_name != nullptr &&
        thread_entry.overload_name != nullptr && thread_entry.hash == hash &&
        *thread_entry.operator_name == operator_name.name &&
        *thread_entry.overload_name == operator_name.overload_name)
    {
        return *thread_entry.argument_names;
    }

    auto& shard = schema_cache().shards[hash % kSchemaCacheShardCount];
    {
        const std::shared_lock<std::shared_mutex> lock{shard.mutex};
        if (const auto operator_it = shard.operators.find(operator_name.name);
            operator_it != shard.operators.end())
        {
            if (const auto overload_it = operator_it->second.find(operator_name.overload_name);
                overload_it != operator_it->second.end())
            {
                thread_entry = {hash, &operator_it->first, &overload_it->first, &overload_it->second};
                return overload_it->second;
            }
        }
    }

    std::vector<std::string> names;
    try
    {
        auto handle = c10::Dispatcher::realSingleton().findSchemaOrThrow(operator_name.name.c_str(),
                                                                         operator_name.overload_name.c_str());

        names = parse_schema_argument_names(handle.entry()->dumpState());
    }
    catch (...)
    {
        warn_schema_failure_once();
    }

    const std::lock_guard<std::shared_mutex> lock{shard.mutex};
    auto [operator_it, operator_inserted] = shard.operators.try_emplace(operator_name.name);
    if (operator_inserted)
    {
        operator_it->second.reserve(kExpectedOverloadsPerOperator);
    }
    auto overload_it =
        operator_it->second.try_emplace(operator_name.overload_name, std::move(names)).first;
    thread_entry = {hash, &operator_it->first, &overload_it->first, &overload_it->second};
    return overload_it->second;
}

struct DtypeName
{
    std::int32_t     dtype;
    std::string_view name;
};

constexpr auto kFallbackDtypeNames = std::to_array<std::string_view>({
    "byte",
    "char",
    "short",
    "int",
    "long",
    "half",
    "float",
    "double",
    "complexhalf",
    "complexfloat",
    "complexdouble",
    "bool",
    "qint8",
    "quint8",
    "qint32",
    "bfloat16",
    "quint4x2",
    "quint2x4",
    "bits1x8",
    "bits2x4",
    "bits4x2",
    "bits8",
    "bits16",
    "float8_e5m2",
    "float8_e4m3fn",
    "float8_e5m2fnuz",
    "float8_e4m3fnuz",
    "uint16",
    "uint32",
    "uint64",
    "uint1",
    "uint2",
    "uint3",
    "uint4",
    "uint5",
    "uint6",
    "uint7",
    "int1",
    "int2",
    "int3",
    "int4",
    "int5",
    "int6",
    "int7",
    "float8_e8m0fnu",
    "float4_e2m1fn_x2",
    "bcomplex32",
});

static_assert(kFallbackDtypeNames.size() == torch_abi::kKnownScalarTypeCount);

const auto& common_dtype_names()
{
    static const auto names = std::to_array<DtypeName>({
        {aoti_torch_dtype_float32(), "float32"},
        {aoti_torch_dtype_float64(), "float64"},
        {aoti_torch_dtype_float16(), "float16"},
        {aoti_torch_dtype_bfloat16(), "bfloat16"},
        {aoti_torch_dtype_int64(), "int64"},
        {aoti_torch_dtype_int32(), "int32"},
        {aoti_torch_dtype_int16(), "int16"},
        {aoti_torch_dtype_int8(), "int8"},
        {aoti_torch_dtype_uint8(), "uint8"},
        {aoti_torch_dtype_bool(), "bool"},
        {aoti_torch_dtype_complex32(), "complex32"},
        {aoti_torch_dtype_complex64(), "complex64"},
        {aoti_torch_dtype_complex128(), "complex128"},
    });
    return names;
}

std::uint64_t runtime_aoti_abi()
{
    static const std::uint64_t abi = aoti_torch_abi_version();
    return abi;
}

void append_integer(BoundedArgumentBuffer& output, std::int64_t value)
{
    std::array<char, kMaxIntegerChars> buffer;
    const auto conversion = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (conversion.ec != std::errc{})
    {
        warn_capture_failure_once();
        output += '?';
        return;
    }
    output.append(
        std::string_view{buffer.data(), static_cast<std::size_t>(conversion.ptr - buffer.data())});
}

void append_tag_name(BoundedArgumentBuffer& output, const c10::IValue& value)
{
    static constexpr auto names = std::to_array<std::string_view>(
        {"None",   "Tensor",    "Storage",     "Double",      "ComplexDouble", "Int",
         "UInt",   "SymInt",    "SymFloat",    "SymBool",     "Bool",          "Tuple",
         "String", "Blob",      "GenericList", "GenericDict", "Future",        "Await",
         "Device", "Stream",    "Object",      "PyObject",    "Uninitialized", "Capsule",
         "RRef",   "Quantizer", "Generator",   "Enum"});
    static_assert(names.size() == torch_abi::kIValueKnownTagCount);
    if (value.tag < names.size())
    {
        output += names[value.tag];
        return;
    }
    output += "InvalidTag(";
    append_integer(output, value.tag);
    output += ')';
}

std::string_view fallback_dtype_name(std::int32_t dtype)
{
    if (dtype >= 0 && static_cast<std::size_t>(dtype) < kFallbackDtypeNames.size())
    {
        if (dtype == torch_abi::kBComplex32ScalarType && runtime_aoti_abi() < torch_abi::kTorch214AotiAbi)
        {
            return "unknown_scalar";
        }
        return kFallbackDtypeNames[static_cast<std::size_t>(dtype)];
    }
    return "unknown_scalar";
}

std::string_view dtype_name(std::int32_t dtype)
{
    const auto& dtypes = common_dtype_names();

    const auto found = std::find_if(dtypes.begin(),
                                    dtypes.end(),
                                    [dtype](const DtypeName& entry) { return entry.dtype == dtype; });
    if (found != dtypes.end())
    {
        return found->name;
    }
    return fallback_dtype_name(dtype);
}

void append_tensor(BoundedArgumentBuffer& output, const c10::IValue& value)
{
    static_assert(torch_abi::kIValuePayloadOff == 0);
    static_assert(torch_abi::kTensorSize <= torch_abi::kIValueSize);
    auto handle  = reinterpret_cast<AtenTensorHandle>(const_cast<c10::IValue*>(&value));
    bool defined = false;
    if (aoti_torch_is_defined(handle, &defined) != 0)
    {
        warn_capture_failure_once();
        output += '?';
        return;
    }
    if (!defined)
    {
        output += "None";
        return;
    }

    std::int64_t  dimension = 0;
    std::int64_t* sizes     = nullptr;
    std::int32_t  dtype     = 0;
    if (aoti_torch_get_dim(handle, &dimension) != 0 || aoti_torch_get_sizes(handle, &sizes) != 0 ||
        aoti_torch_get_dtype(handle, &dtype) != 0 || dimension < 0 ||
        (dimension > 0 && sizes == nullptr))
    {
        warn_capture_failure_once();
        output += '?';
        return;
    }

    output += dtype_name(dtype);
    output += '[';
    for (std::int64_t i = 0; i < dimension && !output.truncated(); ++i)
    {
        if (i > 0)
        {
            output += 'x';
        }
        append_integer(output, sizes[i]);
    }
    if (!output.truncated())
    {
        output += ']';
    }
}

void append_tensor_list(BoundedArgumentBuffer& output, const c10::IValue& value)
{
    struct RenderState
    {
        BoundedArgumentBuffer* output;
        std::size_t            index = 0;
        bool                   root  = true;
    } state{&output};

    output += '[';
    if (output.truncated())
    {
        return;
    }
    value.visit(
        [&state](const c10::IValue& element)
        {
            if (state.root)
            {
                state.root = false;
                return false;
            }
            if (state.index < kMaxNestedArgItems && !state.output->truncated())
            {
                if (state.index > 0)
                {
                    *state.output += ", ";
                }
                if (element.tag == torch_abi::kIValueTensorTag)
                {
                    append_tensor(*state.output, element);
                }
                else
                {
                    append_tag_name(*state.output, element);
                }
            }
            ++state.index;
            return true;
        });
    output += ']';
}

void append_value(BoundedArgumentBuffer& output, const c10::IValue& value)
{
    const auto checkpoint = output.checkpoint();
    try
    {
        if (value.tag == torch_abi::kIValueTensorTag)
        {
            append_tensor(output, value);
            return;
        }
        if (value.isTensorList())
        {
            append_tensor_list(output, value);
            return;
        }
        append_tag_name(output, value);
    }
    catch (...)
    {
        output.restore(checkpoint);
        warn_capture_failure_once();
        output += '?';
    }
}

InputArrayView input_view(const at::RecordFunction& record_function)
{
    InputArrayView result{};
    const auto*    bytes = reinterpret_cast<const std::byte*>(&record_function);
    std::memcpy(&result, bytes + torch_abi::kRecordFunctionInputsOff, sizeof(result));
    return result;
}

void render_inputs(BoundedArgumentBuffer& out, const InputArrayView& inputs, const SchemaNames* names)
{
    out += '(';
    const std::size_t count = std::min(inputs.size, kMaxArgItems);
    for (std::size_t i = 0; i < count; ++i)
    {
        if (i > 0)
        {
            out += ", ";
        }
        const auto& input = inputs.data[i];
        if (names != nullptr && i < names->size())
        {
            out += (*names)[i];
            out += '=';
        }
        if (out.truncated())
        {
            return;
        }
        append_value(out, input);
        if (out.truncated())
        {
            return;
        }
    }
    out += ')';
}

std::size_t capture(const at::RecordFunction& record_function, char* output, std::size_t capacity) noexcept
{
    try
    {
        const InputArrayView inputs = input_view(record_function);
        if (inputs.size == 0)
        {
            return encode_to_output(kUnavailable, output, capacity);
        }
        if (inputs.data == nullptr || inputs.size > kMaxInputCount)
        {
            warn_capture_failure_once();
            return encode_to_output(kUnavailable, output, capacity);
        }

        const SchemaNames* names = nullptr;
        if (const auto operator_name = record_function.operator_name(); operator_name.has_value())
        {
            names = &load_schema_names(*operator_name);
        }

        BoundedArgumentBuffer arguments;
        render_inputs(arguments, inputs, names);
        return encode_to_output(arguments.finish(), output, capacity);
    }
    catch (...)
    {
        warn_capture_failure_once();
        return encode_to_output(kUnavailable, output, capacity);
    }
}

}  // namespace

std::size_t torch_trace_collector::detail::capture_args(const at::RecordFunction& record_function,
                                                        char*                     output,
                                                        std::size_t               capacity) noexcept
{
    return capture(record_function, output, capacity);
}
