// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/lib_c.h"

#include "addc/base_report.hpp"
#include "addc/lib.hpp"
#include "addc/mca/decoder.hpp"
#include "addc/pipeline/output.hpp"
#include "addc/product.hpp"
#include "../core/runtime.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

constexpr const char* LIB_VERSION = ADDC_VERSION;

template <typename Enum>
std::underlying_type_t<Enum> enumStorage(const Enum& value) noexcept
{
    static_assert(std::is_enum_v<Enum>);
    std::underlying_type_t<Enum> storage{};
    static_assert(sizeof(storage) == sizeof(value));
    std::memcpy(&storage, &value, sizeof(storage));
    return storage;
}

char* strdupMalloc(std::string_view s) noexcept
{
    if (s.size() == static_cast<std::size_t>(-1))
    {
        return nullptr;
    }
    auto* p = static_cast<char*>(
        std::malloc(s.size() + 1)); // NOLINT(cppcoreguidelines-no-malloc)
    if (p != nullptr)
    {
        std::memcpy(p, s.data(), s.size());
        p[s.size()] = '\0';
    }
    return p;
}

void setError(char** error_out, std::string_view msg) noexcept
{
    if (error_out != nullptr)
    {
        *error_out = strdupMalloc(msg);
    }
}

template <typename Function>
addc_status_t statusBoundary(addc_error_info_t* error_info,
                             Function&& function) noexcept;

bool validErrorInfo(const addc_error_info_t* error_info) noexcept
{
    return (error_info == nullptr) ||
           error_info->struct_size >= offsetof(addc_error_info_t, reserved);
}

void resetErrorInfo(addc_error_info_t* error_info) noexcept
{
    if (error_info == nullptr)
    {
        return;
    }
    const uint32_t struct_size = error_info->struct_size;
    std::memset(error_info, 0,
                std::min<std::size_t>(struct_size, sizeof(*error_info)));
    error_info->struct_size = struct_size;
}

template <std::size_t N>
void copyErrorText(char (&destination)[N], std::string_view value) noexcept
{
    const auto count = std::min(value.size(), N - 1U);
    std::memcpy(&destination[0], value.data(), count);
    destination[count] = '\0';
}

void setErrorInfo(addc_error_info_t* error_info, addc_status_t status,
                  std::string_view source, std::string_view message,
                  std::optional<std::size_t> offset = std::nullopt) noexcept
{
    if (error_info == nullptr)
    {
        return;
    }
    error_info->status = static_cast<uint32_t>(status);
    copyErrorText(error_info->source, source);
    copyErrorText(error_info->message, message);
    if (offset)
    {
        error_info->flags |= ADDC_ERROR_INFO_HAS_OFFSET;
        error_info->byte_offset = *offset;
    }
}

addc_status_t operationStatus(addc::detail::OperationCode code) noexcept
{
    switch (code)
    {
        case addc::detail::OperationCode::InvalidArgument:
            return ADDC_STATUS_INVALID_ARGUMENT;
        case addc::detail::OperationCode::ParseError:
            return ADDC_STATUS_PARSE_ERROR;
        case addc::detail::OperationCode::DecodeError:
            return ADDC_STATUS_DECODE_ERROR;
        case addc::detail::OperationCode::IoError:
            return ADDC_STATUS_IO_ERROR;
        case addc::detail::OperationCode::OutOfResources:
            return ADDC_STATUS_OUT_OF_RESOURCES;
        case addc::detail::OperationCode::InternalError:
            return ADDC_STATUS_INTERNAL_ERROR;
    }
    return ADDC_STATUS_INTERNAL_ERROR;
}

template <typename Function>
addc_status_t statusBoundary(addc_error_info_t* error_info,
                             Function&& function) noexcept
{
    if (!validErrorInfo(error_info))
    {
        return ADDC_STATUS_INVALID_ARGUMENT;
    }
    resetErrorInfo(error_info);
    try
    {
        const auto status = std::forward<Function>(function)();
        if (error_info)
        {
            error_info->status = static_cast<uint32_t>(status);
        }
        return status;
    }
    catch (const std::bad_alloc&)
    {
        setErrorInfo(error_info, ADDC_STATUS_OUT_OF_RESOURCES, "runtime",
                     "out of resources");
        return ADDC_STATUS_OUT_OF_RESOURCES;
    }
    catch (const std::invalid_argument& error)
    {
        setErrorInfo(error_info, ADDC_STATUS_INVALID_ARGUMENT, "api",
                     error.what());
        return ADDC_STATUS_INVALID_ARGUMENT;
    }
    catch (const std::exception& error)
    {
        setErrorInfo(error_info, ADDC_STATUS_INTERNAL_ERROR, "runtime",
                     error.what());
        return ADDC_STATUS_INTERNAL_ERROR;
    }
    catch (...)
    {
        setErrorInfo(error_info, ADDC_STATUS_INTERNAL_ERROR, "runtime",
                     "internal error");
        return ADDC_STATUS_INTERNAL_ERROR;
    }
}

addc_status_t writeJsonBuffer(std::string_view json, char* json_buffer,
                              size_t* json_buffer_size) noexcept
{
    const size_t required_size = json.size() + 1U;
    const size_t capacity = *json_buffer_size;
    *json_buffer_size = required_size;

    if (json_buffer == nullptr)
    {
        return ADDC_STATUS_SUCCESS;
    }
    if (capacity < required_size)
    {
        return ADDC_STATUS_INSUFFICIENT_SIZE;
    }

    std::memcpy(json_buffer, json.data(), json.size());
    json_buffer[json.size()] = '\0';
    return ADDC_STATUS_SUCCESS;
}

template <typename Function>
int legacyStringBoundary(char** json_out, char** error_out,
                         Function&& function) noexcept
{
    if (json_out)
    {
        *json_out = nullptr;
    }
    if (error_out)
    {
        *error_out = nullptr;
    }
    try
    {
        return std::forward<Function>(function)();
    }
    catch (const std::bad_alloc&)
    {
        setError(error_out, "out of resources");
    }
    catch (const std::exception& error)
    {
        setError(error_out, error.what());
    }
    catch (...)
    {
        setError(error_out, "internal error");
    }
    return -1;
}

} // namespace

struct addc_context
{
    addc::detail::BaseRuntime runtime;
};

namespace
{

addc_status_t decodeCperWithContext(
    addc_context_t* ctx, const uint8_t* cper, size_t cper_size,
    std::string_view filename, addc::base::ResolvedReport* output,
    addc::detail::OperationError* operation_error = nullptr)
{
    auto result = ctx->runtime.decode(std::span<const uint8_t>{cper, cper_size},
                                      filename, LIB_VERSION);
    if (!result)
    {
        const auto status = result.error ? operationStatus(result.error->code)
                                         : ADDC_STATUS_INTERNAL_ERROR;
        if ((operation_error != nullptr) && result.error)
        {
            *operation_error = std::move(*result.error);
        }
        return status;
    }

    *output = addc::base::resolve_report(std::move(*result.value));
    return ADDC_STATUS_SUCCESS;
}

std::vector<addc_cper_summary_entry_t> makeSummaryEntries(
    const addc::base::ResolvedReport& output)
{
    std::vector<addc_cper_summary_entry_t> entries;
    for (const auto& summary : addc::base::summarize(output))
    {
        addc_cper_summary_entry_t entry{};
        entry.afid = summary.afid;
        const auto copyString = [](char* destination,
                                   std::size_t destination_size,
                                   const std::string& source) {
            const auto copy_size =
                std::min(source.size(), destination_size - 1U);
            std::memcpy(destination, source.data(), copy_size);
        };
        copyString(entry.fru_id, sizeof(entry.fru_id), summary.fru_id);
        copyString(entry.fru_text, sizeof(entry.fru_text), summary.fru_text);
        copyString(entry.additional_context, sizeof(entry.additional_context),
                   summary.additional_context);
        entries.push_back(entry);
    }
    return entries;
}

} // namespace

extern "C"
{
addc_context_t* addc_context_create(void)
{
    return addc_context_create_with_options(nullptr);
}

addc_context_t* addc_context_create_with_options(
    const addc_context_options_t* options)
try
{
    addc::DecodeOptions decode_options;
    if (options != nullptr)
    {
        if (options->struct_size < offsetof(addc_context_options_t, reserved))
        {
            return nullptr;
        }
        // A C caller can supply any value representable by the enum storage.
        // Copy the representation into its integer type before validating it;
        // loading an out-of-range C enum through a C++ enum lvalue is UB.
        using ParseModeStorage = std::underlying_type_t<addc_parse_mode_t>;
        const auto parse_mode = enumStorage(options->parse_mode);
        switch (parse_mode)
        {
            case static_cast<ParseModeStorage>(ADDC_PARSE_MODE_TOLERANT):
                decode_options.parse_mode = addc::ParseMode::Tolerant;
                break;
            case static_cast<ParseModeStorage>(ADDC_PARSE_MODE_STRICT):
                decode_options.parse_mode = addc::ParseMode::Strict;
                break;
            default:
                return nullptr;
        }
    }

    auto ctx = std::unique_ptr<addc_context>{new addc_context{
        addc::detail::build_base_runtime(decode_options)}};
    return ctx.release();
}
catch (...)
{
    return nullptr;
}

void addc_context_destroy(addc_context_t* ctx)
{
    delete ctx;
}

addc_status_t addc_cper_decode_json(addc_context_t* ctx, const uint8_t* cper,
                                    size_t cper_size, char* json_buffer,
                                    size_t* json_buffer_size)
{
    return addc_cper_decode_json_ex(ctx, cper, cper_size, nullptr, json_buffer,
                                    json_buffer_size, nullptr);
}

addc_status_t addc_cper_decode_json_ex(
    addc_context_t* ctx, const uint8_t* cper, size_t cper_size,
    const char* filename, char* json_buffer, size_t* json_buffer_size,
    addc_error_info_t* error_info)
{
    return statusBoundary(error_info, [&] {
        if (!ctx || !cper || !json_buffer_size)
        {
            setErrorInfo(error_info, ADDC_STATUS_INVALID_ARGUMENT, "api",
                         "null context, CPER buffer, or size pointer");
            return ADDC_STATUS_INVALID_ARGUMENT;
        }

        addc::base::ResolvedReport output;
        addc::detail::OperationError operation_error;
        const auto status = decodeCperWithContext(
            ctx, cper, cper_size, filename ? filename : "", &output,
            &operation_error);
        if (status != ADDC_STATUS_SUCCESS)
        {
            setErrorInfo(error_info, status, operation_error.source,
                         operation_error.message, operation_error.offset);
            return status;
        }

        const std::string json = addc::base::to_json(output).dump(2);
        return writeJsonBuffer(json, json_buffer, json_buffer_size);
    });
}

addc_status_t addc_cper_get_summary(
    addc_context_t* ctx, const uint8_t* cper, size_t cper_size,
    addc_cper_summary_entry_t* entries, size_t* entry_count)
{
    return addc_cper_get_summary_ex(ctx, cper, cper_size, nullptr, entries,
                                    entry_count, nullptr);
}

addc_status_t addc_cper_get_summary_ex(
    addc_context_t* ctx, const uint8_t* cper, size_t cper_size,
    const char* filename, addc_cper_summary_entry_t* entries,
    size_t* entry_count, addc_error_info_t* error_info)
{
    return statusBoundary(error_info, [&] {
        if (!ctx || !cper || !entry_count)
        {
            setErrorInfo(error_info, ADDC_STATUS_INVALID_ARGUMENT, "api",
                         "null context, CPER buffer, or count pointer");
            return ADDC_STATUS_INVALID_ARGUMENT;
        }

        addc::base::ResolvedReport output;
        addc::detail::OperationError operation_error;
        const auto status = decodeCperWithContext(
            ctx, cper, cper_size, filename ? filename : "", &output,
            &operation_error);
        if (status != ADDC_STATUS_SUCCESS)
        {
            setErrorInfo(error_info, status, operation_error.source,
                         operation_error.message, operation_error.offset);
            return status;
        }

        const auto summary = makeSummaryEntries(output);
        const size_t required_count = summary.size();
        const size_t capacity = *entry_count;
        *entry_count = required_count;

        if (!entries)
        {
            return ADDC_STATUS_SUCCESS;
        }
        if (capacity < required_count)
        {
            return ADDC_STATUS_INSUFFICIENT_SIZE;
        }

        if (required_count != 0U)
        {
            std::memcpy(entries, summary.data(),
                        required_count * sizeof(*entries));
        }
        return ADDC_STATUS_SUCCESS;
    });
}

addc_status_t addc_cper_parse_json(
    addc_context_t* ctx, const uint8_t* cper, size_t cper_size,
    char* json_buffer, size_t* json_buffer_size, addc_error_info_t* error_info)
{
    return statusBoundary(error_info, [&] {
        if (!ctx || !cper || !json_buffer_size)
        {
            setErrorInfo(error_info, ADDC_STATUS_INVALID_ARGUMENT, "api",
                         "null context, CPER buffer, or size pointer");
            return ADDC_STATUS_INVALID_ARGUMENT;
        }

        auto parsed =
            ctx->runtime.parse(std::span<const uint8_t>{cper, cper_size});
        if (!parsed)
        {
            const auto status = parsed.error
                                    ? operationStatus(parsed.error->code)
                                    : ADDC_STATUS_INTERNAL_ERROR;
            if (parsed.error)
            {
                setErrorInfo(error_info, status, parsed.error->source,
                             parsed.error->message, parsed.error->offset);
            }
            return status;
        }

        const std::string json = parsed.value->dump(2);
        return writeJsonBuffer(json, json_buffer, json_buffer_size);
    });
}

const char* addc_status_string(addc_status_t status)
{
    using StatusStorage = std::underlying_type_t<addc_status_t>;
    switch (enumStorage(status))
    {
        case static_cast<StatusStorage>(ADDC_STATUS_SUCCESS):
            return "success";
        case static_cast<StatusStorage>(ADDC_STATUS_INVALID_ARGUMENT):
            return "invalid argument";
        case static_cast<StatusStorage>(ADDC_STATUS_PARSE_ERROR):
            return "CPER parse error";
        case static_cast<StatusStorage>(ADDC_STATUS_DECODE_ERROR):
            return "CPER decode error";
        case static_cast<StatusStorage>(ADDC_STATUS_INSUFFICIENT_SIZE):
            return "insufficient size";
        case static_cast<StatusStorage>(ADDC_STATUS_OUT_OF_RESOURCES):
            return "out of resources";
        case static_cast<StatusStorage>(ADDC_STATUS_INTERNAL_ERROR):
            return "internal error";
        case static_cast<StatusStorage>(ADDC_STATUS_UNSUPPORTED):
            return "unsupported operation";
        case static_cast<StatusStorage>(ADDC_STATUS_IO_ERROR):
            return "I/O error";
        default:
            return "unknown status";
    }
}

int addc_analyze_cper(const uint8_t* buf, size_t len, const char* filename,
                      char** json_out, char** error_out)
{
    return legacyStringBoundary(json_out, error_out, [&] {
        if (!buf || !json_out)
        {
            setError(error_out, "null buf or json_out pointer");
            return -1;
        }

        std::string err;
        auto result = addc::analyze_cper(std::span<const uint8_t>{buf, len},
                                         filename ? filename : "", &err);

        if (!result)
        {
            setError(error_out, err);
            return -1;
        }

        *json_out = strdupMalloc(*result);
        if (!*json_out)
        {
            setError(error_out, "allocation failed");
            return -1;
        }
        return 0;
    });
}

int addc_analyze_cper_ctx(addc_context_t* ctx, const uint8_t* buf, size_t len,
                          const char* filename, char** json_out,
                          char** error_out)
{
    return legacyStringBoundary(json_out, error_out, [&] {
        if (!ctx || !buf || !json_out)
        {
            setError(error_out, "null ctx, buf, or json_out pointer");
            return -1;
        }

        auto result =
            ctx->runtime.decode(std::span<const uint8_t>{buf, len},
                                filename ? filename : "", LIB_VERSION);

        if (!result)
        {
            setError(error_out,
                     result.error ? std::string_view{result.error->message}
                                  : std::string_view{"CPER analysis failed"});
            return -1;
        }

        auto j = addc::base::to_json(
            addc::base::resolve_report(std::move(*result.value)));
        std::string json_str = j.dump(2);
        *json_out = strdupMalloc(json_str);
        if (!*json_out)
        {
            setError(error_out, "allocation failed");
            return -1;
        }
        return 0;
    });
}

int addc_analyze_cper_file(const char* path, char** json_out, char** error_out)
{
    return legacyStringBoundary(json_out, error_out, [&] {
        if (!path || !json_out)
        {
            setError(error_out, "null path or json_out pointer");
            return -1;
        }

        std::string err;
        auto result = addc::analyze_cper_file(path, &err);

        if (!result)
        {
            setError(error_out, err);
            return -1;
        }

        *json_out = strdupMalloc(*result);
        if (!*json_out)
        {
            setError(error_out, "allocation failed");
            return -1;
        }
        return 0;
    });
}

int addc_get_error_summary(const uint8_t* buf, size_t len, const char* filename,
                           char** json_out, char** error_out)
{
    return legacyStringBoundary(json_out, error_out, [&] {
        if (!buf || !json_out)
        {
            setError(error_out, "null buf or json_out pointer");
            return -1;
        }

        std::string err;
        auto result = addc::get_error_summary(
            std::span<const uint8_t>{buf, len}, filename ? filename : "", &err);

        if (!result)
        {
            setError(error_out, err);
            return -1;
        }

        *json_out = strdupMalloc(*result);
        if (!*json_out)
        {
            setError(error_out, "allocation failed");
            return -1;
        }
        return 0;
    });
}

int addc_get_error_summary_ctx(addc_context_t* ctx, const uint8_t* buf,
                               size_t len, const char* filename,
                               char** json_out, char** error_out)
{
    return legacyStringBoundary(json_out, error_out, [&] {
        if (!ctx || !buf || !json_out)
        {
            setError(error_out, "null ctx, buf, or json_out pointer");
            return -1;
        }

        auto result =
            ctx->runtime.decode(std::span<const uint8_t>{buf, len},
                                filename ? filename : "", LIB_VERSION);

        if (!result)
        {
            setError(error_out,
                     result.error ? std::string_view{result.error->message}
                                  : std::string_view{"CPER analysis failed"});
            return -1;
        }

        auto afid_list = nlohmann::ordered_json::array();
        for (const auto& ev : result.value->events)
        {
            if (!ev.analysis.contains("afid"))
            {
                continue;
            }
            const auto& afid_arr = ev.analysis["afid"];
            if (!afid_arr.is_array() || afid_arr.empty())
            {
                continue;
            }

            std::string fru_id;
            if (ev.platform.contains("fru_id") &&
                ev.platform["fru_id"].is_string())
            {
                fru_id = ev.platform["fru_id"].get<std::string>();
            }
            std::string fru_text;
            if (ev.platform.contains("fru") && ev.platform["fru"].is_string())
            {
                fru_text = ev.platform["fru"].get<std::string>();
            }
            for (const auto& id : afid_arr)
            {
                nlohmann::ordered_json entry;
                entry["fru_id"] = fru_id;
                entry["fru_text"] = fru_text;
                entry["afid"] = id.get<int>();
                entry["additional_context"] = "";
                afid_list.push_back(std::move(entry));
            }
        }

        std::string json_str = afid_list.dump(2);
        *json_out = strdupMalloc(json_str);
        if (!*json_out)
        {
            setError(error_out, "allocation failed");
            return -1;
        }
        return 0;
    });
}

int addc_get_error_summary_file(const char* path, char** json_out,
                                char** error_out)
{
    return legacyStringBoundary(json_out, error_out, [&] {
        if (!path || !json_out)
        {
            setError(error_out, "null path or json_out pointer");
            return -1;
        }

        std::string err;
        auto result = addc::get_error_summary_file(path, &err);

        if (!result)
        {
            setError(error_out, err);
            return -1;
        }

        *json_out = strdupMalloc(*result);
        if (!*json_out)
        {
            setError(error_out, "allocation failed");
            return -1;
        }
        return 0;
    });
}

int addc_parse_cper(const uint8_t* buf, size_t len, char** json_out,
                    char** error_out)
{
    return legacyStringBoundary(json_out, error_out, [&] {
        if (!buf || !json_out)
        {
            setError(error_out, "null buf or json_out pointer");
            return -1;
        }

        const auto ir = addc::parse_cper(std::span<const uint8_t>{buf, len});
        if (!ir)
        {
            setError(error_out, "Failed to parse CPER record");
            return -1;
        }

        std::string json_str = ir->dump(2);
        *json_out = strdupMalloc(json_str);
        if (!*json_out)
        {
            setError(error_out, "allocation failed");
            return -1;
        }
        return 0;
    });
}

int addc_parse_cper_file(const char* path, char** json_out, char** error_out)
{
    return legacyStringBoundary(json_out, error_out, [&] {
        if (!path || !json_out)
        {
            setError(error_out, "null path or json_out pointer");
            return -1;
        }

        std::string error;
        const auto ir = addc::parse_cper_file(path, &error);
        if (!ir)
        {
            setError(error_out, error);
            return -1;
        }
        const std::string json = ir->dump(2);
        *json_out = strdupMalloc(json);
        if (!*json_out)
        {
            setError(error_out, "allocation failed");
            return -1;
        }
        return 0;
    });
}

int addc_mca_decode(const char* project, const addc_mca_regs_t* regs,
                    char** json_out, char** error_out)
{
    return legacyStringBoundary(json_out, error_out, [&] {
        if (!project || !regs || !json_out)
        {
            setError(error_out, "null project, regs, or json_out pointer");
            return -1;
        }

        auto json_str = addc::mca::decode_to_json(
            project, regs->status, regs->ipid, regs->synd, regs->addr,
            regs->misc0, regs->misc1);
        *json_out = strdupMalloc(json_str);
        if (!*json_out)
        {
            setError(error_out, "allocation failed");
            return -1;
        }
        return 0;
    });
}

int addc_mca_decode_ctx(addc_context_t* ctx, const char* project,
                        const addc_mca_regs_t* regs, char** json_out,
                        char** error_out)
{
    return legacyStringBoundary(json_out, error_out, [&] {
        if (!ctx || !project || !regs || !json_out)
        {
            setError(error_out, "null ctx, project, regs, or json_out pointer");
            return -1;
        }

        auto json_str = addc::mca::decode_to_json(
            project, regs->status, regs->ipid, regs->synd, regs->addr,
            regs->misc0, regs->misc1);
        *json_out = strdupMalloc(json_str);
        if (!*json_out)
        {
            setError(error_out, "allocation failed");
            return -1;
        }
        return 0;
    });
}

addc_status_t addc_mca_decode_json(
    addc_context_t* ctx, const char* project, const addc_mca_regs_t* regs,
    char* json_buffer, size_t* json_buffer_size, addc_error_info_t* error_info)
{
    return statusBoundary(error_info, [&] {
        if (!ctx || !project || !regs || !json_buffer_size)
        {
            setErrorInfo(error_info, ADDC_STATUS_INVALID_ARGUMENT, "api",
                         "null context, project, registers, or size pointer");
            return ADDC_STATUS_INVALID_ARGUMENT;
        }
        const auto* definition = addc::product::find_compiled_product(project);
        if (definition == nullptr || !definition->capabilities.mca)
        {
            setErrorInfo(error_info, ADDC_STATUS_UNSUPPORTED, "mca",
                         "unsupported_by_build: MCA project is not compiled");
            return ADDC_STATUS_UNSUPPORTED;
        }

        const std::string json = addc::mca::decode_to_json(
            project, regs->status, regs->ipid, regs->synd, regs->addr,
            regs->misc0, regs->misc1);
        return writeJsonBuffer(json, json_buffer, json_buffer_size);
    });
}

int addc_mca_supported_projects(char** json_out, char** error_out)
{
    return legacyStringBoundary(json_out, error_out, [&] {
        if (!json_out)
        {
            setError(error_out, "null json_out pointer");
            return -1;
        }

        *json_out = strdupMalloc(addc::mca_supported_projects_json());
        if (!*json_out)
        {
            setError(error_out, "allocation failed");
            return -1;
        }
        return 0;
    });
}

addc_status_t addc_mca_supported_projects_json(
    char* json_buffer, size_t* json_buffer_size, addc_error_info_t* error_info)
{
    return statusBoundary(error_info, [&] {
        if (!json_buffer_size)
        {
            setErrorInfo(error_info, ADDC_STATUS_INVALID_ARGUMENT, "api",
                         "null size pointer");
            return ADDC_STATUS_INVALID_ARGUMENT;
        }
        const std::string json = addc::mca_supported_projects_json();
        return writeJsonBuffer(json, json_buffer, json_buffer_size);
    });
}

void addc_free_string(char* s)
{
    std::free(s); // NOLINT(cppcoreguidelines-no-malloc)
}

const char* addc_version(void)
{
    return LIB_VERSION;
}

uint32_t addc_api_version(void)
{
    return ADDC_API_VERSION;
}

const char* addc_schema_version(void)
{
    return ADDC_SCHEMA_VERSION;
}

} // extern "C"
