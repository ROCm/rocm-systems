// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "tools/hsa_run_kernel.h"

#include <charconv>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace rocjitsu::tools;

namespace {

constexpr int kUsageError = 1;

struct CliStorage {
  std::vector<std::pair<std::string, std::string>> output_paths;
};

struct CliOptions {
  HsaRunOptions run;
  CliStorage storage;
  bool show_help = false;
};

void print_help() {
  std::cout
      << "Usage: rj_hsa_run --code-object PATH --kernel NAME --grid X[,Y,Z] "
         "--workgroup X[,Y,Z] [options]\n\n"
      << "Options:\n"
      << "  --code-object PATH              AMDGPU code-object ELF\n"
      << "  --kernel NAME                   Kernel name; default symbol is <NAME>.kd\n"
      << "  --kernel-symbol SYMBOL          Override kernel descriptor symbol\n"
      << "  --grid X[,Y,Z]                  Dispatch grid size\n"
      << "  --workgroup X[,Y,Z]             Dispatch workgroup size\n"
      << "  --agent-index N                 Select Nth GPU agent (default: 0)\n"
      << "  --require-agent-isa TEXT        Require selected GPU ISA to contain TEXT\n"
      << "  --timeout-ms N                  Dispatch timeout in milliseconds (default: 5000)\n"
      << "  --buffer SPEC                   NAME:size=BYTES[:input=PATH][:output=PATH][:zero]\n"
      << "  --kernarg-size BYTES            Size of kernarg allocation\n"
      << "  --kernarg-blob PATH             Initial raw kernarg bytes before patches\n"
      << "  --arg-ptr OFFSET=BUFFER         Write device pointer argument\n"
      << "  --arg-u32 OFFSET=VALUE          Write uint32 argument\n"
      << "  --arg-u64 OFFSET=VALUE          Write uint64 argument\n"
      << "  --arg-i32 OFFSET=VALUE          Write int32 argument\n"
      << "  --arg-f32 OFFSET=VALUE          Write float argument\n"
      << "  --arg-bytes OFFSET=HEX          Write raw hex bytes\n"
      << "  --help                          Show this help\n";
}

[[nodiscard]] bool parse_u64(std::string_view text, uint64_t &value) {
  int base = 10;
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2);
    base = 16;
  }
  auto *begin = text.data();
  auto *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value, base);
  return ec == std::errc{} && ptr == end;
}

[[nodiscard]] bool parse_u32(std::string_view text, uint32_t &value) {
  uint64_t parsed = 0;
  if (!parse_u64(text, parsed) || parsed > UINT32_MAX)
    return false;
  value = static_cast<uint32_t>(parsed);
  return true;
}

[[nodiscard]] bool parse_u16(std::string_view text, uint16_t &value) {
  uint64_t parsed = 0;
  if (!parse_u64(text, parsed) || parsed > UINT16_MAX)
    return false;
  value = static_cast<uint16_t>(parsed);
  return true;
}

[[nodiscard]] bool parse_size(std::string_view text, size_t &value) {
  uint64_t parsed = 0;
  if (!parse_u64(text, parsed))
    return false;
  value = static_cast<size_t>(parsed);
  return static_cast<uint64_t>(value) == parsed;
}

[[nodiscard]] bool read_file(const std::string &path, std::vector<uint8_t> &bytes) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
    return false;
  const auto size = in.tellg();
  if (size < 0)
    return false;
  bytes.resize(static_cast<size_t>(size));
  in.seekg(0, std::ios::beg);
  in.read(reinterpret_cast<char *>(bytes.data()), size);
  return static_cast<bool>(in);
}

[[nodiscard]] bool write_file(const std::string &path, const std::vector<uint8_t> &bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    return false;
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(out);
}

[[nodiscard]] bool require_value(int argc, char **argv, int &index, std::string_view flag,
                                 std::string_view &value) {
  if (index + 1 >= argc) {
    std::cerr << "missing value for " << flag << "\n";
    return false;
  }
  ++index;
  value = argv[index];
  return true;
}

[[nodiscard]] std::vector<std::string_view> split(std::string_view text, char delimiter) {
  std::vector<std::string_view> parts;
  while (true) {
    const size_t pos = text.find(delimiter);
    if (pos == std::string_view::npos) {
      parts.push_back(text);
      return parts;
    }
    parts.push_back(text.substr(0, pos));
    text.remove_prefix(pos + 1);
  }
}

[[nodiscard]] bool parse_grid(std::string_view text, uint32_t &x, uint32_t &y, uint32_t &z) {
  const auto parts = split(text, ',');
  if (parts.empty() || parts.size() > 3)
    return false;
  y = 1;
  z = 1;
  if (!parse_u32(parts[0], x))
    return false;
  if (parts.size() > 1 && !parse_u32(parts[1], y))
    return false;
  if (parts.size() > 2 && !parse_u32(parts[2], z))
    return false;
  return x != 0 && y != 0 && z != 0;
}

[[nodiscard]] bool parse_workgroup(std::string_view text, uint16_t &x, uint16_t &y, uint16_t &z) {
  const auto parts = split(text, ',');
  if (parts.empty() || parts.size() > 3)
    return false;
  y = 1;
  z = 1;
  if (!parse_u16(parts[0], x))
    return false;
  if (parts.size() > 1 && !parse_u16(parts[1], y))
    return false;
  if (parts.size() > 2 && !parse_u16(parts[2], z))
    return false;
  return x != 0 && y != 0 && z != 0;
}

[[nodiscard]] std::optional<uint8_t> hex_nibble(char c) {
  if (c >= '0' && c <= '9')
    return static_cast<uint8_t>(c - '0');
  if (c >= 'a' && c <= 'f')
    return static_cast<uint8_t>(10 + c - 'a');
  if (c >= 'A' && c <= 'F')
    return static_cast<uint8_t>(10 + c - 'A');
  return std::nullopt;
}

[[nodiscard]] bool parse_hex_bytes(std::string_view text, std::vector<uint8_t> &bytes) {
  if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X'))
    text.remove_prefix(2);
  if (text.size() % 2 != 0)
    return false;

  bytes.clear();
  bytes.reserve(text.size() / 2);
  for (size_t i = 0; i < text.size(); i += 2) {
    auto hi = hex_nibble(text[i]);
    auto lo = hex_nibble(text[i + 1]);
    if (!hi || !lo)
      return false;
    bytes.push_back(static_cast<uint8_t>((*hi << 4) | *lo));
  }
  return true;
}

template <typename T> std::vector<uint8_t> scalar_bytes(T value) {
  std::vector<uint8_t> bytes(sizeof(T));
  std::memcpy(bytes.data(), &value, sizeof(T));
  return bytes;
}

[[nodiscard]] bool parse_offset_value(std::string_view text, size_t &offset,
                                      std::string_view &value) {
  const size_t eq = text.find('=');
  if (eq == std::string_view::npos)
    return false;
  if (!parse_size(text.substr(0, eq), offset))
    return false;
  value = text.substr(eq + 1);
  return !value.empty();
}

[[nodiscard]] bool parse_buffer(std::string_view spec, CliOptions &options) {
  const auto parts = split(spec, ':');
  if (parts.empty() || parts[0].empty())
    return false;

  HsaBufferSpec buffer;
  buffer.name = std::string(parts[0]);
  std::string output_path;

  for (size_t i = 1; i < parts.size(); ++i) {
    const std::string_view part = parts[i];
    if (part == "zero") {
      buffer.zero_fill = true;
      continue;
    }

    const size_t eq = part.find('=');
    if (eq == std::string_view::npos)
      return false;

    const std::string_view key = part.substr(0, eq);
    const std::string_view value = part.substr(eq + 1);
    if (key == "size") {
      if (!parse_size(value, buffer.size))
        return false;
    } else if (key == "input") {
      if (!read_file(std::string(value), buffer.input)) {
        std::cerr << "failed to read buffer input: " << value << "\n";
        return false;
      }
    } else if (key == "output") {
      buffer.copy_output = true;
      output_path = std::string(value);
    } else {
      return false;
    }
  }

  if (buffer.name.empty() || buffer.size == 0)
    return false;
  if (buffer.copy_output)
    options.storage.output_paths.push_back({buffer.name, output_path});
  options.run.buffers.push_back(buffer);
  return true;
}

[[nodiscard]] bool add_arg_patch(std::string_view spec, KernelArgKind kind, CliOptions &options) {
  size_t offset = 0;
  std::string_view value;
  if (!parse_offset_value(spec, offset, value))
    return false;

  KernelArgPatch patch;
  patch.offset = offset;
  patch.kind = kind;

  switch (kind) {
  case KernelArgKind::Pointer:
    patch.buffer_name = std::string(value);
    break;
  case KernelArgKind::U32: {
    uint32_t parsed = 0;
    if (!parse_u32(value, parsed))
      return false;
    patch.bytes = scalar_bytes(parsed);
    break;
  }
  case KernelArgKind::U64: {
    uint64_t parsed = 0;
    if (!parse_u64(value, parsed))
      return false;
    patch.bytes = scalar_bytes(parsed);
    break;
  }
  case KernelArgKind::I32: {
    int value_i32 = 0;
    auto *begin = value.data();
    auto *end = value.data() + value.size();
    auto [ptr, ec] = std::from_chars(begin, end, value_i32);
    if (ec != std::errc{} || ptr != end)
      return false;
    patch.bytes = scalar_bytes(static_cast<int32_t>(value_i32));
    break;
  }
  case KernelArgKind::F32: {
    const std::string owned(value);
    char *end = nullptr;
    const float parsed = std::strtof(owned.c_str(), &end);
    if (end == nullptr || *end != '\0')
      return false;
    patch.bytes = scalar_bytes(parsed);
    break;
  }
  case KernelArgKind::RawBytes:
    if (!parse_hex_bytes(value, patch.bytes))
      return false;
    break;
  }

  options.run.arg_patches.push_back(std::move(patch));
  return true;
}

[[nodiscard]] bool parse_args(int argc, char **argv, CliOptions &options) {
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    std::string_view value;

    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
      return true;
    }

    if (arg == "--code-object") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      options.run.code_object_path = std::string(value);
    } else if (arg == "--kernel") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      options.run.kernel_name = std::string(value);
    } else if (arg == "--kernel-symbol") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      options.run.kernel_symbol = std::string(value);
    } else if (arg == "--grid") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!parse_grid(value, options.run.grid_x, options.run.grid_y, options.run.grid_z)) {
        std::cerr << "invalid grid: " << value << "\n";
        return false;
      }
    } else if (arg == "--workgroup") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!parse_workgroup(value, options.run.workgroup_x, options.run.workgroup_y,
                           options.run.workgroup_z)) {
        std::cerr << "invalid workgroup: " << value << "\n";
        return false;
      }
    } else if (arg == "--agent-index") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!parse_u32(value, options.run.agent_index)) {
        std::cerr << "invalid agent index: " << value << "\n";
        return false;
      }
    } else if (arg == "--require-agent-isa") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      options.run.require_agent_isa = std::string(value);
    } else if (arg == "--timeout-ms") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!parse_u64(value, options.run.timeout_ms)) {
        std::cerr << "invalid timeout: " << value << "\n";
        return false;
      }
    } else if (arg == "--buffer") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!parse_buffer(value, options)) {
        std::cerr << "invalid buffer spec: " << value << "\n";
        return false;
      }
    } else if (arg == "--kernarg-size") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!parse_size(value, options.run.kernarg_size)) {
        std::cerr << "invalid kernarg size: " << value << "\n";
        return false;
      }
    } else if (arg == "--kernarg-blob") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!read_file(std::string(value), options.run.kernarg_template)) {
        std::cerr << "failed to read kernarg blob: " << value << "\n";
        return false;
      }
    } else if (arg == "--arg-ptr") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!add_arg_patch(value, KernelArgKind::Pointer, options))
        return false;
    } else if (arg == "--arg-u32") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!add_arg_patch(value, KernelArgKind::U32, options))
        return false;
    } else if (arg == "--arg-u64") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!add_arg_patch(value, KernelArgKind::U64, options))
        return false;
    } else if (arg == "--arg-i32") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!add_arg_patch(value, KernelArgKind::I32, options))
        return false;
    } else if (arg == "--arg-f32") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!add_arg_patch(value, KernelArgKind::F32, options))
        return false;
    } else if (arg == "--arg-bytes") {
      if (!require_value(argc, argv, i, arg, value))
        return false;
      if (!add_arg_patch(value, KernelArgKind::RawBytes, options))
        return false;
    } else {
      std::cerr << "unknown option: " << arg << "\n";
      return false;
    }
  }

  return true;
}

[[nodiscard]] std::optional<std::string> output_path_for(const CliStorage &storage,
                                                         const std::string &name) {
  for (const auto &[buffer_name, path] : storage.output_paths) {
    if (buffer_name == name)
      return path;
  }
  return std::nullopt;
}

} // namespace

int main(int argc, char **argv) {
  CliOptions options;
  if (!parse_args(argc, argv, options)) {
    print_help();
    return kUsageError;
  }

  if (options.show_help) {
    print_help();
    return 0;
  }

  if (options.run.code_object_path.empty() || options.run.kernel_name.empty()) {
    std::cerr << "--code-object and --kernel are required\n";
    print_help();
    return kUsageError;
  }

  auto result = run_hsa_kernel(options.run);
  if (!result.ok()) {
    for (const auto &error : result.errors)
      std::cerr << "error: " << error.message << "\n";
    return result.errors.front().exit_code;
  }

  for (const auto &output : result.value.outputs) {
    auto path = output_path_for(options.storage, output.name);
    if (!path) {
      std::cerr << "error: no output path registered for buffer " << output.name << "\n";
      return 5;
    }
    if (!write_file(*path, output.bytes)) {
      std::cerr << "error: failed to write output buffer " << output.name << " to " << *path
                << "\n";
      return 5;
    }
  }

  std::cerr << "agent=" << result.value.agent_isa << " elapsed_ns=" << result.value.elapsed_ns
            << "\n";
  return 0;
}
