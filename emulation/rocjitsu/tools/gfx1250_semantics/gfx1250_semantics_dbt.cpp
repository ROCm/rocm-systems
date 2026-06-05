// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
RJ_DIAGNOSTIC_POP

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/executable.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;

struct InputBuffer {
  std::string name;
  uint64_t addr = 0;
  std::string path;
  std::vector<uint8_t> bytes;
};

struct OutputBuffer {
  std::string name;
  uint64_t addr = 0;
  size_t size = 0;
};

struct Options {
  std::string case_name;
  std::string program;
  std::string kernel;
  std::string output_dir;
  uint32_t grid_size = 32;
  uint16_t block_size = 32;
  std::string kernarg_path;
  std::vector<uint8_t> kernarg_bytes;
  bool native_code_object = false;
  std::vector<InputBuffer> inputs;
  std::vector<OutputBuffer> outputs;
};

struct HostTarget {
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  uint32_t mach = 0;
  char isa_name[128]{};
};

struct HsaState {
  hsa_agent_t cpu{};
  hsa_agent_t gpu{};
  HostTarget target{};
};

struct Allocation {
  uint64_t guest_addr = 0;
  size_t size = 0;
  void *device_ptr = nullptr;
};

[[noreturn]] void fail_usage(std::string_view message) {
  std::cerr << "error: " << message << "\n\n"
            << "usage: gfx1250_semantics_dbt"
            << " --case NAME --program FILE --kernel SYMBOL --output-dir DIR"
            << " --grid N --block N --kernarg FILE [--native-code-object]"
            << " --input NAME:ADDR:FILE [--input ...]"
            << " --output NAME:ADDR:BYTES [--output ...]\n";
  std::exit(2);
}

uint64_t parse_u64(std::string_view value) {
  std::string owned(value);
  size_t pos = 0;
  uint64_t parsed = 0;
  try {
    parsed = std::stoull(owned, &pos, 0);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid integer: " + owned);
  }
  if (pos != owned.size())
    throw std::runtime_error("invalid integer: " + owned);
  return parsed;
}

std::vector<std::string_view> split_spec(std::string_view spec, size_t expected) {
  std::vector<std::string_view> parts;
  size_t start = 0;
  while (parts.size() + 1 < expected) {
    const size_t pos = spec.find(':', start);
    if (pos == std::string_view::npos)
      throw std::runtime_error("invalid spec, expected ':' separators");
    parts.push_back(spec.substr(start, pos - start));
    start = pos + 1;
  }
  parts.push_back(spec.substr(start));
  for (std::string_view part : parts)
    if (part.empty())
      throw std::runtime_error("invalid spec, empty field");
  return parts;
}

Options parse_args(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    auto next = [&]() -> std::string_view {
      if (++i >= argc)
        fail_usage("missing value for " + std::string(arg));
      return argv[i];
    };

    if (arg == "--case") {
      opts.case_name = std::string(next());
    } else if (arg == "--program") {
      opts.program = std::string(next());
    } else if (arg == "--kernel") {
      opts.kernel = std::string(next());
    } else if (arg == "--output-dir") {
      opts.output_dir = std::string(next());
    } else if (arg == "--grid") {
      opts.grid_size = static_cast<uint32_t>(parse_u64(next()));
    } else if (arg == "--block") {
      opts.block_size = static_cast<uint16_t>(parse_u64(next()));
    } else if (arg == "--kernarg") {
      opts.kernarg_path = std::string(next());
    } else if (arg == "--native-code-object") {
      opts.native_code_object = true;
    } else if (arg == "--input") {
      auto parts = split_spec(next(), 3);
      opts.inputs.push_back(
          InputBuffer{std::string(parts[0]), parse_u64(parts[1]), std::string(parts[2]), {}});
    } else if (arg == "--output") {
      auto parts = split_spec(next(), 3);
      opts.outputs.push_back(OutputBuffer{std::string(parts[0]), parse_u64(parts[1]),
                                          static_cast<size_t>(parse_u64(parts[2]))});
    } else {
      fail_usage("unknown argument " + std::string(arg));
    }
  }

  if (opts.case_name.empty())
    fail_usage("--case is required");
  if (opts.program.empty())
    fail_usage("--program is required");
  if (opts.kernel.empty())
    fail_usage("--kernel is required");
  if (opts.output_dir.empty())
    fail_usage("--output-dir is required");
  if (opts.kernarg_path.empty())
    fail_usage("--kernarg is required");
  if (opts.outputs.empty())
    fail_usage("at least one --output is required");
  return opts;
}

std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("failed to open " + path + ": " + std::strerror(errno));
  in.seekg(0, std::ios::end);
  const auto size = in.tellg();
  if (size < 0)
    throw std::runtime_error("failed to size " + path);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  in.seekg(0, std::ios::beg);
  if (!bytes.empty())
    in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!in && !bytes.empty())
    throw std::runtime_error("failed to read " + path);
  return bytes;
}

void write_file(const std::filesystem::path &path, const std::vector<uint8_t> &bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out)
    throw std::runtime_error("failed to open " + path.string() + " for writing");
  if (!bytes.empty())
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  if (!out)
    throw std::runtime_error("failed to write " + path.string());
}

std::string json_escape(std::string_view value) {
  std::string out;
  for (char c : value) {
    switch (c) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out += c;
      break;
    }
  }
  return out;
}

std::string hex_u64(uint64_t value) {
  std::ostringstream os;
  os << "0x" << std::hex << std::uppercase << value;
  return os.str();
}

void write_metadata(const Options &opts, const HostTarget &target,
                    const std::vector<std::string> &warnings) {
  std::filesystem::path path = std::filesystem::path(opts.output_dir) / "dbt_metadata.json";
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::trunc);
  if (!out)
    throw std::runtime_error("failed to open " + path.string() + " for writing");

  out << "{\n";
  out << "  \"schema\": \"gfx1250-semantics-dbt-v1\",\n";
  out << "  \"implementation\": \"gfx1250_to_rdna4_dbt\",\n";
  out << "  \"case\": \"" << json_escape(opts.case_name) << "\",\n";
  out << "  \"program\": \"" << json_escape(opts.program) << "\",\n";
  out << "  \"kernel\": \"" << json_escape(opts.kernel) << "\",\n";
  out << "  \"guest_target\": \"gfx1250\",\n";
  out << "  \"host_isa\": \"" << json_escape(target.isa_name) << "\",\n";
  out << "  \"host_mach\": " << target.mach << ",\n";
  out << "  \"native_code_object\": " << (opts.native_code_object ? "true" : "false") << ",\n";
  out << "  \"grid_size_x\": " << opts.grid_size << ",\n";
  out << "  \"workgroup_size_x\": " << opts.block_size << ",\n";
  out << "  \"translation_warnings\": [";
  for (size_t i = 0; i < warnings.size(); ++i) {
    out << (i == 0 ? "\n" : ",\n") << "    \"" << json_escape(warnings[i]) << "\"";
  }
  out << (warnings.empty() ? "" : "\n  ") << "]\n";
  out << "}\n";
}

void check_hsa(hsa_status_t status, std::string_view action) {
  if (status == HSA_STATUS_SUCCESS)
    return;
  const char *message = nullptr;
  hsa_status_string(status, &message);
  throw std::runtime_error(std::string(action) +
                           " failed: " + (message ? message : "unknown HSA error"));
}

hsa_agent_t find_agent(hsa_device_type_t wanted) {
  struct Ctx {
    hsa_device_type_t wanted;
    hsa_agent_t agent{};
  } ctx{wanted, {}};

  hsa_status_t status = hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        auto *ctx = static_cast<Ctx *>(data);
        hsa_device_type_t type{};
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type == ctx->wanted) {
          ctx->agent = agent;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &ctx);
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
    check_hsa(status, "hsa_iterate_agents");
  if (ctx.agent.handle == 0)
    throw std::runtime_error("required HSA agent not found");
  return ctx.agent;
}

hsa_amd_memory_pool_t find_pool(hsa_agent_t agent, hsa_amd_segment_t segment,
                                bool host_accessible = false) {
  struct Ctx {
    hsa_amd_segment_t segment;
    bool host_accessible;
    hsa_amd_memory_pool_t pool{};
  } ctx{segment, host_accessible, {}};

  hsa_status_t status = hsa_amd_agent_iterate_memory_pools(
      agent,
      [](hsa_amd_memory_pool_t pool, void *data) -> hsa_status_t {
        auto *ctx = static_cast<Ctx *>(data);
        hsa_amd_segment_t segment{};
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
        if (segment != ctx->segment)
          return HSA_STATUS_SUCCESS;
        if (ctx->host_accessible) {
          bool accessible = false;
          hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL,
                                       &accessible);
          if (!accessible)
            return HSA_STATUS_SUCCESS;
        }
        ctx->pool = pool;
        return HSA_STATUS_INFO_BREAK;
      },
      &ctx);
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
    check_hsa(status, "hsa_amd_agent_iterate_memory_pools");
  if (ctx.pool.handle == 0)
    throw std::runtime_error("required HSA memory pool not found");
  return ctx.pool;
}

HostTarget select_host_target(hsa_agent_t gpu) {
  HostTarget target{};
  hsa_isa_t isa{};
  check_hsa(hsa_agent_get_info(gpu, HSA_AGENT_INFO_ISA, &isa), "hsa_agent_get_info ISA");
  check_hsa(hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, target.isa_name),
            "hsa_isa_get_info_alt name");
  if (std::strstr(target.isa_name, "gfx1201")) {
    target.arch = ROCJITSU_CODE_ARCH_RDNA4;
    target.mach = EF_AMDGPU_MACH_AMDGCN_GFX1201;
  } else if (std::strstr(target.isa_name, "gfx1200")) {
    target.arch = ROCJITSU_CODE_ARCH_RDNA4;
    target.mach = EF_AMDGPU_MACH_AMDGCN_GFX1200;
  }
  if (target.mach == 0)
    throw std::runtime_error(std::string("gfx1250 DBT corpus run requires an RDNA4 GPU, found: ") +
                             target.isa_name);
  return target;
}

HsaState init_hsa() {
  check_hsa(hsa_init(), "hsa_init");
  HsaState state{};
  state.cpu = find_agent(HSA_DEVICE_TYPE_CPU);
  state.gpu = find_agent(HSA_DEVICE_TYPE_GPU);
  state.target = select_host_target(state.gpu);
  return state;
}

uint64_t read_le_u64(const uint8_t *bytes) {
  uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
  return value;
}

void write_le_u64(uint8_t *bytes, uint64_t value) {
  for (unsigned i = 0; i < 8; ++i)
    bytes[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xFFu);
}

void rewrite_kernarg_pointers(std::vector<uint8_t> &kernarg,
                              const std::vector<Allocation> &allocations) {
  if (kernarg.size() < sizeof(uint64_t))
    return;

  std::unordered_map<uint64_t, uint64_t> pointer_map;
  for (const auto &allocation : allocations) {
    pointer_map.emplace(allocation.guest_addr, reinterpret_cast<uint64_t>(allocation.device_ptr));
  }

  for (size_t offset = 0; offset + sizeof(uint64_t) <= kernarg.size(); offset += sizeof(uint64_t)) {
    const uint64_t guest_addr = read_le_u64(kernarg.data() + offset);
    auto it = pointer_map.find(guest_addr);
    if (it != pointer_map.end())
      write_le_u64(kernarg.data() + offset, it->second);
  }
}

std::vector<Allocation> plan_allocations(const Options &opts) {
  std::vector<Allocation> allocations;
  auto add_range = [&](uint64_t addr, size_t size) {
    auto it = std::ranges::find_if(allocations, [addr](const Allocation &allocation) {
      return allocation.guest_addr == addr;
    });
    if (it == allocations.end()) {
      allocations.push_back(Allocation{addr, size, nullptr});
    } else {
      it->size = std::max(it->size, size);
    }
  };

  for (const auto &input : opts.inputs)
    add_range(input.addr, input.bytes.size());
  for (const auto &output : opts.outputs)
    add_range(output.addr, output.size);
  return allocations;
}

Allocation *find_allocation(std::vector<Allocation> &allocations, uint64_t addr) {
  auto it = std::ranges::find_if(
      allocations, [addr](const Allocation &allocation) { return allocation.guest_addr == addr; });
  if (it == allocations.end())
    throw std::runtime_error("no HSA allocation for guest address " + hex_u64(addr));
  return &*it;
}

void allocate_and_initialize_buffers(HsaState &state, std::vector<Allocation> &allocations,
                                     const Options &opts) {
  hsa_amd_memory_pool_t gpu_pool = find_pool(state.gpu, HSA_AMD_SEGMENT_GLOBAL);
  hsa_agent_t agents[] = {state.cpu, state.gpu};
  for (auto &allocation : allocations) {
    check_hsa(hsa_amd_memory_pool_allocate(gpu_pool, allocation.size, 0, &allocation.device_ptr),
              "hsa_amd_memory_pool_allocate");
    check_hsa(hsa_amd_agents_allow_access(2, agents, nullptr, allocation.device_ptr),
              "hsa_amd_agents_allow_access");
    std::vector<uint8_t> zero(allocation.size, 0);
    check_hsa(hsa_memory_copy(allocation.device_ptr, zero.data(), zero.size()),
              "hsa_memory_copy zero");
  }

  for (const auto &input : opts.inputs) {
    Allocation *allocation = find_allocation(allocations, input.addr);
    check_hsa(hsa_memory_copy(allocation->device_ptr, input.bytes.data(), input.bytes.size()),
              "hsa_memory_copy input");
  }
}

std::vector<uint8_t> translate_code_object(const Options &opts, const HostTarget &target,
                                           std::vector<std::string> &warnings) {
  if (opts.native_code_object)
    return read_file(opts.program);

  Executable exec(opts.program);
  if (!exec.is_valid())
    throw std::runtime_error("failed to load HIP fatbin from " + opts.program);
  const AmdGpuCodeObject *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX1250, 0);
  if (!co)
    throw std::runtime_error("no gfx1250 code object found in " + opts.program);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, target.arch, target.mach);
  auto result = translator.translate(*co);
  warnings = std::move(result.warnings);
  if (result.elf_bytes.empty())
    throw std::runtime_error("gfx1250 -> RDNA4 translation produced an empty code object");
  if (!warnings.empty())
    throw std::runtime_error("gfx1250 -> RDNA4 translation produced warning: " + warnings.front());
  return std::move(result.elf_bytes);
}

uint64_t load_kernel_object(HsaState &state, const std::vector<uint8_t> &elf_bytes,
                            const std::string &kernel, hsa_executable_t &executable,
                            hsa_code_object_reader_t &reader, uint32_t &group_segment_size,
                            uint32_t &private_segment_size) {
  check_hsa(hsa_code_object_reader_create_from_memory(elf_bytes.data(), elf_bytes.size(), &reader),
            "hsa_code_object_reader_create_from_memory");
  check_hsa(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                      nullptr, &executable),
            "hsa_executable_create_alt");
  check_hsa(hsa_executable_load_agent_code_object(executable, state.gpu, reader, nullptr, nullptr),
            "hsa_executable_load_agent_code_object");
  check_hsa(hsa_executable_freeze(executable, nullptr), "hsa_executable_freeze");

  hsa_executable_symbol_t symbol{};
  const std::string kd_symbol = kernel + ".kd";
  check_hsa(hsa_executable_get_symbol_by_name(executable, kd_symbol.c_str(), &state.gpu, &symbol),
            "hsa_executable_get_symbol_by_name");

  uint64_t kernel_object = 0;
  check_hsa(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                           &kernel_object),
            "hsa_executable_symbol_get_info kernel object");
  check_hsa(hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_segment_size),
            "hsa_executable_symbol_get_info group segment size");
  check_hsa(hsa_executable_symbol_get_info(symbol,
                                           HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                           &private_segment_size),
            "hsa_executable_symbol_get_info private segment size");
  if (kernel_object == 0)
    throw std::runtime_error("HSA returned a null kernel object");
  return kernel_object;
}

void dispatch_kernel(HsaState &state, uint64_t kernel_object, uint32_t group_segment_size,
                     uint32_t private_segment_size, const Options &opts,
                     std::vector<uint8_t> patched_kernarg) {
  hsa_amd_memory_pool_t kernarg_pool = find_pool(state.cpu, HSA_AMD_SEGMENT_GLOBAL, true);
  hsa_agent_t agents[] = {state.cpu, state.gpu};
  void *kernarg = nullptr;
  const size_t kernarg_size = std::max<size_t>(patched_kernarg.size(), 256);
  check_hsa(hsa_amd_memory_pool_allocate(kernarg_pool, kernarg_size, 0, &kernarg),
            "hsa_amd_memory_pool_allocate kernarg");
  check_hsa(hsa_amd_agents_allow_access(2, agents, nullptr, kernarg),
            "hsa_amd_agents_allow_access kernarg");
  std::memset(kernarg, 0, kernarg_size);
  std::memcpy(kernarg, patched_kernarg.data(), patched_kernarg.size());

  hsa_queue_t *queue = nullptr;
  uint32_t queue_size = 0;
  check_hsa(hsa_agent_get_info(state.gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size),
            "hsa_agent_get_info queue max size");
  check_hsa(hsa_queue_create(state.gpu, queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                             UINT32_MAX, UINT32_MAX, &queue),
            "hsa_queue_create");

  hsa_signal_t signal{};
  check_hsa(hsa_signal_create(1, 0, nullptr, &signal), "hsa_signal_create");

  uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
  auto *aql = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
              (write_idx & (queue->size - 1));

  std::memset(aql, 0, sizeof(*aql));
  aql->setup = 1;
  aql->workgroup_size_x = opts.block_size;
  aql->workgroup_size_y = 1;
  aql->workgroup_size_z = 1;
  aql->grid_size_x = opts.grid_size;
  aql->grid_size_y = 1;
  aql->grid_size_z = 1;
  aql->private_segment_size = private_segment_size;
  aql->group_segment_size = group_segment_size;
  aql->kernel_object = kernel_object;
  aql->kernarg_address = kernarg;
  aql->completion_signal = signal;

  uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  header |= 1 << HSA_PACKET_HEADER_BARRIER;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  __atomic_store_n(reinterpret_cast<uint16_t *>(aql), header, __ATOMIC_RELEASE);
  hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);

  hsa_signal_value_t value = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                                       5'000'000'000ULL, HSA_WAIT_STATE_BLOCKED);
  if (value != 0)
    throw std::runtime_error("kernel dispatch timed out or failed");

  check_hsa(hsa_signal_destroy(signal), "hsa_signal_destroy");
  check_hsa(hsa_queue_destroy(queue), "hsa_queue_destroy");
  check_hsa(hsa_amd_memory_pool_free(kernarg), "hsa_amd_memory_pool_free kernarg");
}

void write_outputs(const Options &opts, std::vector<Allocation> &allocations) {
  const std::filesystem::path root(opts.output_dir);
  for (const auto &output : opts.outputs) {
    Allocation *allocation = find_allocation(allocations, output.addr);
    std::vector<uint8_t> bytes(output.size);
    check_hsa(hsa_memory_copy(bytes.data(), allocation->device_ptr, output.size),
              "hsa_memory_copy output");
    write_file(root / "outputs" / (output.name + ".bin"), bytes);
  }
}

void free_allocations(std::vector<Allocation> &allocations) {
  for (auto &allocation : allocations) {
    if (allocation.device_ptr)
      hsa_amd_memory_pool_free(allocation.device_ptr);
    allocation.device_ptr = nullptr;
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    Options opts = parse_args(argc, argv);
    opts.kernarg_bytes = read_file(opts.kernarg_path);
    for (auto &input : opts.inputs)
      input.bytes = read_file(input.path);

    std::filesystem::remove_all(opts.output_dir);
    std::filesystem::create_directories(std::filesystem::path(opts.output_dir) / "outputs");

    HsaState state = init_hsa();
    std::vector<std::string> warnings;
    std::vector<uint8_t> translated = translate_code_object(opts, state.target, warnings);

    hsa_executable_t executable{};
    hsa_code_object_reader_t reader{};
    uint32_t group_segment_size = 0;
    uint32_t private_segment_size = 0;
    uint64_t kernel_object = load_kernel_object(state, translated, opts.kernel, executable, reader,
                                                group_segment_size, private_segment_size);

    std::vector<Allocation> allocations = plan_allocations(opts);
    allocate_and_initialize_buffers(state, allocations, opts);
    rewrite_kernarg_pointers(opts.kernarg_bytes, allocations);

    dispatch_kernel(state, kernel_object, group_segment_size, private_segment_size, opts,
                    opts.kernarg_bytes);
    write_outputs(opts, allocations);
    write_metadata(opts, state.target, warnings);

    free_allocations(allocations);
    check_hsa(hsa_executable_destroy(executable), "hsa_executable_destroy");
    check_hsa(hsa_code_object_reader_destroy(reader), "hsa_code_object_reader_destroy");
    check_hsa(hsa_shut_down(), "hsa_shut_down");

    std::cout << "gfx1250 DBT corpus output: " << opts.output_dir << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    hsa_shut_down();
    return 1;
  }
}
