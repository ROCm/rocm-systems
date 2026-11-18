// sqtt_site_injector: rocprofiler-sdk tool that adds SQTT site markers to GPU
// code objects as they are loaded, so unmodified binaries and libraries can be
// attributed with counter_fit.py.
//
// It wraps hsa_code_object_reader_create_from_memory and _from_file, which
// HIP and the ROCm libraries use for every code object they load. Each AMDGPU
// ELF is passed through sqtt_site_rewrite.py; on success the reader is created
// from the rewritten bytes, otherwise from the original ones.
//
// The application must never break because of this tool, so every step falls
// back to the original code object:
//   * the rewriter fails, finds nothing to mark, or fails its own checks;
//   * the rewriter runs longer than SQTT_SITE_TIMEOUT seconds (it is killed);
//   * the output is not an AMDGPU ELF;
//   * the runtime rejects the rewritten bytes when the reader is created, or
//     when the code object is loaded into an executable (the load is retried
//     with a reader for the original bytes).
//
// Build:
//   g++ -std=c++17 -shared -fPIC -O2 sqtt_site_injector.cpp -o libsqtt_site_injector.so
//       -I<rocprofiler-sdk>/include -I/opt/rocm/include -L<rocprofiler-sdk>/lib -lrocprofiler-sdk
// Use (works alongside rocprofv3, which appends its own tool):
//   ROCP_TOOL_LIBRARIES=/path/libsqtt_site_injector.so
//   SQTT_SITE_REWRITE=/path/sqtt_site_rewrite.py
//   SQTT_SITE_ARGS="--kernel lds_conflict"     (optional rewriter arguments)
//   rocprofv3 --att ... -- ./app
// Optional: SQTT_SITE_PYTHON (default python3), SQTT_SITE_TIMEOUT (seconds,
// default 600), SQTT_SITE_KEEP=1 to keep the temporary files.

#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr uint16_t EM_AMDGPU = 224;

decltype(hsa_code_object_reader_create_from_memory)* original_from_memory = nullptr;
decltype(hsa_code_object_reader_create_from_file)* original_from_file = nullptr;
decltype(hsa_executable_load_agent_code_object)* original_load_agent = nullptr;
decltype(hsa_executable_load_program_code_object)* original_load_program = nullptr;

std::atomic<int> next_object{0};
std::mutex rewrite_mutex;  // one rewrite at a time keeps temp files and logs simple

// Readers created from rewritten bytes, with the original bytes to fall back to.
std::mutex readers_mutex;
std::map<uint64_t, std::shared_ptr<std::vector<char>>> original_of_reader;

const char* env_or(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value && *value ? value : fallback;
}

bool is_amdgpu_elf(const void* data, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  if (size < 64 || std::memcmp(bytes, "\x7f" "ELF", 4) != 0) return false;
  if (bytes[4] != 2 || bytes[5] != 1) return false;  // ELF64, little-endian
  uint16_t machine;
  std::memcpy(&machine, bytes + 18, sizeof(machine));
  return machine == EM_AMDGPU;
}

void log(const char* fmt, const std::string& a, size_t size, const std::string& b) {
  std::fprintf(stderr, fmt, a.c_str(), size, b.c_str());
}

// Runs `python script in out [SQTT_SITE_ARGS...]` with stdout in log (stderr in log.err)
// and returns its exit status, or -1 if it could not start, crashed or timed
// out. It is spawned directly, not through a shell, with only PATH and HOME
// set, so neither the shell nor the rewriter inherits the profiler's injection
// (LD_PRELOAD, ROCP_TOOL_LIBRARIES, ...). It runs in its own process group, so
// a timeout also kills the LLVM tools it started.
int run_rewriter(const char* script, const std::string& in, const std::string& out,
                 const std::string& log) {
  std::vector<std::string> args = {env_or("SQTT_SITE_PYTHON", "python3"), "-I", script, in, out};
  std::istringstream extra(env_or("SQTT_SITE_ARGS", ""));
  for (std::string word; extra >> word;) args.push_back(word);
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  argv.push_back(nullptr);

  std::string path = std::string("PATH=") + env_or("PATH", "/usr/bin:/bin");
  std::string home = std::string("HOME=") + env_or("HOME", "/tmp");
  char* envp[] = {path.data(), home.data(), nullptr};

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC, 0644);
  // Notes and skipped sites go to <log>.err, so the log's first line is the summary.
  const std::string err = log + ".err";
  posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, err.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC, 0644);
  posix_spawnattr_t attr;
  posix_spawnattr_init(&attr);
  posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
  posix_spawnattr_setpgroup(&attr, 0);
  pid_t pid = 0;
  int rc = posix_spawnp(&pid, argv[0], &actions, &attr, argv.data(), envp);
  posix_spawn_file_actions_destroy(&actions);
  posix_spawnattr_destroy(&attr);
  if (rc != 0) return -1;

  const double timeout = std::atof(env_or("SQTT_SITE_TIMEOUT", "600"));
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(static_cast<long>(timeout * 1000));
  int status = 0;
  for (;;) {
    pid_t done = waitpid(pid, &status, WNOHANG);
    if (done == pid) break;
    if (done < 0 && errno != EINTR) return -1;
    if (std::chrono::steady_clock::now() > deadline) {
      kill(-pid, SIGKILL);
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
      }
      std::ofstream(log, std::ios::app) << "timed out after " << timeout << " s\n";
      return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// Runs the rewriter on a copy of the code object. Returns the rewritten bytes,
// or an empty vector if rewriting failed or was skipped. Never throws.
std::vector<char> rewrite(const void* data, size_t size) noexcept {
  try {
    const char* script = std::getenv("SQTT_SITE_REWRITE");
    if (!script) return {};

    std::lock_guard<std::mutex> lock(rewrite_mutex);
    const std::string base = std::string(env_or("TMPDIR", "/tmp")) + "/sqtt_site_" +
                             std::to_string(getpid()) + "_" + std::to_string(next_object++);
    const std::string in = base + ".in.co", out = base + ".out.co", log_path = base + ".log";
    {
      std::ofstream f(in, std::ios::binary);
      f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
      if (!f) return {};
    }

    const int status = run_rewriter(script, in, out, log_path);

    std::vector<char> result;
    if (status == 0) {
      std::ifstream f(out, std::ios::binary);
      result.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
      if (!is_amdgpu_elf(result.data(), result.size())) result.clear();
    }
    // The rewriter's first line says how many sites were marked, or why not.
    std::string first;
    std::getline(std::ifstream(log_path), first);
    if (status != 0 && status != 3 && first.empty()) first = "rewriter failed";
    log("[sqtt_site_injector] code object %s (%zu bytes): %s\n",
        result.empty() ? "unchanged" : "rewritten", size, first);

    if (std::string(env_or("SQTT_SITE_KEEP", "0")) != "1") {
      std::remove(in.c_str());
      std::remove(out.c_str());
      std::remove(log_path.c_str());
      std::remove((log_path + ".err").c_str());
    }
    return result;
  } catch (...) {
    return {};
  }
}

hsa_status_t from_memory(const void* code_object, size_t size,
                         hsa_code_object_reader_t* reader) {
  if (is_amdgpu_elf(code_object, size)) {
    std::vector<char> rewritten = rewrite(code_object, size);
    if (!rewritten.empty()) {
      // The reader keeps referring to this memory until it is destroyed, and
      // the executable may outlive it; keep both copies for the process lifetime.
      auto* copy = new std::vector<char>(std::move(rewritten));
      hsa_status_t status = original_from_memory(copy->data(), copy->size(), reader);
      if (status == HSA_STATUS_SUCCESS) {
        auto original = std::make_shared<std::vector<char>>(
            static_cast<const char*>(code_object), static_cast<const char*>(code_object) + size);
        std::lock_guard<std::mutex> lock(readers_mutex);
        original_of_reader[reader->handle] = original;
        return status;
      }
      std::fprintf(stderr, "[sqtt_site_injector] runtime rejected the rewritten code object "
                           "(status %d); using the original\n", status);
      delete copy;
    }
  }
  return original_from_memory(code_object, size, reader);
}

hsa_status_t from_file(hsa_file_t file, hsa_code_object_reader_t* reader) {
  // Read the whole file without moving its offset; on any problem, defer to
  // the original entry point with the file untouched.
  struct stat st {};
  if (std::getenv("SQTT_SITE_REWRITE") && fstat(file, &st) == 0 && st.st_size >= 64) {
    std::vector<char> bytes(static_cast<size_t>(st.st_size));
    size_t got = 0;
    while (got < bytes.size()) {
      ssize_t n = pread(file, bytes.data() + got, bytes.size() - got, static_cast<off_t>(got));
      if (n <= 0) break;
      got += static_cast<size_t>(n);
    }
    if (got == bytes.size() && is_amdgpu_elf(bytes.data(), bytes.size())) {
      std::vector<char> rewritten = rewrite(bytes.data(), bytes.size());
      if (!rewritten.empty()) {
        auto* copy = new std::vector<char>(std::move(rewritten));
        hsa_status_t status = original_from_memory(copy->data(), copy->size(), reader);
        if (status == HSA_STATUS_SUCCESS) {
          std::lock_guard<std::mutex> lock(readers_mutex);
          original_of_reader[reader->handle] =
              std::make_shared<std::vector<char>>(std::move(bytes));
          return status;
        }
        delete copy;
      }
    }
  }
  return original_from_file(file, reader);
}

// If loading a rewritten code object fails, load the original instead.
std::shared_ptr<std::vector<char>> original_for(hsa_code_object_reader_t reader) {
  std::lock_guard<std::mutex> lock(readers_mutex);
  auto it = original_of_reader.find(reader.handle);
  return it == original_of_reader.end() ? nullptr : it->second;
}

hsa_status_t load_agent(hsa_executable_t executable, hsa_agent_t agent,
                        hsa_code_object_reader_t reader, const char* options,
                        hsa_loaded_code_object_t* loaded) {
  hsa_status_t status = original_load_agent(executable, agent, reader, options, loaded);
  auto original = status == HSA_STATUS_SUCCESS ? nullptr : original_for(reader);
  if (!original) return status;
  std::fprintf(stderr, "[sqtt_site_injector] loading the rewritten code object failed "
                       "(status %d); loading the original\n", status);
  hsa_code_object_reader_t fallback;
  if (original_from_memory(original->data(), original->size(), &fallback) != HSA_STATUS_SUCCESS)
    return status;
  return original_load_agent(executable, agent, fallback, options, loaded);
}

hsa_status_t load_program(hsa_executable_t executable, hsa_code_object_reader_t reader,
                          const char* options, hsa_loaded_code_object_t* loaded) {
  hsa_status_t status = original_load_program(executable, reader, options, loaded);
  auto original = status == HSA_STATUS_SUCCESS ? nullptr : original_for(reader);
  if (!original) return status;
  hsa_code_object_reader_t fallback;
  if (original_from_memory(original->data(), original->size(), &fallback) != HSA_STATUS_SUCCESS)
    return status;
  return original_load_program(executable, fallback, options, loaded);
}

void on_hsa_table(rocprofiler_intercept_table_t type, uint64_t, uint64_t, void** tables,
                  uint64_t num_tables, void*) {
  if (type != ROCPROFILER_HSA_TABLE || num_tables < 1) return;
  auto* core = static_cast<HsaApiTable*>(tables[0])->core_;
  original_from_memory = core->hsa_code_object_reader_create_from_memory_fn;
  original_from_file = core->hsa_code_object_reader_create_from_file_fn;
  original_load_agent = core->hsa_executable_load_agent_code_object_fn;
  original_load_program = core->hsa_executable_load_program_code_object_fn;
  core->hsa_code_object_reader_create_from_memory_fn = from_memory;
  core->hsa_code_object_reader_create_from_file_fn = from_file;
  core->hsa_executable_load_agent_code_object_fn = load_agent;
  core->hsa_executable_load_program_code_object_fn = load_program;
  std::fprintf(stderr, "[sqtt_site_injector] wrapping code object loading (rewriter: %s)\n",
               env_or("SQTT_SITE_REWRITE", "<unset: pass-through>"));
}

}  // namespace

extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
    uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id) {
  id->name = "sqtt_site_injector";
  rocprofiler_at_intercept_table_registration(on_hsa_table, ROCPROFILER_HSA_TABLE, nullptr);
  // No services to start: the table wrapper does all the work.
  static auto cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr, nullptr, nullptr};
  return &cfg;
}
