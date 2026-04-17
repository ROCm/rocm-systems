// LOAD-LEVEL smoke test comparing the legacy byte-level transpiler and
// the Salmon LLVM IR raiser.
//
// Scope (read this carefully)
// ---------------------------
// A "PASS" from this tool means exactly one thing: ROCR accepted the
// translated code object (load + `hsa_executable_freeze` succeeded).
// The kernel is never dispatched, no inputs are supplied, no outputs
// are compared.  This tool CANNOT detect miscompilation — a translator
// that silently produces numerically wrong code will happily PASS here.
//
// For numerical correctness comparison (compile → launch → diff vs
// CPU reference), use the sibling tool `tools/compare_correctness/`.
//
// What it is good for
// -------------------
// - Scanning large pre-existing corpora of `.co`/`.hsaco` files
//   (rocblas library, AITER dumps, ...) for per-engine load coverage.
// - Bisecting which files regress/progress between the two engines.
// - Acting as a fast first filter before investing the effort of
//   authoring a CPU reference for the numerical harness.
//
// Why fork+exec
// -------------
// The two transpilers are mutually exclusive at ROCR init time.  The HSA
// loader selects between them based on the HSA_HOTSWAP_IR_RAISER env var:
//
//     static const char* s_use_salmon = std::getenv("HSA_HOTSWAP_IR_RAISER");
//
// Because that read is a `static` initializer inside the loader, the
// selection is frozen for the lifetime of the process.  A single process
// cannot exercise both engines.  Nor is there an in-process fallback —
// Salmon fails hard on unsupported instructions rather than delegating back
// to the legacy path.
//
// So to compare the two engines honestly, we need one process per mode.
// This tool launches `fork() + execv()` pairs — one child per (file, mode)
// combination — and aggregates their results.
//
// Usage
// -----
//   smoke_test_compare_transpilers <path> [options]
//
//   <path>                   a .co/.hsaco file, or a directory of them.
//
//   --recursive              recurse into subdirectories
//   --isa=<gfx942>           target ISA (forwarded to children as
//                            HSA_HOTSWAP_ISA_OVERRIDE).  default: gfx942.
//   --json                   emit a machine-readable report at the end
//   --quiet                  suppress per-file rows, print only the summary
//   --child-mode=<legacy|salmon>
//                            internal: run as a child process against exactly
//                            one file (expects <path> to be a regular file).
//   -h, --help               show help
//
// Required environment (for children; parent forwards them unchanged):
//   HSA_HOTSWAP_RULES=/dev/null   (loader refuses to activate without this)
//   LD_LIBRARY_PATH               must include $ROCR_BUILD/rocr/lib so that
//                                 the Salmon-enabled libhsa-runtime64.so
//                                 wins over any system ROCR install.
//
// Child-mode output protocol (exactly one line on stdout, then exit):
//   RESULT <PASS|FAIL|ERROR> <duration_ms>
// where ERROR means a process-level problem (HSA init failed, file could not
// be read, ...) rather than a legitimate transpile/load failure.

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Enums and small result types
// ─────────────────────────────────────────────────────────────────────────────

enum class Mode { Legacy, Salmon };

const char *modeName(Mode m) {
  return m == Mode::Legacy ? "legacy" : "salmon";
}

enum class Status {
  Pass,    // load completed successfully
  Fail,    // transpile/load declined the input (normal failure mode)
  Error,   // child could not even attempt the comparison (infra problem)
};

const char *statusName(Status s) {
  switch (s) {
    case Status::Pass: return "PASS";
    case Status::Fail: return "FAIL";
    case Status::Error: return "ERROR";
  }
  return "?";
}

struct RunResult {
  Status status;
  double durationMs;   // wall time for just the load call (child-measured)
  std::string stderrTail;  // up to 2 KiB of child stderr, for diagnostics
  int childExitCode;   // -1 if terminated by signal
  int childSignal;     // 0 if exited normally
};

struct PerFileResult {
  std::string path;
  RunResult legacy;
  RunResult salmon;
};

// ─────────────────────────────────────────────────────────────────────────────
// Filesystem helpers
// ─────────────────────────────────────────────────────────────────────────────

std::vector<uint8_t> readFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  auto sz = f.tellg();
  if (sz < 0) return {};
  f.seekg(0);
  std::vector<uint8_t> buf(static_cast<size_t>(sz));
  if (!f.read(reinterpret_cast<char *>(buf.data()), buf.size())) return {};
  return buf;
}

bool hasCoSuffix(const std::string &name) {
  auto ends = [&](const char *suf) {
    size_t n = std::strlen(suf);
    return name.size() >= n &&
           name.compare(name.size() - n, n, suf) == 0;
  };
  return ends(".co") || ends(".hsaco");
}

void collectCoFiles(const std::string &dir,
                    std::vector<std::string> &out, bool recursive) {
  DIR *d = opendir(dir.c_str());
  if (!d) return;
  while (struct dirent *e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    std::string full = dir + "/" + e->d_name;
    struct stat st;
    if (stat(full.c_str(), &st) != 0) continue;
    if (S_ISDIR(st.st_mode)) {
      if (recursive) collectCoFiles(full, out, recursive);
    } else if (S_ISREG(st.st_mode) && hasCoSuffix(e->d_name)) {
      out.push_back(full);
    }
  }
  closedir(d);
}

// ─────────────────────────────────────────────────────────────────────────────
// Child side: load one file, print RESULT line, exit
// ─────────────────────────────────────────────────────────────────────────────

hsa_agent_t g_gpu_agent = {0};

hsa_status_t findGpuAgent(hsa_agent_t agent, void *) {
  hsa_device_type_t type;
  hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (type == HSA_DEVICE_TYPE_GPU) {
    g_gpu_agent = agent;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

// Returns true if the loader accepted + froze the code object for the agent.
bool tryLoadCodeObject(hsa_agent_t agent, const std::vector<uint8_t> &data) {
  hsa_code_object_reader_t reader;
  hsa_status_t st = hsa_code_object_reader_create_from_memory(
      data.data(), data.size(), &reader);
  if (st != HSA_STATUS_SUCCESS) return false;

  hsa_executable_t exec;
  st = hsa_executable_create_alt(HSA_PROFILE_FULL,
                                 HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                 nullptr, &exec);
  if (st != HSA_STATUS_SUCCESS) {
    hsa_code_object_reader_destroy(reader);
    return false;
  }

  st = hsa_executable_load_agent_code_object(exec, agent, reader, nullptr,
                                             nullptr);
  bool ok = (st == HSA_STATUS_SUCCESS);
  if (ok) {
    st = hsa_executable_freeze(exec, nullptr);
    ok = (st == HSA_STATUS_SUCCESS);
  }

  hsa_executable_destroy(exec);
  hsa_code_object_reader_destroy(reader);
  return ok;
}

// Returns the process exit code: 0 on PASS, 1 on FAIL, 2 on ERROR.
int runChild(Mode mode, const std::string &path) {
  // Select the transpiler for this child.  Must happen before hsa_init.
  if (mode == Mode::Salmon) {
    setenv("HSA_HOTSWAP_IR_RAISER", "1", /*overwrite=*/1);
  } else {
    unsetenv("HSA_HOTSWAP_IR_RAISER");
  }

  auto data = readFile(path);
  if (data.empty()) {
    fprintf(stderr, "smoke_test_compare_transpilers[child]: cannot read %s\n",
            path.c_str());
    fprintf(stdout, "RESULT ERROR 0.0\n");
    return 2;
  }

  hsa_status_t st = hsa_init();
  if (st != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "smoke_test_compare_transpilers[child]: hsa_init failed: %d\n", st);
    fprintf(stdout, "RESULT ERROR 0.0\n");
    return 2;
  }

  hsa_iterate_agents(findGpuAgent, nullptr);
  if (!g_gpu_agent.handle) {
    fprintf(stderr, "smoke_test_compare_transpilers[child]: no GPU agent\n");
    hsa_shut_down();
    fprintf(stdout, "RESULT ERROR 0.0\n");
    return 2;
  }

  auto t0 = std::chrono::steady_clock::now();
  bool ok = tryLoadCodeObject(g_gpu_agent, data);
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  hsa_shut_down();

  fprintf(stdout, "RESULT %s %.3f\n", ok ? "PASS" : "FAIL", ms);
  fflush(stdout);
  return ok ? 0 : 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parent side: fork+exec helpers
// ─────────────────────────────────────────────────────────────────────────────

std::string selfExe() {
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n < 0) return {};
  buf[n] = '\0';
  return std::string(buf);
}

// Read up to maxBytes from fd.  Stops at EOF or limit.
std::string drainFd(int fd, size_t maxBytes) {
  std::string out;
  out.reserve(4096);
  char buf[4096];
  while (out.size() < maxBytes) {
    ssize_t n = read(fd, buf, std::min(sizeof(buf), maxBytes - out.size()));
    if (n < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (n == 0) break;
    out.append(buf, static_cast<size_t>(n));
  }
  return out;
}

// Keep only the last N bytes (so we show the tail of a crashy stderr).
std::string tail(const std::string &s, size_t n) {
  if (s.size() <= n) return s;
  return "…" + s.substr(s.size() - n);
}

RunResult spawnChild(const std::string &exe, Mode mode,
                     const std::string &path) {
  RunResult r{Status::Error, 0.0, {}, -1, 0};

  int outPipe[2];
  int errPipe[2];
  if (pipe(outPipe) != 0 || pipe(errPipe) != 0) {
    r.stderrTail = "parent: pipe() failed: " + std::string(strerror(errno));
    return r;
  }

  pid_t pid = fork();
  if (pid < 0) {
    close(outPipe[0]); close(outPipe[1]);
    close(errPipe[0]); close(errPipe[1]);
    r.stderrTail = "parent: fork() failed: " + std::string(strerror(errno));
    return r;
  }

  if (pid == 0) {
    // Child: redirect stdout/stderr, close unused ends, exec.
    dup2(outPipe[1], STDOUT_FILENO);
    dup2(errPipe[1], STDERR_FILENO);
    close(outPipe[0]); close(outPipe[1]);
    close(errPipe[0]); close(errPipe[1]);

    std::string modeArg = std::string("--child-mode=") + modeName(mode);
    // exe is argv0; path is the input to process.
    char *argv[] = {
        const_cast<char *>(exe.c_str()),
        const_cast<char *>(modeArg.c_str()),
        const_cast<char *>(path.c_str()),
        nullptr,
    };
    execv(exe.c_str(), argv);
    // exec failed
    fprintf(stderr, "smoke_test_compare_transpilers[child-exec]: %s\n", strerror(errno));
    _exit(127);
  }

  // Parent
  close(outPipe[1]);
  close(errPipe[1]);

  std::string out = drainFd(outPipe[0], /*maxBytes=*/64 * 1024);
  std::string err = drainFd(errPipe[0], /*maxBytes=*/64 * 1024);
  close(outPipe[0]);
  close(errPipe[0]);

  int wstatus = 0;
  for (;;) {
    pid_t w = waitpid(pid, &wstatus, 0);
    if (w == pid) break;
    if (w < 0 && errno == EINTR) continue;
    r.stderrTail = "parent: waitpid() failed: " + std::string(strerror(errno));
    return r;
  }

  if (WIFSIGNALED(wstatus)) {
    r.childSignal = WTERMSIG(wstatus);
    r.status = Status::Error;
    r.stderrTail = "child killed by signal " + std::to_string(r.childSignal)
                 + "; stderr: " + tail(err, 1500);
    return r;
  }
  r.childExitCode = WEXITSTATUS(wstatus);

  // Parse the RESULT line.  We scan every line so we do not get confused by
  // engine diagnostics that leaked onto stdout (they should not, but be
  // tolerant).
  bool parsed = false;
  size_t pos = 0;
  while (pos < out.size()) {
    size_t nl = out.find('\n', pos);
    std::string line = out.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
    pos = (nl == std::string::npos) ? out.size() : nl + 1;
    if (line.rfind("RESULT ", 0) != 0) continue;

    char statusBuf[16] = {};
    double ms = 0.0;
    if (std::sscanf(line.c_str(), "RESULT %15s %lf", statusBuf, &ms) == 2) {
      r.durationMs = ms;
      if (std::strcmp(statusBuf, "PASS") == 0) r.status = Status::Pass;
      else if (std::strcmp(statusBuf, "FAIL") == 0) r.status = Status::Fail;
      else r.status = Status::Error;
      parsed = true;
    }
  }

  if (!parsed) {
    r.status = Status::Error;
    r.stderrTail = "no RESULT line; child stderr: " + tail(err, 1500);
    return r;
  }

  // Preserve stderr when the run did not pass, so callers can see why.
  if (r.status != Status::Pass) {
    r.stderrTail = tail(err, 1500);
  }
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parent side: reporting
// ─────────────────────────────────────────────────────────────────────────────

enum class Agreement {
  BothPass,
  BothFail,
  LegacyOnly,   // legacy PASS, salmon FAIL
  SalmonOnly,   // salmon PASS, legacy FAIL
  EitherError,  // at least one side ERROR
};

const char *agreementName(Agreement a) {
  switch (a) {
    case Agreement::BothPass:    return "both-pass";
    case Agreement::BothFail:    return "both-fail";
    case Agreement::LegacyOnly:  return "LEGACY-ONLY";
    case Agreement::SalmonOnly:  return "SALMON-ONLY";
    case Agreement::EitherError: return "error";
  }
  return "?";
}

Agreement classify(const RunResult &l, const RunResult &s) {
  if (l.status == Status::Error || s.status == Status::Error)
    return Agreement::EitherError;
  if (l.status == Status::Pass && s.status == Status::Pass)
    return Agreement::BothPass;
  if (l.status == Status::Fail && s.status == Status::Fail)
    return Agreement::BothFail;
  if (l.status == Status::Pass && s.status == Status::Fail)
    return Agreement::LegacyOnly;
  return Agreement::SalmonOnly;
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

struct Options {
  std::string path;
  bool recursive = false;
  bool quiet = false;
  bool json = false;
  std::string isa = "gfx942";
  std::optional<Mode> childMode;  // set only in child invocation
};

void printHelp(const char *argv0) {
  fprintf(stderr,
    "Load-level smoke test: does the code object load under the legacy\n"
    "byte-level transpiler and under Salmon?  This tool does NOT check\n"
    "numerical correctness — a transpiler can miscompile and still PASS\n"
    "here.  For correctness, use tools/compare_correctness/.\n"
    "\n"
    "Usage: %s <path> [options]\n"
    "\n"
    "  <path>                    a .co/.hsaco file, or a directory\n"
    "\n"
    "  --recursive               recurse into subdirectories\n"
    "  --isa=<gfxNNNN>           target ISA (default: gfx942)\n"
    "  --json                    emit a JSON summary at the end\n"
    "  --quiet                   suppress per-file rows, print summary only\n"
    "  --child-mode=<legacy|salmon>\n"
    "                            internal: run as a child process\n"
    "  -h, --help                show this help\n"
    "\n"
    "Environment (must be set in the parent; children inherit):\n"
    "  HSA_HOTSWAP_RULES=/dev/null\n"
    "  LD_LIBRARY_PATH must point at your Salmon-enabled ROCR build so that\n"
    "  libhsa-runtime64.so resolves to it (not a system ROCR install).\n",
    argv0);
}

bool parseArgs(int argc, char **argv, Options &opt) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      printHelp(argv[0]);
      std::exit(0);
    } else if (a == "--recursive") {
      opt.recursive = true;
    } else if (a == "--quiet") {
      opt.quiet = true;
    } else if (a == "--json") {
      opt.json = true;
    } else if (a.rfind("--isa=", 0) == 0) {
      opt.isa = a.substr(6);
    } else if (a.rfind("--child-mode=", 0) == 0) {
      std::string m = a.substr(13);
      if (m == "legacy") opt.childMode = Mode::Legacy;
      else if (m == "salmon") opt.childMode = Mode::Salmon;
      else {
        fprintf(stderr, "bad --child-mode=%s\n", m.c_str());
        return false;
      }
    } else if (!a.empty() && a[0] == '-') {
      fprintf(stderr, "unknown option: %s\n", a.c_str());
      return false;
    } else if (opt.path.empty()) {
      opt.path = a;
    } else {
      fprintf(stderr, "unexpected positional argument: %s\n", a.c_str());
      return false;
    }
  }
  if (opt.path.empty()) {
    fprintf(stderr, "missing <path>\n\n");
    printHelp(argv[0]);
    return false;
  }
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reporting
// ─────────────────────────────────────────────────────────────────────────────

// Minimal JSON escaping — enough for file paths and stderr tails.
std::string jsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", c);
          out += b;
        } else {
          out += c;
        }
    }
  }
  return out;
}

void printHumanReport(const std::vector<PerFileResult> &results,
                      const Options &opt) {
  size_t bothPass = 0, bothFail = 0, legacyOnly = 0, salmonOnly = 0, errors = 0;
  double totalLegacyMs = 0.0, totalSalmonMs = 0.0;
  int cntLegacyPass = 0, cntSalmonPass = 0;

  if (!opt.quiet) {
    printf("\n%-60s  %-10s  %-10s  %s\n",
           "FILE", "LEGACY", "SALMON", "VERDICT");
    printf("%s\n", std::string(60 + 2 + 10 + 2 + 10 + 2 + 16, '-').c_str());
  }

  for (const auto &r : results) {
    Agreement a = classify(r.legacy, r.salmon);
    switch (a) {
      case Agreement::BothPass:    ++bothPass;    break;
      case Agreement::BothFail:    ++bothFail;    break;
      case Agreement::LegacyOnly:  ++legacyOnly;  break;
      case Agreement::SalmonOnly:  ++salmonOnly;  break;
      case Agreement::EitherError: ++errors;      break;
    }
    if (r.legacy.status == Status::Pass) {
      ++cntLegacyPass;
      totalLegacyMs += r.legacy.durationMs;
    }
    if (r.salmon.status == Status::Pass) {
      ++cntSalmonPass;
      totalSalmonMs += r.salmon.durationMs;
    }

    if (opt.quiet) continue;

    // Shorten the path for the table.
    std::string shortName = r.path;
    if (shortName.size() > 60) {
      shortName = "…" + shortName.substr(shortName.size() - 59);
    }

    auto col = [](const RunResult &x) {
      static thread_local char buf[32];
      std::snprintf(buf, sizeof(buf), "%-5s %5.0fms",
                    statusName(x.status), x.durationMs);
      return std::string(buf);
    };

    printf("%-60s  %-10s  %-10s  %s\n",
           shortName.c_str(),
           col(r.legacy).c_str(),
           col(r.salmon).c_str(),
           agreementName(a));

    // Show stderr tail for any disagreement or error.
    if (a == Agreement::LegacyOnly || a == Agreement::SalmonOnly ||
        a == Agreement::EitherError) {
      if (!r.legacy.stderrTail.empty()) {
        printf("    legacy stderr: %s\n", r.legacy.stderrTail.c_str());
      }
      if (!r.salmon.stderrTail.empty()) {
        printf("    salmon stderr: %s\n", r.salmon.stderrTail.c_str());
      }
    }
  }

  size_t n = results.size();
  printf("\n=== Summary ===\n");
  printf("  target ISA   : %s\n", opt.isa.c_str());
  printf("  inputs       : %zu\n", n);
  printf("  both pass    : %zu\n", bothPass);
  printf("  both fail    : %zu\n", bothFail);
  printf("  LEGACY only  : %zu  (salmon regressed relative to legacy)\n", legacyOnly);
  printf("  SALMON only  : %zu  (salmon handles cases legacy does not)\n", salmonOnly);
  printf("  errors       : %zu  (process-level failures, excluded from above)\n", errors);
  printf("\n");
  printf("  legacy pass rate: %d / %zu  (avg %.1f ms)\n",
         cntLegacyPass, n, cntLegacyPass ? totalLegacyMs / cntLegacyPass : 0.0);
  printf("  salmon pass rate: %d / %zu  (avg %.1f ms)\n",
         cntSalmonPass, n, cntSalmonPass ? totalSalmonMs / cntSalmonPass : 0.0);
}

void printJsonReport(const std::vector<PerFileResult> &results,
                     const Options &opt) {
  printf("{\n");
  printf("  \"scope\": \"load-level smoke test; PASS does not imply "
         "numerical correctness\",\n");
  printf("  \"isa\": \"%s\",\n", jsonEscape(opt.isa).c_str());
  printf("  \"files\": [\n");
  for (size_t i = 0; i < results.size(); ++i) {
    const auto &r = results[i];
    Agreement a = classify(r.legacy, r.salmon);
    auto emit = [](const char *key, const RunResult &x) {
      printf("      \"%s\": {\n", key);
      printf("        \"status\": \"%s\",\n", statusName(x.status));
      printf("        \"duration_ms\": %.3f,\n", x.durationMs);
      printf("        \"exit_code\": %d,\n", x.childExitCode);
      printf("        \"signal\": %d,\n", x.childSignal);
      printf("        \"stderr_tail\": \"%s\"\n",
             jsonEscape(x.stderrTail).c_str());
      printf("      }");
    };
    printf("    {\n");
    printf("      \"path\": \"%s\",\n", jsonEscape(r.path).c_str());
    printf("      \"verdict\": \"%s\",\n", agreementName(a));
    emit("legacy", r.legacy);  printf(",\n");
    emit("salmon", r.salmon);  printf("\n");
    printf("    }%s\n", (i + 1 == results.size()) ? "" : ",");
  }
  printf("  ]\n");
  printf("}\n");
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
  Options opt;
  if (!parseArgs(argc, argv, opt)) return 2;

  // Child invocation: run once against exactly one file and exit.
  if (opt.childMode.has_value()) {
    struct stat st;
    if (stat(opt.path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
      fprintf(stderr, "smoke_test_compare_transpilers[child]: --child-mode requires a "
                      "regular file path, got: %s\n", opt.path.c_str());
      fprintf(stdout, "RESULT ERROR 0.0\n");
      return 2;
    }
    return runChild(*opt.childMode, opt.path);
  }

  // Parent invocation: forward HSA_HOTSWAP_ISA_OVERRIDE for both children.
  setenv("HSA_HOTSWAP_ISA_OVERRIDE", opt.isa.c_str(), /*overwrite=*/1);

  const char *rules = std::getenv("HSA_HOTSWAP_RULES");
  if (!rules || !rules[0]) {
    fprintf(stderr,
            "warning: HSA_HOTSWAP_RULES is not set; the ROCR hotswap hook\n"
            "         will not activate and both modes will look identical.\n"
            "         Set HSA_HOTSWAP_RULES=/dev/null (or a real rules file)\n"
            "         before running this tool.\n\n");
  }

  std::string exe = selfExe();
  if (exe.empty()) {
    fprintf(stderr, "cannot resolve /proc/self/exe; is /proc mounted?\n");
    return 2;
  }

  // Collect inputs: single file, or directory.
  std::vector<std::string> files;
  struct stat st;
  if (stat(opt.path.c_str(), &st) != 0) {
    fprintf(stderr, "cannot stat %s: %s\n",
            opt.path.c_str(), strerror(errno));
    return 2;
  }
  if (S_ISREG(st.st_mode)) {
    files.push_back(opt.path);
  } else if (S_ISDIR(st.st_mode)) {
    collectCoFiles(opt.path, files, opt.recursive);
    std::sort(files.begin(), files.end());
  } else {
    fprintf(stderr, "%s is neither a regular file nor a directory\n",
            opt.path.c_str());
    return 2;
  }

  if (files.empty()) {
    fprintf(stderr, "no .co/.hsaco files under %s\n", opt.path.c_str());
    return 2;
  }

  // The caveat block is the FIRST thing on stderr so it is impossible
  // to miss when reading output or scrollback.  This tool is a
  // load-level smoke test; PASS does not imply numerical correctness.
  fprintf(stderr,
          "!! CAVEAT: LOAD-LEVEL SMOKE TEST ONLY !!\n"
          "   A 'PASS' here means ROCR accepted the code object\n"
          "   (load + hsa_executable_freeze succeeded).  It does NOT\n"
          "   mean the kernel will produce correct results when\n"
          "   dispatched — a miscompilation is invisible to this tool.\n"
          "   For numerical correctness, use tools/compare_correctness/.\n"
          "\n");

  fprintf(stderr, "=== smoke_test_compare_transpilers (load-level smoke test) ===\n");
  fprintf(stderr, "  target ISA   : %s\n", opt.isa.c_str());
  fprintf(stderr, "  inputs       : %zu\n", files.size());
  fprintf(stderr, "  executable   : %s\n", exe.c_str());
  fprintf(stderr, "  strategy     : per-file fork+exec isolation, both modes\n\n");

  std::vector<PerFileResult> results;
  results.reserve(files.size());

  for (const auto &f : files) {
    PerFileResult pf;
    pf.path = f;
    // Run legacy first so stderr ordering is stable.
    pf.legacy = spawnChild(exe, Mode::Legacy, f);
    pf.salmon = spawnChild(exe, Mode::Salmon, f);
    results.push_back(std::move(pf));

    // Progress ticker on stderr so the user sees something during long runs.
    fprintf(stderr, "  [%zu/%zu] %s  legacy=%s  salmon=%s\n",
            results.size(), files.size(), f.c_str(),
            statusName(results.back().legacy.status),
            statusName(results.back().salmon.status));
  }

  if (opt.json) {
    printJsonReport(results, opt);
  } else {
    printHumanReport(results, opt);
  }

  // Exit nonzero only on process-level errors.  Disagreements between the
  // two engines are the signal this tool is designed to surface; they do
  // not indicate that the tool itself failed.
  for (const auto &r : results) {
    if (r.legacy.status == Status::Error || r.salmon.status == Status::Error)
      return 1;
  }
  return 0;
}
