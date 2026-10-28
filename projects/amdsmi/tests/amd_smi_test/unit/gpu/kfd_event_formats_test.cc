// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hardware-free tests for the KFD SMI event messages reported by the public
// amdsmi_get_gpu_event_notification(). Each test registers a synthetic GPU and runs the same public
// flow as the functional events test (init, set mask, get, stop) against a fake /dev/kfd and KFD
// event stream (ScopedFakeKfdEventStream), queuing a kernel event line between the mask and get
// calls. Each payload is built with the kernel's own KFD_EVENT_FMT_* macros (kfd_event_formats.h),
// so a parser format that drifts from them (wrong base, width, or spacing) fails here instead of
// printing stale stack values. Two more tests compare that header, and the event and trigger enums
// in rocm_smi's kfd_ioctl.h, with the kernel's upstream master, so a kernel change is caught too.
// If upstream cannot be fetched, those comparisons are not run and a message says why. The download
// uses the first of curl, wget and python3 found on PATH, and tries again (up to 3 times) after a
// rate limit (429) or server error (5xx). The two local headers are read from
// kfd_headers/ next to the test binary, where the tests package installs them, and from the source
// tree when the tests run from the build directory.
//
// See docs/design/faking-external-interfaces.md for the fake-device pattern.
//
// Environment:
//   AMDSMI_KFD_IOCTL_MAINLINE_FILE=<path> read upstream's kfd_ioctl.h from a local file

#include "kfd_event_formats.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "test_base.h"
#include "test_common.h"
#include "test_fixture_utils.h"

extern char** environ;

namespace {

using amd::smi::AMDSmiDrm;
using amd::smi::AMDSmiGPUDevice;
using amd::smi::testing::ScopedProcessorRegistration;

constexpr size_t kPayloadBufferSize = 256;  // larger than any message the tests build
constexpr size_t kDownloadChunkSize = 4096;
constexpr int kConnectTimeoutSeconds = 5;
constexpr int kDownloadTimeoutSeconds = 20;
constexpr int kMaxDownloadAttempts = 3;
constexpr int kRetryDelaySeconds = 5;  // doubles after each failed attempt

// The reader strips the trailing newline before handing the payload to the parser.
__attribute__((format(printf, 1, 2))) std::string Payload(const char* fmt, ...) {
  char buf[kPayloadBufferSize];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  std::string out(buf);
  if (!out.empty() && out.back() == '\n') {
    out.pop_back();
  }
  return out;
}

constexpr const char* kMainlineHeaderUrl =
    "https://raw.githubusercontent.com/torvalds/linux/master/include/uapi/linux/kfd_ioctl.h";

// python3 fallback: argv[1] is the URL and argv[2] the timeout; refuses a redirect off https.
constexpr const char* kPythonFetchScript =
    "import sys, urllib.request\n"
    "r = urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2]))\n"
    "if not r.geturl().startswith('https://'):\n"
    "    sys.exit('redirected to a non-https URL: ' + r.geturl())\n"
    "sys.stdout.write(r.read().decode())\n";

std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Where the tests package installs the two local headers, relative to the test binary.
constexpr const char* kInstalledHeadersDir = "kfd_headers";

// kfd_event_formats.h sits next to this source file.
std::string CommittedHeaderPath() {
  const std::string source = __FILE__;
  return FindTestDataFile(kInstalledHeadersDir, "kfd_event_formats.h",
                          source.substr(0, source.find_last_of('/') + 1) + "kfd_event_formats.h");
}

struct FetchResult {
  std::string text;
  std::string unable_reason;  // empty when text was obtained
  std::string source;         // where text came from, for verbose output
};

struct DownloadTool {
  std::string name;
  std::vector<std::string> argv;  // run directly, never through a shell
};

bool OnPath(const std::string& tool) {
  const char* path = std::getenv("PATH");
  std::istringstream dirs(path != nullptr ? path : "");
  std::string dir;
  while (std::getline(dirs, dir, ':')) {
    if (!dir.empty() && access((dir + "/" + tool).c_str(), X_OK) == 0) {
      return true;
    }
  }
  return false;
}

struct CommandResult {
  bool started = false;
  int status = 0;      // as returned by pclose(); 0 on success
  std::string output;  // the body on success; the tool's error text, or why it did not start
};

// Runs argv[0] (found on PATH) directly with no shell, stdin from /dev/null and stderr merged
// into the captured output.
CommandResult RunCommand(const std::vector<std::string>& argv) {
  CommandResult result;
  int fds[2];
  if (pipe2(fds, O_CLOEXEC) != 0) {
    result.output = std::strerror(errno);
    return result;
  }
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_adddup2(&actions, fds[1], STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, fds[1], STDERR_FILENO);

  std::vector<char*> args;
  for (const std::string& arg : argv) {
    args.push_back(const_cast<char*>(arg.c_str()));  // posix_spawnp does not modify it
  }
  args.push_back(nullptr);

  pid_t pid = 0;
  const int spawn_error = posix_spawnp(&pid, args[0], &actions, nullptr, args.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(fds[1]);  // the child holds the only write end now
  if (spawn_error != 0) {
    close(fds[0]);
    result.output = std::strerror(spawn_error);
    return result;
  }

  result.started = true;
  char buf[kDownloadChunkSize];
  for (;;) {
    const ssize_t n = read(fds[0], buf, sizeof(buf));
    if (n > 0) {
      result.output.append(buf, static_cast<size_t>(n));
    } else if (n == 0 || errno != EINTR) {
      break;
    }
  }
  close(fds[0]);
  while (waitpid(pid, &result.status, 0) < 0) {
    if (errno != EINTR) {
      result.status = -1;
      break;
    }
  }
  return result;
}

// A python traceback ends with the exception, so the last line is the useful one.
std::string LastLine(std::string text) {
  while (!text.empty() && text.back() == '\n') {
    text.pop_back();
  }
  const size_t last_newline = text.find_last_of('\n');
  return last_newline == std::string::npos ? text : text.substr(last_newline + 1);
}

// A rate limit (429) or a server error (5xx) can pass on its own; any other failure will not.
bool IsTransientHttpError(const std::string& error_text) {
  static const std::regex kTransient(R"(\b(429|5[0-9]{2})\b)");
  return std::regex_search(error_text, kTransient);
}

FetchResult FetchMainlineHeader() {
  if (const char* local = std::getenv("AMDSMI_KFD_IOCTL_MAINLINE_FILE")) {
    std::string text = ReadFile(local);
    if (text.empty()) {
      return {"", std::string("AMDSMI_KFD_IOCTL_MAINLINE_FILE=") + local + " is missing or empty",
              ""};
    }
    return {text, "", std::string("file ") + local};
  }
  // Quiet modes print only errors, and stderr joins the output, so a failure keeps its reason.
  // https only: curl and python3 refuse a redirect to http; wget (whose --https-only affects only
  // recursive downloads) follows none, which is enough because the URL does not redirect.
  const std::string url = kMainlineHeaderUrl;
  const std::string connect_timeout = std::to_string(kConnectTimeoutSeconds);
  const std::string timeout = std::to_string(kDownloadTimeoutSeconds);
  const std::vector<DownloadTool> kTools = {
      {"curl",
       {"curl", "-fsSL", "--proto", "=https", "--proto-redir", "=https", "--connect-timeout",
        connect_timeout, "--max-time", timeout, url}},
      {"wget",
       {"wget", "-q", "-O", "-", "--max-redirect=0", "--timeout=" + timeout, "--tries=1", url}},
      {"python3", {"python3", "-c", kPythonFetchScript, url, timeout}},
  };
  const DownloadTool* tool = nullptr;
  for (const DownloadTool& candidate : kTools) {
    if (OnPath(candidate.name)) {
      tool = &candidate;
      break;
    }
  }
  if (tool == nullptr) {
    return {"", "no download tool on PATH (tried curl, wget, python3)", ""};
  }
  // Retry only a rate limit or server error, with a growing pause, so an offline run is not slowed.
  CommandResult attempt;
  int attempts = 0;
  while (true) {
    ++attempts;
    attempt = RunCommand(tool->argv);
    if (!attempt.started || attempt.status == 0 || attempts == kMaxDownloadAttempts ||
        !IsTransientHttpError(LastLine(attempt.output))) {
      break;
    }
    sleep(static_cast<unsigned int>(kRetryDelaySeconds << (attempts - 1)));
  }
  if (!attempt.started) {
    return {"", "could not start " + tool->name + ": " + attempt.output, ""};
  }
  if (attempt.status != 0) {
    std::string reason =
        tool->name + " exited with status " +
        std::to_string(WIFEXITED(attempt.status) ? WEXITSTATUS(attempt.status) : attempt.status);
    if (attempts > 1) {
      reason += " after " + std::to_string(attempts) + " attempts";
    }
    const std::string error_line = LastLine(attempt.output);
    if (!error_line.empty()) {
      reason += ": " + error_line;
    }
    return {"", reason, ""};
  }
  const std::string& out = attempt.output;
  return {out, "", url + " (via " + tool->name + ")"};
}

// One download serves every upstream comparison in this binary.
const FetchResult& MainlineHeader() {
  static const FetchResult result = FetchMainlineHeader();
  return result;
}

std::string CollapseSpaces(const std::string& s) {
  std::string out;
  bool pending_space = false;
  for (char c : s) {
    if (c == ' ' || c == '\t' || c == '\r') {
      pending_space = !out.empty();
    } else {
      if (pending_space) {
        out += ' ';
        pending_space = false;
      }
      out += c;
    }
  }
  return out;
}

// KFD_EVENT_FMT_* name -> definition, with continuation lines joined and whitespace collapsed so
// reformatting the header does not read as drift.
std::map<std::string, std::string> ExtractFormats(const std::string& text) {
  const std::string kDefine = "#define KFD_EVENT_FMT_";
  std::map<std::string, std::string> formats;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    const size_t start = line.find_first_not_of(" \t");
    if (start == std::string::npos || line.compare(start, kDefine.size(), kDefine) != 0) {
      continue;
    }
    std::string def = line.substr(start);
    while (!def.empty() && def.back() == '\\') {
      def.pop_back();
      std::string next;
      if (!std::getline(in, next)) {
        break;
      }
      def += ' ' + next;
    }
    def = CollapseSpaces(def);
    const size_t name_begin = std::string("#define ").size();
    const size_t paren = def.find('(', name_begin);
    if (paren != std::string::npos) {
      formats[def.substr(name_begin, paren - name_begin)] = def;
    }
  }
  return formats;
}

// An enum as {entry name -> value}, e.g. {"KFD_SMI_EVENT_VMFAULT", 1}.
using EnumValues = std::map<std::string, long>;

// Stands in for a value that is not a plain integer, so a comparison reports it instead of
// quietly reading a wrong number.
constexpr long kUnreadableValue = LONG_MIN;
// strtol base 0: "0x10" is read as hex and "16" as decimal, as in C source.
constexpr int kAutoDetectBase = 0;

// The enums whose numbers the library relies on: the event ids and the trigger values.
constexpr const char* kKfdEnums[] = {"kfd_smi_event", "KFD_MIGRATE_TRIGGERS",
                                     "KFD_QUEUE_EVICTION_TRIGGERS", "KFD_SVM_UNMAP_TRIGGERS"};

// The library's copy of kfd_ioctl.h, relative to this test directory.
constexpr const char* kTestDirectory = "/tests/amd_smi_test/";
constexpr const char* kLibraryKfdHeader = "/rocm_smi/include/rocm_smi/kfd_ioctl.h";

std::string LibraryKfdHeaderPath() {
  const std::string source = __FILE__;
  return FindTestDataFile(kInstalledHeadersDir, "kfd_ioctl.h",
                          source.substr(0, source.find(kTestDirectory)) + kLibraryKfdHeader);
}

std::string StripComments(const std::string& code) {
  static const std::regex kComment(R"(/\*[\s\S]*?\*/|//[^\n]*)");
  return std::regex_replace(code, kComment, " ");
}

// Reads one enum out of a C header:
//
//     enum kfd_smi_event {
//         KFD_SMI_EVENT_NONE = 0,  /* not used */
//         KFD_SMI_EVENT_VMFAULT = 1,
//     };
//
// An entry without "= value" is the previous value plus one, as in C.
EnumValues ParseEnum(const std::string& header, const std::string& name) {
  EnumValues values;
  const size_t keyword = header.find("enum " + name + " {");
  if (keyword == std::string::npos) {
    return values;
  }
  // Only the enum body is scanned, not the whole header. Enum bodies hold no nested braces.
  const size_t open = header.find('{', keyword);
  const size_t close = header.find('}', open);
  const std::string body = StripComments(header.substr(open + 1, close - open - 1));

  long value = 0;
  for (const std::string& entry : split_string(body, ',')) {  // trimmed, none empty
    const size_t equals = entry.find('=');
    if (equals != std::string::npos) {
      const std::string text = entry.substr(equals + 1);
      char* rest = nullptr;
      const long parsed = std::strtol(text.c_str(), &rest, kAutoDetectBase);
      const bool plain_integer = rest != text.c_str() && amd::smi::trim(std::string(rest)).empty();
      value = plain_integer ? parsed : kUnreadableValue;
    }
    values[amd::smi::trim(entry.substr(0, equals))] = value;
    if (value != kUnreadableValue) {
      ++value;
    }
  }
  return values;
}

std::string DescribeValue(long value) {
  return value == kUnreadableValue ? "not a plain integer" : std::to_string(value);
}

// What differs between the library's copy of an enum and the kernel's, one line per entry; empty
// when they are the same.
std::string DescribeEnumDifferences(const EnumValues& library, const EnumValues& kernel) {
  std::ostringstream differences;
  for (const auto& [entry, value] : library) {
    const auto in_kernel = kernel.find(entry);
    if (in_kernel == kernel.end()) {
      differences << "  removed upstream: " << entry << " = " << DescribeValue(value) << "\n";
    } else if (in_kernel->second != value) {
      differences << "  changed: " << entry << " = " << DescribeValue(value) << " here, "
                  << DescribeValue(in_kernel->second) << " upstream\n";
    }
  }
  for (const auto& [entry, value] : kernel) {
    if (library.find(entry) == library.end()) {
      differences << "  added upstream: " << entry << " = " << DescribeValue(value) << "\n";
    }
  }
  return differences.str();
}

void ReportUnableToVerify(const std::string& what, const std::string& reason) {
  std::cerr << "[  UNVERIFIED ] Unable to compare " << what << " with upstream "
            << kMainlineHeaderUrl << "\n                Reason: " << reason
            << "\n                To check offline, set AMDSMI_KFD_IOCTL_MAINLINE_FILE=<path to a "
               "downloaded kfd_ioctl.h>.\n";
}

// The upstream header was fetched, so the missing piece is the local copy; say so.
std::string LocalFileMissingReason(const std::string& path, const FetchResult& upstream) {
  return "upstream was fetched (" + upstream.source + ") but " + path +
         " could not be read; the tests package installs it in " + kInstalledHeadersDir +
         "/ next to the test binary, and runs from the build directory use the source tree";
}

bool Verbose() { return GetTestVerbosity() >= TestBase::VERBOSE_STANDARD; }

// A payload plus the source text that built it and where, for verbose and failure output.
struct KfdPayload {
  std::string text;
  const char* built_with;
  const char* file = "";
  int line = 0;
};

// Builds a payload with the kernel's own KFD_EVENT_FMT_* macro; the macro stringifies the call.
#define KFD_PAYLOAD(call) \
  KfdPayload { Payload(call), #call, __FILE__, __LINE__ }

// A synthetic GPU with a fake KFD event stream, reached by the public event APIs as a real one is.
class FakeGpuWithKfdEvents {
 public:
  FakeGpuWithKfdEvents()
      : rocm_device_(tree_.root() + "/card0"),
        gpu_device_(rocm_device_.index(), "card0", amdsmi_bdf_t{}, drm_),
        registration_(&gpu_device_),
        stream_(rocm_device_.index()) {}

  amdsmi_processor_handle handle() const { return registration_.handle(); }
  ScopedFakeKfdEventStream& stream() { return stream_; }

 private:
  ScopedAmdsmiInit smi_init_;
  FakeSysfsTree tree_;
  ScopedRocmSmiDevice rocm_device_;
  AMDSmiDrm drm_;
  AMDSmiGPUDevice gpu_device_;
  ScopedProcessorRegistration registration_;
  ScopedFakeKfdEventStream stream_;
};

// Every event type, as the functional events test selects them.
constexpr uint64_t AllEventsMask() {
  uint64_t mask = 0;
  for (int event = AMDSMI_EVT_NOTIF_FIRST; event <= AMDSMI_EVT_NOTIF_LAST; ++event) {
    mask |= AMDSMI_EVENT_MASK_FROM_INDEX(event);
  }
  return mask;
}

amdsmi_status_t InitEventNotification(FakeGpuWithKfdEvents& gpu) {
  const bool verbose = Verbose();
  DISPLAY_AMDSMI_API("amdsmi_init_gpu_event_notification", "gpu=0", verbose);
  const amdsmi_status_t ret = amdsmi_init_gpu_event_notification(gpu.handle());
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
  return ret;
}

amdsmi_status_t StopEventNotification(FakeGpuWithKfdEvents& gpu) {
  const bool verbose = Verbose();
  DISPLAY_AMDSMI_API("amdsmi_stop_gpu_event_notification", "gpu=0", verbose);
  const amdsmi_status_t ret = amdsmi_stop_gpu_event_notification(gpu.handle());
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
  return ret;
}

amdsmi_status_t SetEventMask(FakeGpuWithKfdEvents& gpu, uint64_t mask) {
  const bool verbose = Verbose();
  std::ostringstream desc;
  desc << "mask=0x" << std::hex << mask;
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_event_notification_mask", desc.str(), verbose);
  const amdsmi_status_t ret = amdsmi_set_gpu_event_notification_mask(gpu.handle(), mask);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
  return ret;
}

// Reads the queued events without waiting, since they are already queued.
amdsmi_status_t GetEvents(uint32_t capacity, std::vector<amdsmi_evt_notification_data_t>* data,
                          uint32_t* num_elem, amdsmi_status_t expected) {
  constexpr int kNoWaitMs = 0;
  const bool verbose = Verbose();
  data->assign(capacity, amdsmi_evt_notification_data_t{});
  *num_elem = capacity;
  DISPLAY_AMDSMI_API(
      "amdsmi_get_gpu_event_notification",
      "timeout_ms=" + std::to_string(kNoWaitMs) + ", num_elem=" + std::to_string(capacity),
      verbose);
  const amdsmi_status_t ret = amdsmi_get_gpu_event_notification(kNoWaitMs, num_elem, data->data());
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, ret, expected);
  return ret;
}

// Selects every event with amdsmi_set_gpu_event_notification_mask(), as a real caller must (the
// kernel sends nothing until then), queues `payload` as kernel event number `event` on a synthetic
// GPU's fake KFD event stream, and reads it back through amdsmi_get_gpu_event_notification().
// Checks the event, the GPU handle and that the reported message equals `expected`. With verbose
// output on, prints what is verified, how, the APIs called and their status, and whether the
// message matched.
void ExpectEvtMessage(const char* what, uint32_t event, const KfdPayload& payload,
                      const std::string& expected) {
  ::testing::ScopedTrace trace(payload.file, payload.line, what);
  constexpr uint32_t kBufferSize = 4;  // room for more than the one event queued

  FakeGpuWithKfdEvents gpu;
  const bool verbose = Verbose();
  if (verbose) {
    std::cout << "\t### " << what << "\n"
              << "\t      built with               : " << payload.built_with << "\n"
              << "\t      kernel line (input)      : " << payload.text << " (event " << event << ")"
              << std::endl;
  }
  ASSERT_EQ(InitEventNotification(gpu), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(SetEventMask(gpu, AllEventsMask()), AMDSMI_STATUS_SUCCESS);
  gpu.stream().Write(event, payload.text);

  uint32_t num_elem = 0;
  std::vector<amdsmi_evt_notification_data_t> data;
  ASSERT_EQ(GetEvents(kBufferSize, &data, &num_elem, AMDSMI_STATUS_SUCCESS), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_elem, 1u);
  EXPECT_EQ(data[0].event, static_cast<amdsmi_evt_notification_type_t>(event));
  EXPECT_EQ(data[0].processor_handle, gpu.handle());
  const std::string actual = data[0].message;
  if (verbose) {
    std::cout << "\t      library message (output) : " << actual << "\n"
              << "\t      test requires            : " << expected << "\n"
              << "\t===> " << (actual == expected ? "VERIFIED" : "MISMATCH") << std::endl;
  }
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(StopEventNotification(gpu), AMDSMI_STATUS_SUCCESS);
}

}  // namespace

// The tests below pass each value to the kernel's macro with its parameter name in a comment
// (/*pid=*/...). The kernel sends most numbers in hex and the library reports some of them in
// decimal, so a test notes both forms where they differ. Addresses and sizes are wider than 32
// bits, so reading them into a 32-bit field would cut them.
//
// Keep these in step with FormatEvtNotifMessage() in rocm_smi_evt_notif.cc. For each event add a
// test for a full line and, when older kernels send fewer fields, one for the older line.

TEST(GpuUnit, KFD_EVENT_FMT__GpuPreResetReportsSequenceAndCause) {
  // Sequence number: sent as 0x1f, reported as 31.
  ExpectEvtMessage("GPU_PRE_RESET: reset sequence number (hex) and reset cause text",
                   AMDSMI_EVT_NOTIF_GPU_PRE_RESET,
                   KFD_PAYLOAD(KFD_EVENT_FMT_UPDATE_GPU_RESET(/*reset_seq_num=*/0x1fu,
                                                              /*reset_cause=*/"GPU hang")),
                   "reset sequence number: 31  reset cause: GPU hang");
}

TEST(GpuUnit, KFD_EVENT_FMT__GpuPreResetWithEmptyCause) {
  ExpectEvtMessage("GPU_PRE_RESET: an empty reset cause still reports the sequence number",
                   AMDSMI_EVT_NOTIF_GPU_PRE_RESET,
                   KFD_PAYLOAD(KFD_EVENT_FMT_UPDATE_GPU_RESET(/*reset_seq_num=*/0x1fu,
                                                              /*reset_cause=*/"")),
                   "reset sequence number: 31  reset cause: ");
}

TEST(GpuUnit, KFD_EVENT_FMT__GpuPostResetReportsOnlySequence) {
  ExpectEvtMessage("GPU_POST_RESET: only the reset sequence number is reported",
                   AMDSMI_EVT_NOTIF_GPU_POST_RESET,
                   KFD_PAYLOAD(KFD_EVENT_FMT_UPDATE_GPU_RESET(/*reset_seq_num=*/0x1fu,
                                                              /*reset_cause=*/"GPU hang")),
                   "reset sequence number: 31");
}

TEST(GpuUnit, KFD_EVENT_FMT__ThermalThrottle) {
  ExpectEvtMessage("THERMAL_THROTTLE: 64-bit hex bitmask and counter",
                   AMDSMI_EVT_NOTIF_THERMAL_THROTTLE,
                   KFD_PAYLOAD(KFD_EVENT_FMT_THERMAL_THROTTLING(/*bitmask=*/0xdeadbeefcafeULL,
                                                                /*counter=*/0x10ULL)),
                   "bitmask: 0xdeadbeefcafe  counter: 0x10");
}

TEST(GpuUnit, KFD_EVENT_FMT__VmFault) {
  // Pid: sent as 0x1a2b, reported as 6699.
  ExpectEvtMessage("VMFAULT: pid (hex) and task name", AMDSMI_EVT_NOTIF_VMFAULT,
                   KFD_PAYLOAD(KFD_EVENT_FMT_VMFAULT(/*pid=*/0x1a2bu, /*task_name=*/"myapp")),
                   "PID: 6699  task name: myapp");
}

TEST(GpuUnit, KFD_EVENT_FMT__VmFaultWithEmptyTaskName) {
  ExpectEvtMessage("VMFAULT: a process with an empty name reports an empty task name, not N/A",
                   AMDSMI_EVT_NOTIF_VMFAULT,
                   KFD_PAYLOAD(KFD_EVENT_FMT_VMFAULT(/*pid=*/0x1a2bu, /*task_name=*/"")),
                   "PID: 6699  task name: ");
}

TEST(GpuUnit, KFD_EVENT_FMT__VmFaultTaskNameWithSpace) {
  ExpectEvtMessage("VMFAULT: a task name with a space is reported whole, not cut at the space",
                   AMDSMI_EVT_NOTIF_VMFAULT,
                   KFD_PAYLOAD(KFD_EVENT_FMT_VMFAULT(/*pid=*/0x1a2bu, /*task_name=*/"Web Content")),
                   "PID: 6699  task name: Web Content");
}

TEST(GpuUnit, KFD_EVENT_FMT__MigrateStart) {
  // Start 0x7f12abcd1234 and size 0x200001000 (8 GiB + 4 KiB) are both wider than 32 bits. The
  // trigger is the enum value the kernel sends, here 3 (KFD_MIGRATE_TRIGGER_TTM_EVICTION).
  ExpectEvtMessage(
      "MIGRATE_START: 64-bit hex start and size (not decimal, not truncated to 32 bits), "
      "from/to and locations",
      AMDSMI_EVT_NOTIF_MIGRATE_START,
      KFD_PAYLOAD(KFD_EVENT_FMT_MIGRATE_START(
          /*ns=*/123456789012LL, /*pid=*/4321, /*start=*/0x7f12abcd1234UL, /*size=*/0x200001000UL,
          /*from=*/1u, /*to=*/2u, /*prefetch_loc=*/0xau, /*preferred_loc=*/0xbu,
          /*trigger=*/KFD_MIGRATE_TRIGGER_TTM_EVICTION)),
      "ns: 123456789012  pid: 4321  start: 0x7f12abcd1234  size: 0x200001000  from: 0x1  to: 0x2"
      "  prefetch_loc: 0xa  preferred_loc: 0xb  migrate_trigger: 3");
}

TEST(GpuUnit, KFD_EVENT_FMT__MigrateEndReportsNegativeErrorCode) {
  // The trigger is 2 (KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU). The error code is a negative errno, -12
  // (ENOMEM), so reading it as unsigned would print a huge number.
  ExpectEvtMessage(
      "MIGRATE_END: 64-bit hex start and size, and a negative error_code reported as signed",
      AMDSMI_EVT_NOTIF_MIGRATE_END,
      KFD_PAYLOAD(KFD_EVENT_FMT_MIGRATE_END(
          /*ns=*/5LL, /*pid=*/77, /*start=*/0xabcdef012345UL, /*size=*/0x100000010UL, /*from=*/3u,
          /*to=*/0u, /*trigger=*/KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU, /*error_code=*/-12)),
      "ns: 5  pid: 77  start: 0xabcdef012345  size: 0x100000010  from: 0x3  to: 0x0"
      "  migrate_trigger: 2  error_code: -12");
}

TEST(GpuUnit, KFD_EVENT_FMT__PageFaultStart) {
  ExpectEvtMessage(
      "PAGE_FAULT_START: 64-bit hex fault address, node and read/write flag",
      AMDSMI_EVT_NOTIF_PAGE_FAULT_START,
      KFD_PAYLOAD(KFD_EVENT_FMT_PAGEFAULT_START(/*ns=*/9LL, /*pid=*/100,
                                                /*addr=*/0x7ffd12345678UL, /*node=*/0xab12u,
                                                /*rw=*/'W')),
      "ns: 9  pid: 100  addr: 0x7ffd12345678  node: 0xab12  rw: W");
}

TEST(GpuUnit, KFD_EVENT_FMT__PageFaultEnd) {
  ExpectEvtMessage(
      "PAGE_FAULT_END: 64-bit hex fault address, node and the migrate_update flag (label "
      "spelled migrate_update)",
      AMDSMI_EVT_NOTIF_PAGE_FAULT_END,
      KFD_PAYLOAD(KFD_EVENT_FMT_PAGEFAULT_END(/*ns=*/9LL, /*pid=*/100, /*addr=*/0x7ffd12345678UL,
                                              /*node=*/0xab12u, /*migrate_update=*/'M')),
      "ns: 9  pid: 100  addr: 0x7ffd12345678  node: 0xab12  migrate_update: M");
}

TEST(GpuUnit, KFD_EVENT_FMT__QueueEviction) {
  // The trigger is 3 (KFD_QUEUE_EVICTION_TRIGGER_SUSPEND), what a GPU reset reports.
  ExpectEvtMessage("QUEUE_EVICTION: pid, node and trigger are parsed, not left uninitialized",
                   AMDSMI_EVT_NOTIF_QUEUE_EVICTION,
                   KFD_PAYLOAD(KFD_EVENT_FMT_QUEUE_EVICTION(
                       /*ns=*/42LL, /*pid=*/2057, /*node=*/0xabcu,
                       /*evict_trigger=*/KFD_QUEUE_EVICTION_TRIGGER_SUSPEND)),
                   "ns: 42  pid: 2057  node: 0xabc  evict_trigger: 3");
}

TEST(GpuUnit, KFD_EVENT_FMT__QueueRestore) {
  ExpectEvtMessage(
      "QUEUE_RESTORE: pid, node and rescheduled flag are parsed, not left uninitialized",
      AMDSMI_EVT_NOTIF_QUEUE_RESTORE,
      KFD_PAYLOAD(KFD_EVENT_FMT_QUEUE_RESTORE(/*ns=*/42LL, /*pid=*/2057, /*node=*/0xabcu,
                                              /*rescheduled=*/'R')),
      "ns: 42  pid: 2057  node: 0xabc  rescheduled: R");
}

TEST(GpuUnit, KFD_EVENT_FMT__UnmapFromGpu) {
  // Size 0x1deadbeef is wider than 32 bits. The trigger is 2
  // (KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU).
  ExpectEvtMessage(
      "UNMAP_FROM_GPU: 64-bit hex address and size, node and unmap trigger",
      AMDSMI_EVT_NOTIF_UNMAP_FROM_GPU,
      KFD_PAYLOAD(KFD_EVENT_FMT_UNMAP_FROM_GPU(
          /*ns=*/7LL, /*pid=*/8, /*addr=*/0x123456789abUL, /*size=*/0x1deadbeefUL, /*node=*/0xfu,
          /*unmap_trigger=*/KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU)),
      "ns: 7  pid: 8  addr: 0x123456789ab  size: 0x1deadbeef  node: 0xf  unmap_trigger: 2");
}

TEST(GpuUnit, KFD_EVENT_FMT__ProcessStartAndEnd) {
  // Pid: sent as 0x3039, reported as 12345.
  ExpectEvtMessage("PROCESS_START: pid (hex) and task name", AMDSMI_EVT_NOTIF_PROCESS_START,
                   KFD_PAYLOAD(KFD_EVENT_FMT_PROCESS(/*pid=*/0x3039u, /*task_name=*/"bash")),
                   "PID: 12345  task: bash");
  ExpectEvtMessage("PROCESS_END: pid (hex) and task name", AMDSMI_EVT_NOTIF_PROCESS_END,
                   KFD_PAYLOAD(KFD_EVENT_FMT_PROCESS(/*pid=*/0x3039u, /*task_name=*/"bash")),
                   "PID: 12345  task: bash");
}

TEST(GpuUnit, KFD_EVENT_FMT__ProcessTaskNameWithSpace) {
  ExpectEvtMessage("PROCESS_START: a task name with a space is reported whole",
                   AMDSMI_EVT_NOTIF_PROCESS_START,
                   KFD_PAYLOAD(KFD_EVENT_FMT_PROCESS(/*pid=*/0x3039u, /*task_name=*/"Web Content")),
                   "PID: 12345  task: Web Content");
  ExpectEvtMessage("PROCESS_END: a task name with a space is reported whole",
                   AMDSMI_EVT_NOTIF_PROCESS_END,
                   KFD_PAYLOAD(KFD_EVENT_FMT_PROCESS(/*pid=*/0x3039u, /*task_name=*/"Web Content")),
                   "PID: 12345  task: Web Content");
}

// Literal payloads, not built from kfd_event_formats.h, so they also catch a wrong macro there.
// Modelled on a GPU reset run on a Navi 21 host, with the values copied by hand; node 0x56c7 is
// the KFD gpu_id of that host's second GPU. Kernel line layouts:
//   QUEUE_EVICTION  "<ns> -<pid> <node hex> <trigger>"
//   GPU_PRE_RESET / GPU_POST_RESET  "<reset sequence hex> <cause>"
TEST(GpuUnit, KFD_EVENT_FMT__LiteralPayloads) {
  ExpectEvtMessage("QUEUE_EVICTION: ns, pid, node (hex) and trigger, as seen in a reset run",
                   AMDSMI_EVT_NOTIF_QUEUE_EVICTION,
                   {"510812858298103 -999198 56c7 3", "literal, modelled on a Navi 21 reset run"},
                   "ns: 510812858298103  pid: 999198  node: 0x56c7  evict_trigger: 3");
  ExpectEvtMessage("GPU_PRE_RESET: sequence number with an empty cause",
                   AMDSMI_EVT_NOTIF_GPU_PRE_RESET,
                   {"2 ", "literal, modelled on a Navi 21 reset run"},
                   "reset sequence number: 2  reset cause: ");
  ExpectEvtMessage("GPU_POST_RESET: the same sequence number", AMDSMI_EVT_NOTIF_GPU_POST_RESET,
                   {"2 ", "literal, modelled on a Navi 21 reset run"}, "reset sequence number: 2");
}

// Older kernels (checked against the 5.18 tree and the amdgpu 6.7.0 DKMS driver) send a prefix of
// the current layout for two events. The fields they omit read as N/A instead of stray stack
// values.
//   MIGRATE_END    "<ns> -<pid> @<start>(<size>) <from>-><to> <trigger>"      (no error_code)
//   QUEUE_RESTORE  "<ns> -<pid> <node>"                                      (no rescheduled)
TEST(GpuUnit, KFD_EVENT_FMT__OlderKernelLayoutsReportNotAvailable) {
  ExpectEvtMessage("MIGRATE_END: a kernel without error_code reports it as N/A",
                   AMDSMI_EVT_NOTIF_MIGRATE_END,
                   {"123 -45 @7f1234567000(200000) 1->2 3", "literal, older kernel layout"},
                   "ns: 123  pid: 45  start: 0x7f1234567000  size: 0x200000  from: 0x1  to: 0x2  "
                   "migrate_trigger: 3  error_code: N/A");
  ExpectEvtMessage("QUEUE_RESTORE: a kernel without the rescheduled flag reports it as N/A",
                   AMDSMI_EVT_NOTIF_QUEUE_RESTORE, {"42 -2057 abc", "literal, older kernel layout"},
                   "ns: 42  pid: 2057  node: 0xabc  rescheduled: N/A");
}

// A line shorter than any known layout keeps the fields it does have and reports the rest as N/A.
TEST(GpuUnit, KFD_EVENT_FMT__ShortOrUnparsablePayloadsReportNotAvailable) {
  ExpectEvtMessage("QUEUE_EVICTION: a line cut after the pid keeps ns and pid",
                   AMDSMI_EVT_NOTIF_QUEUE_EVICTION, {"42 -7", "literal, truncated line"},
                   "ns: 42  pid: 7  node: N/A  evict_trigger: N/A");
  ExpectEvtMessage("VMFAULT: an empty payload reports the pid as N/A and an empty task name",
                   AMDSMI_EVT_NOTIF_VMFAULT, {"", "literal, empty payload"},
                   "PID: N/A  task name: ");
  ExpectEvtMessage("MIGRATE_START: text that is not a payload reports every field as N/A",
                   AMDSMI_EVT_NOTIF_MIGRATE_START, {"not a payload", "literal, unparsable text"},
                   "ns: N/A  pid: N/A  start: N/A  size: N/A  from: N/A  to: N/A  "
                   "prefetch_loc: N/A  preferred_loc: N/A  migrate_trigger: N/A");
}

// Event 0 and 99 have no bit in the mask and the kernel never sends them; the fake passes them
// through so the library's handling can be checked.
TEST(GpuUnit, KFD_EVENT_FMT__NoneAndUnknownEvents) {
  ExpectEvtMessage("NONE: reports its fixed text", AMDSMI_EVT_NOTIF_NONE,
                   {"", "no kernel format (empty payload)"}, "Event type None received");
  ExpectEvtMessage("unknown event number: reports a fixed text", 99,
                   {"anything", "no kernel format (event number 99)"}, "Unknown event received");
}

// The kernel starts with an empty mask and queues only the events the mask selects.
TEST(GpuUnit, KFD_EVENT_FMT__OnlyEventsInTheMaskAreReported) {
  constexpr uint32_t kBufferSize = 4;
  const std::string process = KFD_PAYLOAD(KFD_EVENT_FMT_PROCESS(/*pid=*/0x3039u, "bash")).text;
  const std::string vmfault = KFD_PAYLOAD(KFD_EVENT_FMT_VMFAULT(/*pid=*/0x1a2bu, "myapp")).text;

  FakeGpuWithKfdEvents gpu;
  ASSERT_EQ(InitEventNotification(gpu), AMDSMI_STATUS_SUCCESS);
  gpu.stream().Write(AMDSMI_EVT_NOTIF_PROCESS_START, process);  // no mask set yet
  ASSERT_EQ(SetEventMask(gpu, AMDSMI_EVENT_MASK_FROM_INDEX(AMDSMI_EVT_NOTIF_VMFAULT)),
            AMDSMI_STATUS_SUCCESS);
  gpu.stream().Write(AMDSMI_EVT_NOTIF_PROCESS_START, process);  // not in the mask
  gpu.stream().Write(AMDSMI_EVT_NOTIF_VMFAULT, vmfault);

  uint32_t num_elem = 0;
  std::vector<amdsmi_evt_notification_data_t> data;
  ASSERT_EQ(GetEvents(kBufferSize, &data, &num_elem, AMDSMI_STATUS_SUCCESS), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_elem, 1u);
  EXPECT_EQ(data[0].event, AMDSMI_EVT_NOTIF_VMFAULT);
  EXPECT_STREQ(data[0].message, "PID: 6699  task name: myapp");

  // Everything queued has been read, so a second read finds nothing.
  EXPECT_EQ(GetEvents(kBufferSize, &data, &num_elem, AMDSMI_STATUS_NO_DATA), AMDSMI_STATUS_NO_DATA);
  EXPECT_EQ(StopEventNotification(gpu), AMDSMI_STATUS_SUCCESS);
}

// The error text each download tool prints; only a rate limit or server error is worth retrying.
TEST(GpuUnit, KFD_EVENT_FMT__DownloadRetriesOnlyTransientHttpErrors) {
  EXPECT_TRUE(IsTransientHttpError("curl: (22) The requested URL returned error: 429"));
  EXPECT_TRUE(IsTransientHttpError("curl: (22) The requested URL returned error: 503"));
  EXPECT_TRUE(IsTransientHttpError("ERROR 429: Too Many Requests."));
  EXPECT_TRUE(IsTransientHttpError("urllib.error.HTTPError: HTTP Error 502: Bad Gateway"));

  EXPECT_FALSE(IsTransientHttpError("curl: (22) The requested URL returned error: 404"));
  EXPECT_FALSE(IsTransientHttpError("curl: (6) Could not resolve host: raw.githubusercontent.com"));
  EXPECT_FALSE(IsTransientHttpError("curl: (28) Operation timed out after 20001 milliseconds"));
  EXPECT_FALSE(IsTransientHttpError("ERROR 403: Forbidden."));
  EXPECT_FALSE(IsTransientHttpError(""));
}

TEST(GpuUnit, KFD_EVENT_FMT__MatchesMainlineKernelHeader) {
  const FetchResult& upstream = MainlineHeader();
  if (!upstream.unable_reason.empty()) {
    ReportUnableToVerify("kfd_event_formats.h", upstream.unable_reason);
    return;
  }
  const auto mainline = ExtractFormats(upstream.text);
  ASSERT_FALSE(mainline.empty()) << "Fetched " << kMainlineHeaderUrl
                                 << " but it defines no KFD_EVENT_FMT_* macros; the "
                                 << "layouts may have moved or been renamed upstream.";

  const std::string committed_path = CommittedHeaderPath();
  const auto committed = ExtractFormats(ReadFile(committed_path));
  if (committed.empty()) {
    ReportUnableToVerify("kfd_event_formats.h", LocalFileMissingReason(committed_path, upstream));
    return;
  }

  std::ostringstream diff;
  std::ostringstream detail;  // one line per macro, printed in verbose mode
  size_t identical = 0;
  for (const auto& [name, def] : committed) {
    const auto it = mainline.find(name);
    if (it == mainline.end()) {
      diff << "  removed upstream: " << name << "\n";
    } else if (it->second != def) {
      diff << "  changed: " << name << "\n    kfd_event_formats.h: " << def
           << "\n    upstream:            " << it->second << "\n";
    } else {
      ++identical;
    }
    detail << "\t      " << name << ": "
           << (it == mainline.end() ? "removed upstream"
                                    : (it->second != def ? "CHANGED" : "identical"))
           << "\n";
  }
  for (const auto& entry : mainline) {
    if (committed.find(entry.first) == committed.end()) {
      diff << "  added upstream: " << entry.first << "\n";
      detail << "\t      " << entry.first << ": ADDED upstream\n";
    }
  }
  if (Verbose()) {
    std::cout << "\t### drift check: kfd_event_formats.h vs " << upstream.source << "\n"
              << detail.str() << "\t      " << identical << " of " << committed.size()
              << " KFD_EVENT_FMT_* macros identical (whitespace and line breaks ignored)\n"
              << "\t===> " << (diff.str().empty() ? "VERIFIED" : "MISMATCH") << std::endl;
  }

  EXPECT_TRUE(diff.str().empty())
      << "kfd_event_formats.h differs from " << kMainlineHeaderUrl << ":\n"
      << diff.str()
      << "Copy the KFD_EVENT_FMT_* block from upstream into kfd_event_formats.h, then update "
         "FormatEvtNotifMessage() until the tests above pass.";
}

// What this checks: the event numbers and trigger values in rocm_smi's kfd_ioctl.h (the enums in
// kKfdEnums) against the same enums in the Linux kernel's master kfd_ioctl.h.
// How a mismatch shows: the test FAILS and lists each entry as "added upstream" (a new event or
// trigger the library does not know), "removed upstream" or "changed" (a different number). It is
// UNVERIFIED, and passes, only when the kernel header cannot be downloaded.
TEST(GpuUnit, KFD_EVENT_FMT__EnumsMatchMainlineKernelHeader) {
  const std::string what = "the event and trigger enums in rocm_smi's kfd_ioctl.h";
  const FetchResult& upstream = MainlineHeader();
  if (!upstream.unable_reason.empty()) {
    ReportUnableToVerify(what, upstream.unable_reason);
    return;
  }
  const std::string library_path = LibraryKfdHeaderPath();
  const std::string library_text = ReadFile(library_path);
  if (library_text.empty()) {
    ReportUnableToVerify(what, LocalFileMissingReason(library_path, upstream));
    return;
  }

  std::ostringstream all_differences;
  for (const char* name : kKfdEnums) {
    const EnumValues kernel = ParseEnum(upstream.text, name);
    const EnumValues library = ParseEnum(library_text, name);
    ASSERT_FALSE(kernel.empty()) << "Fetched " << kMainlineHeaderUrl << " but found no enum "
                                 << name << "; it may have been renamed upstream.";
    ASSERT_FALSE(library.empty()) << "Found no enum " << name << " in " << library_path;

    const std::string differences = DescribeEnumDifferences(library, kernel);
    if (Verbose()) {
      std::cout << "\t### enum " << name << ": " << library.size() << " entries here, "
                << kernel.size() << " upstream (" << upstream.source << ")\n"
                << "\t===> " << (differences.empty() ? "VERIFIED" : "MISMATCH") << std::endl;
    }
    if (!differences.empty()) {
      all_differences << "enum " << name << ":\n" << differences;
    }
  }

  EXPECT_TRUE(all_differences.str().empty())
      << "rocm_smi's kfd_ioctl.h differs from " << kMainlineHeaderUrl << ":\n"
      << all_differences.str()
      << "Update the enums in kfd_ioctl.h, then handle the new entries in "
         "FormatEvtNotifMessage() and the event notification types.";
}
