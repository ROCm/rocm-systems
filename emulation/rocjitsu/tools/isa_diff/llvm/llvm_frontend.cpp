// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "llvm_frontend.h"

#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

extern char **environ;

namespace rocjitsu::isa_diff::llvm {
namespace {

bool is_executable(const std::string &path) { return ::access(path.c_str(), X_OK) == 0; }

std::string first_line(const std::string &text) {
  const size_t nl = text.find('\n');
  std::string line = text.substr(0, nl);
  while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
    line.erase(line.begin());
  return line;
}

std::string tail(const std::string &text, size_t max_bytes = 2000) {
  return text.size() <= max_bytes ? text : text.substr(text.size() - max_bytes);
}

/// Run and record a command; returns its exit code.
int record(Provenance &prov, const std::vector<std::string> &argv, const std::string &stdout_path,
           const std::string &stderr_path) {
  ToolCommand cmd;
  cmd.argv = argv;
  const auto t0 = std::chrono::steady_clock::now();
  cmd.exit_code = run_tool(argv, stdout_path, stderr_path);
  cmd.elapsed_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
          .count());
  cmd.stderr_tail = tail(read_text_file(stderr_path).value_or(""));
  prov.commands.push_back(cmd);
  return cmd.exit_code;
}

std::string trim(const std::string &s) {
  const size_t b = s.find_first_not_of(" \t\r");
  if (b == std::string::npos)
    return {};
  const size_t e = s.find_last_not_of(" \t\r");
  return s.substr(b, e - b + 1);
}

bool is_label(std::string t) {
  if (t.empty() || t.front() == ';')
    return false;
  t = trim(t.substr(0, t.find(';'))); // ".LBB0_2:   ; %Flow"
  if (t.empty() || t.back() != ':')
    return false;
  for (size_t i = 0; i + 1 < t.size(); ++i) {
    const char c = t[i];
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '$'))
      return false;
  }
  return true;
}

/// String body for .ascii: backslash, quote, tab and newline escaped.
std::string asm_string(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '\\' || c == '"')
      out += '\\', out += c;
    else if (c == '\t')
      out += "\\t";
    else if (c == '\n')
      out += "\\n";
    else
      out += c;
  }
  return out;
}

} // namespace

std::string mark_inline_asm(const std::string &source, size_t &marked) {
  marked = 0;
  std::vector<std::string> lines;
  {
    std::istringstream in(source);
    std::string line;
    while (std::getline(in, line))
      lines.push_back(line);
  }
  std::string out;
  std::string records;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (trim(lines[i]) != ";;#ASMSTART") {
      out += lines[i] + "\n";
      continue;
    }
    size_t end = i + 1;
    while (end < lines.size() && trim(lines[end]) != ";;#ASMEND")
      ++end;
    bool comment_only = end < lines.size();
    std::string text;
    for (size_t k = i + 1; k < end && comment_only; ++k) {
      const std::string t = trim(lines[k]);
      if (t.empty())
        continue;
      if (t.front() != ';')
        comment_only = false;
      else
        text += (text.empty() ? "" : "\n") + t;
    }
    if (comment_only && !text.empty()) {
      // Does a label come before the next instruction? Skip blank and comment lines,
      // other comment-only asm blocks, and directives.
      bool before_label = false;
      for (size_t k = end + 1; k < lines.size(); ++k) {
        const std::string t = trim(lines[k]);
        if (t.empty() || t.front() == ';')
          continue;
        if (is_label(t))
          before_label = true;
        else if (t.front() == '.')
          continue;
        break;
      }
      const std::string name = std::string(kInlineAsmMarkerPrefix) + std::to_string(marked++);
      out += name + ":\n";
      records += "\t.asciz\t\"" +
                 asm_string(name + "\t" + (before_label ? "1" : "0") + "\t" + text) + "\"\n";
    }
    for (size_t k = i; k <= end && k < lines.size(); ++k)
      out += lines[k] + "\n";
    i = end;
  }
  if (marked)
    out += "\t.section\t" + std::string(kInlineAsmSection) + ",\"\",@progbits\n" + records;
  return out;
}

int run_tool(const std::vector<std::string> &argv, const std::string &stdout_path,
             const std::string &stderr_path) {
  if (argv.empty())
    return -1;
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  if (!stdout_path.empty())
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, stdout_path.c_str(),
                                     O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (!stderr_path.empty())
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, stderr_path.c_str(),
                                     O_WRONLY | O_CREAT | O_TRUNC, 0644);
  std::vector<char *> args;
  args.reserve(argv.size() + 1);
  for (const auto &a : argv)
    args.push_back(const_cast<char *>(a.c_str()));
  args.push_back(nullptr);
  pid_t pid = 0;
  const int rc = posix_spawn(&pid, argv[0].c_str(), &actions, nullptr, args.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (rc != 0)
    return -1;
  int status = 0;
  if (waitpid(pid, &status, 0) < 0)
    return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status)
                           : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
}

std::vector<std::string> split_args(const std::string &text) {
  std::vector<std::string> out;
  std::istringstream in(text);
  std::string word;
  while (in >> word)
    out.push_back(word);
  return out;
}

std::optional<std::string> check_side_args(const std::vector<std::string> &args) {
  for (const auto &a : args)
    for (const char *forbidden : {"-mtriple", "--mtriple", "-mcpu", "--mcpu", "-march", "--march",
                                  "-mattr", "--mattr", "-o", "-filetype", "--filetype"})
      if (a == forbidden || a.rfind(std::string(forbidden) + "=", 0) == 0)
        return "argument '" + a +
               "' would change the target, features or outputs between the sides; the tool "
               "fixes -mtriple/-mcpu/-filetype/-o, and -mattr belongs in the common arguments";
  return std::nullopt;
}

std::optional<Toolchain> Toolchain::discover(const std::string &hint, std::string &error) {
  std::vector<std::string> candidates;
  if (!hint.empty())
    candidates.push_back(hint);
  if (const char *env = std::getenv("RJ_ISA_DIFF_LLVM_BIN"))
    candidates.emplace_back(env);
  candidates.emplace_back("/opt/rocm/llvm/bin");
  for (const auto &dir : candidates) {
    Toolchain tc;
    tc.bin_dir = dir;
    tc.llc = dir + "/llc";
    tc.llvm_mc = dir + "/llvm-mc";
    tc.ld_lld = dir + "/ld.lld";
    tc.llvm_objcopy = dir + "/llvm-objcopy";
    if (!is_executable(tc.llc) || !is_executable(tc.llvm_mc) || !is_executable(tc.ld_lld))
      continue;
    const std::string tmp =
        (std::filesystem::temp_directory_path() / ("rj_isa_diff_llvm_" + std::to_string(getpid())))
            .string();
    if (run_tool({tc.llc, "--version"}, tmp + ".out", tmp + ".err") == 0) {
      const std::string text = read_text_file(tmp + ".out").value_or("");
      std::istringstream in(text);
      std::string line;
      while (std::getline(in, line))
        if (line.find("version") != std::string::npos) {
          tc.version = first_line(line);
          break;
        }
    }
    const std::string clang = dir + "/clang";
    if (is_executable(clang) && run_tool({clang, "--version"}, tmp + ".out", tmp + ".err") == 0)
      tc.revision = first_line(read_text_file(tmp + ".out").value_or(""));
    std::filesystem::remove(tmp + ".out");
    std::filesystem::remove(tmp + ".err");
    return tc;
  }
  error = "no LLVM toolchain (llc, llvm-mc, ld.lld) found; pass --llvm-bin DIR or set "
          "RJ_ISA_DIFF_LLVM_BIN";
  return std::nullopt;
}

Json Toolchain::to_json() const {
  Json j = Json::object();
  j.set("bin_dir", bin_dir);
  j.set("version", version);
  if (!revision.empty())
    j.set("revision", revision);
  return j;
}

Json Provenance::to_json() const {
  Json j = Json::object();
  j.set("input", input_path);
  j.set("input_sha256", input_sha256);
  j.set("input_kind", input_kind);
  j.set("triple", triple);
  j.set("mcpu", mcpu);
  if (wave_size)
    j.set("wave_size", static_cast<uint64_t>(wave_size));
  if (!mattr.empty())
    j.set("mattr", mattr);
  Json common = Json::array();
  for (const auto &a : common_args)
    common.push(a);
  j.set("common_args", std::move(common));
  Json side = Json::array();
  for (const auto &a : side_args)
    side.push(a);
  j.set("side_args", std::move(side));
  Json cmds = Json::array();
  for (const auto &c : commands) {
    Json cj = Json::object();
    Json argv = Json::array();
    for (const auto &a : c.argv)
      argv.push(a);
    cj.set("argv", std::move(argv));
    cj.set("exit_code", c.exit_code);
    cj.set("elapsed_ns", c.elapsed_ns);
    if (!c.stderr_tail.empty())
      cj.set("stderr_tail", c.stderr_tail);
    cmds.push(std::move(cj));
  }
  j.set("commands", std::move(cmds));
  uint64_t total = 0;
  for (const auto &c : commands)
    total += c.elapsed_ns;
  j.set("compile_ns", total);
  j.set("executed_object", executed_object);
  j.set("note", note);
  if (inline_asm_blocks)
    j.set("inline_asm_blocks_marked", static_cast<uint64_t>(inline_asm_blocks));
  j.set("toolchain", toolchain);
  return j;
}

namespace {

/// -mattr: the features asked for, then the wave size on a target that has both (GFX10+).
std::optional<std::string> wave_attr(const Provenance &prov) {
  std::string attr = prov.mattr;
  const auto t = target_by_name(prov.mcpu);
  if (prov.wave_size && t && t->major >= 10)
    attr +=
        (attr.empty() ? "" : ",") + std::string("+wavefrontsize") + std::to_string(prov.wave_size);
  if (attr.empty())
    return std::nullopt;
  return attr;
}

} // namespace

bool assemble(const Toolchain &tc, const std::string &s_path, const std::string &work_prefix,
              const std::string &kernel, Artifact &out, Provenance &prov, std::string &error) {
  auto text = read_text_file(s_path);
  if (!text) {
    error = "cannot read " + s_path;
    return false;
  }
  if (prov.input_path.empty()) {
    prov.input_path = s_path;
    prov.input_sha256 = sha256_hex(*text);
    prov.input_kind = "s";
  }
  prov.toolchain = tc.to_json();
  const std::string obj = work_prefix + ".o";
  const std::string hsaco = work_prefix + ".hsaco";
  const std::string err = work_prefix + ".stderr";
  // Comment-only inline-asm blocks assemble to nothing: mark where they are and record
  // their text in the object, so the code object carries them.
  size_t marked = 0;
  const std::string marked_text = mark_inline_asm(*text, marked);
  std::string mc_input = s_path;
  if (marked) {
    mc_input = work_prefix + ".marked.s";
    std::ofstream f(mc_input, std::ios::binary);
    f << marked_text;
    if (!f) {
      error = "cannot write " + mc_input;
      return false;
    }
    prov.inline_asm_blocks = marked;
  }
  std::vector<std::string> mc = {tc.llvm_mc, "-triple=" + prov.triple, "-mcpu=" + prov.mcpu};
  if (auto attr = wave_attr(prov))
    mc.push_back("-mattr=" + *attr);
  mc.insert(mc.end(), {"-filetype=obj", "-o", obj, mc_input});
  if (record(prov, mc, "", err) != 0) {
    error = "llvm-mc failed on " + s_path + ": " + prov.commands.back().stderr_tail;
    return false;
  }
  // A linked code object keeps the kernel descriptor and symbol size; bare instruction
  // listings have neither and fall back to the raw .text bytes.
  // --emit-relocs keeps .rela.text: the executed bytes are unchanged, and instructions whose
  // bytes came from relocation processing stay identifiable (handoff 4).
  if (record(prov, {tc.ld_lld, "-shared", "--emit-relocs", "-o", hsaco, obj}, "", err) == 0) {
    std::string load_error;
    if (load_artifact(hsaco, kernel, out, load_error)) {
      prov.executed_object = "linked_code_object";
      prov.note = "executed bytes are the code object linked from the re-assembled source, "
                  "decoded by RocJITsu; they are not necessarily an original binary emission";
      out.label = hsaco;
      return true;
    }
    prov.note = "linked object has no usable kernel (" + load_error + "); using raw .text";
  }
  const std::string bin = work_prefix + ".text.bin";
  if (!is_executable(tc.llvm_objcopy) ||
      record(prov, {tc.llvm_objcopy, "-O", "binary", "--only-section=.text", obj, bin}, "", err) !=
          0) {
    error = "cannot extract .text from " + obj;
    return false;
  }
  if (!load_artifact(bin, "", out, error))
    return false;
  if (auto t = target_by_name(prov.mcpu)) { // raw words carry no target: the one assembled for
    out.target = *t;
    if (prov.wave_size)
      if (auto w = with_wave_size(*t, prov.wave_size))
        out.target = *w;
  }
  prov.executed_object = "raw_text";
  if (prov.note.empty())
    prov.note = "raw .text of the re-assembled source (no kernel descriptor)";
  return true;
}

bool compile(const Toolchain &tc, const std::string &ll_path,
             const std::vector<std::string> &common_args, const std::vector<std::string> &side_args,
             const std::string &work_prefix, const std::string &kernel, Artifact &out,
             Provenance &prov, std::string &error) {
  auto text = read_text_file(ll_path);
  if (!text) {
    error = "cannot read " + ll_path;
    return false;
  }
  if (auto bad = check_side_args(side_args)) {
    error = *bad;
    return false;
  }
  for (const auto &a : common_args)
    if (a.rfind("-mtriple", 0) == 0 || a.rfind("-mcpu", 0) == 0 || a == "-o" ||
        a.rfind("-filetype", 0) == 0) {
      error = "common argument '" + a + "' is fixed by the tool";
      return false;
    }
  prov.input_path = ll_path;
  prov.input_sha256 = sha256_hex(*text);
  prov.input_kind = "ll";
  prov.common_args = common_args;
  prov.side_args = side_args;
  prov.toolchain = tc.to_json();
  const std::string s_path = work_prefix + ".s";
  std::vector<std::string> argv = {tc.llc, "-mtriple=" + prov.triple, "-mcpu=" + prov.mcpu};
  if (auto attr = wave_attr(prov))
    argv.push_back("-mattr=" + *attr);
  argv.insert(argv.end(), common_args.begin(), common_args.end());
  argv.insert(argv.end(), side_args.begin(), side_args.end());
  argv.insert(argv.end(), {"-filetype=asm", "-o", s_path, ll_path});
  if (record(prov, argv, "", work_prefix + ".stderr") != 0) {
    error = "llc failed: " + prov.commands.back().stderr_tail;
    return false;
  }
  return assemble(tc, s_path, work_prefix, kernel, out, prov, error);
}

} // namespace rocjitsu::isa_diff::llvm
