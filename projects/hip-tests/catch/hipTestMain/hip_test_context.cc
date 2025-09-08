/*
 * Copyright (C) Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cstdlib>
#include <hip_test_common.hh>
#include <picojson.h>
#include <fstream>
#include <sstream>
#include <regex>
#include "hip_test_context.hh"
#include "hip_test_filesystem.hh"
#include "hip_test_features.hh"

void TestContext::detectOS() {
#if (HT_WIN == 1)
  p_windows = true;
#elif (HT_LINUX == 1)
  p_linux = true;
#endif
}

void TestContext::detectPlatform() {
#if (HT_AMD == 1)
  amd = true;
#elif (HT_NVIDIA == 1)
  nvidia = true;
#endif
}

std::string TestContext::substringFound(std::vector<std::string> list, std::string filename) {
  std::string match = "";
  for (unsigned int i = 0; i < list.size(); i++) {
    if (filename.find(list.at(i)) != std::string::npos) {
      match = list.at(i);
      break;
    }
  }
  return match;
}

std::string TestContext::getCurrentArch() {
#if HT_LINUX
  const char* cmd =
      "/opt/rocm/bin/rocm_agent_enumerator | awk '$0 != \"gfx000\"' | xargs | sed -e 's/ /;/g' | "
      "tr -d '\n'";
  std::array<char, 1024> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    printf("popen() failed!");
    return "";
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    std::string res = buffer.data();
    result = res;
  }

  std::string s_visible_devices = TestContext::getEnvVar("HIP_VISIBLE_DEVICES");

  auto parser = [](std::string input, char c) -> std::vector<std::string> {
    std::vector<std::string> ret;
    auto loc = input.find(c);
    while (loc != std::string::npos) {
      auto t_str = input.substr(0, loc);
      ret.push_back(t_str);
      input.erase(0, loc + 1);
      loc = input.find(c);
    }
    if (input.size() > 0) {
      ret.push_back(input);
    }
    return ret;
  };

  std::vector<std::string> archs = parser(result, ';');
  std::vector<std::string> v_visible_devices = parser(s_visible_devices, ',');
  std::vector<int> visible_devices;
  std::for_each(v_visible_devices.begin(), v_visible_devices.end(),
                [&](const std::string& in) { visible_devices.push_back(std::stoi(in)); });

  if (archs.size() == 0) {
    return "";  // rocm_agent_enum gave us garbage
  }

  auto first_arch = archs[0];
  if (!std::all_of(archs.begin(), archs.end(),
                   [&](const std::string& in) { return in == first_arch; })) {
    // We have multiple archs in rocm_agent_enum
    // Check if they are same or not by applying HIP_VISIBLE_DEVICES filter
    std::vector<std::string> filtered_archs;
    if (visible_devices.size() > 0) {
      for (size_t i = 0; i < visible_devices.size(); i++) {
        filtered_archs.push_back(archs[visible_devices[i]]);
      }
    } else {
      filtered_archs = archs;
    }
    auto first_filtered_arch = filtered_archs[0];
    if (!std::all_of(filtered_archs.begin(), filtered_archs.end(),
                     [&](const std::string& in) { return in == first_filtered_arch; })) {
      LogPrintf("%s",
                "[ERROR] Cannot run tests on Hetrogenous Architecture. Please set "
                "HIP_VISIBLE_DEVICES with devices of same arch");
      std::abort();
    }
    return first_filtered_arch;
  }
  return first_arch;
#else
  return "";
#endif
}

std::string TestContext::getMatchingConfigFile(std::string config_dir) {
  std::string configFileToUse = "";
  if (isLinux() && isAmd()) {
    std::string cur_arch = getCurrentArch();
    LogPrintf("The arch present: %s", cur_arch.c_str());
    configFileToUse = config_dir + "/config_" + getConfig().platform + "_" + getConfig().os + "_" +
                      cur_arch + ".json";
  } else {
    configFileToUse =
        config_dir + "/config_" + getConfig().platform + "_" + getConfig().os + ".json";
  }
  if (fs::exists(configFileToUse)) {
    return configFileToUse;
  }
  return "";
}

std::string& TestContext::getCommonJsonFile() {
  fs::path config_dir = exe_path;
  config_dir = config_dir.parent_path();
  int levels = 0;
  bool configFolderFound = false;
  std::vector<std::string> configList;
  std::string configFile;
  // check a max of 5 levels down the executable path
  while (levels < 5) {
    fs::path temp_path = config_dir;
    temp_path /= "config";
    if (fs::exists(temp_path)) {
      config_dir = fs::absolute(temp_path);
      configFolderFound = true;
      break;
    } else {
      config_dir = config_dir.parent_path();
      levels++;
    }
  }

  // get config.json files if config folder.
  if (configFolderFound) {
    json_file_ = getMatchingConfigFile(config_dir.string());
  }
  return json_file_;
}


void TestContext::getConfigFiles() {
  config_.platform = (amd ? "amd" : (nvidia ? "nvidia" : "unknown"));
  config_.os = (p_windows ? "windows" : (p_linux ? "linux" : "unknown"));

  if (config_.os == "unknown" || config_.platform == "unknown") {
    LogPrintf("%s", "Either Config or Os is unknown, this wont end well");
    abort();
  }

  std::string env_config = TestContext::getEnvVar("HIP_CATCH_EXCLUDE_FILE");
  LogPrintf("Env Config file: %s", (!env_config.empty()) ? env_config.c_str() : "Not found");
  // HIP_CATCH_EXCLUDE_FILE is set for custom file path
  if (!env_config.empty()) {
    if (fs::exists(env_config)) {
      config_.json_files.push_back(env_config);
    }
  } else {
    std::string jsonFile = getCommonJsonFile();
    // get common json file
    if (jsonFile != "") {
      config_.json_files.push_back(getCommonJsonFile());
    }
  }

  for (const auto& fl : config_.json_files) {
    LogPrintf("Config file path: %s", fl.c_str());
  }
}

TestContext::TestContext(int argc, char** argv) {
  detectOS();
  detectPlatform();
  setExePath(argc, argv);
  getConfigFiles();
  parseJsonFiles();
  parseOptions(argc, argv);
}

void TestContext::setExePath(int argc, char** argv) {
  if (argc == 0) return;
  fs::path p = std::string(argv[0]);
  if (p.has_filename()) p.remove_filename();
  exe_path = p.string();
}

bool TestContext::isWindows() const { return p_windows; }
bool TestContext::isLinux() const { return p_linux; }

bool TestContext::isNvidia() const { return nvidia; }
bool TestContext::isAmd() const { return amd; }

void TestContext::parseOptions(int argc, char** argv) {
  // Test name is at [1] position
  if (argc != 2) return;
  current_test = std::string(argv[1]);
}

bool TestContext::skipTest() const {
  // Direct Match
  auto flags = std::regex::ECMAScript;
  for (const auto& i : skip_test) {
    auto regex = std::regex(i.c_str(), flags);
    if (std::regex_match(current_test, regex)) {
      return true;
    }
  }
  // TODO add test case skip as well
  return false;
}

std::string TestContext::currentPath() const { return fs::current_path().string(); }

bool TestContext::parseJsonFiles() {
  // Check if file exists
  for (const auto& fl : config_.json_files) {
    if (!fs::exists(fl)) {
      LogPrintf("Unable to find the file: %s", fl.c_str());
      return true;
    }
    // Open the file
    std::ifstream js_file(fl);
    std::string json_str((std::istreambuf_iterator<char>(js_file)),
                         std::istreambuf_iterator<char>());
    LogPrintf("Json contents:: %s", json_str.data());

    picojson::value v;
    std::string err = picojson::parse(v, json_str);
    if (err.size() > 1) {
      LogPrintf("Error from PicoJson: %s", err.data());
      return false;
    }

    if (!v.is<picojson::object>()) {
      LogPrintf("%s", "Data in json is not in correct format, it should be an object");
      return false;
    }

    const picojson::object& o = v.get<picojson::object>();
    for (picojson::object::const_iterator i = o.begin(); i != o.end(); ++i) {
      // Processing for DisabledTests
      if (i->first == "DisabledTests") {
        // Value should contain list of values
        if (!i->second.is<picojson::array>()) return false;

        auto& val = i->second.get<picojson::array>();
        for (auto ai = val.begin(); ai != val.end(); ai++) {
          std::string tmp = ai->get<std::string>();
          std::string newRegexName;
          for (const auto& c : tmp) {
            if (c == '*')
              newRegexName += ".*";
            else
              newRegexName += c;
          }
          skip_test.insert(newRegexName);
        }
      }
    }
  }
  return true;
}

void TestContext::cleanContext() {
  for (auto& pair : compiledKernels) {
    hipError_t error = hipModuleUnload(pair.second.module);
    if (error != hipSuccess) {
      throw std::runtime_error("Unable to unload rtc module");
    }
  }
}

void TestContext::trackRtcState(std::string kernelNameExpression, hipModule_t loadedModule,
                                hipFunction_t kernelFunction) {
  rtcState state{loadedModule, kernelFunction};
  compiledKernels[kernelNameExpression] = state;
}

hipFunction_t TestContext::getFunction(const std::string kernelNameExpression) {
  auto it{compiledKernels.find(kernelNameExpression)};

  if (it != compiledKernels.end()) {
    return it->second.kernelFunction;
  } else {
    return nullptr;
  }
}

void TestContext::addResults(HCResult r) {
  std::unique_lock<std::mutex> lock(resultMutex);
  results.push_back(r);
  if ((!r.conditionsResult) ||
      ((r.result != hipSuccess) && (r.result != hipErrorPeerAccessAlreadyEnabled))) {
    hasErrorOccured_.store(true);
  }
}

void TestContext::finalizeResults() {
  std::unique_lock<std::mutex> lock(resultMutex);
  // clear the results whatever happens
  std::shared_ptr<void> emptyVec(nullptr, [this](auto) { results.clear(); });

  for (const auto& i : results) {
    INFO("HIP API Result check\n    File:: "
         << i.file << "\n    Line:: " << i.line << "\n    API:: " << i.call
         << "\n    Result:: " << i.result << "\n    Result Str:: " << hipGetErrorString(i.result));
    REQUIRE(((i.result == hipSuccess) || (i.result == hipErrorPeerAccessAlreadyEnabled) ||
             (i.result == hipErrorNotSupported)));
    REQUIRE(i.conditionsResult);
  }
  hasErrorOccured_.store(false);  // Clear the flag
}

bool TestContext::hasErrorOccured() { return hasErrorOccured_.load(); }

TestContext::~TestContext() {
  // Show this message when there are unchecked results
  if (results.size() != 0) {
    std::cerr << "HIP_CHECK_THREAD_FINALIZE() has not been called after HIP_CHECK_THREAD\n"
              << "Please call HIP_CHECK_THREAD_FINALIZE after joining threads\n"
              << "There is/are " << results.size() << " unchecked results from threads."
              << std::endl;
    std::abort();  // Crash to bring users attention to this message and avoid accidental passing of
                   // tests without checking for errors
  }
}
