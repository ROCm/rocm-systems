/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018-2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */
#define _GNU_SOURCE 1 // REQUIRED: to utilize some GNU features/functions, see
                      // _GNU_SOURCE functions which check
#include <assert.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <glob.h>
#include <sys/utsname.h>
#include <dlfcn.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace amd {
namespace smi {
const std::string kTmpFilePrefix = "rocmsmi_";

// Return 0 if same file, 1 if not, and -1 for error
int SameFile(const std::string fileA, const std::string fileB) {
  struct stat aStat;
  struct stat bStat;
  int ret;

  ret = stat(fileA.c_str(), &aStat);
  if (ret) {
      return -1;
  }

  ret = stat(fileB.c_str(), &bStat);
  if (ret) {
      return -1;
  }

  if (aStat.st_dev != bStat.st_dev) {
      return 1;
  }

  if (aStat.st_ino != bStat.st_ino) {
      return 1;
  }

  return 0;
}

bool FileExists(char const *filename) {
  struct stat buf;
  return (stat(filename, &buf) == 0);
}

static inline void debugFilesDiscovered(std::vector<std::string> files) {
  std::ostringstream ss;
  int numberOfFilesFound = static_cast<int>(files.size());
  ss << "fileName.size() = " << numberOfFilesFound
     << "; Files discovered = {";
  if(numberOfFilesFound > 0) {
    for (auto it = begin(files); it != end(files); ++it) {
      auto nextElement = std::next(it);
      if (nextElement != files.end()) {
        ss << *it << ", ";
      } else {
        ss << *it;
      }
    }
  } else {
    ss << "<none>";
  }
  ss << "}";
  LOG_DEBUG(ss);
}

// Input: string filePattern = can put in * file searches (see example)
// example: globFilesExist("/etc/*release")
// Return a vector containing file paths that matched
// You can obtain if files exist by doing globFilesExist(...).size() > 0
std::vector<std::string> globFilesExist(const std::string& filePattern) {
  std::ostringstream ss;
  std::vector<std::string> fileNames;
  glob_t result_glob;
  memset(&result_glob, 0, sizeof(result_glob));

  if (glob(filePattern.c_str(), GLOB_TILDE, nullptr, &result_glob) != 0) {
    globfree(&result_glob);
    // Leaving below to help debug issues discovering future glob file searches
    // debugFilesDiscovered(fileNames);
    return fileNames;
  }

  for(size_t i = 0; i < result_glob.gl_pathc; ++i) {
    fileNames.emplace_back(result_glob.gl_pathv[i]);
  }
  globfree(&result_glob);

  // Leaving below to help debug issues discovering future glob file searches
  // debugFilesDiscovered(fileNames);
  return fileNames;
}

int isRegularFile(std::string fname, bool *is_reg) {
  struct stat file_stat;
  int ret;

  ret = stat(fname.c_str(), &file_stat);
  if (ret) {
    return errno;
  }

  if (is_reg != nullptr) {
    *is_reg = S_ISREG(file_stat.st_mode);
  }

  return 0;
}

int isReadOnlyForAll(const std::string& fname, bool *is_read_only){
  struct stat file_stat;
  int ret;

  ret = stat(fname.c_str(), &file_stat);
  if (ret) {
    return errno;
  }

  if (is_read_only != nullptr) {
    *is_read_only = (file_stat.st_mode & (S_IRUSR | S_IRGRP | S_IROTH)) && !(file_stat.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH));
  } else {
    ret = 1;
  }

  return ret;
}

int WriteSysfsStr(std::string path, std::string val) {
  //  On success, zero is returned.  On error, -1 is returned, and
  //  errno is set to indicate the error.
  auto is_regular_file_result = isRegularFile(path, nullptr);
  if (is_regular_file_result != 0) {
    return ENOENT;
  }

  std::ofstream fs;
  int ret = 0;
  std::ostringstream ss;

  fs.open(path);
  if (!fs.is_open()) {
    ret = errno;
    errno = 0;
    ss << "Could not write/open SYSFS file (" << path << ") string = " << val
       << ", returning " << std::to_string(ret) << " ("
       << std::strerror(ret) << ")";
    LOG_ERROR(ss);
    return ret;
  }

  fs << val;
  fs.close();
  if (!fs) {
    return ENOENT;  // Map to NOT_SUPPORT if errors
  }
  ss << "Successfully wrote to SYSFS file (" << path << ") string = " << val;
  LOG_INFO(ss);
  return ret;
}

int ReadSysfsStr(std::string path, std::string *retStr) {
  //  On success, zero is returned.  On error, -1 is returned, and
  //  errno is set to indicate the error.
  auto is_regular_file_result = isRegularFile(path, nullptr);
  if (is_regular_file_result != 0) {
    return ENOENT;
  }

  std::stringstream ss;
  int ret = 0;
  std::ostringstream oss;

  assert(retStr != nullptr);

  std::ifstream fs;
  fs.open(path);

  if (!fs.is_open()) {
    ret = errno;
    errno = 0;
    oss << __PRETTY_FUNCTION__
      << " | Fail | Cause: file does not exist or permissions issue"
      << " | SYSFS file: " << path
      << " | Returning: " <<  std::strerror(ret) << " |";
    LOG_ERROR(oss);
    return ret;
  }
  ss << fs.rdbuf();
  fs.close();

  *retStr = ss.str();

  retStr->erase(std::remove(retStr->begin(), retStr->end(), '\n'),
                                                               retStr->end());
  oss << "Successfully read SYSFS file (" << path << ")"
      << ", returning str = " << *retStr;
  LOG_INFO(oss);
  return ret;
}

bool IsInteger(const std::string & n_str) {
  if (n_str.empty() || ((!isdigit(n_str[0])) && (n_str[0] != '-')
                                                      && (n_str[0] != '+'))) {
    return false;
  }

  char * tmp;
  strtol(n_str.c_str(), &tmp, 10);

  return (*tmp == 0);
}

rsmi_status_t handleException() {
  try {
    throw;
  } catch (const std::bad_alloc& e) {
    debug_print("RSMI exception: BadAlloc\n");
    return RSMI_STATUS_OUT_OF_RESOURCES;
  } catch (const amd::smi::rsmi_exception& e) {
    debug_print("Exception caught: %s.\n", e.what());
    return e.error_code();
  } catch (const std::exception& e) {
    debug_print("Exception caught: %s\n", e.what());
    return RSMI_STATUS_INTERNAL_EXCEPTION;
  } catch (const std::nested_exception& e) {
    debug_print("Callback threw.\n");
    return RSMI_STATUS_INTERNAL_EXCEPTION;
  } catch (...) {
    debug_print("Unknown exception caught.\n");
    return RSMI_STATUS_INTERNAL_EXCEPTION;
  }
}

pthread_mutex_t *GetMutex(uint32_t dv_ind) {
  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  if (dv_ind >= smi.devices().size()) {
    return nullptr;
  }
  std::shared_ptr<amd::smi::Device> dev = smi.devices()[dv_ind];
  assert(dev != nullptr);

  return dev->mutex();
}

rsmi_status_t GetDevValueVec(amd::smi::DevInfoTypes type,
                         uint32_t dv_ind, std::vector<std::string> *val_vec) {
  assert(val_vec != nullptr);
  if (val_vec == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  GET_DEV_FROM_INDX

  int ret = dev->readDevInfo(type, val_vec);
  return ErrnoToRsmiStatus(ret);
}

rsmi_status_t
GetDevBinaryBlob(amd::smi::DevInfoTypes type,
  uint32_t dv_ind, std::size_t b_size, void* p_binary_data) {
  assert(p_binary_data != nullptr);
  if (p_binary_data == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  GET_DEV_FROM_INDX

  int ret = dev->readDevInfo(type, b_size, p_binary_data);
  return ErrnoToRsmiStatus(ret);
}

rsmi_status_t ErrnoToRsmiStatus(int err) {
  switch (err) {
    case 0:      return RSMI_STATUS_SUCCESS;
    case ESRCH:  return RSMI_STATUS_NOT_FOUND;
    case EACCES: return RSMI_STATUS_PERMISSION;
    case EPERM:
    case ENOENT: return RSMI_STATUS_NOT_SUPPORTED;
    case EBADF:
    case EISDIR: return RSMI_STATUS_FILE_ERROR;
    case EINTR:  return RSMI_STATUS_INTERRUPT;
    case EIO:    return RSMI_STATUS_UNEXPECTED_SIZE;
    case ENXIO:  return RSMI_STATUS_UNEXPECTED_DATA;
    case EBUSY:  return RSMI_STATUS_BUSY;
    case EINVAL: return RSMI_STATUS_INVALID_ARGS;
    default:     return RSMI_STATUS_UNKNOWN_ERROR;
  }
}

bool is_vm_guest() {
  // the cpuinfo will set hypervisor flag in VM guest
  const std::string hypervisor = "hypervisor";
  std::string line;

  // default to false if cannot find the file
  std::ifstream infile("/proc/cpuinfo");
  if (infile.fail()) {
    return false;
  }

  while (std::getline(infile, line)) {
    if (line.find(hypervisor) != std::string::npos) {
      return true;
    }
  }
  return false;
}

std::string leftTrim(const std::string &s) {
  if (!s.empty()) {
    return std::regex_replace(s, std::regex("^\\s+"), "");
  }
  return s;
}

std::string rightTrim(const std::string &s) {
  if (!s.empty()) {
    return std::regex_replace(s, std::regex("\\s+$"), "");
  }
  return s;
}

std::string removeNewLines(const std::string &s) {
  if (!s.empty()) {
    return std::regex_replace(s, std::regex("\n+"), "");
  }
  return s;
}

// Trims white space from both ends of string
std::string trim(const std::string &s) {
  if (!s.empty()) {
    // remove new lines -> trim white space at ends
    std::string noNewLines = removeNewLines(s);
    return leftTrim(rightTrim(noNewLines));
  }
  return s;
}

// Trims white space from both ends of string and removes all white space
std::string trimAllWhiteSpace(const std::string &s) {
  if (!s.empty()) {
    // remove new lines -> trim white space at ends
    std::string noNewLines = trim(s);
    return removeWhitespace(noNewLines);
  }
  return s;
}

std::string removeWhitespace(const std::string &s) {
  if (!s.empty()) {
    return std::regex_replace(s, std::regex("\\s+"), "");
  }
  return s;
}

// Given original string and string to remove (removeMe)
// Return will provide the resulting modified string with the removed string(s)
std::string removeString(const std::string origStr,
                         const std::string &removeMe) {
  std::string modifiedStr = origStr;
  std::string::size_type l = removeMe.length();
  for (std::string::size_type i = modifiedStr.find(removeMe);
       i != std::string::npos;
       i = modifiedStr.find(removeMe)) {
    modifiedStr.erase(i, l);
  }
  return modifiedStr;
}

// defaults to trim stdOut
std::pair<bool, std::string> executeCommand(std::string command, bool stdOut) {
  char buffer[128];
  std::string stdoutAndErr;
  bool successfulRun = true;
  command = "stdbuf -i0 -o0 -e0 " + command;  // remove stdOut and err buffering

  FILE *pipe = popen(command.c_str(), "r");
  if (!pipe) {
    stdoutAndErr = "[ERROR] popen failed to call " + command;
    successfulRun = false;
  } else {
    // read until end of process
    while (!feof(pipe)) {
      // use buffer to read and add to stdoutAndErr
      if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        stdoutAndErr += buffer;
      }
    }
  }

  // any return code other than 0, is a failed execution
  if (pclose(pipe) != 0) {
    successfulRun = false;
  }

  if (stdOut) {
    // remove leading and trailing spaces of output and new lines
    stdoutAndErr = trim(stdoutAndErr);
  }
  return std::make_pair(successfulRun, stdoutAndErr);
}

// originalString - string to search for substring
// substring - string looking to find
// displayComparisons = defaults to false, set to true to see debug prints
bool containsString(std::string originalString, std::string substring,
                  bool displayComparisons) {
  std::ostringstream ss;
  bool found = originalString.find(substring) != std::string::npos;
  if (displayComparisons) {
    ss << __PRETTY_FUNCTION__
       << " | originalString: " << originalString
       << " | substring: " << substring
       << " | found: " << (found ? "True": "False");
    LOG_TRACE(ss);
  }
  return found;
}

// Creates and stores supplied data into a temporary file (within /tmp/).
// All temporary files are removed upon reboot.
// Allows all users/groups to read the temporary file.
//
// For more detail, refer to mkstemp manpage:
// https://man7.org/linux/man-pages/man3/mkstemp.3.html
//
// Temporary file name format:
// <app prefix>_<state name>_<parameter name>_<device id>
// <app prefix> - prefix for our application's identifier (see kTmpFilePrefix)
// <parameter name> - name of parameter being stored
// <state name> - state at which the stored value captures
// <device index> - device identifier
//
// dv_ind - device index
// parameterName - name of parameter stored
// stateName - state at which the stored value captures
// storageData - string value of data to be stored
rsmi_status_t storeTmpFile(uint32_t dv_ind, std::string parameterName,
                           std::string stateName, std::string storageData) {
  // Required tags needed to store our files
  // Files name format:
  // <app prefix>_<stateName>_<parameterName>_<device id>
  std::string fullFileName = kTmpFilePrefix + stateName + "_" +
                             parameterName + "_" + std::to_string(dv_ind);
  bool doesFileExist;
  std::tie(doesFileExist, std::ignore) =
        readTmpFile(dv_ind, stateName, parameterName);
  if (doesFileExist) {
    // do not store, if file already exists
    return RSMI_STATUS_SUCCESS;
  }
  // template for our file
  std::string fullTempFilePath = "/tmp/" + fullFileName + ".XXXXXX";
  char *fileName = &fullTempFilePath[0];
  int fd = mkstemp(fileName);
  if (fd == -1) {
    return RSMI_STATUS_FILE_ERROR;
  }

  chmod(fileName, S_IRUSR|S_IRGRP|S_IROTH);
  ssize_t rc_write = write(fd, storageData.c_str(), storageData.size());
  close(fd);
  if (rc_write == -1) {
    return RSMI_STATUS_FILE_ERROR;
  }
  return RSMI_STATUS_SUCCESS;
}

std::vector<std::string> getListOfAppTmpFiles() {
  std::string path = "/tmp";
  DIR *dir;
  struct dirent *ent;
  std::vector<std::string> tmpFiles;

  dir = opendir(path.c_str());
  if (dir == nullptr) {
    return tmpFiles;
  }
  // captures all files & directories under specified path
  while ((ent = readdir(dir)) != nullptr) {
    std::string fileDirName = ent->d_name;
    // we only want our app specific files
    if (containsString(fileDirName, kTmpFilePrefix)) {
      tmpFiles.emplace_back(path + "/" + fileDirName);
    } else {
      continue;
    }
  }
  closedir(dir);
  return tmpFiles;
}

// Reads a file in path provided
// If file does not exist, returns an empty string
// If file exists, returns content (which could be an empty string)
std::string readFile(std::string path) {
  std::string fileContent;
  std::ifstream inFileStream(path);
  if (inFileStream.is_open()) {
    inFileStream >> fileContent;
  }
  return fileContent;
}

// Reads a file in path provided
// If file does not exist, returns an empty vector
// If file exists, returns content (each line put into a vector; which
// could be an empty string)
std::vector<std::string> readEntireFile(std::string path) {
  std::vector<std::string> fileContent;
  std::ifstream inFileStream(path);
  if (inFileStream.is_open()) {
    std::string line;
    while (std::getline(inFileStream, line)) {
      std::istringstream ss(line);
      if (!line.empty()) {
        fileContent.push_back(line);
      }
    }
  }
  return fileContent;
}

// Used to debug application temporary files (identified by kTmpFilePrefix)
// and their content
void displayAppTmpFilesContent() {
  std::vector<std::string> tmpFiles = getListOfAppTmpFiles();
  if (!tmpFiles.empty()) {
    for (auto &x : tmpFiles) {
      std::string out = readFile(x);
      std::cout << __PRETTY_FUNCTION__ << " | Temporary file: " << x
                << "; Contained content: " << out << std::endl;
    }
  } else {
    std::cout << __PRETTY_FUNCTION__ << " | No temporary files were found"
              << std::endl;
  }
}

// Used to debug vector string list and their content
std::string debugVectorContent(std::vector<std::string> v) {
  std::ostringstream ss;
  ss << "Vector = {";
  if (!v.empty()) {
    for (auto it=v.begin(); it < v.end(); it++) {
      ss << *it;
      auto temp_it = it;
      if (++temp_it != v.end()) {
        ss << ", ";
      }
    }
  }
  ss << "}" << std::endl;

  return ss.str();
}

// Used to debug vector string list and their content
std::string displayAllDevicePaths(std::vector<std::shared_ptr<Device>> v) {
  std::ostringstream ss;
  ss << "Vector = {";
  if (!v.empty()) {
    for (auto it=v.begin(); it < v.end(); it++) {
      ss << (*it)->path();
      auto temp_it = it;
      if (++temp_it != v.end()) {
        ss << ", ";
      }
    }
  }
  ss << "}" << std::endl;

  return ss.str();
}

// Attempts to read application specific temporary file
// This method is to be used for reading (or determining if it exists),
// in order to keep file naming scheme consistent.
//
// dv_ind - device index
// parameterName - name of parameter stored
// stateName - state at which the stored value captures
// Returns:
// boolean - if temporary file exists
// string - content of temporary file, if it exists (otherwise, an empty
// string is returned)
std::tuple<bool, std::string> readTmpFile(uint32_t dv_ind,
                                          std::string stateName,
                                          std::string parameterName) {
  bool fileExists = false;
  std::string tmpFileName = kTmpFilePrefix + stateName + "_" +parameterName +
                            "_" + std::to_string(dv_ind);
  std::string fileContent;
  std::vector<std::string> tmpFiles = getListOfAppTmpFiles();
  if (!tmpFiles.empty()) {
    for (auto &x: tmpFiles) {
      if (containsString(x, tmpFileName)) {
        fileContent = readFile(x);
        fileExists = true;
        break;
      }
    }
  }
  return std::make_tuple(fileExists, fileContent);
}

// wrapper to return string expression of a rsmi_status_t return
// rsmi_status_t ret - return value of RSMI API function
// bool fullStatus - defaults to true, set to false to chop off description
// Returns:
// string - if fullStatus == true, returns full decription of return value
//      ex. 'RSMI_STATUS_SUCCESS: The function has been executed successfully.'
// string - if fullStatus == false, returns a minimalized return value
//      ex. 'RSMI_STATUS_SUCCESS'
std::string getRSMIStatusString(rsmi_status_t ret, bool fullStatus) {
  const char *err_str;
  rsmi_status_string(ret, &err_str);
  if (!fullStatus) {
    return splitString(std::string(err_str), ':');
  }
  return std::string(err_str);
}

// Returns a tuple:
// boolean errorDetected = returns true, if error found retrieving system
// details
// string sysname = system name (os name)
// string nodename = name of the system's node on the network
// string release = os's release level
// string version = os's version level
// string machine = hardware type system is running on
// string domainName = domain name of the the system's node on the network
// string os_distribution = pretty name of os distribution
// (typically found in /etc/*-release file)
// string endianness = system's endianness.
// Expressed as big endian or little endian.
// Big Endian (BE), multi-bit symbols encoded as big endian (MSB first)
// Little Endian (LE), multi-bit symbols encoded as little endian (LSB first)
// string rocm_lib_path = Path to library
// string rocm_build_type = Release or debug
// string rocm_build_date = Creation date of library
// string dev_gfx_versions = GPU target graphics version
std::tuple<bool, std::string, std::string, std::string, std::string,
           std::string, std::string, std::string, std::string,
           std::string, std::string, std::string, std::string, std::string>
           getSystemDetails(void) {
  struct utsname buf;
  bool errorDetected = false;
  std::string temp_data;
  std::string sysname;
  std::string nodename;
  std::string release;
  std::string version;
  std::string machine;
  std::string domainName = "<undefined>";
  std::string os_distribution = "<undefined>";
  std::string endianness = "<undefined>";
  std::string rocm_lib_path = "<undefined>";
  std::string rocm_build_type = "<undefined>";
  std::string rocm_build_date = "<undefined>";
  std::string rocm_env_variables = "<undefined>";
  std::string dev_gfx_versions = "<undefined>";

  if (uname(&buf) < 0) {
    errorDetected = true;
  } else {
    sysname = buf.sysname;
    nodename = buf.nodename;
    release = buf.release;
    version = buf.version;
    machine = buf.machine;
    #ifdef _GNU_SOURCE
      domainName = buf.domainname;
    #endif
  }

  std::string filePath = "/etc/os-release";
  bool fileExists = FileExists(filePath.c_str());
  if (fileExists) {
    std::vector<std::string> fileContent = readEntireFile(filePath);
    for (auto &line: fileContent) {
      if (line.find("PRETTY_NAME=") != std::string::npos) {
        temp_data = removeString(line, "PRETTY_NAME=");
        temp_data = removeString(temp_data, "\"");
        os_distribution = temp_data;
        break;
      }
    }
  }
  if (isSystemBigEndian()) {
    endianness = "Big Endian, multi-bit symbols encoded as"
                 " big endian (MSB first)";
  } else {
    endianness = "Little Endian, multi-bit symbols encoded as"
                 " little endian (LSB first)";
  }
  rocm_build_type = getBuildType();
  rocm_lib_path = getMyLibPath();
  rocm_build_date = getFileCreationDate(rocm_lib_path);
  rocm_env_variables = RocmSMI::getInstance().getRSMIEnvVarInfo();
  std::queue<std::string> devGraphicsVersions = getAllDeviceGfxVers();
  if (devGraphicsVersions.empty() == false) {
    dev_gfx_versions = "";
    while (devGraphicsVersions.empty() == false) {
      dev_gfx_versions += "\n\t" + devGraphicsVersions.front();
      devGraphicsVersions.pop();
    }
  }
  return std::make_tuple(errorDetected, sysname, nodename, release,
                         version, machine, domainName, os_distribution,
                         endianness, rocm_build_type, rocm_lib_path,
                         rocm_build_date, rocm_env_variables, dev_gfx_versions);
}

// If logging is enabled through RSMI_LOGGING environment variable.
// We display helpful system metrics for debug purposes.
void logSystemDetails(void) {
  std::ostringstream ss;
  bool errorDetected;
  std::string sysname, node, release, version, machine, domain, distName,
              endianness, rocm_build_type, lib_path, build_date, rocm_env_vars,
              dev_gfx_versions;
  std::tie(errorDetected, sysname, node, release, version, machine, domain,
           distName, endianness, rocm_build_type, lib_path, build_date,
           rocm_env_vars, dev_gfx_versions) = getSystemDetails();
  if (errorDetected == false) {
    ss << "====== Gathered system details ============\n"
       << "SYSTEM NAME: " << sysname << "\n"
       << "OS DISTRIBUTION: " << distName << "\n"
       << "NODE NAME: " << node << "\n"
       << "RELEASE: " << release << "\n"
       << "VERSION: " << version << "\n"
       << "MACHINE TYPE: " << machine << "\n"
       << "DOMAIN: " << domain << "\n"
       << "ENDIANNESS: " << endianness << "\n"
       << "ROCM BUILD TYPE: " << rocm_build_type << "\n"
       << "ROCM-SMI-LIB PATH: " << lib_path << "\n"
       << "ROCM-SMI-LIB BUILD DATE: " << build_date << "\n"
       << "ROCM ENV VARIABLES: " << rocm_env_vars
       << "AMD GFX VERSIONS: " << dev_gfx_versions << "\n";
    LOG_INFO(ss);
  } else {
    ss << "====== Gathered system details ============\n"
       << "Could not retrieve system details";
    LOG_ERROR(ss);
  }
}

// Usage:
//     logHexDump(desc, addr, len, bytesPerLine);
//         desc:    if non-NULL, printed as a description before hex dump.
//         addr:    the address to start dumping from.
//         len:     the number of bytes to dump.
//         bytesPerLine: number of bytes on each output line.
void logHexDump(
  const char *desc, const void *addr, const size_t len, size_t bytesPerLine) {
  // UNCOMMENT: printf lines if you want to see directly to stdout
  std::ostringstream ss;
  // Silently ignore per-line values.
  if (bytesPerLine < 4 || bytesPerLine > 64) bytesPerLine = 16;

  size_t i;
  unsigned char buff[bytesPerLine + 1];
  const unsigned char *pc           // ptr to data (char, 1 byte sized data)
                          = (const unsigned char *) addr;

  // Output description if given.
  // if (desc != NULL) printf("%s:\n", desc);
  if (desc != nullptr) ss << "\n" << desc << "\n";

  // Length checks.
  if (len == 0) {
    // printf("  ZERO LENGTH\n");
    ss << "  ZERO LENGTH\n";
    LOG_ERROR(ss);
    return;
  }
  std::string endianness = "<undefined>";
  if (isSystemBigEndian()) {
    endianness = "** System is Big Endian, multi-bit symbols encoded as"
                 " big endian (MSB first) **";
  } else {
    endianness = "** System is Little Endian, multi-bit symbols encoded as"
                 " little endian (LSB first) **";
  }
  ss << "\t" << endianness << "\n";

  // Process every byte in the data.
  for (i = 0; i < len; i++) {
    // Multiple of bytesPerLine means new or first line (with line offset).
    if ((i % bytesPerLine) == 0) {
      // Only print previous-line ASCII buffer for lines beyond first.
      // if (i != 0) printf("  %s\n", buff);
      if (i != 0) ss << "  " << buff << "\n";
      // Output the offset of current line.
      // printf("  %08lx ", i);
      ss << "  " << std::setw(8) << std::setfill('0') << std::hex << i << " ";
    }

    // Now the hex code for the specific character.
    // printf(" %02x", pc[i]);

    ss << " " << std::setw(2) << std::setfill('0') << std::hex
       << static_cast<unsigned>(pc[i]);

    // And buffer a printable ASCII character for later.
    // x20 = 32 || x7e = 126 (ascii table range)
    if ((pc[i] < 0x20) || (pc[i] > 0x7e)) { // isprint() may be better.
      buff[i % bytesPerLine] = '.';
    } else {
      buff[i % bytesPerLine] = pc[i];
    }
    buff[(i % bytesPerLine) + 1] = '\0';
  }

  // Pad out last line if not exactly bytesPerLine characters.
  while ((i % bytesPerLine) != 0) {
    // printf("   ");
    ss << "   ";
    i++;
  }

  // And print the final ASCII buffer.
  // printf("  %s\n", buff);
  ss << "  " << buff << "\n";
  LOG_DEBUG(ss);
}

bool isSystemBigEndian() {
  int n = 1;
  bool isBigEndian = true;
  if (*(char *)&n == 1) {
    isBigEndian = false;
  }
  return isBigEndian;
}

std::string getBuildType() {
  std::string build = "<unknown>";
#ifndef DEBUG
  build = "release";
#else
  build = "debug";
#endif
  return build;
}

const char *my_fname(void) {
#ifdef _GNU_SOURCE
  Dl_info dl_info;
  dladdr(reinterpret_cast<void *>(my_fname), &dl_info);
  return (dl_info.dli_fname);
#else
  std::string emptyRet = "";
  return emptyRet.c_str();
#endif
}

std::string getMyLibPath(void) {
  std::string libName = "rocm-smi-lib";
  std::string path = std::string(my_fname());
  if (path.empty()) {
    path = "Could not find library path for " + libName;
  }
  return path;
}

std::string getFileCreationDate(std::string path) {
  struct stat t_stat;
  stat(path.c_str(), &t_stat);
  struct tm *timeinfo = localtime(&t_stat.st_ctime); // NOLINT
  return removeNewLines(std::string(asctime(timeinfo))); // NOLINT
}

rsmi_status_t getBDFString(uint64_t bdf_id, std::string& bfd_str)
{
  auto result = rsmi_status_t::RSMI_STATUS_SUCCESS;
  auto bus_id = static_cast<uint8_t>((bdf_id & 0x0000FF00) >> 8);
  auto dev_id = static_cast<uint8_t>((bdf_id & 0x000000F8) >> 3);
  auto func_id = static_cast<uint8_t>(bdf_id & 0x00000003);

  bfd_str = std::string();
  if (!(bus_id > 0)) {
    result = rsmi_status_t::RSMI_STATUS_NO_DATA;
    return result;
  }

  std::stringstream bdf_sstream;
  bdf_sstream << std::hex << std::setfill('0') << std::setw(sizeof(uint8_t) * 2) << +bus_id << ":";
  bdf_sstream << std::hex << std::setfill('0') << std::setw(sizeof(uint8_t) * 2) << +dev_id << ".";
  bdf_sstream << std::hex << std::setfill('0') << +func_id;
  bfd_str = bdf_sstream.str();
  return result;
}

int subDirectoryCountInPath(const std::string path) {
  int dir_count = 0;
  struct dirent *dent;
  DIR *srcdir = opendir(path.c_str());

  if (srcdir == NULL) {
    perror("opendir");
    return -1;
  }

  while ((dent = readdir(srcdir)) != NULL) {
    struct stat st;

    if (strcmp(dent->d_name, ".") == 0 || strcmp(dent->d_name, "..") == 0) {
      continue;
    }

    if (fstatat(dirfd(srcdir), dent->d_name, &st, 0) < 0) {
      perror(dent->d_name);
      continue;
    }

    if (S_ISDIR(st.st_mode)) {
      dir_count++;
    }
  }
  closedir(srcdir);
  return dir_count;
}

std::string monitor_type_string(MonitorTypes type) {
  const std::map<MonitorTypes, std::string> monitorTypesToString{
      {kMonName,
       "MonitorTypes::kMonName"},
      {kMonTemp,
       "MonitorTypes::kMonTemp"},
      {kMonFanSpeed,
       "MonitorTypes::kMonFanSpeed"},
      {kMonMaxFanSpeed,
      "MonitorTypes::kMonMaxFanSpeed"},
      {kMonFanRPMs,
      "MonitorTypes::kMonFanRPMs"},
      {kMonFanCntrlEnable,
      "MonitorTypes::kMonFanCntrlEnable"},
      {kMonPowerCap,
      "MonitorTypes::kMonPowerCap"},
      {kMonPowerCapDefault,
      "MonitorTypes::kMonPowerCapDefault"},
      {kMonPowerCapMax,
      "MonitorTypes::kMonPowerCapMax"},
      {kMonPowerCapMin,
      "MonitorTypes::kMonPowerCapMin"},
      {kMonPowerAve,
      "MonitorTypes::kMonPowerAve"},
      {kMonPowerInput,
      "MonitorTypes::kMonPowerInput"},
      {kMonPowerLabel,
      "MonitorTypes::kMonPowerLabel"},
      {kMonTempMax,
      "MonitorTypes::kMonTempMax"},
      {kMonTempMin,
      "MonitorTypes::kMonTempMin"},
      {kMonTempMaxHyst,
      "MonitorTypes::kMonTempMaxHyst"},
      {kMonTempMinHyst,
      "MonitorTypes::kMonTempMinHyst"},
      {kMonTempCritical,
      "MonitorTypes::kMonTempCritical"},
      {kMonTempCriticalHyst,
      "MonitorTypes::kMonTempCriticalHyst"},
      {kMonTempEmergency,
      "MonitorTypes::kMonTempEmergency"},
      {kMonTempEmergencyHyst,
       "MonitorTypes::kMonTempEmergencyHyst"},
      {kMonTempCritMin,
      "MonitorTypes::kMonTempCritMin"},
      {kMonTempCritMinHyst,
      "MonitorTypes::kMonTempCritMinHyst"},
      {kMonTempOffset,
      "MonitorTypes::kMonTempOffset"},
      {kMonTempLowest,
      "MonitorTypes::kMonTempLowest"},
      {kMonTempHighest,
      "MonitorTypes::kMonTempHighest"},
      {kMonTempLabel,
      "MonitorTypes::kMonTempLabel"},
      {kMonVolt,
      "MonitorTypes::kMonVolt"},
      {kMonVoltMax,
      "MonitorTypes::kMonVoltMax"},
      {kMonVoltMinCrit,
      "MonitorTypes::kMonVoltMinCrit"},
      {kMonVoltMin,
      "MonitorTypes::kMonVoltMin"},
      {kMonVoltMaxCrit,
      "MonitorTypes::kMonVoltMaxCrit"},
      {kMonVoltAverage,
      "MonitorTypes::kMonVoltAverage"},
      {kMonVoltLowest,
      "MonitorTypes::kMonVoltLowest"},
      {kMonVoltHighest,
      "MonitorTypes::kMonVoltHighest"},
      {kMonVoltLabel,
      "MonitorTypes::kMonVoltLabel"},
      {kMonInvalid,
      "MonitorTypes::kMonInvalid"},
  };
  return monitorTypesToString.at(type);
}

std::string power_type_string(RSMI_POWER_TYPE type) {
  const std::map<RSMI_POWER_TYPE, std::string> powerTypesToString{
      {RSMI_AVERAGE_POWER,
      "RSMI_POWER_TYPE::RSMI_AVERAGE_POWER"},
      {RSMI_CURRENT_POWER,
      "RSMI_POWER_TYPE::RSMI_CURRENT_POWER"},
      {RSMI_INVALID_POWER,
      "RSMI_POWER_TYPE::RSMI_INVALID_POWER"},
  };
  return powerTypesToString.at(type);
}

std::string splitString(std::string str, char delim) {
  std::vector<std::string> tokens;
  std::stringstream ss(str);
  std::string token;

  if (str.empty()) {
    return "";
  }

  while (std::getline(ss, token, delim)) {
    tokens.push_back(token);
    return token;  // return 1st match
  }
    return "";
}

static std::string pt_rng_Mhz(std::string title, rsmi_range *r) {
  std::ostringstream ss;
  if (r == nullptr) {
    ss << "pt_rng_Mhz | rsmi_range r = nullptr\n";
    return ss.str();
  }

  ss << title;
  ss << r->lower_bound/1000000 << " to "
     << r->upper_bound/1000000 << " MHz" << "\n";
  return ss.str();
}

static std::string pt_rng_mV(std::string title, rsmi_range *r) {
  std::ostringstream ss;
  if (r == nullptr) {
    ss << "pt_rng_mV | rsmi_range r = nullptr\n";
    return ss.str();
  }

  ss << title;
  ss << r->lower_bound << " to " << r->upper_bound
     << " mV" << "\n";
  return ss.str();
}

std::string print_rsmi_od_volt_freq_data_t(rsmi_od_volt_freq_data_t *odv) {
  std::ostringstream ss;
  if (odv == nullptr) {
    ss << "rsmi_od_volt_freq_data_t odv = nullptr\n";
    return ss.str();
  }

  ss << pt_rng_Mhz("\t**Current SCLK frequency range: ", &odv->curr_sclk_range);
  ss << pt_rng_Mhz("\t**Current MCLK frequency range: ", &odv->curr_mclk_range);
  ss << pt_rng_Mhz("\t**Min/Max Possible SCLK frequency range: ",
                   &odv->sclk_freq_limits);
  ss << pt_rng_Mhz("\t**Min/Max Possible MCLK frequency range: ",
                   &odv->mclk_freq_limits);

  ss << "\t**Current Freq/Volt. curve: " << "\n";
  ss << "\t\t N/A" << "\n";

  ss << "\t**Number of Freq./Volt. regions: " << odv->num_regions << "\n\n";
  return ss.str();
}

std::string print_odv_region(rsmi_freq_volt_region_t *region) {
  std::ostringstream ss;
  ss << pt_rng_Mhz("\t\tFrequency range: ", &region->freq_range);
  ss << pt_rng_mV("\t\tVoltage range: ", &region->volt_range);
  return ss.str();
}

std::string print_rsmi_od_volt_freq_regions(uint32_t num_regions,
                                            rsmi_freq_volt_region_t *regions) {
  std::ostringstream ss;
  if (regions == nullptr) {
    ss << "rsmi_freq_volt_region_t regions = nullptr\n";
    return ss.str();
  }
  for (uint32_t i = 0; i < num_regions; ++i) {
    ss << "\tRegion " << i << ": " << "\n";
    ss << print_odv_region(&regions[i]);
  }
  return ss.str();
}

bool is_sudo_user() {
  std::ostringstream ss;
  bool isRunningWithSudo = false;
  auto myUID = getuid();
  auto myPrivledges = geteuid();
  if ((myUID == myPrivledges) && (myPrivledges == 0)) {
    isRunningWithSudo = true;
  }
  ss << __PRETTY_FUNCTION__ << (isRunningWithSudo ? " | running as sudoer" :
     " | NOT running as sudoer");
  LOG_DEBUG(ss);
  return isRunningWithSudo;
}

// string output of gfx_<version>
rsmi_status_t rsmi_get_gfx_target_version(uint32_t dv_ind, std::string *gfx_version) {
  std::ostringstream ss;
  uint64_t kfd_gfx_version = 0;
  GET_DEV_AND_KFDNODE_FROM_INDX

  int ret = kfd_node->get_gfx_target_version(&kfd_gfx_version);
  uint64_t orig_target_version = 0;
  uint64_t major = 0;
  uint64_t minor = 0;
  uint64_t rev = 0;
  if (ret == 0) {
    orig_target_version = std::stoull(std::to_string(kfd_gfx_version));
    // separate out parts -> put back into normal graphics version format
    major = static_cast<uint64_t>((orig_target_version / 10000) * 100);
    minor = static_cast<uint64_t>((orig_target_version % 10000 / 100) * 10);
    rev = static_cast<uint64_t>(orig_target_version % 100);

    ss << std::hex << rev;
    std::string revision = ss.str();
    *gfx_version = "gfx" + std::to_string((major + minor)/10) + revision;

    ss.str("");
    ss << __PRETTY_FUNCTION__
    << " | " << std::dec << "kfd_target_version = " << orig_target_version
    << "; major = " << major << "; minor = " << minor << "; rev = "
    << rev << "\nReporting rsmi_get_gfx_target_version = " << *gfx_version
    << "\n";
    LOG_INFO(ss);
    return RSMI_STATUS_SUCCESS;
  } else {
    *gfx_version = "Unknown";
    return RSMI_STATUS_NOT_SUPPORTED;
  }
}

std::queue<std::string> getAllDeviceGfxVers() {
  uint32_t num_monitor_devs = 0;
  rsmi_status_t ret;
  std::queue<std::string> deviceGfxVersions;
  std::string response = "";
  std::string dev_gfx_ver = "";

  ret = rsmi_num_monitor_devices(&num_monitor_devs);
  if (ret != RSMI_STATUS_SUCCESS || num_monitor_devs == 0) {
    response = "N/A - No AMD devices detected";
    deviceGfxVersions.push(response);
    return deviceGfxVersions;
  }

  for (uint32_t i = 0; i < num_monitor_devs; ++i) {
    ret = amd::smi::rsmi_get_gfx_target_version(i , &dev_gfx_ver);
    response = "Device[" + std::to_string(i) + "]: ";
    if (ret != RSMI_STATUS_SUCCESS) {
      deviceGfxVersions.push(response + getRSMIStatusString(ret, false));
    } else {
      deviceGfxVersions.push(response + std::string(dev_gfx_ver));
    }
  }
  return deviceGfxVersions;
}

// milli_seconds: time to wait, in milliseconds
// 1 sec = 1000ms
// .5 sec = 500ms
void system_wait(int milli_seconds) {
  std::ostringstream ss;
  auto start = std::chrono::high_resolution_clock::now();
  // 1 ms = 1000 us
  int waitTime = milli_seconds * 1000;
  ss << __PRETTY_FUNCTION__ << " | "
     << "** Waiting for " << std::dec << waitTime
     << " us (" << waitTime/1000 << " milli-seconds) **";
  LOG_DEBUG(ss);
  usleep(waitTime);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  ss << __PRETTY_FUNCTION__ << " | "
     << "** Waiting took " << duration.count() / 1000
     << " milli-seconds **";
  LOG_DEBUG(ss);
}

int countDigit(uint64_t n) {
  return static_cast<int>(std::floor(log10(static_cast<double>(n)) + 1));
}

std::string find_file_in_folder(const std::string& folder,
               const std::string& regex) {
  std::string file_name;
  // TODO(amdsmi_dev): The closedir function has some non-standard attributes
  // that are being ignored here which is causing a warning to be thrown
  using dir_ptr = std::unique_ptr<DIR, decltype(&closedir)>;

  struct dirent *dir = nullptr;
  std::regex file_regex(regex);
  auto drm_dir = dir_ptr(opendir(folder.c_str()), &closedir);
  if (drm_dir == nullptr) return file_name;
  std::cmatch m;
  while ((dir = readdir(drm_dir.get())) != NULL) {
    if (std::regex_search(dir->d_name, m, file_regex)) {
        file_name = dir->d_name;
        break;
    }
  }
  return file_name;
}

uint64_t get_multiplier_from_char(char units_char) {
  uint32_t multiplier = 0;

  switch (units_char) {
    case 'G':   // GT or GHz
      multiplier = 1000000000;
      break;

    case 'M':   // MT or MHz
      multiplier = 1000000;
      break;

    case 'K':   // KT or KHz
    case 'V':   // default unit for voltage is mV
      multiplier = 1000;
      break;

    case 'T':   // Transactions
    case 'H':   // Hertz
    case 'm':   // mV (we will make mV the default unit for voltage)
      multiplier = 1;
      break;

    default:
      assert(false);  // Unexpected units for frequency
      throw amd::smi::rsmi_exception(RSMI_STATUS_UNEXPECTED_DATA, __FUNCTION__);
  }
  return multiplier;
}

}  // namespace smi
}  // namespace amd
