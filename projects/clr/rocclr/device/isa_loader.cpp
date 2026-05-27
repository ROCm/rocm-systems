/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "isa_loader.hpp"
#include "device.hpp"
#include "utils/debug.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <sys/stat.h>

#ifdef __linux__
#include <dlfcn.h>
#include <libgen.h>
#endif

namespace amd {

// Simple JSON parser for ISA table
// This is a minimal implementation specific to our schema
namespace {

// Trim whitespace from string
std::string trim(const std::string& str) {
  size_t start = str.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return "";
  size_t end = str.find_last_not_of(" \t\n\r");
  return str.substr(start, end - start + 1);
}

// Extract string value from JSON (handles "value" format)
bool extractString(const std::string& json, size_t& pos, std::string& value) {
  pos = json.find('"', pos);
  if (pos == std::string::npos) return false;
  pos++;  // Skip opening quote

  size_t end = pos;
  while (end < json.length()) {
    if (json[end] == '"' && (end == 0 || json[end-1] != '\\')) {
      value = json.substr(pos, end - pos);
      pos = end + 1;
      return true;
    }
    end++;
  }
  return false;
}

// Extract integer value from JSON
bool extractInt(const std::string& json, size_t& pos, uint32_t& value) {
  size_t start = json.find_first_of("0123456789", pos);
  if (start == std::string::npos) return false;

  size_t end = start;
  while (end < json.length() && isdigit(json[end])) {
    end++;
  }

  try {
    value = std::stoul(json.substr(start, end - start));
    pos = end;
    return true;
  } catch (...) {
    return false;
  }
}

// Extract boolean value from JSON
bool extractBool(const std::string& json, size_t& pos, bool& value) {
  size_t truePos = json.find("true", pos);
  size_t falsePos = json.find("false", pos);

  if (truePos != std::string::npos && (falsePos == std::string::npos || truePos < falsePos)) {
    if (truePos - pos < 20) {  // Reasonable proximity
      value = true;
      pos = truePos + 4;
      return true;
    }
  }
  if (falsePos != std::string::npos && (truePos == std::string::npos || falsePos < truePos)) {
    if (falsePos - pos < 20) {
      value = false;
      pos = falsePos + 5;
      return true;
    }
  }
  return false;
}

// Find next occurrence of a key in JSON
size_t findKey(const std::string& json, const std::string& key, size_t start = 0) {
  std::string searchStr = "\"" + key + "\"";
  return json.find(searchStr, start);
}

}  // anonymous namespace

bool IsaLoader::parseJson(const std::string& jsonContent,
                         std::vector<IsaConfigEntry>& entries,
                         std::string& errorMsg) {
  // Find the "isas" array
  size_t isasPos = findKey(jsonContent, "isas");
  if (isasPos == std::string::npos) {
    errorMsg = "Missing required 'isas' array in JSON";
    return false;
  }

  // Find the array start
  size_t arrayStart = jsonContent.find('[', isasPos);
  if (arrayStart == std::string::npos) {
    errorMsg = "Invalid 'isas' array format";
    return false;
  }

  size_t arrayEnd = jsonContent.find(']', arrayStart);
  if (arrayEnd == std::string::npos) {
    errorMsg = "Unterminated 'isas' array";
    return false;
  }

  // Parse each ISA entry (objects within the array)
  size_t pos = arrayStart + 1;
  int entryIndex = 0;

  while (pos < arrayEnd) {
    // Find next object start
    size_t objStart = jsonContent.find('{', pos);
    if (objStart == std::string::npos || objStart >= arrayEnd) {
      break;  // No more entries
    }

    // Find corresponding object end
    int braceCount = 1;
    size_t objEnd = objStart + 1;
    while (objEnd < arrayEnd && braceCount > 0) {
      if (jsonContent[objEnd] == '{') braceCount++;
      else if (jsonContent[objEnd] == '}') braceCount--;
      objEnd++;
    }

    if (braceCount != 0) {
      errorMsg = "Malformed JSON object at entry " + std::to_string(entryIndex);
      return false;
    }

    // Parse this entry
    IsaConfigEntry entry;
    size_t entryPos = objStart;

    // Parse each required field
    #define PARSE_STRING_FIELD(name) \
      { \
        size_t keyPos = findKey(jsonContent, #name, entryPos); \
        if (keyPos == std::string::npos || keyPos >= objEnd) { \
          errorMsg = "Missing required field '" #name "' in ISA entry " + std::to_string(entryIndex); \
          return false; \
        } \
        size_t valPos = keyPos; \
        if (!extractString(jsonContent, valPos, entry.name)) { \
          errorMsg = "Invalid value for '" #name "' in ISA entry " + std::to_string(entryIndex); \
          return false; \
        } \
      }

    #define PARSE_INT_FIELD(name) \
      { \
        size_t keyPos = findKey(jsonContent, #name, entryPos); \
        if (keyPos == std::string::npos || keyPos >= objEnd) { \
          errorMsg = "Missing required field '" #name "' in ISA entry " + std::to_string(entryIndex); \
          return false; \
        } \
        size_t valPos = keyPos; \
        if (!extractInt(jsonContent, valPos, entry.name)) { \
          errorMsg = "Invalid value for '" #name "' in ISA entry " + std::to_string(entryIndex); \
          return false; \
        } \
      }

    #define PARSE_BOOL_FIELD(name) \
      { \
        size_t keyPos = findKey(jsonContent, #name, entryPos); \
        if (keyPos == std::string::npos || keyPos >= objEnd) { \
          errorMsg = "Missing required field '" #name "' in ISA entry " + std::to_string(entryIndex); \
          return false; \
        } \
        size_t valPos = keyPos; \
        if (!extractBool(jsonContent, valPos, entry.name)) { \
          errorMsg = "Invalid value for '" #name "' in ISA entry " + std::to_string(entryIndex); \
          return false; \
        } \
      }

    PARSE_STRING_FIELD(targetId);
    PARSE_BOOL_FIELD(runtimeRocSupported);
    PARSE_BOOL_FIELD(runtimePalSupported);
    PARSE_INT_FIELD(versionMajor);
    PARSE_INT_FIELD(versionMinor);
    PARSE_INT_FIELD(versionStepping);
    PARSE_STRING_FIELD(sramecc);
    PARSE_STRING_FIELD(xnack);
    PARSE_INT_FIELD(simdPerCU);
    PARSE_INT_FIELD(simdWidth);
    PARSE_INT_FIELD(simdInstructionWidth);
    PARSE_INT_FIELD(memChannelBankWidth);
    PARSE_INT_FIELD(localMemSizePerCU);
    PARSE_INT_FIELD(localMemBanks);
    PARSE_INT_FIELD(ldsAlignment);

    #undef PARSE_STRING_FIELD
    #undef PARSE_INT_FIELD
    #undef PARSE_BOOL_FIELD

    entries.push_back(entry);
    entryIndex++;
    pos = objEnd;
  }

  return true;
}

int IsaLoader::parseFeature(const std::string& str) {
  if (str == "Unsupported") return static_cast<int>(Isa::Feature::Unsupported);
  if (str == "Any") return static_cast<int>(Isa::Feature::Any);
  if (str == "Disabled") return static_cast<int>(Isa::Feature::Disabled);
  if (str == "Enabled") return static_cast<int>(Isa::Feature::Enabled);
  return static_cast<int>(Isa::Feature::Unsupported);  // Default fallback
}

bool IsaLoader::validateIsaEntry(const IsaConfigEntry& entry,
                                const std::vector<IsaConfigEntry>& existingIsas,
                                std::string& error) {
  // Validate targetId format
  if (entry.targetId.empty() || entry.targetId.substr(0, 3) != "gfx") {
    error = "Invalid targetId '" + entry.targetId + "': must start with 'gfx'";
    return false;
  }

  // Validate version ranges
  if (entry.versionMajor > 255 || entry.versionMinor > 255 || entry.versionStepping > 255) {
    error = "Invalid version " + std::to_string(entry.versionMajor) + "." +
            std::to_string(entry.versionMinor) + "." +
            std::to_string(entry.versionStepping) + ": values must be 0-255";
    return false;
  }

  // Validate feature values
  if (entry.sramecc != "Unsupported" && entry.sramecc != "Any" &&
      entry.sramecc != "Disabled" && entry.sramecc != "Enabled") {
    error = "Invalid sramecc value '" + entry.sramecc + "': must be Unsupported/Any/Disabled/Enabled";
    return false;
  }

  if (entry.xnack != "Unsupported" && entry.xnack != "Any" &&
      entry.xnack != "Disabled" && entry.xnack != "Enabled") {
    error = "Invalid xnack value '" + entry.xnack + "': must be Unsupported/Any/Disabled/Enabled";
    return false;
  }

  // Validate feature/targetId consistency
  bool hasSrameccPlus = entry.targetId.find(":sramecc+") != std::string::npos;
  bool hasSrameccMinus = entry.targetId.find(":sramecc-") != std::string::npos;
  bool hasXnackPlus = entry.targetId.find(":xnack+") != std::string::npos;
  bool hasXnackMinus = entry.targetId.find(":xnack-") != std::string::npos;

  if (hasSrameccPlus && entry.sramecc != "Enabled") {
    error = "targetId '" + entry.targetId + "' contains 'sramecc+' but field is '" + entry.sramecc + "'";
    return false;
  }

  if (hasSrameccMinus && entry.sramecc != "Disabled") {
    error = "targetId '" + entry.targetId + "' contains 'sramecc-' but field is '" + entry.sramecc + "'";
    return false;
  }

  if (hasXnackPlus && entry.xnack != "Enabled") {
    error = "targetId '" + entry.targetId + "' contains 'xnack+' but field is '" + entry.xnack + "'";
    return false;
  }

  if (hasXnackMinus && entry.xnack != "Disabled") {
    error = "targetId '" + entry.targetId + "' contains 'xnack-' but field is '" + entry.xnack + "'";
    return false;
  }

  // Check for duplicates in loaded entries
  for (const auto& existing : existingIsas) {
    if (existing.targetId == entry.targetId) {
      error = "Duplicate targetId '" + entry.targetId + "' found in JSON";
      return false;
    }
  }

  // Validate hardware properties
  if (entry.simdPerCU != 2 && entry.simdPerCU != 4) {
    ClPrint(amd::LOG_WARNING, amd::LOG_INIT,
            "Unusual simdPerCU value %u for ISA '%s' (expected 2 or 4)",
            entry.simdPerCU, entry.targetId.c_str());
  }

  if (entry.simdWidth != 16 && entry.simdWidth != 32 && entry.simdWidth != 64) {
    ClPrint(amd::LOG_WARNING, amd::LOG_INIT,
            "Unusual simdWidth value %u for ISA '%s' (expected 16, 32, or 64)",
            entry.simdWidth, entry.targetId.c_str());
  }

  return true;
}

bool IsaLoader::fileExists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

bool IsaLoader::readFile(const std::string& path, std::string& content) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  content = buffer.str();
  return true;
}

std::string IsaLoader::parentDir(const std::string& path) {
#ifdef __linux__
  char* pathCopy = strdup(path.c_str());
  char* dir = dirname(pathCopy);
  std::string result(dir);
  free(pathCopy);
  return result;
#else
  size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) return ".";
  return path.substr(0, pos);
#endif
}

std::string IsaLoader::findConfigFile() {
  // 1. Check environment variable override
  const char* envPath = std::getenv("ROCCLR_ISA_CONFIG_FILE");
  if (envPath && fileExists(envPath)) {
    return std::string(envPath);
  }

  // 2. Check ROCm install location
  const char* rocmPath = std::getenv("ROCM_PATH");
  std::string installPath = rocmPath ? rocmPath : "/opt/rocm";
  std::string configPath = installPath + "/share/rocclr/isa_table_internal.json";
  if (fileExists(configPath)) {
    return configPath;
  }

  // 3. Check development build location (relative to library)
#ifdef __linux__
  Dl_info dlInfo;
  if (dladdr(reinterpret_cast<void*>(&IsaLoader::findConfigFile), &dlInfo)) {
    std::string libPath = dlInfo.dli_fname;
    std::string devPath = parentDir(libPath) + "/../share/rocclr/isa_table_internal.json";
    if (fileExists(devPath)) {
      return devPath;
    }
  }
#endif

  // Not found - this is OK, hardcoded table is sufficient
  return "";
}

bool IsaLoader::loadFromFile(const std::string& filePath,
                            std::vector<Isa>& isas,
                            std::vector<std::string>& stringStorage,
                            std::string& errorMsg) {
  // Read file content
  std::string jsonContent;
  if (!readFile(filePath, jsonContent)) {
    errorMsg = "Failed to read file: " + filePath;
    return false;
  }

  // Parse JSON
  std::vector<IsaConfigEntry> entries;
  if (!parseJson(jsonContent, entries, errorMsg)) {
    return false;
  }

  // Validate and convert each entry
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& entry = entries[i];

    // Validate against previously loaded entries
    std::vector<IsaConfigEntry> previousEntries(entries.begin(), entries.begin() + i);
    if (!validateIsaEntry(entry, previousEntries, errorMsg)) {
      return false;
    }

    // Store targetId string for pointer stability
    stringStorage.push_back(entry.targetId);
    const char* targetIdPtr = stringStorage.back().c_str();

    // Create Isa with stable pointer
    isas.emplace_back(
      targetIdPtr,
      entry.runtimeRocSupported,
      entry.runtimePalSupported,
      entry.versionMajor,
      entry.versionMinor,
      entry.versionStepping,
      static_cast<Isa::Feature>(parseFeature(entry.sramecc)),
      static_cast<Isa::Feature>(parseFeature(entry.xnack)),
      entry.simdPerCU,
      entry.simdWidth,
      entry.simdInstructionWidth,
      entry.memChannelBankWidth,
      entry.localMemSizePerCU,
      entry.localMemBanks,
      entry.ldsAlignment
    );
  }

  return true;
}

}  // namespace amd
