/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ISA_LOADER_HPP_
#define ISA_LOADER_HPP_

#include <string>
#include <vector>
#include <cstdint>

namespace amd {

// Forward declaration
class Isa;

// Intermediate representation for ISA entries from JSON
struct IsaConfigEntry {
  std::string targetId;
  bool runtimeRocSupported;
  bool runtimePalSupported;
  uint32_t versionMajor;
  uint32_t versionMinor;
  uint32_t versionStepping;
  std::string sramecc;  // "Unsupported", "Any", "Disabled", "Enabled"
  std::string xnack;    // "Unsupported", "Any", "Disabled", "Enabled"
  uint32_t simdPerCU;
  uint32_t simdWidth;
  uint32_t simdInstructionWidth;
  uint32_t memChannelBankWidth;
  uint32_t localMemSizePerCU;
  uint32_t localMemBanks;
  uint32_t ldsAlignment;
};

/**
 * @brief ISA configuration loader
 *
 * Responsible for loading, parsing, and validating ISA configurations
 * from internal JSON files. These ISAs are appended to the hardcoded
 * ISA table at runtime.
 */
class IsaLoader {
 public:
  /**
   * @brief Load ISAs from JSON file
   *
   * @param filePath Path to JSON configuration file
   * @param isas Output vector to append loaded ISA objects
   * @param stringStorage String storage for targetId stability
   * @param errorMsg Output error message if loading fails
   * @return true if loading succeeded, false otherwise
   */
  static bool loadFromFile(const std::string& filePath,
                          std::vector<Isa>& isas,
                          std::vector<std::string>& stringStorage,
                          std::string& errorMsg);

  /**
   * @brief Find ISA configuration file location
   *
   * Searches in order:
   * 1. ROCCLR_ISA_CONFIG_FILE environment variable
   * 2. $ROCM_PATH/share/rocclr/isa_table_internal.json
   * 3. Relative to library location (for development builds)
   *
   * @return Path to config file, or empty string if not found
   */
  static std::string findConfigFile();

  /**
   * @brief Validate ISA entry for consistency
   *
   * Checks:
   * - Feature/targetId consistency (e.g., sramecc+ must have Enabled)
   * - Version ranges
   * - Hardware property ranges
   *
   * @param entry ISA configuration entry to validate
   * @param existingIsas Existing ISAs to check for duplicates
   * @param error Output error message if validation fails
   * @return true if valid, false otherwise
   */
  static bool validateIsaEntry(const IsaConfigEntry& entry,
                               const std::vector<IsaConfigEntry>& existingIsas,
                               std::string& error);

 private:
  /**
   * @brief Parse JSON content to intermediate representation
   *
   * @param jsonContent JSON file content as string
   * @param entries Output vector of parsed ISA entries
   * @param errorMsg Output error message if parsing fails
   * @return true if parsing succeeded, false otherwise
   */
  static bool parseJson(const std::string& jsonContent,
                       std::vector<IsaConfigEntry>& entries,
                       std::string& errorMsg);

  /**
   * @brief Convert feature string to Feature enum
   *
   * @param str Feature string ("Unsupported", "Any", "Disabled", "Enabled")
   * @return Corresponding Isa::Feature enum value
   */
  static int parseFeature(const std::string& str);

  /**
   * @brief Check if file exists
   *
   * @param path File path to check
   * @return true if file exists and is readable
   */
  static bool fileExists(const std::string& path);

  /**
   * @brief Read entire file into string
   *
   * @param path File path to read
   * @param content Output file content
   * @return true if read succeeded, false otherwise
   */
  static bool readFile(const std::string& path, std::string& content);

  /**
   * @brief Get parent directory of a path
   *
   * @param path File or directory path
   * @return Parent directory path
   */
  static std::string parentDir(const std::string& path);
};

}  // namespace amd

#endif  // ISA_LOADER_HPP_
