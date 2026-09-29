/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_level.hh>

#include <regex>
#include <sstream>

namespace HipTestLevel {
namespace {

/**
 * @brief Split a Catch2 test spec into its OR'd filters.
 *
 * Only unescaped commas separate filters. CatchAddTests.cmake escapes '\', ',',
 * '[', ']' and ';' in generated test names, so a name may legitimately contain
 * "\," - splitting on it would corrupt the filter.
 */
std::vector<std::string> splitFilters(const std::string& spec) {
  std::vector<std::string> filters;
  std::string filter;
  for (std::size_t i = 0; i < spec.size(); ++i) {
    if (spec[i] == '\\' && i + 1 < spec.size()) {
      // Escaped character: copy the pair verbatim, never split on it.
      filter += spec[i];
      filter += spec[i + 1];
      ++i;
    } else if (spec[i] == ',') {
      filters.push_back(filter);
      filter.clear();
    } else {
      filter += spec[i];
    }
  }
  filters.push_back(filter);
  return filters;
}

/// @brief Join @p parts into one Catch2 spec string, i.e. OR them together.
std::string joinWithCommas(const std::vector<std::string>& parts) {
  std::string joined;
  for (const auto& part : parts) {
    if (!joined.empty()) {
      joined += ',';
    }
    joined += part;
  }
  return joined;
}

}  // namespace

bool isSupportedLevel(const std::string& level) {
  for (const char* supported : kSupportedLevels) {
    if (level == supported) {
      return true;
    }
  }
  return false;
}

std::set<int> collectLevels(const std::string& text, bool allowBareTags) {
  static const std::regex bracketedPattern("(~?)\\[([^\\[\\]]*)\\]");
  static const std::regex barePattern("\\s*(~?)\\s*([^\\s]+)\\s*");

  // Level named by a pattern, or -1 when it does not name one. Anything but
  // an exact "level_N" is rejected; the digit count is bounded so that the
  // conversion cannot overflow.
  const auto parseLevel = [](const std::string& tag) {
    static const std::regex levelRegex("level_([0-9]{1,9})");
    std::smatch match;
    return std::regex_match(tag, match, levelRegex) ? std::stoi(match[1].str()) : -1;
  };

  std::set<int> levels;
  std::stringstream stream(text);
  std::string filter;
  while (std::getline(stream, filter, ',')) {
    std::set<int> included, excluded;

    // Sort this filter's level patterns into included and excluded.
    const auto sortPattern = [&](const std::string& negation, const std::string& tag) {
      const int level = parseLevel(tag);
      if (level >= 0) {
        (negation.empty() ? included : excluded).insert(level);
      }
    };

    bool bracketed = false;
    for (auto it = std::sregex_iterator(filter.begin(), filter.end(), bracketedPattern);
         it != std::sregex_iterator(); ++it) {
      bracketed = true;
      sortPattern((*it)[1].str(), (*it)[2].str());
    }
    if (!bracketed && allowBareTags) {
      std::smatch match;
      if (std::regex_match(filter, match, barePattern)) {
        sortPattern(match[1].str(), match[2].str());
      }
    }

    if (included.empty()) {
      if (excluded.empty()) {
        continue;  // filter says nothing about the level
      }
      // Exclusion only: everything the suite supports is still in play.
      for (const char* supported : kSupportedLevels) {
        const int level = parseLevel(supported);
        if (level >= 0) {
          included.insert(level);
        }
      }
    }
    for (const int level : included) {
      if (excluded.count(level) == 0) {
        levels.insert(level);
      }
    }
  }
  return levels;
}

std::string highestLevel(const std::set<int>& levels) {
  if (levels.empty()) {
    return "";
  }
  return "level_" + std::to_string(*levels.rbegin());
}

LevelResolution resolveLevel(const std::vector<std::string>& testsOrTags, const char* envLevel) {
  LevelResolution resolution;

  // Priority 1: Catch2 command-line tag filter (e.g. ./test "[level_2]").
  std::set<int> cliLevels;
  for (const auto& arg : testsOrTags) {
    const auto found = collectLevels(arg, false);
    cliLevels.insert(found.begin(), found.end());
  }
  if (!cliLevels.empty()) {
    resolution.levels = cliLevels;
    resolution.level = highestLevel(cliLevels);
    resolution.source = LevelSource::kCommandLine;
    return resolution;
  }

  // Priority 2: HIP_TEST_LEVEL environment variable.
  if (envLevel != nullptr) {
    resolution.envValue = envLevel;
    const auto envLevels = collectLevels(envLevel, true);
    if (!envLevels.empty()) {
      resolution.levels = envLevels;
      resolution.level = highestLevel(envLevels);
      resolution.source = LevelSource::kEnvironment;
      return resolution;
    }
    resolution.envRejected = true;
  }

  // Priority 3: Hardcoded default. Parsed rather than hardcoded as a number so
  // kDefaultLevel stays the single source of truth for the default.
  resolution.levels = collectLevels(kDefaultLevel, true);
  resolution.level = kDefaultLevel;
  resolution.source = LevelSource::kDefault;
  return resolution;
}

bool LevelResolution::multipleSelected() const { return levels.size() > 1; }

std::string LevelResolution::firstUnsupportedLevel() const {
  for (const int selected : levels) {
    const std::string name = "level_" + std::to_string(selected);
    if (!isSupportedLevel(name)) {
      return name;
    }
  }
  return "";
}

std::vector<std::string> LevelResolution::describe() const {
  const char* sourceName = "default";
  switch (source) {
    case LevelSource::kCommandLine:
      sourceName = "command line";
      break;
    case LevelSource::kEnvironment:
      sourceName = "HIP_TEST_LEVEL";
      break;
    case LevelSource::kDefault:
      break;
  }

  std::vector<std::string> lines;
  if (envRejected) {
    lines.push_back("[Level Filter] HIP_TEST_LEVEL='" + envValue +
                    "' has no valid level, ignoring");
  }
  if (multipleSelected()) {
    std::string selected;
    for (const int level : levels) {
      if (!selected.empty()) {
        selected += ", ";
      }
      selected += "level_" + std::to_string(level);
    }
    lines.push_back("[Level Filter] Multiple levels selected via " + std::string(sourceName) +
                    " (" + selected + "); all of them run, parameters use the highest (" + level +
                    ")");
  }
  lines.push_back("[Level Filter] Detected from " + std::string(sourceName) + ": " + level);
  return lines;
}

bool applyLevelFilter(std::vector<std::string>& testsOrTags, const LevelResolution& resolution) {
  // The spec already names the level; re-applying would AND in a second one.
  if (resolution.source == LevelSource::kCommandLine) {
    return false;
  }

  // The default level never narrows the run - it only supplies parameters. A
  // bare "./AtomicsTest" therefore runs every test in the binary, as it always
  // has, and so does plain ctest. Only an explicit request selects.
  if (resolution.source != LevelSource::kEnvironment) {
    return false;
  }

  // One tag per selected level. Catch2 has no OR inside a filter, so these are
  // combined by repeating filters, not by concatenating tags.
  std::vector<std::string> tags;
  tags.reserve(resolution.levels.size());
  for (const int level : resolution.levels) {
    tags.push_back("[level_" + std::to_string(level) + "]");
  }
  if (tags.empty()) {
    return false;
  }

  // Nothing else was asked for, so the levels are the only selector available.
  if (testsOrTags.empty()) {
    testsOrTags.push_back(joinWithCommas(tags));
    return true;
  }

  // Cross product: every filter of the spec, once per selected level.
  for (auto& spec : testsOrTags) {
    std::vector<std::string> rewritten;
    for (const auto& filter : splitFilters(spec)) {
      // An empty filter constrains nothing; Catch2 discards it too.
      if (filter.empty()) {
        continue;
      }
      for (const auto& tag : tags) {
        rewritten.push_back(filter + tag);
      }
    }
    spec = joinWithCommas(rewritten);
  }
  return true;
}

int mapLevelFilterExitCode(int exitCode, bool levelFilterApplied) {
  // Catch2 exit codes are file-local constants in catch_session.cpp:37-42 and
  // are not exported, so they are restated here.
  constexpr int kCatchNoTestsRunExitCode = 2;
  constexpr int kCatchAllTestsSkippedExitCode = 4;

  if (levelFilterApplied && exitCode == kCatchNoTestsRunExitCode) {
    return kCatchAllTestsSkippedExitCode;
  }
  return exitCode;
}

}  // namespace HipTestLevel
