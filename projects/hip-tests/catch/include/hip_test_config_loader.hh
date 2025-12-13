/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

/**
 * @brief Configuration file loader for test parameters
 * 
 * Loads test parameters from simple text files with support for:
 * - Memory sizes with K/M/G suffixes
 * - Centralized level configuration (test_levels.txt)
 * - Key=Value parameter files
 */
class ConfigFileLoader {
public:
    static std::vector<size_t> loadMemorySizes(const std::string& filepath) {
        std::vector<size_t> sizes;
        std::ifstream file(filepath);
        if (!file.is_open()) return sizes;
        
        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            
            size_t size = parseSizeString(line);
            if (size > 0) sizes.push_back(size);
        }
        return sizes;
    }
    
    static std::vector<int> loadBlockSizes(const std::string& filepath) {
        std::vector<int> sizes;
        std::ifstream file(filepath);
        if (!file.is_open()) return sizes;
        
        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            
            try {
                int size = std::stoi(line);
                if (size > 0) sizes.push_back(size);
            } catch (...) {}
        }
        return sizes;
    }
    
    static std::map<std::string, std::string> loadKeyValueParams(const std::string& filepath) {
        std::map<std::string, std::string> params;
        std::ifstream file(filepath);
        if (!file.is_open()) return params;
        
        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = trim(line.substr(0, pos));
                std::string value = trim(line.substr(pos + 1));
                if (!key.empty()) params[key] = value;
            }
        }
        return params;
    }
    
    static std::string findConfigFile(const std::string& filename, 
                                       const std::string& executablePath = "") {
        std::vector<std::string> searchPaths;
        
        if (filename[0] == '/' || filename.find(":\\") != std::string::npos) {
            if (fileExists(filename)) return filename;
        }
        
        searchPaths.push_back(filename);
        searchPaths.push_back("./config/" + filename);
        searchPaths.push_back("../config/" + filename);
        searchPaths.push_back("../../config/" + filename);
        
        if (!executablePath.empty()) {
            size_t pos = executablePath.find_last_of("/\\");
            if (pos != std::string::npos) {
                std::string execDir = executablePath.substr(0, pos);
                searchPaths.push_back(execDir + "/config/" + filename);
                searchPaths.push_back(execDir + "/../config/" + filename);
            }
        }
        
        for (const auto& path : searchPaths) {
            if (fileExists(path)) return path;
        }
        return "";
    }

    static bool loadCentralizedLevelConfig(
        const std::string& filepath,
        std::map<std::string, std::vector<size_t>>& outMemorySizes,
        std::map<std::string, std::vector<int>>& outBlockSizes,
        std::map<std::string, int>& outIterations,
        std::map<std::string, int>& outWarmups) {
        
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        
        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            
            size_t dotPos = line.find('.');
            size_t equalsPos = line.find('=');
            if (dotPos == std::string::npos || equalsPos == std::string::npos) continue;
            
            std::string levelName = trim(line.substr(0, dotPos));
            std::string paramType = trim(line.substr(dotPos + 1, equalsPos - dotPos - 1));
            std::string values = trim(line.substr(equalsPos + 1));
            
            std::vector<std::string> valueList = splitByComma(values);
            
            if (paramType == "memory_sizes") {
                std::vector<size_t> sizes;
                for (const auto& val : valueList) {
                    size_t size = parseSizeString(trim(val));
                    if (size > 0) sizes.push_back(size);
                }
                if (!sizes.empty()) outMemorySizes[levelName] = sizes;
            }
            else if (paramType == "block_sizes") {
                std::vector<int> sizes;
                for (const auto& val : valueList) {
                    try {
                        int size = std::stoi(trim(val));
                        if (size > 0) sizes.push_back(size);
                    } catch (...) {}
                }
                if (!sizes.empty()) outBlockSizes[levelName] = sizes;
            }
            else if (paramType == "iterations") {
                try {
                    outIterations[levelName] = std::stoi(trim(values));
                } catch (...) {}
            }
            else if (paramType == "warmups") {
                try {
                    outWarmups[levelName] = std::stoi(trim(values));
                } catch (...) {}
            }
        }
        
        return !outMemorySizes.empty() || !outBlockSizes.empty();
    }

private:
    static size_t parseSizeString(const std::string& str) {
        if (str.empty()) return 0;
        
        std::string numStr;
        char suffix = 0;
        
        for (char c : str) {
            if (std::isdigit(c) || c == '.') {
                numStr += c;
            } else if (std::isalpha(c)) {
                suffix = std::toupper(c);
                break;
            }
        }
        
        if (numStr.empty()) return 0;
        
        double value;
        try {
            value = std::stod(numStr);
        } catch (...) {
            return 0;
        }
        
        size_t multiplier = 1;
        switch (suffix) {
            case 'K': multiplier = 1024; break;
            case 'M': multiplier = 1024 * 1024; break;
            case 'G': multiplier = 1024 * 1024 * 1024; break;
            case 0:   multiplier = 1; break;
            default:  return 0;
        }
        
        return static_cast<size_t>(value * multiplier);
    }
    
    static std::string trim(const std::string& str) {
        size_t start = 0;
        size_t end = str.length();
        
        while (start < end && std::isspace(str[start])) start++;
        while (end > start && std::isspace(str[end - 1])) end--;
        
        return str.substr(start, end - start);
    }
    
    static bool fileExists(const std::string& path) {
        std::ifstream file(path);
        return file.good();
    }
    
    static std::vector<std::string> splitByComma(const std::string& str) {
        std::vector<std::string> result;
        std::string current;
        
        for (char c : str) {
            if (c == ',') {
                if (!current.empty()) {
                    result.push_back(current);
                    current.clear();
                }
            } else {
                current += c;
            }
        }
        
        if (!current.empty()) result.push_back(current);
        return result;
    }
};

