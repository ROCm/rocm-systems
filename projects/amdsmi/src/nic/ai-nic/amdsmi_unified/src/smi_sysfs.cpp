/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "smi_sysfs.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

SmiSysfsReader::SysfsStatus SmiSysfsReader::readAll(const std::string& filepath, std::vector<SysfsValue>& content)
{
	std::ifstream file(filepath);
	std::string line;

	if (!file.is_open()) {
		return SmiSysfsReader::SysfsStatus::FileNotFound;
	}

	if (!SmiSysfsReader::exists(filepath)) {
		return SmiSysfsReader::SysfsStatus::IOError;
	}

	content.clear();
	while (std::getline(file, line)) {
		if (line.find(' ') != std::string::npos) {
			content.push_back(line);
			continue;
		}
		std::stringstream ss(line);
		std::string token;
		while (ss >> token) {
			try {
				if (token.find("0x") == 0 || token.find("0X") == 0) {
					int hex_value = std::stoi(token, nullptr, 16);
					content.push_back(hex_value);
				} else if (std::all_of(token.begin(), token.end(), ::isdigit)) {
					content.push_back(std::stoi(token));
				} else {
					content.push_back(token);
				}
			} catch (const std::invalid_argument&) {
				return SmiSysfsReader::SysfsStatus::ParseError;
			} catch (const std::out_of_range&) {
				return SmiSysfsReader::SysfsStatus::ParseError;
			}
		}
	}

	return SmiSysfsReader::SysfsStatus::Success;
}

SmiSysfsReader::SysfsStatus SmiSysfsReader::readLine(const std::string& filepath, SmiSysfsReader::SysfsValue& content)
{
	std::ifstream file(filepath);
	std::string line;

	if (!file.is_open()) {
		return SmiSysfsReader::SysfsStatus::FileNotFound;
	}

	if (!SmiSysfsReader::exists(filepath)) {
		return SmiSysfsReader::SysfsStatus::IOError;
	}

	if (std::getline(file, line)) {
		if (line.find(' ') != std::string::npos) {
			content = line;
			return SmiSysfsReader::SysfsStatus::Success;
		}
		std::stringstream ss(line);
		std::string token;
		if (ss >> token) {
			try {
				if (token.find("0x") == 0 || token.find("0X") == 0) {
					int hex_value = std::stoi(token, nullptr, 16);
					content = hex_value;
				} else if (std::all_of(token.begin(), token.end(), ::isdigit)) {
					content = std::stoi(token);
				} else {
					content = token;
				}
				return SmiSysfsReader::SysfsStatus::Success;
			} catch (...) {
				return SmiSysfsReader::SysfsStatus::ParseError;
			}
		}
	}

	return SmiSysfsReader::SysfsStatus::Success;
}

bool SmiSysfsReader::exists(const std::string& filepath)
{
	std::ifstream file(filepath);
	return file.good();
}
