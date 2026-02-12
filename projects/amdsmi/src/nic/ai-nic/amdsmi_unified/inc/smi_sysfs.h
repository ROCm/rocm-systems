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

#ifndef __SMI_SYSFS_H__
#define __SMI_SYSFS_H__

#include <vector>
#include <string>
#include <variant>

class SmiSysfsReader {
public:
	using SysfsValue = std::variant<int, std::string>;
	enum class SysfsStatus {
		Success = 0,
		FileNotFound,
		IOError,
		ParseError
	};

	static SysfsStatus readAll(const std::string& filepath, std::vector<SysfsValue>& content);
	static SysfsStatus readLine(const std::string& filepath, SysfsValue& content);
	static bool exists(const std::string& filepath);

	SmiSysfsReader() = delete;
};

#endif // __SMI_SYSFS_H__
