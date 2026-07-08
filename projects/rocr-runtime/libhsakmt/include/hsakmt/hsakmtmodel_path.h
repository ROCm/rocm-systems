/*
 * Copyright © 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef HSAKMTMODEL_PATH_H_
#define HSAKMTMODEL_PATH_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static inline bool hsakmt_path_contains_dotdot(const char *path)
{
	const char *p = path;

	while (*p) {
		if (p[0] == '.' && p[1] == '.' &&
		    (p[2] == '\0' || p[2] == '/'))
			return true;
		p = strchr(p, '/');
		if (!p)
			break;
		p++;
	}
	return false;
}

static inline bool hsakmt_model_lib_path_allowed(const char *libname,
						 const char *prefix,
						 char *resolved,
						 size_t resolved_size)
{
	size_t prefix_len = strlen(prefix);

	if (!libname || !*libname || !prefix || !*prefix ||
	    hsakmt_path_contains_dotdot(libname))
		return false;

	if (libname[0] == '/') {
		if (strncmp(libname, prefix, prefix_len) != 0)
			return false;
		if (libname[prefix_len] != '\0' && libname[prefix_len] != '/')
			return false;
		return snprintf(resolved, resolved_size, "%s", libname) <
		       (int)resolved_size;
	}

	return snprintf(resolved, resolved_size, "%s/%s", prefix, libname) <
	       (int)resolved_size;
}

#endif  // HSAKMTMODEL_PATH_H_
