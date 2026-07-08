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

#include <assert.h>
#include <string.h>

#include "hsakmt/hsakmtmodel_path.h"

int main(void)
{
	char resolved[256];
	const char *prefix = "/opt/rocm/lib";

	assert(hsakmt_path_contains_dotdot("../evil.so"));
	assert(hsakmt_path_contains_dotdot("/opt/rocm/lib/../evil.so"));
	assert(!hsakmt_path_contains_dotdot("/opt/rocm/lib/model.so"));

	assert(hsakmt_model_lib_path_allowed("model.so", prefix, resolved,
					     sizeof(resolved)));
	assert(strcmp(resolved, "/opt/rocm/lib/model.so") == 0);

	assert(hsakmt_model_lib_path_allowed("/opt/rocm/lib/model.so", prefix,
					     resolved, sizeof(resolved)));
	assert(strcmp(resolved, "/opt/rocm/lib/model.so") == 0);

	assert(!hsakmt_model_lib_path_allowed("/tmp/evil.so", prefix, resolved,
					      sizeof(resolved)));
	assert(!hsakmt_model_lib_path_allowed("../evil.so", prefix, resolved,
					      sizeof(resolved)));
	assert(!hsakmt_model_lib_path_allowed("/opt/rocm-evil/lib/model.so",
					      prefix, resolved,
					      sizeof(resolved)));

	return 0;
}
