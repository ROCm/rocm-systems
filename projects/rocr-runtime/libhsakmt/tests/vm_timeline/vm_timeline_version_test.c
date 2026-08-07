/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 */

#include "drm_version.h"

int main(void)
{
	if (hsakmt_drm_supports_vm_timeline(2, 99) ||
	    hsakmt_drm_supports_vm_timeline(3, 63) ||
	    !hsakmt_drm_supports_vm_timeline(3, 64) ||
	    !hsakmt_drm_supports_vm_timeline(3, 65) ||
	    !hsakmt_drm_supports_vm_timeline(4, 0))
		return 1;

	return 0;
}
