#include <stdio.h>
#include "oam/oam_mapi.h"
#include "oam/amd_oam.h"

const oam_ops_t amd_oam_ops = {
	.init = amdoam_init,
	.free = amdoam_free,
//	.get_mapi_version = amdoam_get_mapi_version,
	.discover_devices = amdoam_discover_devices,
};

int main()
{
	uint32_t dev_cnt = 0;
	oam_mapi_version_t version;

	if (amd_oam_ops.init(version)) {
		printf("init failed\n");
		return -1;
	}

//	amd_oam_ops.get_mapi_version(&version);
	if (!amd_oam_ops.discover_devices(&dev_cnt))
		printf("%d AMD devices are discovered\n", dev_cnt);

	amd_oam_ops.free();

	return 0;
}

