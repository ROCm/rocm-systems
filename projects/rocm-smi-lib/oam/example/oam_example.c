#include <stdio.h>
#include "oam/oam_mapi.h"
#include "oam/amd_oam.h"

const oam_ops_t amd_oam_ops = {
	.init = amdoam_init,
	.free = amdoam_free,
//	.get_mapi_version = amdoam_get_mapi_version,
	.discover_devices = amdoam_discover_devices,
	.get_dev_properties = amdoam_get_dev_properties,
	.get_pci_properties = amdoam_get_pci_properties,
	.get_sensors_count = amdoam_get_sensors_count,
	.get_error_description = amdoam_get_error_description,
};

int main()
{
	uint32_t dev_cnt = 0;
	oam_mapi_version_t version;
	oam_dev_properties_t *devs_prop;
	int i;
	oam_pci_info_t pci_info;
	oam_sensor_count_t sensor_count;
	const char *string;

	if (amd_oam_ops.init()) {
		printf("init failed\n");
		return -1;
	}

//	amd_oam_ops.get_mapi_version(&version);
	if (!amd_oam_ops.discover_devices(&dev_cnt))
		printf("%d AMD devices are discovered\n", dev_cnt);
	if (!dev_cnt) {
		printf("No devices are  found.\n");
		return -1;
	}
	devs_prop = calloc(dev_cnt, sizeof(oam_dev_properties_t));
	if (!devs_prop) {
		printf("Allocating dev_prop failed\n");
		return -1;
	}

	amd_oam_ops.get_dev_properties(dev_cnt, devs_prop);
	for (i = 0; i < dev_cnt; i++) {
		printf("Device %d:\n", i);
		printf("   device id %d\n", devs_prop[i].device_id);
		printf("   device_vendor %s\n", devs_prop[i].device_vendor);
		printf("   device_name %s\n", devs_prop[i].device_name);
		printf("   sku_name %s\n", devs_prop[i].sku_name);
		printf("   board_name %s\n", devs_prop[i].board_name);
		printf("   board_revision %s\n", devs_prop[i].board_revision);
		printf("   board_serial_number %s\n",
					devs_prop[i].board_serial_number);

		if (!amd_oam_ops.get_pci_properties(devs_prop[i].device_id, &pci_info)){
			printf("   PCI Domain : 0x%d \n", pci_info.domain);
			printf("   PCI bus : 0x%d \n", pci_info.bus);
			printf("   PCI device : 0x%d \n", pci_info.device);
			printf("   PCI function : 0x%d \n", pci_info.function);
		}
		if(!amd_oam_ops.get_sensors_count(devs_prop[i].device_id, &sensor_count)) {
			printf("   Temperature Sensors : %d \n", sensor_count.num_temperature_sensors);
			printf("   Power Sensors : %d \n", sensor_count.num_power_sensors);
			printf("   Voltage Sensors : %d \n", sensor_count.num_voltage_sensors);
			printf("   Current Sensors : %d \n", sensor_count.num_current_sensors);
			printf("   Fan Sensors : %d \n", sensor_count.num_fans);
		}
	}

	amd_oam_ops.get_error_description(1, &string);
	printf("error code 1: %s\n", string);

	free(devs_prop);
	amd_oam_ops.free();

	return 0;
}

